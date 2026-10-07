// (fork) weight prefetch into the Infinity Cache (RDNA2: 96 MB memory-side cache, ~1 TB/s on hits vs ~370 GB/s DRAM).
// While the CPU computes a layer's experts the GPU is idle once the overlap prefix is done; this kernel, queued right
// after the prefix, reads the next split's weights in the order they are used so the following kernels find them in
// the cache. It only loads (results discarded): no computed value changes. It stops as soon as the host signals that
// the CPU split is done (pinned host flag), so it never delays the next split by more than one chunk.

#include "common.cuh"

#define PF_MAXR 48
#define PF_NT   256

struct pf_args {
    const char * p[PF_MAXR];
    unsigned long long end[PF_MAXR];   // cumulative byte offsets (multiples of 16)
    int n;
    unsigned int seq;
    const volatile unsigned int * stop;
    int * sink;
    unsigned long long * count;   // LLAMA_GPU_PREFETCH_LOG: chunks read (pinned host counter), else null
};

#define PF_U 16   // 16-byte loads per thread per chunk: 64 KB per workgroup and chunk

static __global__ void __launch_bounds__(PF_NT) pf_kernel(const pf_args a) {
    __shared__ int quit;
    const unsigned long long total = a.end[a.n - 1];
    const unsigned long long chunk = PF_NT * 16 * PF_U;
    int4 acc = make_int4(0, 0, 0, 0);
    int r = 0;
    unsigned long long done = 0;
    if (threadIdx.x == 0) quit = 0;
    __syncthreads();
    for (unsigned long long off = (unsigned long long) blockIdx.x * chunk; off < total; off += (unsigned long long) gridDim.x * chunk) {
        // the stop flag (host memory, slow to read) is fetched alongside this chunk's loads and acted on after them
        unsigned int flag = 0;
        if (threadIdx.x == 0) flag = __hip_atomic_load(a.stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        while (r < a.n - 1 && off >= a.end[r]) r++;
        const unsigned long long base = r ? a.end[r - 1] : 0;
        const unsigned long long lim  = a.end[r];
#pragma unroll
        for (int k = 0; k < PF_U; k++) {
            const unsigned long long o = off + (unsigned long long) (k * PF_NT + threadIdx.x) * 16;
            if (o < lim) {
                const int4 v = *(const int4 *) (a.p[r] + (o - base));
                acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
            }
        }
        done++;
        if (threadIdx.x == 0 && (int) (flag - a.seq) >= 0) quit = 1;
        __syncthreads();
        if (quit) break;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7f3a5c21) a.sink[0] = 1;   // keeps the loads
    if (a.count && threadIdx.x == 0) __hip_atomic_fetch_add(a.count, done * PF_U / 4, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// LLAMA_GPU_PREFETCH_STREAM=1: the prefetch runs on its own stream, concurrently with the overlap prefix
extern "C" int ggml_cuda_fork_prefetch_concurrent(void) {
    static const int on = [] { const char * e = getenv("LLAMA_GPU_PREFETCH_STREAM"); return e ? atoi(e) : 0; }();
    return on;
}

static unsigned int * g_pf_stop = nullptr;
static int * g_pf_sink = nullptr;
static unsigned int g_pf_seq = 0;

// queue a prefetch of the given device ranges on the backend's stream; returns its sequence number
extern "C" unsigned int ggml_cuda_fork_prefetch(void * backend_ctx, int n, const void * const * ptrs, const size_t * sizes) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend_ctx;
    if (!g_pf_stop) {
        CUDA_CHECK(cudaMallocHost((void **) &g_pf_stop, sizeof(unsigned int)));
        *(volatile unsigned int *) g_pf_stop = 0;
        CUDA_CHECK(cudaMalloc((void **) &g_pf_sink, sizeof(int)));
    }
    pf_args a = {};
    unsigned long long tot = 0;
    for (int i = 0; i < n && a.n < PF_MAXR; i++) {
        const unsigned long long sz = sizes[i] & ~15ull;
        if (sz == 0) continue;
        a.p[a.n] = (const char *) ptrs[i];
        tot += sz;
        a.end[a.n++] = tot;
    }
    const unsigned int seq = ++g_pf_seq;
    if (a.n == 0) return seq;
    static unsigned long long * cnt = nullptr; static unsigned long long req = 0, calls = 0;
    static const bool log = getenv("LLAMA_GPU_PREFETCH_LOG") != nullptr;
    if (log && !cnt) { CUDA_CHECK(cudaMallocHost((void **) &cnt, sizeof(*cnt))); *cnt = 0; }
    a.count = cnt;
    if (log) {
        req += tot; calls++;
        if (calls % 4800 == 0) {
            fprintf(stderr, "[prefetch] per call: requested %.1f MB, read %.1f MB\n", req / 4800.0 / 1e6, __atomic_load_n(cnt, __ATOMIC_RELAXED) * (double) (PF_NT*16*4) / 4800.0 / 1e6);
            req = 0; __atomic_store_n(cnt, 0ull, __ATOMIC_RELAXED);
        }
    }
    a.seq  = seq;
    a.stop = g_pf_stop;
    a.sink = g_pf_sink;
    static const int grid = [] { const char * e = getenv("LLAMA_GPU_PREFETCH_GRID"); return e ? atoi(e) : 40; }();
    if (ggml_cuda_fork_prefetch_concurrent()) {
        // on its own low-priority stream, after the work queued so far (the previous split): runs alongside the
        // overlap prefix that is queued next on the main stream
        static cudaStream_t st = nullptr; static cudaEvent_t ev = nullptr;
        if (!st) {
            int lo, hi;
            CUDA_CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(hipStreamCreateWithPriority(&st, cudaStreamNonBlocking, lo));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(ev, ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(st, ev, 0));
        pf_kernel<<<grid, PF_NT, 0, st>>>(a);
    } else {
        pf_kernel<<<grid, PF_NT, 0, ctx->stream()>>>(a);
    }
    return seq;
}

// the CPU split is done: running prefetches with a sequence number <= seq stop at their next chunk
extern "C" void ggml_cuda_fork_prefetch_stop(unsigned int seq) {
    static const bool nostop = getenv("LLAMA_GPU_PREFETCH_NOSTOP") != nullptr;   // debug: always prefetch everything
    if (g_pf_stop && !nostop) __atomic_store_n(g_pf_stop, seq, __ATOMIC_RELEASE);
}
