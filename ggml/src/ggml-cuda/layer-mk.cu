// (fork) layer megakernel: a run of single-token graph nodes executed as stages of ONE persistent kernel. Workgroups
// stay resident and stages are separated by a grid barrier (~0.5 us) instead of a kernel boundary (~2.6 us in a HIP
// graph). Every stage computes exactly the values of the kernels it replaces: same per-thread / per-lane expressions,
// same accumulation order, same reduction trees (a stage may run the original blocks as virtual blocks, or one wave
// per row with the same per-lane chain). The stage list travels in the kernel arguments (captured by the HIP graph).
//
// Matched runs (decode, n_tokens = 1):
//   HC chain   [DSV4_HC_POST ->] RMS_NORM -> MUL -> MUL_MAT -> SCALE -> SILU -> MUL_MAT -> DSV4_HC_PRE
//   linear attention (after an HC chain): conv state, qkv/z/alpha/beta products, SSM_CONV+SILU, q/k norms, alpha/beta
//   gates, GATED_DELTA_NET (state read and written in the one-cell cache), gated norm, output product, hc inject

#include "layer-mk.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <vector>
#include <algorithm>

static std::vector<const ggml_tensor *> g_mk_check;   // LLAMA_MK_CHECK (debug)

#define MK_NT        256
#define MK_NW        (MK_NT/32)
#define MK_MAXST     80
#define MK_MAXQB     32               // q8_1 blocks of the hc up input (K = 320 -> 10)

enum mk_op : int {
    MK_HC_POST, MK_RMS_BIG, MK_QUANT, MK_MATVEC, MK_SILU_UP, MK_HC_PRE,
    MK_LIN_CONV, MK_ALPHA, MK_SIGMOID, MK_SSS, MK_RMS_SMALL, MK_GDN, MK_COPY,
};

struct mk_stage {
    int op, bar;                      // bar: grid barrier after this stage
    int n0, n1, n2, n3;
    float f0, f1, f2, f3;
    const void * a; const void * b; const void * c; const void * d; const void * e;
    void * x; void * y;
};

struct mk_prog {
    int n;
    unsigned * bar;
    unsigned long long * ts;          // LLAMA_MK_TS: per-stage end times (workgroup 0)
    mk_stage s[MK_MAXST];
};
static_assert(sizeof(mk_prog) <= 8192, "kernel argument size (ROCm takes up to 32 KB)");

// grid barrier: arrival counter + generation, self-resetting; release/acquire at agent scope (L0/L1 invalidated)
static __device__ __forceinline__ void mk_grid_sync(unsigned * bar) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        const unsigned gen = __hip_atomic_load(&bar[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned arrived = __hip_atomic_fetch_add(&bar[0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == gridDim.x - 1) {
            __hip_atomic_store(&bar[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(&bar[1], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            while (__hip_atomic_load(&bar[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}

// ---------------------------------------------------------------- stages

// dsv4_hc_post_f32<false>: dst[i0, k] = x[i0]*post[k] + res[i0, k]
static __device__ void mk_hc_post(const mk_stage & s) {
    const float * x = (const float *) s.a, * res = (const float *) s.b, * post = (const float *) s.c;
    float * dst = (float *) s.x;
    const int n = s.n0, ne = s.n0*s.n1;
    for (int ir = blockIdx.x*MK_NT + threadIdx.x; ir < ne; ir += gridDim.x*MK_NT) {
        const int i0 = ir % n, idst = ir / n;
        float sum = x[i0] * post[idst];
        sum += res[i0 + idst*n];
        dst[i0 + idst*n] = sum;
    }
}

// rms_norm_f32<1024, true> (1024 < n <= 3072): one 1024-thread virtual block per row, emulated with 4 virtual threads
// per thread (v = tid + 256*j, virtual warp 8*j + wave); columns kept in registers between the two passes
static __device__ void mk_rms_big(const mk_stage & s, float * s_red) {
    const int tid = threadIdx.x, lane = tid % 32, wv = tid / 32, n = s.n0;
    for (int row = blockIdx.x; row < s.n1; row += gridDim.x) {
        const float * x = (const float *) s.a + row*n;
        const float * w = (const float *) s.b + row*n;
        float xv[4][3], wv3[4][3], tmp[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const int col = tid + 256*j + 1024*c;
                xv[j][c]  = col < n ? x[col] : 0.0f;
                wv3[j][c] = col < n ? w[col] : 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            tmp[j] = 0.0f;
#pragma unroll
            for (int c = 0; c < 3; c++) {
                if (tid + 256*j + 1024*c < n) {
                    const float xi = xv[j][c];
                    tmp[j] += xi * xi;
                }
            }
            tmp[j] = warp_reduce_sum(tmp[j]);
        }
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < 4; j++) s_red[8*j + wv] = tmp[j];
        }
        __syncthreads();
        float t = s_red[lane];
        t = warp_reduce_sum(t);
        const float mean  = t / n;
        const float scale = rsqrtf(mean + s.f0);
        float * dst = (float *) s.x + row*n;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const int col = tid + 256*j + 1024*c;
                if (col < n) {
                    dst[col] = scale * xv[j][c] * wv3[j][c];
                }
            }
        }
        __syncthreads();
    }
}

// quantize_q8_1, one wave per 32-value block
static __device__ void mk_quant(const mk_stage & s) {
    const float * x = (const float *) s.a;
    block_q8_1 * y = (block_q8_1 *) s.x;
    for (int i0 = blockIdx.x*MK_NT + threadIdx.x; i0 < s.n0; i0 += gridDim.x*MK_NT) {
        const float xi = x[i0];
        float amax = fabsf(xi);
        float sum = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        y[i0 / QK8_1].qs[i0 % QK8_1] = q;
        if (i0 % QK8_1 == 0) {
            y[i0 / QK8_1].ds = make_half2(d, sum);
        }
    }
}

// q8_0 x q8_1 matrix-vector product, the values of mul_mat_vec_q<Q8_0, 1> on RDNA2 (one wave per row; also of its
// k-split variant): per lane the contracted fma chain over its blocks in order, then the warp reduction
static __device__ void mk_matvec(const mk_stage & s) {
    constexpr int qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ, bpi = vdr*32/qi, U = 8;
    const int lane = threadIdx.x % 32, gw = blockIdx.x*MK_NW + threadIdx.x/32;
    const int bpr = s.n1 / QK8_0;
    const int kbx0 = lane / (qi/vdr), kqs = vdr * (lane % (qi/vdr));
    const int niter = (bpr - kbx0 + bpi - 1) / bpi;
    const block_q8_1 * y = (const block_q8_1 *) s.b;
    float * dst = (float *) s.x;
    for (int row = gw; row < s.n0; row += gridDim.x*MK_NW) {
        const block_q8_0 * x = (const block_q8_0 *) s.a + (size_t) row*bpr;
        float tmp = 0.0f;
        for (int i0 = 0; i0 < niter; i0 += U) {
            float f[U], sv[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (i0 + u < niter) {
                    const int kbx = kbx0 + (i0 + u)*bpi;
                    int sumi = 0;
#pragma unroll
                    for (int v = 0; v < vdr; ++v) {
                        sumi = ggml_cuda_dp4a(get_int_b2(x[kbx].qs, kqs + v), get_int_b4(y[kbx].qs, kqs + v), sumi);
                    }
                    const float d8_0 = x[kbx].d;
                    const float d8_1 = __low2half(y[kbx].ds);
                    f[u]  = d8_0*d8_1;
                    sv[u] = (float) sumi;
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (i0 + u < niter) {
                    tmp = __fmaf_rn(f[u], sv[u], tmp);
                }
            }
        }
        tmp = warp_reduce_sum<32>(tmp);
        if (lane == 0) {
            dst[row] = tmp;
        }
    }
}

// scale + silu + q8_1 (mul_mat_vec_q8_0_multirow_scale_silu's prologue, once per workgroup), then its multirow loop
static __device__ void mk_silu_up(const mk_stage & s, block_q8_1 * ys) {
    const int tid = threadIdx.x, lane = tid % 32, wv = tid / 32;
    const float * lo = (const float *) s.a;
    const int n_lo = s.n0, nqb = n_lo / QK8_1;
    for (int b = wv; b < nqb; b += MK_NW) {
        const int i = b*QK8_1 + lane;
        float xi = 0.0f;
        if (i < n_lo) {
            const float y = s.f0 * lo[i] + s.f1;
            xi = ggml_cuda_op_silu_single(y);
        }
        float amax = fabsf(xi);
        float sum = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        ys[b].qs[lane] = q;
        if (lane == 0) {
            ys[b].ds = make_half2(d, sum);
        }
    }
    __syncthreads();
    constexpr int R = 4, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ, bpi = vdr*32/qi;
    const int kqs = vdr * (lane % (qi/vdr));
    float * gate = (float *) s.x;
    for (int row0 = (blockIdx.x*MK_NW + wv)*R; row0 < s.n1; row0 += gridDim.x*MK_NW*R) {
        float tmp[R] = {0.0f};
        for (int kbx = lane / (qi/vdr); kbx < nqb; kbx += bpi) {
            const float d8_1 = __low2half(ys[kbx].ds);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const block_q8_0 * x = (const block_q8_0 *) s.b + (size_t) (row0 + r)*nqb + kbx;
                int sumi = 0;
#pragma unroll
                for (int v = 0; v < vdr; ++v) {
                    sumi = ggml_cuda_dp4a(get_int_b2(x->qs, kqs + v), get_int_b4(ys[kbx].qs, kqs + v), sumi);
                }
                const float d8_0 = x->d;
                tmp[r] = __fmaf_rn(d8_0*d8_1, (float) sumi, tmp[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            tmp[r] = warp_reduce_sum<32>(tmp[r]);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                gate[row0 + r] = tmp[r];
            }
        }
    }
    __syncthreads();   // ys is reused by a later stage of this workgroup
}

// dsv4_hc_pre_f32<true>, one token
static __device__ void mk_hc_pre(const mk_stage & s) {
    const float * xn = (const float *) s.a, * gate = (const float *) s.b;
    float * mixed = (float *) s.x;
    for (int i0 = blockIdx.x*MK_NT + threadIdx.x; i0 < s.n0; i0 += gridDim.x*MK_NT) {
        float sum = 0.0f;
        for (int ih = 0; ih < s.n1; ++ih) {
            const float xv = xn[i0 + ih*s.n0];
            float wv;
            wv = 1.0f / (1.0f + expf(-gate[i0 + ih*s.n0]));
            sum += xv * wv;
        }
        mixed[i0] = s.f0 * sum;
    }
}

// conv state of one token: concat(state[c, 0..2], qkv[c]) -> the new state [1..3] written back in place by the thread
// that read it; ssm_conv_f32<silu> (d_conv 4, n_t 1): sumf += x[j]*w[j], sumf += b, silu
static __device__ void mk_lin_conv(const mk_stage & s) {
    const float * qkv = (const float *) s.a, * w = (const float *) s.c;
    float * st = (float *) s.b, * out = (float *) s.x;
    const float b = s.f0;
    for (int c = blockIdx.x*MK_NT + threadIdx.x; c < s.n0; c += gridDim.x*MK_NT) {
        float x[4];
        x[0] = st[3*c + 0]; x[1] = st[3*c + 1]; x[2] = st[3*c + 2]; x[3] = qkv[c];
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            sumf += x[j] * w[4*c + j];
        }
        sumf += b;
        out[c] = ggml_cuda_op_silu_single(sumf);
        st[3*c + 0] = x[1]; st[3*c + 1] = x[2]; st[3*c + 2] = x[3];
    }
}

// alpha gate: ADD (alpha + dt bias), then the fused SOFTPLUS*MUL (softplus(x) * a)
static __device__ void mk_alpha(const mk_stage & s) {
    const float * al = (const float *) s.a, * bias = (const float *) s.b, * aa = (const float *) s.c;
    float * g = (float *) s.x;
    for (int i = blockIdx.x*MK_NT + threadIdx.x; i < s.n0; i += gridDim.x*MK_NT) {
        const float t = al[i] + bias[i];
        const float sp = (t > 20.0f) ? t : logf(1.0f + expf(t));
        g[i] = sp * aa[i];
    }
}

static __device__ void mk_sigmoid(const mk_stage & s) {
    const float * x = (const float *) s.a;
    float * y = (float *) s.x;
    for (int i = blockIdx.x*MK_NT + threadIdx.x; i < s.n0; i += gridDim.x*MK_NT) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// scale_unary_scale_f32<op_sigmoid, true>
static __device__ void mk_sss(const mk_stage & s) {
    const float * x = (const float *) s.a;
    float * dst = (float *) s.x;
    for (int i = blockIdx.x*MK_NT + threadIdx.x; i < s.n0; i += gridDim.x*MK_NT) {
        float y = s.f0 * x[i] + s.f1;
        y = 1.0f / (1.0f + expf(-y));
        y = s.f2 * y + s.f3;
        dst[i] = y;
    }
}

// rms_norm_f32<256, ...> for rows of n <= 256 columns: one workgroup per row running the original code (block_reduce)
// n2 = 0: fused SCALE: dst = f1 * (scale * x);   n2 = 1: fused MUL by w (b), then SIGMOID(z)*that (c = z)
static __device__ void mk_rms_small(const mk_stage & s, float * s_sum) {
    const int tid = threadIdx.x, n = s.n0;
    for (int row = blockIdx.x; row < s.n1; row += gridDim.x) {
        const float * x = (const float *) s.a + row*n;
        float tmp = 0.0f;
        for (int col = tid; col < n; col += 256) {
            const float xi = x[col];
            tmp += xi * xi;
        }
        tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, s_sum);
        const float mean = tmp / n;
        const float scale = rsqrtf(mean + s.f0);
        float * dst = (float *) s.x + row*n;
        for (int col = tid; col < n; col += 256) {
            if (s.n2 == 0) {
                dst[col] = s.f1 * (scale * x[col]);
            } else {
                const float nm = scale * x[col] * ((const float *) s.b)[col];
                const float z  = ((const float *) s.c)[row*n + col];
                dst[col] = (1.0f / (1.0f + expf(-z))) * nm;
            }
        }
        __syncthreads();   // s_sum reused by the next row
    }
}

// gated_delta_net_cuda<128, false, false>, one token, one sequence: one wave per (head, column), state in place
static __device__ void mk_gdn(const mk_stage & s) {
    constexpr int S_v = 128, warp_size = 32, rows_per_lane = S_v / warp_size;
    const int lane = threadIdx.x % 32, gw = blockIdx.x*MK_NW + threadIdx.x/32;
    const int H = s.n0, Hqk = s.n1;
    const float * q = (const float *) s.a, * k = (const float *) s.b, * v = (const float *) s.c;
    const float * g = (const float *) s.d, * beta = (const float *) s.e;
    float * attn = (float *) s.x, * state = (float *) s.y;
    const float scale = s.f0;
    for (int u = gw; u < H*S_v; u += gridDim.x*MK_NW) {
        const int h_idx = u / S_v, col = u % S_v;
        const int iq1 = h_idx % Hqk;
        float * st = state + (size_t) h_idx*S_v*S_v + col*S_v;
        float s_shard[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = st[r*warp_size + lane];
        }
        const float * q_t = q + iq1*S_v;
        const float * k_t = k + iq1*S_v;
        const float * v_t = v + h_idx*S_v;
        const float beta_val = beta[h_idx];
        float k_reg[rows_per_lane], q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r*warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
        const float g_val = expf(g[h_idx]);
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col = warp_reduce_sum<warp_size>(kv_shard);
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        float attn_col = warp_reduce_sum<warp_size>(attn_partial);
        if (lane == 0) {
            attn[h_idx*S_v + col] = attn_col * scale;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            st[r*warp_size + lane] = s_shard[r];
        }
    }
}

static __device__ void mk_copy(const mk_stage & s) {   // n0 floats
    const float * a = (const float *) s.a;
    float * x = (float *) s.x;
    for (int i = blockIdx.x*MK_NT + threadIdx.x; i < s.n0; i += gridDim.x*MK_NT) {
        x[i] = a[i];
    }
}

__launch_bounds__(MK_NT, 1)
static __global__ void layer_mk_kernel(const mk_prog p) {
    __shared__ float s_red[32];
    __shared__ block_q8_1 ys[MK_MAXQB];
    if (p.ts && blockIdx.x == 0 && threadIdx.x == 0) p.ts[0] = wall_clock64();
    for (int k = 0; k < p.n; k++) {
        const mk_stage & s = p.s[k];
        switch (s.op) {
            case MK_HC_POST:   mk_hc_post(s);        break;
            case MK_RMS_BIG:   mk_rms_big(s, s_red); break;
            case MK_QUANT:     mk_quant(s);          break;
            case MK_MATVEC:    mk_matvec(s);         break;
            case MK_SILU_UP:   mk_silu_up(s, ys);    break;
            case MK_HC_PRE:    mk_hc_pre(s);         break;
            case MK_LIN_CONV:  mk_lin_conv(s);       break;
            case MK_ALPHA:     mk_alpha(s);          break;
            case MK_SIGMOID:   mk_sigmoid(s);        break;
            case MK_SSS:       mk_sss(s);            break;
            case MK_RMS_SMALL: mk_rms_small(s, s_red); break;
            case MK_GDN:       mk_gdn(s);            break;
            case MK_COPY:      mk_copy(s);           break;
        }
        if (s.bar) {
            mk_grid_sync(p.bar);
            if (p.ts && blockIdx.x == 0 && threadIdx.x == 0) p.ts[1 + k] = wall_clock64();
        }
    }
}

// ------------------------------------------------------------------ graph matching

namespace {

struct mk_walker {
    const ggml_cgraph * g;
    int j;
    // next node that does work (views, no-ops and empty tensors skipped); nullptr at the end
    const ggml_tensor * next() {
        for (++j; j < g->n_nodes; ++j) {
            const ggml_tensor * n = g->nodes[j];
            if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
                n->op == GGML_OP_NONE || ggml_nelements(n) == 0) {
                continue;
            }
            return n;
        }
        return nullptr;
    }
};

bool mk_is(const ggml_tensor * t, const ggml_tensor * base) {   // t is base or a view of it
    for (; t; t = t->view_src) if (t == base) return true;
    return false;
}

bool mk_q8_0_mat(const ggml_tensor * w, int64_t k) {
    return w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == k && w->ne[2] == 1 && w->ne[3] == 1 &&
           ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

bool mk_vec(const ggml_tensor * t, int64_t n) {   // contiguous f32 with n elements
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
}

const float * fdata(const ggml_tensor * t) { return (const float *) t->data; }

// Values computed by the stages live in the scratch buffer (the graph allocator may give overlapping memory to tensors
// whose lifetimes do not overlap in the original node order, and the stages run in another order). in(t) is where a
// stage reads t (scratch if produced here, else its graph data); out(t) reserves t's scratch copy and queues its write
// back to the graph data, done at the end in the original node order (so the final memory is the original's).
struct mk_wb { int j; const ggml_tensor * t; void * src; size_t bytes; };
struct mk_builder {
    std::vector<mk_stage> st;
    char * scratch;
    size_t used = 0;
    std::vector<std::pair<const ggml_tensor *, void *>> val;
    std::vector<mk_wb> wb;
    void * alloc(size_t bytes) { void * r = scratch + used; used += GGML_PAD(bytes, 256); return r; }
    mk_stage & add(int op, int bar = 1) { st.push_back({}); st.back().op = op; st.back().bar = bar; return st.back(); }
    void * in(const ggml_tensor * t) {
        size_t off = 0;
        const ggml_tensor * b = t;
        for (; b; b = b->view_src) {
            for (auto & v : val) if (v.first == b) return (char *) v.second + off;
            off += b->view_src ? b->view_offs : 0;
        }
        return t->data;
    }
    // node j produces t; bytes: what the original kernel writes (written back), 0: not materialized originally
    void * out(int j, const ggml_tensor * t, size_t bytes, size_t alloc_bytes = 0) {
        void * p = alloc(alloc_bytes ? alloc_bytes : ggml_nbytes(t));
        val.push_back({t, p});
        if (bytes) wb.push_back({j, t, p, bytes});
        return p;
    }
};

// HC chain starting at node i (HC_POST or RMS_NORM); on success *walker is at HC_PRE, outputs returned
struct hc_out { const ggml_tensor * xn = nullptr; const ggml_tensor * mixed = nullptr; void * xq = nullptr; };

bool mk_match_hc(mk_walker & w, const ggml_tensor * first, mk_builder & B, hc_out & o) {
    const ggml_tensor * post = nullptr, * n = first;
    const int j_post = w.j;
    if (n->op == GGML_OP_DSV4_HC_POST) {
        post = n;
        n = w.next();
        if (!n) return false;
    }
    const ggml_tensor * rms = n;
    if (rms->op != GGML_OP_RMS_NORM || (post && rms->src[0] != post)) return false;
    const ggml_tensor * mul  = w.next(); if (!mul  || mul->op  != GGML_OP_MUL      || mul->src[0] != rms) return false;
    const int j_mul = w.j;
    const ggml_tensor * down = w.next(); if (!down || down->op != GGML_OP_MUL_MAT  || !mk_is(down->src[1], mul)) return false;
    const int j_down = w.j;
    const ggml_tensor * scl  = w.next(); if (!scl  || scl->op  != GGML_OP_SCALE    || scl->src[0] != down) return false;
    const ggml_tensor * slu  = w.next(); if (!slu  || slu->op  != GGML_OP_UNARY    || ggml_get_unary_op(slu) != GGML_UNARY_OP_SILU || slu->src[0] != scl) return false;
    const ggml_tensor * up   = w.next(); if (!up   || up->op   != GGML_OP_MUL_MAT  || up->src[1] != slu) return false;
    const int j_up = w.j;
    const ggml_tensor * pre  = w.next(); if (!pre  || pre->op  != GGML_OP_DSV4_HC_PRE || !mk_is(pre->src[0], mul) || !mk_is(pre->src[1], up)) return false;
    const int j_pre = w.j;

    const int64_t n_embd = rms->ne[0], hc = rms->ne[1], ne = n_embd*hc, n_lo = down->ne[0];
    if (rms->type != GGML_TYPE_F32 || !ggml_is_contiguous(rms->src[0]) || rms->ne[2] != 1 || rms->ne[3] != 1 || hc > 8 ||
        n_embd < 1024 || n_embd > 3072 || n_embd % 32 != 0 || !ggml_are_same_shape(mul, rms) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(mul->src[1], mul) || !ggml_is_contiguous(mul->src[1]) || mul->src[1]->type != GGML_TYPE_F32) return false;
    if (!mk_q8_0_mat(down->src[0], ne) || down->src[0]->ne[1] != n_lo || ggml_nelements(down->src[1]) != ne || ne % 512 != 0) return false;
    const int niter_down = (int) ((ne/QK8_0 + 7)/8);
    if (n_lo > 1024 || niter_down < 8 || niter_down > 64) return false;                             // k-split path
    if (!mk_q8_0_mat(up->src[0], n_lo) || up->src[0]->ne[1] != ne || n_lo % QK8_1 != 0 || n_lo/QK8_1 > MK_MAXQB) return false;
    if (!((n_lo/QK8_0 + 7)/8 <= 3 && ne > 1024 && ne % 16 == 0)) return false;                      // multirow path
    if (!ggml_is_contiguous(up) || !ggml_is_contiguous(down) || ggml_get_op_params_i32(pre, 1) == 0) return false;   // gated
    const ggml_tensor * px = pre->src[0], * pw = pre->src[1];
    if (px->nb[0] != 4 || px->nb[1] != (size_t) n_embd*4 || pw->nb[0] != 4 || pw->nb[1] != (size_t) n_embd*4 || px->ne[2] != 1 || !ggml_is_contiguous(pre)) return false;
    if (post) {
        const ggml_tensor * x = post->src[0], * r = post->src[1], * pw2 = post->src[2];
        if (post->src[3] || x->ne[0] != n_embd || x->ne[1] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(r) || r->ne[1] != hc || r->ne[2] != 1 ||
            !mk_vec(pw2, hc) || !ggml_is_contiguous(post) || post->ne[1] != hc) return false;
    }

    if (post) {
        mk_stage & s = B.add(MK_HC_POST);
        s.a = B.in(post->src[0]); s.b = B.in(post->src[1]); s.c = B.in(post->src[2]); s.n0 = (int) n_embd; s.n1 = (int) hc;
        s.x = B.out(j_post, post, ggml_nbytes(post));
    }
    { mk_stage & s = B.add(MK_RMS_BIG); s.a = B.in(rms->src[0]); s.b = B.in(mul->src[1]); s.n0 = (int) n_embd; s.n1 = (int) hc;
      memcpy(&s.f0, rms->op_params, sizeof(float)); s.x = B.out(j_mul, mul, ggml_nbytes(mul)); }
    void * xq = B.alloc((size_t) ne/QK8_1*sizeof(block_q8_1));
    { mk_stage & s = B.add(MK_QUANT); s.a = B.in(mul); s.x = xq; s.n0 = (int) ne; }
    { mk_stage & s = B.add(MK_MATVEC); s.a = down->src[0]->data; s.b = xq; s.n0 = (int) n_lo; s.n1 = (int) ne;
      s.x = B.out(j_down, down, ggml_nbytes(down)); }
    { mk_stage & s = B.add(MK_SILU_UP); s.a = B.in(down); s.b = up->src[0]->data; s.n0 = (int) n_lo; s.n1 = (int) ne;
      s.f0 = ggml_get_op_params_f32(scl, 0); s.f1 = ggml_get_op_params_f32(scl, 1); s.x = B.out(j_up, up, ggml_nbytes(up)); }
    { mk_stage & s = B.add(MK_HC_PRE); s.a = B.in(mul); s.b = B.in(up); s.n0 = (int) n_embd; s.n1 = (int) hc;
      s.f0 = ggml_get_op_params_f32(pre, 0); s.x = B.out(j_pre, pre, ggml_nbytes(pre)); }
    o.xn = mul; o.mixed = pre; o.xq = xq;
    return true;
}

// linear attention after an HC chain (mixed = its output, xn/xq = its normed input and q8_1)
struct lin_cut { int j = -1; size_t nst = 0; const ggml_tensor * sgr = nullptr; };

bool mk_match_lin(mk_walker & w, const hc_out & hc, mk_builder & B, lin_cut * cut = nullptr) {
    const ggml_tensor * mixed = hc.mixed;
    auto jof = [&](const ggml_tensor * t) { for (int k = w.j; k >= 0; k--) if (w.g->nodes[k] == t) return k; return -1; };
    const int64_t n_embd = mixed->ne[0];
    const ggml_tensor * gr = w.next();          // conv state (one cell): GET_ROWS(cache_r, s_copy)
    if (!gr || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32) return false;
    const ggml_tensor * cache_r = gr->src[0];
    if (cache_r->type != GGML_TYPE_F32 || cache_r->ne[1] != 1 || !ggml_is_contiguous(cache_r) || gr->ne[1] != 1) return false;
    const ggml_tensor * qkv = w.next();
    if (!qkv || qkv->op != GGML_OP_MUL_MAT || qkv->src[1] != mixed) return false;
    const int64_t C = qkv->ne[0];
    if (!mk_q8_0_mat(qkv->src[0], n_embd) || ggml_nelements(gr) != 3*C) return false;
    const ggml_tensor * cat = w.next();
    if (!cat || cat->op != GGML_OP_CONCAT || !mk_is(cat->src[0], gr) || !mk_is(cat->src[1], qkv) || cat->ne[0] != 4 || cat->ne[1] != C) return false;
    const ggml_tensor * cont = w.next();
    if (!cont || cont->op != GGML_OP_CONT || !mk_is(cont->src[0], cat) || cont->src[0]->view_offs != sizeof(float)) return false;
    const ggml_tensor * cpy = w.next();
    if (!cpy || cpy->op != GGML_OP_CPY || cpy->src[0] != cont || cpy->src[1]->data != cache_r->data || cpy->src[1]->type != GGML_TYPE_F32 ||
        ggml_nelements(cpy->src[1]) != 3*C || !ggml_is_contiguous(cpy->src[1])) return false;
    const ggml_tensor * sgr = w.next();         // recurrent state (one cell)
    if (!sgr || sgr->op != GGML_OP_GET_ROWS || sgr->src[0]->ne[1] != 1 || !ggml_is_contiguous(sgr->src[0]) || sgr->type != GGML_TYPE_F32) return false;
    const ggml_tensor * conv = w.next();
    if (!conv || conv->op != GGML_OP_SSM_CONV || conv->src[0] != cat || !mk_vec(conv->src[1], 4*C) || conv->src[1]->ne[0] != 4) return false;
    const ggml_tensor * csilu = w.next();
    if (!csilu || csilu->op != GGML_OP_UNARY || ggml_get_unary_op(csilu) != GGML_UNARY_OP_SILU || csilu->src[0] != conv) return false;
    const ggml_tensor * rq = w.next(), * sq = w.next(), * rk = w.next(), * sk = w.next();
    if (!rq || rq->op != GGML_OP_RMS_NORM || !mk_is(rq->src[0], csilu) || !sq || sq->op != GGML_OP_SCALE || sq->src[0] != rq) return false;
    if (!rk || rk->op != GGML_OP_RMS_NORM || !mk_is(rk->src[0], csilu) || !sk || sk->op != GGML_OP_SCALE || sk->src[0] != rk) return false;
    const ggml_tensor * alpha = w.next();
    if (!alpha || alpha->op != GGML_OP_MUL_MAT || alpha->src[1] != mixed || !mk_q8_0_mat(alpha->src[0], n_embd)) return false;
    const ggml_tensor * aadd = w.next(), * asp = w.next(), * amul = w.next();
    if (!aadd || aadd->op != GGML_OP_ADD || !mk_is(aadd->src[0], alpha) || !asp || asp->op != GGML_OP_UNARY || ggml_get_unary_op(asp) != GGML_UNARY_OP_SOFTPLUS ||
        asp->src[0] != aadd || !amul || amul->op != GGML_OP_MUL || amul->src[0] != asp) return false;
    const ggml_tensor * beta = w.next(), * bsig = w.next();
    if (!beta || beta->op != GGML_OP_MUL_MAT || beta->src[1] != mixed || !mk_q8_0_mat(beta->src[0], n_embd) ||
        !bsig || bsig->op != GGML_OP_UNARY || ggml_get_unary_op(bsig) != GGML_UNARY_OP_SIGMOID || !mk_is(bsig->src[0], beta)) return false;
    if (cut) cut->j = w.j;   // last node before the GDN
    const ggml_tensor * gdn = w.next();
    if (!gdn || gdn->op != GGML_OP_GATED_DELTA_NET || !mk_is(gdn->src[0], sq) || !mk_is(gdn->src[1], sk) || !mk_is(gdn->src[2], csilu) ||
        !mk_is(gdn->src[3], amul) || !mk_is(gdn->src[4], bsig) || !mk_is(gdn->src[5], sgr)) return false;
    const ggml_tensor * scpy = w.next();        // state snapshot back into the cache cell
    if (!scpy || scpy->op != GGML_OP_CPY || !mk_is(scpy->src[0], gdn) || scpy->src[1]->data != sgr->src[0]->data ||
        ggml_nelements(scpy->src[1]) != ggml_nelements(sgr->src[0])) return false;
    const ggml_tensor * rn = w.next(), * rmul = w.next();
    if (!rn || rn->op != GGML_OP_RMS_NORM || !mk_is(rn->src[0], gdn) || !rmul || rmul->op != GGML_OP_MUL || rmul->src[0] != rn) return false;
    const ggml_tensor * z = w.next(), * zsig = w.next(), * zmul = w.next();
    if (!z || z->op != GGML_OP_MUL_MAT || z->src[1] != mixed || !mk_q8_0_mat(z->src[0], n_embd) ||
        !zsig || zsig->op != GGML_OP_UNARY || ggml_get_unary_op(zsig) != GGML_UNARY_OP_SIGMOID || !mk_is(zsig->src[0], z) ||
        !zmul || zmul->op != GGML_OP_MUL || zmul->src[0] != rmul || zmul->src[1] != zsig) return false;
    const ggml_tensor * out = w.next();
    if (!out || out->op != GGML_OP_MUL_MAT || !mk_is(out->src[1], zmul) || out->ne[0] != n_embd) return false;
    const ggml_tensor * inj = w.next();
    if (!inj || inj->op != GGML_OP_MUL_MAT || !mk_is(inj->src[1], hc.xn) || ggml_nelements(inj->src[1]) != ggml_nelements(hc.xn)) return false;
    const ggml_tensor * s1 = w.next(), * sg = w.next(), * s2 = w.next();
    if (!s1 || s1->op != GGML_OP_SCALE || s1->src[0] != inj || !sg || sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
        sg->src[0] != s1 || !s2 || s2->op != GGML_OP_SCALE || s2->src[0] != sg) return false;

    // shapes of the replaced kernels' paths: S_v 128 heads, q/k norm rows of 128, one token
    const ggml_tensor * vq = rq->src[0], * vk = rk->src[0], * vv = gdn->src[2], * vo = rn->src[0];
    const int64_t S = vv->ne[0], H = vv->ne[1], Hqk = vq->ne[1];
    if (S != 128 || vq->ne[0] != S || vk->ne[0] != S || vk->ne[1] != Hqk || H % Hqk != 0 || vv->ne[2] != 1 || vv->ne[3] != 1 ||
        vq->nb[1] != (size_t) S*4 || vk->nb[1] != (size_t) S*4 || vv->nb[1] != (size_t) S*4 || !ggml_is_contiguous(csilu) ||
        !ggml_is_contiguous(sq) || !ggml_is_contiguous(sk) || ggml_get_op_params_f32(sq, 1) != 0.0f || ggml_get_op_params_f32(sk, 1) != 0.0f ||
        gdn->src[3]->ne[0] != 1 || ggml_get_op_params_i32(gdn, 0) > 1 || gdn->src[5]->ne[0] != S ||
        vo->ne[0] != S || vo->ne[1] != H || vo->nb[1] != (size_t) S*4 || !mk_vec(rmul->src[1], S) || !ggml_is_contiguous(rmul) ||
        !mk_vec(zsig, H*S) || !mk_vec(zmul, H*S) || !mk_q8_0_mat(out->src[0], H*S) ||
        !mk_vec(aadd->src[1], H) || !mk_vec(amul->src[1], H) || !mk_vec(alpha, H) || !mk_vec(beta, H) || !mk_vec(bsig, H) ||
        !mk_q8_0_mat(inj->src[0], hc.xn->ne[0]*hc.xn->ne[1]) || !mk_vec(s2, inj->ne[0]) || ggml_nelements(sgr->src[0]) != S*S*H ||
        ggml_get_op_params_i32(rq, 0) != ggml_get_op_params_i32(rk, 0)) return false;
    const int64_t niter_inj = (hc.xn->ne[0]*hc.xn->ne[1]/QK8_0 + 7)/8;
    if (inj->ne[0] > 1024 || niter_inj < 8 || niter_inj > 64) return false;   // k-split path (same values as one wave per row)

    void * qm = B.alloc((size_t) n_embd/QK8_1*sizeof(block_q8_1));
    { mk_stage & s = B.add(MK_QUANT); s.a = B.in(mixed); s.x = qm; s.n0 = (int) n_embd; }
    const size_t f4 = sizeof(float);
    { mk_stage & s = B.add(MK_MATVEC, 0); s.a = qkv->src[0]->data;   s.b = qm; s.n0 = (int) C;          s.n1 = (int) n_embd; s.x = B.out(jof(qkv), qkv, C*f4); }
    { mk_stage & s = B.add(MK_MATVEC, 0); s.a = z->src[0]->data;     s.b = qm; s.n0 = (int) z->ne[0];   s.n1 = (int) n_embd; s.x = B.out(jof(z), z, z->ne[0]*f4); }
    { mk_stage & s = B.add(MK_MATVEC, 0); s.a = alpha->src[0]->data; s.b = qm; s.n0 = (int) H;          s.n1 = (int) n_embd; s.x = B.out(jof(alpha), alpha, H*f4); }
    { mk_stage & s = B.add(MK_MATVEC, 0); s.a = beta->src[0]->data;  s.b = qm; s.n0 = (int) H;          s.n1 = (int) n_embd; s.x = B.out(jof(beta), beta, H*f4); }
    { mk_stage & s = B.add(MK_MATVEC);    s.a = inj->src[0]->data;   s.b = hc.xq; s.n0 = (int) inj->ne[0]; s.n1 = (int) (hc.xn->ne[0]*hc.xn->ne[1]);
      s.x = B.out(jof(inj), inj, inj->ne[0]*f4); }
    { mk_stage & s = B.add(MK_LIN_CONV, 0); s.a = B.in(qkv); s.b = cache_r->data; s.c = conv->src[1]->data; s.n0 = (int) C; s.f0 = 0.0f;
      s.x = B.out(jof(csilu), csilu, C*f4); }
    { mk_stage & s = B.add(MK_ALPHA, 0);  s.a = B.in(alpha); s.b = aadd->src[1]->data; s.c = amul->src[1]->data; s.n0 = (int) H; s.x = B.out(jof(amul), amul, H*f4); }
    { mk_stage & s = B.add(MK_SIGMOID, 0); s.a = B.in(beta); s.n0 = (int) H; s.x = B.out(jof(bsig), bsig, H*f4); }
    { mk_stage & s = B.add(MK_SSS);       s.a = B.in(inj); s.n0 = (int) inj->ne[0];
      s.f0 = ggml_get_op_params_f32(s1, 0); s.f1 = ggml_get_op_params_f32(s1, 1); s.f2 = ggml_get_op_params_f32(s2, 0); s.f3 = ggml_get_op_params_f32(s2, 1);
      s.x = B.out(jof(s2), s2, inj->ne[0]*f4); }
    float eps_qk; memcpy(&eps_qk, rq->op_params, sizeof(float));
    { mk_stage & s = B.add(MK_RMS_SMALL, 0); s.a = B.in(vq); s.n0 = (int) S; s.n1 = (int) Hqk; s.n2 = 0; s.f0 = eps_qk;
      s.f1 = ggml_get_op_params_f32(sq, 0); s.x = B.out(jof(sq), sq, ggml_nbytes(sq)); }
    { mk_stage & s = B.add(MK_RMS_SMALL);    s.a = B.in(vk); s.n0 = (int) S; s.n1 = (int) Hqk; s.n2 = 0; s.f0 = eps_qk;
      s.f1 = ggml_get_op_params_f32(sk, 0); s.x = B.out(jof(sk), sk, ggml_nbytes(sk)); }
    if (cut) { cut->nst = B.st.size(); cut->sgr = sgr; }
    g_mk_check = { qkv, z, alpha, beta, inj, out };
    { mk_stage & s = B.add(MK_GDN); s.a = B.in(sq); s.b = B.in(sk); s.c = B.in(vv); s.d = B.in(amul); s.e = B.in(bsig);
      s.y = sgr->src[0]->data; s.n0 = (int) H; s.n1 = (int) Hqk; s.f0 = 1.0f / sqrtf((float) S);
      s.x = B.out(jof(gdn), gdn, S*H*f4, S*H*f4); }
    float eps_o; memcpy(&eps_o, rn->op_params, sizeof(float));
    { mk_stage & s = B.add(MK_RMS_SMALL); s.a = B.in(vo); s.b = rmul->src[1]->data; s.c = B.in(z); s.n0 = (int) S; s.n1 = (int) H;
      s.n2 = 1; s.f0 = eps_o; s.x = B.out(jof(zmul), zmul, H*S*f4); }
    void * qf = B.alloc((size_t) H*S/QK8_1*sizeof(block_q8_1));
    { mk_stage & s = B.add(MK_QUANT); s.a = B.in(zmul); s.x = qf; s.n0 = (int) (H*S); }
    { mk_stage & s = B.add(MK_MATVEC); s.a = out->src[0]->data; s.b = qf; s.n0 = (int) n_embd; s.n1 = (int) (H*S); s.x = B.out(jof(out), out, n_embd*f4); }
    // the GDN output's attention scores sit at the start of gdn->data ([S*H] then the unused state tail)
    if (vo->data != gdn->data || vv->data != (const char *) csilu->data + 2*Hqk*S*sizeof(float)) return false;
    return true;
}

} // namespace

void ggml_cuda_rs_elide(const ggml_tensor * get_rows);   // ggml-cuda.cu
bool ggml_cuda_compute_node(ggml_backend_cuda_context & ctx, ggml_tensor * t);
void ggml_cuda_tdump(const char * dir, const char * tag, const ggml_tensor * t, const void * data, size_t bytes);

// LLAMA_MK_CHECK=<call>: after that megakernel call, recompute listed nodes with the original kernels into temporary
// buffers (their inputs are already written) and compare bit by bit
static void mk_check(ggml_backend_cuda_context & ctx) {
    for (const ggml_tensor * t : g_mk_check) {
        ggml_tensor c = *t;
        const size_t nb = ggml_nbytes(t);
        void * tmp; CUDA_CHECK(cudaMalloc(&tmp, nb + 4096));
        c.data = tmp;
        ggml_cuda_compute_node(ctx, &c);
        std::vector<char> a(nb), b(nb);
        CUDA_CHECK(cudaMemcpyAsync(a.data(), t->data, nb, cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), tmp, nb, cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        long bad = 0, first = -1;
        for (size_t k = 0; k < nb/4; k++) if (memcmp(&a[4*k], &b[4*k], 4)) { bad++; if (first < 0) first = (long) k; }
        fprintf(stderr, "[mk-check] %-12s %-28s %6ld of %6zu differ", ggml_op_desc(t), t->name, bad, nb/4);
        if (first >= 0) fprintf(stderr, " (first %ld: mk %.9g orig %.9g)", first, ((float *) a.data())[first], ((float *) b.data())[first]);
        fprintf(stderr, "\n");
        CUDA_CHECK(cudaFree(tmp));
    }
}

int ggml_cuda_try_layer_mk(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    static const bool off = getenv("LLAMA_NO_LAYER_MK") != nullptr || (getenv("LLAMA_MMVQ_KSPLIT") && atoi(getenv("LLAMA_MMVQ_KSPLIT")) != 1) ||
        getenv("LLAMA_MMVQ_KSPLIT_NW") || getenv("LLAMA_MMVQ_KSPLIT_OP") || getenv("LLAMA_MMVQ_KSPLIT_MAXROWS") || getenv("LLAMA_MMVQ_KSPLIT_MINITER") ||
        getenv("LLAMA_MMVQ_MULTIROW_R") || getenv("LLAMA_MMVQ_MULTIROW_MAXITER") || getenv("LLAMA_NO_SSM_FUSE");
    static const int level = getenv("LLAMA_LAYER_MK") ? atoi(getenv("LLAMA_LAYER_MK")) : 2;   // 1: HC chains only
    const ggml_tensor * n0 = g->nodes[i];
    if (off || (n0->op != GGML_OP_DSV4_HC_POST && n0->op != GGML_OP_RMS_NORM) ||
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32 || ctx.curr_stream_no != 0) {
        return 0;
    }
    static char * scratch = nullptr;
    static unsigned * bar = nullptr;
    static int grid = 0;
    if (!bar) {
        CUDA_CHECK(cudaMalloc(&bar, 2*sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(bar, 0, 2*sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&scratch, 1 << 20));
        // the grid barrier needs every workgroup resident: 94 VGPRs -> 10 waves per SIMD -> at most 5 workgroups of 8
        // waves per WGP (4 SIMDs; nsm counts WGPs on RDNA). Kernels of other streams can run alongside (copies, concurrent
        // graph branches): with 4 or 5 per WGP some workgroups wait for them and tokens stall; 3 per WGP leaves room.
        const int per_sm = getenv("LLAMA_LAYER_MK_PER_SM") ? atoi(getenv("LLAMA_LAYER_MK_PER_SM")) : 3;
        grid = per_sm * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        if (getenv("LLAMA_LAYER_MK_GRID")) grid = std::min(grid, atoi(getenv("LLAMA_LAYER_MK_GRID")));
    }
    mk_builder B;
    B.scratch = scratch;
    mk_walker w{g, i - 1};
    const ggml_tensor * first = w.next();
    hc_out hc;
    if (!first || !mk_match_hc(w, first, B, hc)) {
        return 0;
    }
    int last = w.j;
    // rms output (skipped intermediate) must have no other use
    for (int k = i; k <= last; k++) {
        if (g->nodes[k]->op == GGML_OP_RMS_NORM && !ggml_node_has_n_uses(g, k, 1)) return 0;
    }
    if (level >= 2) {
        mk_builder B2 = B;
        mk_walker w2 = w;
        lin_cut cut;
        if (level == 3 && mk_match_lin(w2, hc, B2, &cut)) {   // debug: stop before the GDN (originals run from there)
            B2.st.resize(cut.nst);
            B = B2; last = cut.j;
            ggml_cuda_rs_elide(cut.sgr);
        } else if (level == 2 && mk_match_lin(w2, hc, B2)) {
            // and the FFN-side HC chain that follows
            const ggml_tensor * nx = w2.next();
            hc_out hc2;
            mk_builder B3 = B2;
            mk_walker w3 = w2;
            if (nx && mk_match_hc(w3, nx, B3, hc2) && B3.st.size() <= MK_MAXST) {
                B = B3; last = w3.j;
            } else if (B2.st.size() <= MK_MAXST) {
                B = B2; last = w2.j;
            }
        }
    }
    // write back, original node order; a copy overlapping an earlier one goes to a later barrier group
    {
        std::vector<mk_wb> wb = B.wb;
        std::stable_sort(wb.begin(), wb.end(), [](const mk_wb & x, const mk_wb & y) { return x.j < y.j; });
        std::vector<int> grp(wb.size(), 0);
        int ng = 0;
        for (size_t k = 0; k < wb.size(); k++) {
            const char * a0 = (const char *) wb[k].t->data, * a1 = a0 + wb[k].bytes;
            for (size_t m = 0; m < k; m++) {
                const char * b0 = (const char *) wb[m].t->data, * b1 = b0 + wb[m].bytes;
                if (a0 < b1 && b0 < a1) grp[k] = std::max(grp[k], grp[m] + 1);
            }
            ng = std::max(ng, grp[k] + 1);
        }
        for (int gi = 0; gi < ng; gi++) {
            for (size_t k = 0; k < wb.size(); k++) {
                if (grp[k] != gi) continue;
                GGML_ASSERT(wb[k].bytes % 4 == 0);
                mk_stage & s = B.add(MK_COPY, 0);
                s.a = wb[k].src; s.x = wb[k].t->data; s.n0 = (int) (wb[k].bytes/4);
            }
            if (!B.st.empty()) B.st.back().bar = 1;
        }
    }
    {
        static const bool dbg = getenv("LLAMA_MK_DBG") != nullptr;
        static int left = 6;
        if (dbg && left > 0 && B.st.size() > 15) { left--; fprintf(stderr, "[layer-mk] epoch %llu node %d:", (unsigned long long) ggml_cuda_graph_epoch, i); fprintf(stderr, " node %d: %zu stages (%zu write-backs), nodes %d..%d, scratch %zu\n", i, B.st.size(), B.wb.size(), i, last, B.used); }
    }
    if (B.st.size() > MK_MAXST || B.used > (1u << 20)) return 0;

    static mk_prog p;
    p.n = (int) B.st.size();
    p.bar = bar;
    static unsigned long long * ts = nullptr;
    static const bool ts_on = getenv("LLAMA_MK_TS") != nullptr;
    if (ts_on && !ts) CUDA_CHECK(cudaMalloc(&ts, (MK_MAXST + 1)*sizeof(unsigned long long)));
    p.ts = ts_on ? ts : nullptr;
    for (int k = 0; k < p.n; k++) p.s[k] = B.st[k];
    p.s[p.n - 1].bar = 1;
    layer_mk_kernel<<<grid, MK_NT, 0, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
    {
        static const long check_at = getenv("LLAMA_MK_CHECK") ? atol(getenv("LLAMA_MK_CHECK")) : -1;
        static long calls = 0;
        if (check_at >= 0 && calls++ == check_at && p.n >= 20) mk_check(ctx);
        static const char * tdir = getenv("LLAMA_TDUMP");
        static const long tat = getenv("LLAMA_TDUMP_AT") ? atol(getenv("LLAMA_TDUMP_AT")) : -1;
        static const long tat2 = getenv("LLAMA_TDUMP_TO") ? atol(getenv("LLAMA_TDUMP_TO")) : tat;
        if (tdir && (long) ggml_cuda_graph_epoch >= tat && (long) ggml_cuda_graph_epoch <= tat2) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            for (const mk_wb & e : B.wb) ggml_cuda_tdump(tdir, "mk", e.t, e.src, e.bytes);
        }
    }
    if (ts_on) {   // per stage group, averaged
        static std::vector<double> acc(MK_MAXST + 1, 0.0); static long calls = 0;
        unsigned long long h[MK_MAXST + 1];
        CUDA_CHECK(cudaMemcpyAsync(h, ts, sizeof(h), cudaMemcpyDeviceToHost, ctx.stream())); CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        if (p.n >= 20) {
            unsigned long long prev = h[0];
            for (int k = 0; k < p.n; k++) if (p.s[k].bar) { acc[k] += (h[1 + k] - prev)/100.0; prev = h[1 + k]; }
            if (++calls % 360 == 0) {
                fprintf(stderr, "[layer-mk] grid %d, %d stages, us per call:", grid, p.n);
                for (int k = 0; k < p.n; k++) if (p.s[k].bar) fprintf(stderr, " %d:%.1f", k, acc[k]/360);
                fprintf(stderr, "\n");
                std::fill(acc.begin(), acc.end(), 0.0);
            }
        }
    }
    return last - i;
}
