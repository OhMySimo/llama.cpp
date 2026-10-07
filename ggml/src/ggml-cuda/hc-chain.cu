// (fork) hyper-connection chain in one persistent kernel: [DSV4_HC_POST ->] RMS_NORM -> MUL -> q8_1 quantization ->
// MUL_MAT(hc down, k-split) -> SCALE -> SILU -> MUL_MAT(hc up, multirow) -> DSV4_HC_PRE, single token.
// Each stage runs the blocks of the kernel it replaces as virtual blocks of a grid of resident workgroups, with the
// same thread mapping, the same expressions and the same reduction trees, so every value is identical to the separate
// kernels; stages are separated by a grid barrier (~0.5 us) instead of a kernel boundary (~2.6 us in a HIP graph).

#include "hc-chain.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#define HCC_NT       256              // threads per workgroup
#define HCC_KS_NW    8                // k-split waves per row (mul_mat_vec_q8_0_ksplit<8, 0>)
#define HCC_KS_MAXIT 64
#define HCC_MAXQB    32               // q8_1 blocks of the hc up input (K = 320 -> 10)

struct hcc_args {
    // HC_POST (optional): l_last[i0, k] = x[i0]*post[k] + res[i0, k]
    const float * post_x; const float * post_res; const float * post_w; float * l_last;
    // RMS_NORM of l_last rows (n_embd) and MUL by w_norm -> xn [n_embd*hc]
    const float * norm_x; const float * norm_w; float * xn; float eps;
    // q8_1 of xn, then hc down (k-split) -> lo, scale, silu, hc up (multirow) -> gate, HC_PRE -> mixed
    void * xq; const void * w_down; float * lo; float s1, b1; const void * w_up; float * gate;
    float pre_scale; float * mixed;
    int n_embd, hc, n_lo;   // n_lo: hc down rows (320)
    unsigned * bar;
    unsigned long long * ts;   // LLAMA_HC_CHAIN_TS: stage end times (workgroup 0)
};

// grid barrier: arrival counter + generation, self-resetting; release/acquire at agent scope (L0/L1 invalidated)
static __device__ __forceinline__ void hcc_grid_sync(unsigned * bar, unsigned long long * ts = nullptr, int k = 0) {
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
        if (ts && blockIdx.x == 0) ts[k] = wall_clock64();
    }
    __syncthreads();
}

__launch_bounds__(HCC_NT, 1)
static __global__ void hc_chain_kernel(const hcc_args a) {
    __shared__ float s_red[32];
    __shared__ block_q8_1 ys[HCC_MAXQB];
    const int tid = threadIdx.x, lane = tid % 32, wv = tid / 32;
    const int ne = a.n_embd * a.hc;
    if (a.ts && blockIdx.x == 0 && threadIdx.x == 0) { a.ts[0] = wall_clock64(); if (!a.l_last) a.ts[1] = a.ts[0]; }

    // ---- HC_POST (dsv4_hc_post_f32<false>): ir -> i0 = ir % n_embd, idst = ir / n_embd
    if (a.l_last) {
        for (int ir = blockIdx.x*HCC_NT + tid; ir < ne; ir += gridDim.x*HCC_NT) {
            const int i0 = ir % a.n_embd, idst = ir / a.n_embd;
            float sum = a.post_x[i0] * a.post_w[idst];
            sum += a.post_res[i0 + idst*a.n_embd];
            a.l_last[i0 + idst*a.n_embd] = sum;
        }
        hcc_grid_sync(a.bar, a.ts, 1);
    }

    // ---- RMS_NORM + MUL (rms_norm_f32<1024, true>): one virtual block of 1024 threads per row, emulated with 4
    // partial sums per thread (virtual thread v = tid + 256*j, virtual warp 8*j + wv)
    // (n_embd <= 3072: at most 3 columns per virtual thread, kept in registers between the two passes)
    for (int row = blockIdx.x; row < a.hc; row += gridDim.x) {
        const float * x = a.norm_x + row*a.n_embd;
        const float * w = a.norm_w + row*a.n_embd;
        float xv[4][3], wv3[4][3];
        float tmp[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const int col = tid + 256*j + 1024*c;
                xv[j][c]  = col < a.n_embd ? x[col] : 0.0f;
                wv3[j][c] = col < a.n_embd ? w[col] : 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            tmp[j] = 0.0f;
#pragma unroll
            for (int c = 0; c < 3; c++) {
                if (tid + 256*j + 1024*c < a.n_embd) {
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
        const float mean  = t / a.n_embd;
        const float scale = rsqrtf(mean + a.eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int c = 0; c < 3; c++) {
                const int col = tid + 256*j + 1024*c;
                if (col < a.n_embd) {
                    a.xn[row*a.n_embd + col] = scale * xv[j][c] * wv3[j][c];
                }
            }
        }
        __syncthreads();
    }
    hcc_grid_sync(a.bar, a.ts, 2);

    // ---- quantize_q8_1 of xn (256-thread blocks, one wave per 32-value block)
    {
        block_q8_1 * y = (block_q8_1 *) a.xq;
        for (int i0 = blockIdx.x*HCC_NT + tid; i0 < ne; i0 += gridDim.x*HCC_NT) {
            const float xi = a.xn[i0];
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
    hcc_grid_sync(a.bar, a.ts, 3);

    // ---- hc down: values of mul_mat_vec_q8_0_ksplit<8, 0>: per lane the fma chain over its blocks in order (wave 0's
    // loop there), then the warp reduction; here one wave per row runs the whole chain, loads unrolled ahead
    {
        constexpr int qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ, bpi = vdr*32/qi, U = 8;
        const int blocks_per_row_x = ne / QK8_0;
        const int kbx0 = lane / (qi/vdr);
        const int kqs  = vdr * (lane % (qi/vdr));
        const int niter = (blocks_per_row_x - kbx0 + bpi - 1) / bpi;
        const block_q8_1 * y = (const block_q8_1 *) a.xq;
        for (int row = blockIdx.x*(HCC_NT/32) + wv; row < a.n_lo; row += gridDim.x*(HCC_NT/32)) {
            const block_q8_0 * x = (const block_q8_0 *) a.w_down + (size_t) row*blocks_per_row_x;
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
                a.lo[row] = tmp;
            }
        }
    }
    hcc_grid_sync(a.bar, a.ts, 4);

    // ---- scale + silu + q8_1 (as mul_mat_vec_q8_0_multirow_scale_silu's prologue), once per workgroup
    const int nqb = a.n_lo / QK8_1;
    for (int b = wv; b < nqb; b += HCC_NT/32) {
        const int i = b*QK8_1 + lane;
        float xi = 0.0f;
        if (i < a.n_lo) {
            const float y = a.s1 * a.lo[i] + a.b1;
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

    // ---- hc up: mul_mat_vec_q8_0_multirow<4> loop (each wave 4 consecutive rows)
    {
        constexpr int R = 4, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ, bpi = vdr*32/qi;
        const int kqs = vdr * (lane % (qi/vdr));
        for (int row0 = (blockIdx.x*(HCC_NT/32) + wv)*R; row0 < ne; row0 += gridDim.x*(HCC_NT/32)*R) {
            float tmp[R] = {0.0f};
            for (int kbx = lane / (qi/vdr); kbx < nqb; kbx += bpi) {
                const float d8_1 = __low2half(ys[kbx].ds);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const block_q8_0 * x = (const block_q8_0 *) a.w_up + (size_t) (row0 + r)*nqb + kbx;
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
                    a.gate[row0 + r] = tmp[r];
                }
            }
        }
    }
    hcc_grid_sync(a.bar, a.ts, 5);

    // ---- HC_PRE gated (dsv4_hc_pre_f32<true>), one token
    for (int i0 = blockIdx.x*HCC_NT + tid; i0 < a.n_embd; i0 += gridDim.x*HCC_NT) {
        float sum = 0.0f;
        for (int ih = 0; ih < a.hc; ++ih) {
            const float xv = a.xn[i0 + ih*a.n_embd];
            float wv2;
            wv2 = 1.0f / (1.0f + expf(-a.gate[i0 + ih*a.n_embd]));
            sum += xv * wv2;
        }
        a.mixed[i0] = a.pre_scale * sum;
    }
    hcc_grid_sync(a.bar, a.ts, 6);
}

// ------------------------------------------------------------------ graph matcher

static const ggml_tensor * hcc_next(const ggml_cgraph * g, int & j) {   // next non-view node after j
    for (++j; j < g->n_nodes; ++j) {
        const ggml_tensor * n = g->nodes[j];
        if (n->op != GGML_OP_RESHAPE && n->op != GGML_OP_VIEW && n->op != GGML_OP_PERMUTE && n->op != GGML_OP_TRANSPOSE && n->op != GGML_OP_NONE) {
            return n;
        }
    }
    return nullptr;
}

static bool hcc_is(const ggml_tensor * t, const ggml_tensor * base) {   // t is base or a view of it
    for (; t; t = t->view_src) if (t == base) return true;
    return false;
}

static bool hcc_q8_0_mat(const ggml_tensor * w, int64_t k, int64_t rows) {
    return w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == k && w->ne[1] == rows && w->ne[2] == 1 && w->ne[3] == 1 &&
           ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

int ggml_cuda_try_hc_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    static const bool off = getenv("LLAMA_NO_HC_CHAIN") != nullptr || (getenv("LLAMA_MMVQ_KSPLIT") && atoi(getenv("LLAMA_MMVQ_KSPLIT")) != 1) ||
        getenv("LLAMA_MMVQ_KSPLIT_NW") || getenv("LLAMA_MMVQ_KSPLIT_OP") || getenv("LLAMA_MMVQ_KSPLIT_MAXROWS") || getenv("LLAMA_MMVQ_KSPLIT_MINITER") ||
        getenv("LLAMA_MMVQ_MULTIROW_R") || getenv("LLAMA_MMVQ_MULTIROW_MAXITER") || getenv("LLAMA_NO_SSM_FUSE");
    if (off || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32 || ctx.curr_stream_no != 0) {
        return 0;
    }
    int j = i;
    const ggml_tensor * post = nullptr;
    const ggml_tensor * n = g->nodes[i];
    if (n->op == GGML_OP_DSV4_HC_POST) {
        post = n;
        n = hcc_next(g, j);
        if (!n) return 0;
    }
    const ggml_tensor * rms = n;
    if (rms->op != GGML_OP_RMS_NORM || (post && rms->src[0] != post)) return 0;
    const ggml_tensor * mul  = hcc_next(g, j); if (!mul  || mul->op  != GGML_OP_MUL      || mul->src[0] != rms) return 0;
    const ggml_tensor * down = hcc_next(g, j); if (!down || down->op != GGML_OP_MUL_MAT  || !hcc_is(down->src[1], mul)) return 0;
    const ggml_tensor * scl  = hcc_next(g, j); if (!scl  || scl->op  != GGML_OP_SCALE    || scl->src[0] != down) return 0;
    const ggml_tensor * slu  = hcc_next(g, j); if (!slu  || slu->op  != GGML_OP_UNARY    || ggml_get_unary_op(slu) != GGML_UNARY_OP_SILU || slu->src[0] != scl) return 0;
    const ggml_tensor * up   = hcc_next(g, j); if (!up   || up->op   != GGML_OP_MUL_MAT  || up->src[1] != slu) return 0;
    const ggml_tensor * pre  = hcc_next(g, j); if (!pre  || pre->op  != GGML_OP_DSV4_HC_PRE || !hcc_is(pre->src[0], mul) || !hcc_is(pre->src[1], up)) return 0;
    const int last = j;

    const int64_t n_embd = rms->ne[0], hc = rms->ne[1], ne = n_embd*hc, n_lo = down->ne[0];
    // shapes and layouts of the replaced kernels' fast paths, single token
    if (rms->type != GGML_TYPE_F32 || !ggml_is_contiguous(rms->src[0]) || rms->ne[2] != 1 || rms->ne[3] != 1 || hc > 8 ||
        n_embd < 1024 || n_embd > 3072 || n_embd % 32 != 0 || !ggml_are_same_shape(mul, rms) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(mul->src[1], mul) || !ggml_is_contiguous(mul->src[1]) || mul->src[1]->type != GGML_TYPE_F32) return 0;
    if (!hcc_q8_0_mat(down->src[0], ne, n_lo) || ggml_nelements(down->src[1]) != ne || ne % 512 != 0) return 0;
    const int niter_down = (int) ((ne/QK8_0 + 7)/8);
    if (n_lo > 1024 || niter_down < 8 || niter_down > HCC_KS_MAXIT) return 0;                     // k-split path
    if (!hcc_q8_0_mat(up->src[0], n_lo, ne) || n_lo % QK8_1 != 0 || n_lo/QK8_1 > HCC_MAXQB) return 0;
    if (!((n_lo/QK8_0 + 7)/8 <= 3 && ne > 1024 && ne % 16 == 0)) return 0;                        // multirow path
    if (!ggml_is_contiguous(up) || !ggml_is_contiguous(down) || ggml_get_op_params_i32(pre, 1) == 0) return 0;   // gated HC_PRE
    const ggml_tensor * px = pre->src[0], * pw = pre->src[1];
    if (px->nb[0] != 4 || px->nb[1] != (size_t) n_embd*4 || pw->nb[0] != 4 || pw->nb[1] != (size_t) n_embd*4 || px->ne[2] != 1 || !ggml_is_contiguous(pre)) return 0;
    if (post) {
        const ggml_tensor * x = post->src[0], * r = post->src[1], * w = post->src[2];
        if (post->src[3] || x->ne[0] != n_embd || x->ne[1] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(r) || r->ne[1] != hc || r->ne[2] != 1 ||
            !ggml_is_contiguous(w) || ggml_nelements(w) != hc || !ggml_is_contiguous(post) || post->ne[1] != hc) return 0;
    }
    // intermediates that are skipped must have no other use
    if (!ggml_node_has_n_uses(g, (int) (std::find(g->nodes, g->nodes + g->n_nodes, rms) - g->nodes), 1)) return 0;

    static unsigned * bar = nullptr;
    static int grid = 0;
    if (!bar) {
        CUDA_CHECK(cudaMalloc(&bar, 2*sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(bar, 0, 2*sizeof(unsigned)));
        int per_cu = 0;
        CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, hc_chain_kernel, HCC_NT, 0));
        grid = std::max(1, per_cu) * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        static const int cap = getenv("LLAMA_HC_CHAIN_GRID") ? atoi(getenv("LLAMA_HC_CHAIN_GRID")) : 0;
        if (cap > 0) grid = std::min(grid, cap);
        GGML_LOG_INFO("%s: persistent hc chain kernel, %d workgroups (%d per CU)\n", __func__, grid, per_cu);
    }

    hcc_args a = {};
    if (post) {
        a.post_x = (const float *) post->src[0]->data; a.post_res = (const float *) post->src[1]->data;
        a.post_w = (const float *) post->src[2]->data; a.l_last = (float *) post->data;
    }
    a.norm_x = (const float *) rms->src[0]->data; a.norm_w = (const float *) mul->src[1]->data; a.xn = (float *) mul->data;
    memcpy(&a.eps, rms->op_params, sizeof(float));
    a.xq = ggml_cuda_q8_memo_claim(down->src[1], (size_t) ne/QK8_1*sizeof(block_q8_1));
    a.w_down = down->src[0]->data; a.lo = (float *) down->data;
    a.s1 = ggml_get_op_params_f32(scl, 0); a.b1 = ggml_get_op_params_f32(scl, 1);
    a.w_up = up->src[0]->data; a.gate = (float *) up->data;
    a.pre_scale = ggml_get_op_params_f32(pre, 0); a.mixed = (float *) pre->data;
    a.n_embd = (int) n_embd; a.hc = (int) hc; a.n_lo = (int) n_lo; a.bar = bar;
    if (!a.xq) return 0;
    static unsigned long long * ts = nullptr; static double acc[8]; static long calls = 0;
    static const bool ts_on = getenv("LLAMA_HC_CHAIN_TS") != nullptr;
    if (ts_on && !ts) { CUDA_CHECK(cudaMalloc(&ts, 8*sizeof(unsigned long long))); fprintf(stderr, "[hc-chain] grid %d\n", grid); }
    a.ts = ts_on ? ts : nullptr;
    hc_chain_kernel<<<grid, HCC_NT, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
    if (ts_on) {
        unsigned long long h[8]; CUDA_CHECK(cudaMemcpyAsync(h, ts, sizeof(h), cudaMemcpyDeviceToHost, ctx.stream())); CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (int k = 1; k < 7; k++) acc[k] += (h[k] - h[k-1]) / 100.0;   // 100 MHz
        if (++calls % 960 == 0) {
            fprintf(stderr, "[hc-chain] us per call: post %.2f norm %.2f quant %.2f down %.2f silu+up %.2f pre %.2f  total %.2f\n",
                acc[1]/960, acc[2]/960, acc[3]/960, acc[4]/960, acc[5]/960, acc[6]/960, (acc[1]+acc[2]+acc[3]+acc[4]+acc[5]+acc[6])/960);
            for (double & x : acc) x = 0;
        }
    }
    return last - i;
}
