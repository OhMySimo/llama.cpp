#include "cpu-exact.cuh"

// no implicit a*b+c fusion anywhere in this file: the CPU code rounds products and sums separately unless it
// uses an explicit fma (written here as __fmaf_rn)
#pragma clang fp contract(off)
#include <vector>
#include <cstdio>

// Every float operation below mirrors one instruction of the CPU code it replaces, with explicit rounding
// intrinsics (no contraction, no reassociation). Integer parts are computed with any order: they are exact.

// ---------------------------------------------------------------- activation quantization

// quantize_row_q8_K_ref (x86 quantize_row_q8_K): amax and the first element reaching it, iscale = -127/max,
// q = min(127, nearest_int(iscale*x)) with the 1.5*2^23 trick, d = 1/iscale, bsums of 16
template <bool fma_trick>
static __global__ void cx_quantize_q8_K(const char * x, size_t col_stride, block_q8_K * y, int nb) {
    const int ib  = blockIdx.x;
    const int col = blockIdx.y;
    const int t   = threadIdx.x;   // 256
    const float v  = ((const float *) (x + col*col_stride))[ib*QK_K + t];
    const float ax = fabsf(v);

    __shared__ float s_max[256];
    __shared__ int   s_first;
    __shared__ int   s_q[256];
    s_max[t] = ax;
    if (t == 0) {
        s_first = QK_K;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (t < s) {
            s_max[t] = fmaxf(s_max[t], s_max[t + s]);
        }
        __syncthreads();
    }
    const float amax = s_max[0];
    if (ax == amax) {
        atomicMin(&s_first, t);
    }
    __syncthreads();

    block_q8_K & b = y[col*nb + ib];
    if (amax == 0.0f) {
        b.qs[t] = 0;
        if (t < QK_K/16) {
            b.bsums[t] = 0;
        }
        if (t == 0) {
            b.d = 0.0f;
        }
        return;
    }
    const float max    = ((const float *) (x + col*col_stride))[ib*QK_K + s_first];
    const float iscale = ((-127.f) / (max));
    const float val    = fma_trick ? __fmaf_rn(iscale, v, 12582912.f) : ((((iscale) * (v))) + (12582912.f));
    const int   q      = min(127, (__float_as_int(val) & 0x007fffff) - 0x00400000);
    b.qs[t] = (int8_t) q;
    s_q[t]  = q;
    __syncthreads();
    if (t < QK_K/16) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += s_q[t*16 + i];
        }
        b.bsums[t] = (int16_t) sum;
    }
    if (t == 0) {
        b.d = ((1.0f) / (iscale));
    }
}

// x86 AVX2 quantize_row_q8_0: d = maxabs/127 (to fp16), id = 127/maxabs, q = round_nearest_even(x*id)
static __global__ void cx_quantize_q8_0(const char * x, size_t col_stride, block_q8_0 * y, int nb) {
    const int ib  = blockIdx.x;
    const int col = blockIdx.y;
    const int t   = threadIdx.x;   // 32
    const float v = ((const float *) (x + col*col_stride))[ib*QK8_0 + t];
    __shared__ float s_max[32];
    s_max[t] = fabsf(v);
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (t < s) {
            s_max[t] = fmaxf(s_max[t], s_max[t + s]);
        }
        __syncthreads();
    }
    const float maxs = s_max[0];
    block_q8_0 & b = y[col*nb + ib];
    const float id = maxs != 0.0f ? ((127.f) / (maxs)) : 0.0f;
    b.qs[t] = (int8_t) (int) rintf(((v) * (id)));
    if (t == 0) {
        b.d = __float2half_rn(((maxs) / (127.f)));
    }
}

// ---------------------------------------------------------------- dot products: int32 lane L of super-block i

static __device__ __forceinline__ uint32_t cx_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

// ggml_vec_dot_iq3_xxs_q8_K, AVX2: lane L of a 32-value sub-block = bytes 4L..4L+3 (grid entry q3[L]),
// signs from the 7-bit group L/2 of the sub-block word, scale 2*ls+1; lanes summed over the 8 sub-blocks
static __device__ __forceinline__ int cx_lane_iq3_xxs(const block_iq3_xxs * bx, const block_q8_K * by, int L) {
    const uint8_t * q3  = bx->qs;
    const uint8_t * gas = bx->qs + QK_K/4;
    int sum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t aux  = cx_u32(gas + 4*ib32);
        const int      ls   = 2*(int)(aux >> 28) + 1;
        const uint8_t  sgn  = ksigns_iq2xs[(aux >> (7*(L/2))) & 127];
        const uint32_t grid = iq3xxs_grid[q3[8*ib32 + L]];
        int s = 0;
        for (int b = 0; b < 4; ++b) {
            const int g = (grid >> (8*b)) & 0xff;
            const int q = by->qs[32*ib32 + 4*L + b];
            s += ((sgn >> ((L%2)*4 + b)) & 1) ? -g*q : g*q;
        }
        sum += ls*s;
    }
    return sum;
}

// ggml_vec_dot_iq2_xxs_q8_K, AVX2: per sub-block a 32-bit word of 4 grid indices (8 values each) and a word of
// 4 sign groups + the scale; lane L = half L%2 of grid entry L/2
static __device__ __forceinline__ int cx_lane_iq2_xxs(const block_iq2_xxs * bx, const block_q8_K * by, int L) {
    const uint8_t * q2 = (const uint8_t *) bx->qs;
    int sum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t a0  = cx_u32(q2 + 8*ib32);
        const uint32_t a1  = cx_u32(q2 + 8*ib32 + 4);
        const int      ls  = 2*(int)(a1 >> 28) + 1;
        const int      m   = L/2;
        const uint8_t  sgn = ksigns_iq2xs[(a1 >> (7*m)) & 127];
        const uint64_t grid = iq2xxs_grid[(a0 >> (8*m)) & 0xff];
        int s = 0;
        for (int b = 0; b < 4; ++b) {
            const int pos = (L%2)*4 + b;
            const int g = (int) ((grid >> (8*pos)) & 0xff);
            const int q = by->qs[32*ib32 + 4*L + b];
            s += ((sgn >> pos) & 1) ? -g*q : g*q;
        }
        sum += ls*s;
    }
    return sum;
}

static __device__ __forceinline__ void cx_q4_K_scales(const block_q4_K * bx, uint8_t * sc, uint8_t * mn) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    utmp[0] = cx_u32(bx->scales + 0);
    utmp[1] = cx_u32(bx->scales + 4);
    utmp[2] = cx_u32(bx->scales + 8);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    for (int i = 0; i < 4; ++i) {
        sc[i] = (utmp[0] >> (8*i)) & 0xff; sc[4 + i] = (utmp[1] >> (8*i)) & 0xff;
        mn[i] = (utmp[2] >> (8*i)) & 0xff; mn[4 + i] = (utmp[3] >> (8*i)) & 0xff;
    }
}

// ggml_vec_dot_q4_K_q8_K, AVX2: lane L = sum over the 4 64-value groups of scale * (4 low-nibble products) +
// scale * (4 high-nibble products)
static __device__ __forceinline__ int cx_lane_q4_K(const block_q4_K * bx, const block_q8_K * by, int L) {
    uint8_t sc[8], mn[8];
    cx_q4_K_scales(bx, sc, mn);
    int sum = 0;
    for (int j = 0; j < QK_K/64; ++j) {
        int sl = 0, sh = 0;
        for (int b = 0; b < 4; ++b) {
            const int q4 = bx->qs[32*j + 4*L + b];
            sl += (q4 & 0xf) * by->qs[64*j + 4*L + b];
            sh += (q4 >> 4)  * by->qs[64*j + 32 + 4*L + b];
        }
        sum += sc[2*j]*sl + sc[2*j + 1]*sh;
    }
    return sum;
}

// the mins part: 4 int32 lanes p = m[2p]*s[2p] + m[2p+1]*s[2p+1], s[k] = bsums[2k] + bsums[2k+1]
static __device__ __forceinline__ int cx_mins_q4_K(const block_q4_K * bx, const block_q8_K * by, int p) {
    uint8_t sc[8], mn[8];
    cx_q4_K_scales(bx, sc, mn);
    const int s0 = by->bsums[4*p + 0] + by->bsums[4*p + 1];
    const int s1 = by->bsums[4*p + 2] + by->bsums[4*p + 3];
    return mn[2*p]*s0 + mn[2*p + 1]*s1;
}

// ggml_vec_dot_iq4_nl_q8_0, AVX2: lane L = elements 4L..4L+3 of a 32-value block
static __device__ __forceinline__ int cx_lane_iq4_nl(const block_iq4_nl * bx, const block_q8_0 * by, int L) {
    int s = 0;
    for (int b = 0; b < 4; ++b) {
        const int e   = 4*L + b;
        const int nib = e < 16 ? (bx->qs[e] & 0xf) : (bx->qs[e - 16] >> 4);
        s += kvalues_iq4nl[nib] * by->qs[e];
    }
    return s;
}

// hsum_float_8: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))
static __device__ __forceinline__ float cx_hsum8(const float * a) {
    const float r0 = ((a[4]) + (a[0])), r1 = ((a[5]) + (a[1])), r2 = ((a[6]) + (a[2])), r3 = ((a[7]) + (a[3]));
    return ((((r0) + (r2))) + (((r1) + (r3))));
}

#define CX_MAX_NB 64
#define CX_ROWS   4

// one wave per output row: int lanes in parallel, then the CPU's sequential fma per lane and its horizontal sum
template <ggml_type type>
static __global__ void cx_mmid_dot(
        const char * src0, size_t nb01, size_t nb02, int ne01, int ne00, int n_as, bool zero_last,
        const char * qy, size_t qy_col_bytes, int ne11,
        const char * ids, size_t ids_nb0, size_t ids_nb1,
        char * dst, size_t dst_nb1, size_t dst_nb2) {
    constexpr bool is_q8_0 = type == GGML_TYPE_IQ4_NL;
    constexpr int  qk      = is_q8_0 ? QK8_0 : QK_K;
    const int nb   = ne00 / qk;
    const int w    = threadIdx.y;
    const int lane = threadIdx.x;
    const int row  = blockIdx.x*CX_ROWS + w;
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;

    const int expert = *(const int32_t *) (ids + slot*ids_nb0 + tok*ids_nb1);
    float * out = (float *) (dst + slot*dst_nb1 + tok*dst_nb2);
    if (zero_last && expert == n_as - 1) {
        if (row < ne01 && lane == 0) {
            out[row] = 0.0f;
        }
        return;   // uniform over the block: same slot and token
    }

    __shared__ int   s_int[CX_ROWS][CX_MAX_NB][8];
    __shared__ int   s_min[CX_ROWS][CX_MAX_NB][4];
    __shared__ float s_acc[CX_ROWS][8];
    __shared__ float s_accm[CX_ROWS][4];

    const bool active = row < ne01;
    const char * xrow = src0 + (size_t) expert*nb02 + (size_t) (active ? row : 0)*nb01;
    const char * ycol = qy + (size_t) ((slot % ne11) + tok*ne11)*qy_col_bytes;

    for (int idx = lane; idx < nb*8; idx += 32) {
        const int i = idx / 8, L = idx % 8;
        int v;
        if constexpr (type == GGML_TYPE_IQ3_XXS) {
            v = cx_lane_iq3_xxs((const block_iq3_xxs *) xrow + i, (const block_q8_K *) ycol + i, L);
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            v = cx_lane_iq2_xxs((const block_iq2_xxs *) xrow + i, (const block_q8_K *) ycol + i, L);
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            v = cx_lane_q4_K((const block_q4_K *) xrow + i, (const block_q8_K *) ycol + i, L);
        } else {
            v = cx_lane_iq4_nl((const block_iq4_nl *) xrow + i, (const block_q8_0 *) ycol + i, L);
        }
        s_int[w][i][L] = v;
    }
    if constexpr (type == GGML_TYPE_Q4_K) {
        for (int idx = lane; idx < nb*4; idx += 32) {
            s_min[w][idx/4][idx%4] = cx_mins_q4_K((const block_q4_K *) xrow + idx/4, (const block_q8_K *) ycol + idx/4, idx%4);
        }
    }
    __syncthreads();

    if (lane < 8) {
        const int L = lane;
        if constexpr (is_q8_0) {
            // two accumulators (even / odd blocks), summed lane by lane before the horizontal sum
            float acc1 = 0.0f, acc2 = 0.0f;
            for (int i = 0; i + 1 < nb; i += 2) {
                const block_iq4_nl * bx = (const block_iq4_nl *) xrow;
                const block_q8_0   * by = (const block_q8_0 *) ycol;
                const float d1 = ((__half2float(by[i].d)) * (__half2float(bx[i].d)));
                const float d2 = ((__half2float(by[i + 1].d)) * (__half2float(bx[i + 1].d)));
                acc1 = __fmaf_rn(d1, (float) s_int[w][i][L],     acc1);
                acc2 = __fmaf_rn(d2, (float) s_int[w][i + 1][L], acc2);
            }
            s_acc[w][L] = ((acc1) + (acc2));
        } else {
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const block_q8_K * by = (const block_q8_K *) ycol + i;
                float d;
                if constexpr (type == GGML_TYPE_IQ3_XXS) {
                    d = ((__half2float(((const block_iq3_xxs *) xrow)[i].d)) * (by->d));
                } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
                    d = ((__half2float(((const block_iq2_xxs *) xrow)[i].d)) * (by->d));
                } else {
                    d = ((by->d) * (__low2float(((const block_q4_K *) xrow)[i].dm)));
                }
                acc = __fmaf_rn(d, (float) s_int[w][i][L], acc);
            }
            s_acc[w][L] = acc;
            if constexpr (type == GGML_TYPE_Q4_K) {
                if (L < 4) {
                    float accm = 0.0f;
                    for (int i = 0; i < nb; ++i) {
                        const block_q8_K * by = (const block_q8_K *) ycol + i;
                        const float dmin = ((-by->d) * (__high2float(((const block_q4_K *) xrow)[i].dm)));
                        accm = __fmaf_rn(dmin, (float) s_min[w][i][L], accm);
                    }
                    s_accm[w][L] = accm;
                }
            }
        }
    }
    __syncthreads();

    if (lane == 0 && active) {
        const float h = cx_hsum8(s_acc[w]);
        float r;
        if constexpr (type == GGML_TYPE_IQ3_XXS) {
            r = ((0.25f) * (h));
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            r = ((0.125f) * (h));
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            const float * m = s_accm[w];
            r = ((h) + (((((m[0]) + (m[2]))) + (((m[1]) + (m[3]))))));
        } else {
            r = h;
        }
        out[row] = r;
    }
}

bool ggml_cuda_cpu_exact_supported(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_Q4_K:
            return src0->ne[0] % QK_K == 0 && src0->ne[0] / QK_K <= CX_MAX_NB;
        case GGML_TYPE_IQ4_NL:
            return src0->ne[0] % (2*QK8_0) == 0 && src0->ne[0] / QK8_0 <= CX_MAX_NB;
        default:
            return false;
    }
}

void ggml_cuda_mul_mat_id_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float) && src1->ne[3] == 1);
    GGML_ASSERT(ggml_cuda_cpu_exact_supported(dst));

    cudaStream_t stream = ctx.stream();
    const int ne00 = src0->ne[0], ne01 = src0->ne[1], n_as = src0->ne[2];
    const int ne11 = src1->ne[1], ne12 = src1->ne[2];
    const int n_ids = ids->ne[0];
    const bool is_q8_0 = src0->type == GGML_TYPE_IQ4_NL;
    const ggml_type vdt = is_q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
    const size_t col_bytes = ggml_row_size(vdt, ne00);
    const int ncols = ne11*ne12;

    ggml_cuda_pool_alloc<char> qy(ctx.pool(), (size_t) ncols*col_bytes);
    // src1 columns (i11, i12) -> column index i11 + i12*ne11; a uniform stride is needed: ne11 == 1 or contiguous
    GGML_ASSERT(ne11 == 1 || src1->nb[2] == (size_t) ne11*src1->nb[1]);
    const size_t col_stride = ne11 == 1 ? src1->nb[2] : src1->nb[1];
    if (is_q8_0) {
        const dim3 grid(ne00/QK8_0, ncols);
        cx_quantize_q8_0<<<grid, 32, 0, stream>>>((const char *) src1->data, col_stride, (block_q8_0 *) qy.get(), ne00/QK8_0);
    } else {
        static const bool fma_trick = [] { const char * e = getenv("LLAMA_CX_Q8K_FMA"); return e && atoi(e) != 0; }();
        const dim3 grid(ne00/QK_K, ncols);
        if (fma_trick) {
            cx_quantize_q8_K<true><<<grid, 256, 0, stream>>>((const char *) src1->data, col_stride, (block_q8_K *) qy.get(), ne00/QK_K);
        } else {
            cx_quantize_q8_K<false><<<grid, 256, 0, stream>>>((const char *) src1->data, col_stride, (block_q8_K *) qy.get(), ne00/QK_K);
        }
    }

    if (const char * dump = getenv("LLAMA_CX_DUMP"); dump && !is_q8_0) {   // debug: the quantized activations (q8_K calls)
        std::vector<char> h((size_t) ncols*col_bytes);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), qy.get(), h.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        FILE * f = fopen(dump, "wb"); if (f) { fwrite(h.data(), 1, h.size(), f); fclose(f); }
    }
    const bool zero_last = (src0->flags & GGML_TENSOR_FLAG_ZERO_LAST_EXPERT) != 0;
    const dim3 grid((ne01 + CX_ROWS - 1)/CX_ROWS, n_ids, ne12);
    const dim3 block(32, CX_ROWS);
#define CX_LAUNCH(T) cx_mmid_dot<T><<<grid, block, 0, stream>>>((const char *) src0->data, src0->nb[1], src0->nb[2], ne01, ne00, n_as, zero_last, \
        qy.get(), col_bytes, ne11, (const char *) ids->data, ids->nb[0], ids->nb[1], (char *) dst->data, dst->nb[1], dst->nb[2])
    switch (src0->type) {
        case GGML_TYPE_IQ3_XXS: CX_LAUNCH(GGML_TYPE_IQ3_XXS); break;
        case GGML_TYPE_IQ2_XXS: CX_LAUNCH(GGML_TYPE_IQ2_XXS); break;
        case GGML_TYPE_Q4_K:    CX_LAUNCH(GGML_TYPE_Q4_K);    break;
        case GGML_TYPE_IQ4_NL:  CX_LAUNCH(GGML_TYPE_IQ4_NL);  break;
        default: GGML_ABORT("cpu-exact: unsupported type");
    }
#undef CX_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- SwiGLU: ggml_vec_swiglu_f32 with AVX2 ggml_v_silu

static __device__ __forceinline__ float cx_v_expf(float x) {
    const float r = 0x1.8p23f;
    const float z = __fmaf_rn(x, 0x1.715476p+0f, r);
    const float n = ((z) - (r));
    const float b = __fmaf_rn(-n, 0x1.7f7d1cp-20f, __fmaf_rn(-n, 0x1.62e4p-1f, x));
    const uint32_t e = (uint32_t) __float_as_int(z) << 23;
    const float k = __int_as_float((int) (e + (uint32_t) __float_as_int(1.0f)));
    const bool  c = fabsf(n) > 126.0f;
    const float u = ((b) * (b));
    const float j = __fmaf_rn(__fmaf_rn(__fmaf_rn(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
                                        __fmaf_rn(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                              u, ((0x1.ffffecp-1f) * (b)));
    if (!c) {
        return __fmaf_rn(k, j, k);
    }
    const uint32_t g  = n <= 0.0f ? 0x82000000u : 0u;
    const float    s1 = __int_as_float((int) (g + 0x7f000000u));
    const float    s2 = __int_as_float((int) (e - g));
    if (fabsf(n) > 192.0f) {
        return ((s1) * (s1));
    }
    return ((__fmaf_rn(s2, j, s2)) * (s1));
}

static __global__ void cx_swiglu(const char * g, size_t g_nb1, const char * u, size_t u_nb1, char * dst, size_t dst_nb1, int nc) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= nc) {
        return;
    }
    const float x  = ((const float *) (g + r*g_nb1))[i];
    const float gv = ((const float *) (u + r*u_nb1))[i];
    const float silu = ((x) / (((1.0f) + (cx_v_expf(((0.0f) - (x)))))));
    ((float *) (dst + r*dst_nb1))[i] = ((silu) * (gv));
}

void ggml_cuda_swiglu_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(src1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
    const int nc = dst->ne[0];
    GGML_ASSERT(nc % 8 == 0);   // the CPU takes its vector path for every element
    const int nr = ggml_nrows(dst);
    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(src1) && ggml_is_contiguous_1(dst));
    GGML_ASSERT(src0->nb[2] == src0->ne[1]*src0->nb[1] && src1->nb[2] == src1->ne[1]*src1->nb[1] && dst->nb[2] == dst->ne[1]*dst->nb[1]);
    const ggml_tensor * gt = src0;   // with a separate src1 the CPU ignores the swapped flag
    const ggml_tensor * ut = src1;
    const dim3 grid((nc + 255)/256, nr);
    cx_swiglu<<<grid, 256, 0, ctx.stream()>>>((const char *) gt->data, gt->nb[1], (const char *) ut->data, ut->nb[1],
                                              (char *) dst->data, dst->nb[1], nc);
    CUDA_CHECK(cudaGetLastError());
}
