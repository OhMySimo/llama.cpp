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
// Each lane value is an exact integer (4-byte dot products through dp4a); only the float part follows the CPU order.

static __device__ __forceinline__ uint32_t cx_ld4(const uint8_t * p) {   // 2-byte aligned 32-bit load
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

// byte b = 0xff when bit b of the 4-bit sign group is set
static __device__ __forceinline__ uint32_t cx_mask4(uint32_t sb) {
    return ((sb & 1) ? 0x000000ffu : 0u) | ((sb & 2) ? 0x0000ff00u : 0u) | ((sb & 4) ? 0x00ff0000u : 0u) | ((sb & 8) ? 0xff000000u : 0u);
}

// sum of g[b]*q[b], with the bytes of g selected by neg taken negative (g bytes < 128)
static __device__ __forceinline__ int cx_signed_dot(uint32_t g, uint32_t neg, int q) {
    return ggml_cuda_dp4a((int) (g & ~neg), q, 0) - ggml_cuda_dp4a((int) (g & neg), q, 0);
}

// ggml_vec_dot_iq3_xxs_q8_K, AVX2: lane L of a 32-value sub-block = bytes 4L..4L+3 (grid entry q3[L]), signs from
// the 7-bit group L/2 of the sub-block word, scale 2*ls+1; lanes summed over the 8 sub-blocks
static __device__ __forceinline__ int cx_lane_iq3_xxs(const block_iq3_xxs * bx, const block_q8_K * by, int L) {
    const uint8_t * q3  = bx->qs;
    const uint8_t * gas = bx->qs + QK_K/4;
    int sum = 0;
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t aux = cx_ld4(gas + 4*ib32);
        const uint32_t neg = cx_mask4((ksigns_iq2xs[(aux >> (7*(L >> 1))) & 127] >> ((L & 1)*4)) & 0xf);
        const int      q   = *(const int *) (by->qs + 32*ib32 + 4*L);
        sum += (2*(int)(aux >> 28) + 1) * cx_signed_dot(iq3xxs_grid[q3[8*ib32 + L]], neg, q);
    }
    return sum;
}

// ggml_vec_dot_iq2_xxs_q8_K, AVX2: per sub-block a word of 4 grid indices (8 values each) and a word of 4 sign
// groups + the scale; lane L = half L%2 of grid entry L/2
static __device__ __forceinline__ int cx_lane_iq2_xxs(const block_iq2_xxs * bx, const block_q8_K * by, int L) {
    const uint8_t * q2 = (const uint8_t *) bx->qs;
    const int m = L >> 1;
    int sum = 0;
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t a0   = cx_ld4(q2 + 8*ib32);
        const uint32_t a1   = cx_ld4(q2 + 8*ib32 + 4);
        const uint64_t grid = iq2xxs_grid[(a0 >> (8*m)) & 0xff];
        const uint32_t g    = (L & 1) ? (uint32_t) (grid >> 32) : (uint32_t) grid;
        const uint32_t neg  = cx_mask4((ksigns_iq2xs[(a1 >> (7*m)) & 127] >> ((L & 1)*4)) & 0xf);
        const int      q    = *(const int *) (by->qs + 32*ib32 + 4*L);
        sum += (2*(int)(a1 >> 28) + 1) * cx_signed_dot(g, neg, q);
    }
    return sum;
}

static __device__ __forceinline__ void cx_q4_K_scales(const block_q4_K * bx, uint8_t * sc, uint8_t * mn) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    utmp[0] = *(const uint32_t *) (bx->scales + 0);
    utmp[1] = *(const uint32_t *) (bx->scales + 4);
    utmp[2] = *(const uint32_t *) (bx->scales + 8);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc[i] = (utmp[0] >> (8*i)) & 0xff; sc[4 + i] = (utmp[1] >> (8*i)) & 0xff;
        mn[i] = (utmp[2] >> (8*i)) & 0xff; mn[4 + i] = (utmp[3] >> (8*i)) & 0xff;
    }
}

// ggml_vec_dot_q4_K_q8_K, AVX2: lane L = sum over the 4 64-value groups of scale * (4 low-nibble products) +
// scale * (4 high-nibble products); *mins (L < 4): m[2L]*s[2L] + m[2L+1]*s[2L+1], s[k] = bsums[2k] + bsums[2k+1]
static __device__ __forceinline__ int cx_lane_q4_K(const block_q4_K * bx, const block_q8_K * by, int L, int * mins) {
    uint8_t sc[8], mn[8];
    cx_q4_K_scales(bx, sc, mn);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < QK_K/64; ++j) {
        const uint32_t q4 = *(const uint32_t *) (bx->qs + 32*j + 4*L);
        sum += sc[2*j]     * ggml_cuda_dp4a((int) (q4 & 0x0f0f0f0f),        *(const int *) (by->qs + 64*j + 4*L),      0);
        sum += sc[2*j + 1] * ggml_cuda_dp4a((int) ((q4 >> 4) & 0x0f0f0f0f), *(const int *) (by->qs + 64*j + 32 + 4*L), 0);
    }
    if (L < 4) {
        *mins = mn[2*L]*(by->bsums[4*L] + by->bsums[4*L + 1]) + mn[2*L + 1]*(by->bsums[4*L + 2] + by->bsums[4*L + 3]);
    }
    return sum;
}

// ggml_vec_dot_iq4_nl_q8_0, AVX2: lane L = elements 4L..4L+3 of a 32-value block
static __device__ __forceinline__ int cx_lane_iq4_nl(const block_iq4_nl * bx, const block_q8_0 * by, int L) {
    const uint32_t q4  = cx_ld4(bx->qs + 4*(L & 3));
    const uint32_t nib = L < 4 ? (q4 & 0x0f0f0f0f) : ((q4 >> 4) & 0x0f0f0f0f);
    const uint32_t v   = (uint32_t) (uint8_t) kvalues_iq4nl[nib & 0xf]         | ((uint32_t) (uint8_t) kvalues_iq4nl[(nib >> 8) & 0xf] << 8) |
                         ((uint32_t) (uint8_t) kvalues_iq4nl[(nib >> 16) & 0xf] << 16) | ((uint32_t) (uint8_t) kvalues_iq4nl[nib >> 24] << 24);
    return ggml_cuda_dp4a((int) v, (int) cx_ld4((const uint8_t *) by->qs + 4*L), 0);
}

// hsum_float_8: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))
static __device__ __forceinline__ float cx_hsum8(const float * a) {
    const float r0 = a[4] + a[0], r1 = a[5] + a[1], r2 = a[6] + a[2], r3 = a[7] + a[3];
    return (r0 + r2) + (r1 + r3);
}

// one row = 8 threads (lane L = thread & 7), each running the CPU's sequence for its lane; result valid in lane 0
template <ggml_type type>
static __device__ __forceinline__ float cx_row(const char * xrow, const char * ycol, int nb, int L) {
    float a[8];
    float acc = 0.0f, accm = 0.0f;
    if constexpr (type == GGML_TYPE_IQ4_NL) {
        // two accumulators (even / odd blocks), summed lane by lane before the horizontal sum
        const block_iq4_nl * bx = (const block_iq4_nl *) xrow;
        const block_q8_0   * by = (const block_q8_0 *) ycol;
        float acc1 = 0.0f, acc2 = 0.0f;
        for (int i = 0; i + 1 < nb; i += 2) {
            acc1 = __fmaf_rn(__half2float(by[i].d)     * __half2float(bx[i].d),     (float) cx_lane_iq4_nl(bx + i,     by + i,     L), acc1);
            acc2 = __fmaf_rn(__half2float(by[i + 1].d) * __half2float(bx[i + 1].d), (float) cx_lane_iq4_nl(bx + i + 1, by + i + 1, L), acc2);
        }
        acc = acc1 + acc2;
    } else {
        const block_q8_K * by = (const block_q8_K *) ycol;
        for (int i = 0; i < nb; ++i) {
            if constexpr (type == GGML_TYPE_IQ3_XXS) {
                const block_iq3_xxs * bx = (const block_iq3_xxs *) xrow + i;
                acc = __fmaf_rn(__half2float(bx->d) * by[i].d, (float) cx_lane_iq3_xxs(bx, by + i, L), acc);
            } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
                const block_iq2_xxs * bx = (const block_iq2_xxs *) xrow + i;
                acc = __fmaf_rn(__half2float(bx->d) * by[i].d, (float) cx_lane_iq2_xxs(bx, by + i, L), acc);
            } else {
                const block_q4_K * bx = (const block_q4_K *) xrow + i;
                int mins = 0;
                const int v = cx_lane_q4_K(bx, by + i, L, &mins);
                acc = __fmaf_rn(by[i].d * __low2float(bx->dm), (float) v, acc);
                if (L < 4) {
                    accm = __fmaf_rn(-by[i].d * __high2float(bx->dm), (float) mins, accm);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j] = __shfl(acc, j, 8);
    }
    const float h = cx_hsum8(a);
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return 0.25f * h;
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return 0.125f * h;
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        const float m0 = __shfl(accm, 0, 8), m1 = __shfl(accm, 1, 8), m2 = __shfl(accm, 2, 8), m3 = __shfl(accm, 3, 8);
        return h + ((m0 + m2) + (m1 + m3));
    } else {
        return h;
    }
}

static __device__ __forceinline__ float cx_v_expf(float x);

#define CX_MAX_NB      64
#define CX_ROWS_BLOCK  32   // 8 threads per row, 256 threads per block

// MUL_MAT_ID rows; FUSED: gate and up rows of the same expert, written as swiglu(gate, up)
template <ggml_type type, bool fused>
static __global__ void cx_mmid_rows(
        const char * src0, const char * src0_up, size_t nb01, size_t nb02, int ne01, int nb, int n_as, bool zero_last,
        const char * qy, size_t qy_col_bytes, int ne11,
        const char * ids, size_t ids_nb0, size_t ids_nb1,
        char * dst, size_t dst_nb1, size_t dst_nb2) {
    const int L    = threadIdx.x & 7;
    const int row  = blockIdx.x*CX_ROWS_BLOCK + (threadIdx.x >> 3);
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;
    if (row >= ne01) {
        return;   // the whole 8-thread group
    }
    const int expert = *(const int32_t *) (ids + slot*ids_nb0 + tok*ids_nb1);
    float * out = (float *) (dst + slot*dst_nb1 + tok*dst_nb2);
    if (zero_last && expert == n_as - 1) {
        if (L == 0) {
            out[row] = 0.0f;   // the CPU writes zeros for the dummy (and swiglu(0, 0) = +0)
        }
        return;
    }
    const size_t off  = (size_t) expert*nb02 + (size_t) row*nb01;
    const char * ycol = qy + (size_t) ((slot % ne11) + tok*ne11)*qy_col_bytes;
    const float r = cx_row<type>(src0 + off, ycol, nb, L);
    if constexpr (fused) {
        const float u = cx_row<type>(src0_up + off, ycol, nb, L);
        if (L == 0) {
            out[row] = (r / (1.0f + cx_v_expf(0.0f - r))) * u;
        }
    } else if (L == 0) {
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

static void cx_launch_rows(const ggml_tensor * src0, const ggml_tensor * src0_up, ggml_tensor * dst, const char * qy, size_t col_bytes, int ne11,
                           const ggml_tensor * ids, cudaStream_t stream) {
    const int ne00 = src0->ne[0], ne01 = src0->ne[1], n_as = src0->ne[2];
    const int nb = ne00 / (src0->type == GGML_TYPE_IQ4_NL ? QK8_0 : QK_K);
    const bool zero_last = (src0->flags & GGML_TENSOR_FLAG_ZERO_LAST_EXPERT) != 0;
    const dim3 grid((ne01 + CX_ROWS_BLOCK - 1)/CX_ROWS_BLOCK, ids->ne[0], ids->ne[1]);
    const char * up = src0_up ? (const char *) src0_up->data : nullptr;
#define CX_LAUNCH(T) do { if (up) cx_mmid_rows<T, true><<<grid, 8*CX_ROWS_BLOCK, 0, stream>>>((const char *) src0->data, up, src0->nb[1], src0->nb[2], ne01, nb, n_as, zero_last, \
        qy, col_bytes, ne11, (const char *) ids->data, ids->nb[0], ids->nb[1], (char *) dst->data, dst->nb[1], dst->nb[2]); \
        else cx_mmid_rows<T, false><<<grid, 8*CX_ROWS_BLOCK, 0, stream>>>((const char *) src0->data, nullptr, src0->nb[1], src0->nb[2], ne01, nb, n_as, zero_last, \
        qy, col_bytes, ne11, (const char *) ids->data, ids->nb[0], ids->nb[1], (char *) dst->data, dst->nb[1], dst->nb[2]); } while (0)
    switch (src0->type) {
        case GGML_TYPE_IQ3_XXS: CX_LAUNCH(GGML_TYPE_IQ3_XXS); break;
        case GGML_TYPE_IQ2_XXS: CX_LAUNCH(GGML_TYPE_IQ2_XXS); break;
        case GGML_TYPE_Q4_K:    CX_LAUNCH(GGML_TYPE_Q4_K);    break;
        case GGML_TYPE_IQ4_NL:  if (up) GGML_ABORT("cpu-exact: no fused iq4_nl"); CX_LAUNCH(GGML_TYPE_IQ4_NL); break;
        default: GGML_ABORT("cpu-exact: unsupported type");
    }
#undef CX_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

static void cx_mmid(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src0_up,
                    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float) && src1->ne[3] == 1);

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
    cx_launch_rows(src0, src0_up, dst, qy.get(), col_bytes, ne11, ids, stream);
}

void ggml_cuda_mul_mat_id_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_cpu_exact_supported(dst));
    cx_mmid(ctx, dst->src[0], nullptr, dst->src[1], dst->src[2], dst);
}

bool ggml_cuda_moe_gate_up_swiglu_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu) {
    if (!ggml_cuda_cpu_exact_supported(gate) || gate->src[0]->type == GGML_TYPE_IQ4_NL || up->src[0]->type != gate->src[0]->type ||
        up->src[1] != gate->src[1] || up->src[2] != gate->src[2] || up->src[0]->nb[1] != gate->src[0]->nb[1] ||
        up->src[0]->nb[2] != gate->src[0]->nb[2] || up->src[0]->ne[1] != gate->src[0]->ne[1] ||
        glu->src[0] != gate || glu->src[1] != up || glu->type != GGML_TYPE_F32 ||
        glu->nb[1] != gate->nb[1] || glu->nb[2] != gate->nb[2]) {
        return false;
    }
    cx_mmid(ctx, gate->src[0], up->src[0], gate->src[1], gate->src[2], glu);
    return true;
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
