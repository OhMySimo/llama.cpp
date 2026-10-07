#include "cpu-exact.cuh"
#include "vecdotq.cuh"

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
struct cx_tabs { const uint32_t * g3; const uint64_t * g2; const uint32_t * m; };   // grids, sign masks (idx*2 + half)

static __device__ __forceinline__ int cx_lane_iq3_xxs(const block_iq3_xxs * bx, const block_q8_K * by, int L, const cx_tabs & T) {
    const uint8_t * q3  = bx->qs;
    const uint8_t * gas = bx->qs + QK_K/4;
    int sum = 0;
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t aux = cx_ld4(gas + 4*ib32);
        const uint32_t neg = T.m[((aux >> (7*(L >> 1))) & 127)*2 + (L & 1)];
        const int      q   = *(const int *) (by->qs + 32*ib32 + 4*L);
        sum += (2*(int)(aux >> 28) + 1) * cx_signed_dot(T.g3[q3[8*ib32 + L]], neg, q);
    }
    return sum;
}

// ggml_vec_dot_iq2_xxs_q8_K, AVX2: per sub-block a word of 4 grid indices (8 values each) and a word of 4 sign
// groups + the scale; lane L = half L%2 of grid entry L/2
static __device__ __forceinline__ int cx_lane_iq2_xxs(const block_iq2_xxs * bx, const block_q8_K * by, int L, const cx_tabs & T) {
    const uint8_t * q2 = (const uint8_t *) bx->qs;
    const int m = L >> 1;
    int sum = 0;
#pragma unroll
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const uint32_t a0   = cx_ld4(q2 + 8*ib32);
        const uint32_t a1   = cx_ld4(q2 + 8*ib32 + 4);
        const uint64_t grid = T.g2[(a0 >> (8*m)) & 0xff];
        const uint32_t g    = (L & 1) ? (uint32_t) (grid >> 32) : (uint32_t) grid;
        const uint32_t neg  = T.m[((a1 >> (7*m)) & 127)*2 + (L & 1)];
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
    const int2 v = get_int_from_table_16((int) cx_ld4(bx->qs + 4*(L & 3)), kvalues_iq4nl);   // byte values via v_perm
    return ggml_cuda_dp4a(L < 4 ? v.x : v.y, (int) cx_ld4((const uint8_t *) by->qs + 4*L), 0);
}

// hsum_float_8: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))
static __device__ __forceinline__ float cx_hsum8(const float * a) {
    const float r0 = a[4] + a[0], r1 = a[5] + a[1], r2 = a[6] + a[2], r3 = a[7] + a[3];
    return (r0 + r2) + (r1 + r3);
}

// one row = 8 threads (lane L = thread & 7), each running the CPU's sequence for its lane; result valid in lane 0
template <ggml_type type>
static __device__ __forceinline__ float cx_row(const char * xrow, const char * ycol, int nb, int L, const cx_tabs & T) {
    float a[8];
    float acc = 0.0f, accm = 0.0f;
    if constexpr (type == GGML_TYPE_IQ4_NL) {
        // two accumulators (even / odd blocks), summed lane by lane before the horizontal sum
        const block_iq4_nl * bx = (const block_iq4_nl *) xrow;
        const block_q8_0   * by = (const block_q8_0 *) ycol;
        float acc1 = 0.0f, acc2 = 0.0f;
#pragma unroll 2
        for (int i = 0; i + 1 < nb; i += 2) {
            acc1 = __fmaf_rn(__half2float(by[i].d)     * __half2float(bx[i].d),     (float) cx_lane_iq4_nl(bx + i,     by + i,     L), acc1);
            acc2 = __fmaf_rn(__half2float(by[i + 1].d) * __half2float(bx[i + 1].d), (float) cx_lane_iq4_nl(bx + i + 1, by + i + 1, L), acc2);
        }
        acc = acc1 + acc2;
    } else {
        const block_q8_K * by = (const block_q8_K *) ycol;
#pragma unroll 2
        for (int i = 0; i < nb; ++i) {
            if constexpr (type == GGML_TYPE_IQ3_XXS) {
                const block_iq3_xxs * bx = (const block_iq3_xxs *) xrow + i;
                acc = __fmaf_rn(__half2float(bx->d) * by[i].d, (float) cx_lane_iq3_xxs(bx, by + i, L, T), acc);
            } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
                const block_iq2_xxs * bx = (const block_iq2_xxs *) xrow + i;
                acc = __fmaf_rn(__half2float(bx->d) * by[i].d, (float) cx_lane_iq2_xxs(bx, by + i, L, T), acc);
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

// shared-memory copies of the lookup tables, filled by all threads of the block (call before any early return)
struct cx_stabs { uint32_t g3[256]; uint64_t g2[256]; uint32_t m[256]; };
static __device__ __forceinline__ cx_tabs cx_load_tabs(cx_stabs & S) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        S.g3[i] = iq3xxs_grid[i];
        S.g2[i] = iq2xxs_grid[i];
        S.m[i]  = cx_mask4((ksigns_iq2xs[i >> 1] >> ((i & 1)*4)) & 0xf);
    }
    __syncthreads();
    return { S.g3, S.g2, S.m };
}

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
    __shared__ cx_stabs S;
    const cx_tabs T = cx_load_tabs(S);
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
    const float r = cx_row<type>(src0 + off, ycol, nb, L, T);
    if constexpr (fused) {
        const float u = cx_row<type>(src0_up + off, ycol, nb, L, T);
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


// ---------------------------------------------------------------- whole cached-expert block in one op
// dst (op MUL_MAT_ID, flag CPU_EXACT, op_params[0] = GGML_CX_BLOCK_MAGIC, [1] cap of group 1 (0 = none), [2] cap of
// group 2): srcs 0 g1 gate (or g2 gate), 1 x [n_embd, 1, n_tok], 2 router top-k ids [k, n_tok], 3/4 g1 up/down,
// 5/6/7 g2 gate/up/down, 8/9 F32 maps global id -> slot (cap = not cached). Output [n_embd, k, n_tok]: the expert's
// output for slots cached in either group, 0 elsewhere. Three launches: q8_K of x, gate+up+swiglu, down (with the
// q8_0 of the swiglu output done per block). Same arithmetic as the separate ops.

static __device__ __forceinline__ float cx_row_dyn(int type, const char * xrow, const char * ycol, int nb, int L, const cx_tabs & T) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: return cx_row<GGML_TYPE_IQ3_XXS>(xrow, ycol, nb, L, T);
        case GGML_TYPE_IQ2_XXS: return cx_row<GGML_TYPE_IQ2_XXS>(xrow, ycol, nb, L, T);
        default:                return cx_row<GGML_TYPE_Q4_K>(xrow, ycol, nb, L, T);
    }
}


struct cx_grp { const char * gate; const char * up; const char * down; const float * map; size_t nb01, nb02, dnb01, dnb02; int type, cap; };

static __device__ __forceinline__ int cx_find(const cx_grp & g1, const cx_grp & g2, int e, int & cs) {
    if (g1.cap) { cs = (int) g1.map[e]; if (cs < g1.cap) return 0; }
    if (g2.cap) { cs = (int) g2.map[e]; if (cs < g2.cap) return 1; }
    return -1;
}

static __global__ void cx_block_gateup(cx_grp g1, cx_grp g2, const char * ids, size_t ids_nb0, size_t ids_nb1,
                                       const char * qy, size_t qy_col_bytes, int nb, int n_ff, int k, block_q8_0 * hq) {
    const int L = threadIdx.x & 7, row = blockIdx.x*CX_ROWS_BLOCK + (threadIdx.x >> 3), slot = blockIdx.y, tok = blockIdx.z;
    const int e = *(const int32_t *) (ids + slot*ids_nb0 + tok*ids_nb1);
    {   int cs0 = 0; if (cx_find(g1, g2, e, cs0) < 0) return; }   // whole block: not cached
    __shared__ cx_stabs S;
    const cx_tabs T = cx_load_tabs(S);
    int cs = 0;
    const int grp = cx_find(g1, g2, e, cs);
    const cx_grp & g = grp == 0 ? g1 : g2;
    const size_t off = (size_t) cs*g.nb02 + (size_t) row*g.nb01;
    const char * ycol = qy + (size_t) tok*qy_col_bytes;
    const float r = cx_row_dyn(g.type, g.gate + off, ycol, nb, L, T);
    const float u = cx_row_dyn(g.type, g.up   + off, ycol, nb, L, T);
    // this block's 32 rows are one q8_0 block of the swiglu output: quantize it here (x86 AVX2 quantize_row_q8_0),
    // once, instead of in every block of the down projection
    __shared__ float sh[CX_ROWS_BLOCK];
    if (L == 0) sh[threadIdx.x >> 3] = (r / (1.0f + cx_v_expf(0.0f - r))) * u;
    __syncthreads();
    if (threadIdx.x < 32) {
        const float v = sh[threadIdx.x];
        float m = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
        const float id = m != 0.0f ? 127.f / m : 0.0f;
        block_q8_0 & b = hq[((size_t) tok*k + slot)*(n_ff/QK8_0) + blockIdx.x];
        b.qs[threadIdx.x] = (int8_t) (int) rintf(v * id);
        if (threadIdx.x == 0) b.d = __float2half_rn(m / 127.f);
    }
}

static __global__ void cx_block_down(cx_grp g1, cx_grp g2, const char * ids, size_t ids_nb0, size_t ids_nb1,
                                     const block_q8_0 * hq, int n_ff, int n_embd, int k, char * dst, size_t dst_nb1, size_t dst_nb2) {
    const int L = threadIdx.x & 7, row = blockIdx.x*CX_ROWS_BLOCK + (threadIdx.x >> 3), slot = blockIdx.y, tok = blockIdx.z;
    if (row >= n_embd) return;
    const int e = *(const int32_t *) (ids + slot*ids_nb0 + tok*ids_nb1);
    float * out = (float *) (dst + slot*dst_nb1 + tok*dst_nb2);
    int cs = 0;
    const int grp = cx_find(g1, g2, e, cs);
    if (grp < 0) {
        if (L == 0) out[row] = 0.0f;
        return;
    }
    const int nbq = n_ff / QK8_0;
    const cx_grp & g = grp == 0 ? g1 : g2;
    const float r = cx_row<GGML_TYPE_IQ4_NL>(g.down + (size_t) cs*g.dnb02 + (size_t) row*g.dnb01,
                                             (const char *) (hq + ((size_t) tok*k + slot)*nbq), nbq, L, cx_tabs{});
    if (L == 0) out[row] = r;
}

static cudaEvent_t  g_cx_join_ev   = nullptr;
static cudaStream_t g_cx_join_main = nullptr;   // set while side work waits to be joined

// end of a graph evaluation: the main stream waits for the side-stream expert work (inside the capture if any)
void ggml_cuda_cx_side_join() {
    if (g_cx_join_main) {
        CUDA_CHECK(cudaStreamWaitEvent(g_cx_join_main, g_cx_join_ev, 0));
        g_cx_join_main = nullptr;
    }
}

void ggml_cuda_ecache_block_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t cap1 = ggml_get_op_params_i32(dst, 1), cap2 = ggml_get_op_params_i32(dst, 2);
    const ggml_tensor * x = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    auto grp = [&](const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down, const ggml_tensor * map, int cap) {
        cx_grp g = {};
        if (!cap) return g;
        GGML_ASSERT(gate->type == up->type && down->type == GGML_TYPE_IQ4_NL);
        GGML_ASSERT(gate->type == GGML_TYPE_IQ3_XXS || gate->type == GGML_TYPE_IQ2_XXS || gate->type == GGML_TYPE_Q4_K);
        g.gate = (const char *) gate->data; g.up = (const char *) up->data; g.down = (const char *) down->data;
        g.map = (const float *) map->data; g.nb01 = gate->nb[1]; g.nb02 = gate->nb[2]; g.dnb01 = down->nb[1]; g.dnb02 = down->nb[2];
        g.type = gate->type; g.cap = cap;
        return g;
    };
    const cx_grp g1 = grp(dst->src[0], dst->src[3], dst->src[4], dst->src[8], cap1);
    const cx_grp g2 = grp(cap1 ? dst->src[5] : dst->src[0], dst->src[6], dst->src[7], dst->src[9], cap2);
    const ggml_tensor * gt = dst->src[0];
    const int n_embd = x->ne[0], n_ff = gt->ne[1], k = ids->ne[0], n_tok = ids->ne[1];
    GGML_ASSERT(gt->ne[0] == n_embd && n_embd % QK_K == 0 && n_ff % QK8_0 == 0 && n_ff / QK8_0 <= CX_MAX_NB && n_embd / QK_K <= CX_MAX_NB);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && dst->ne[0] == n_embd);
    cudaStream_t stream = ctx.stream();

    const size_t col_bytes = ggml_row_size(GGML_TYPE_Q8_K, n_embd);
    GGML_ASSERT(n_ff % CX_ROWS_BLOCK == 0 && CX_ROWS_BLOCK == QK8_0);
    // LLAMA_CX_SIDE=1 (fork, decode only): gate/up and down run on a second stream, concurrently with the rest of the
    // overlap prefix (shared expert, router weights...); the main stream waits for them at the end of this graph
    // (ggml_cuda_cx_side_join). They read only private copies (q8_K of x, ids), so no later node can overwrite them.
    static const bool side_on = getenv("LLAMA_CX_SIDE") && atoi(getenv("LLAMA_CX_SIDE")) > 0;
    const bool side = side_on && n_tok == 1 && ctx.curr_stream_no == 0 && k <= 64;
    ggml_cuda_pool_alloc<char>  qy_pool;
    ggml_cuda_pool_alloc<block_q8_0> h_pool;
    char * qy_p; block_q8_0 * h_p; const char * ids_p = (const char *) ids->data;
    size_t ids_nb0 = ids->nb[0], ids_nb1 = ids->nb[1];
    static char * s_qy = nullptr; static block_q8_0 * s_h = nullptr; static int32_t * s_ids = nullptr; static size_t s_qy_n = 0, s_h_n = 0;
    if (side) {
        const size_t hn = (size_t) k*(n_ff/QK8_0);
        if (s_qy_n < col_bytes) { if (s_qy) CUDA_CHECK(cudaFree(s_qy)); CUDA_CHECK(cudaMalloc(&s_qy, col_bytes)); s_qy_n = col_bytes; }
        if (s_h_n < hn) { if (s_h) CUDA_CHECK(cudaFree(s_h)); CUDA_CHECK(cudaMalloc(&s_h, hn*sizeof(block_q8_0))); s_h_n = hn; }
        if (!s_ids) CUDA_CHECK(cudaMalloc(&s_ids, 64*sizeof(int32_t)));
        qy_p = s_qy; h_p = s_h;
        CUDA_CHECK(cudaMemcpy2DAsync(s_ids, sizeof(int32_t), ids->data, ids->nb[0], sizeof(int32_t), k, cudaMemcpyDeviceToDevice, stream));
        ids_p = (const char *) s_ids; ids_nb0 = sizeof(int32_t); ids_nb1 = k*sizeof(int32_t);
    } else {
        qy_pool.pool = &ctx.pool(); qy_p = qy_pool.alloc((size_t) n_tok*col_bytes);
        h_pool.pool  = &ctx.pool(); h_p  = h_pool.alloc((size_t) n_tok*k*(n_ff/QK8_0));
    }
    // LLAMA_CX_BLOCK_PROF=1: GPU time of the three launches, summed per 48 calls (one token)
    static const bool bprof = getenv("LLAMA_CX_BLOCK_PROF") != nullptr;
    static cudaEvent_t ev[4]; static bool ev_init = false; static double acc[3]; static int calls = 0;
    if (bprof && !ev_init) { for (auto & e : ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, 0)); ev_init = true; }
    if (bprof) CUDA_CHECK(cudaEventRecord(ev[0], stream));
    cx_quantize_q8_K<false><<<dim3(n_embd/QK_K, n_tok), 256, 0, stream>>>((const char *) x->data, x->nb[2], (block_q8_K *) qy_p, n_embd/QK_K);
    if (side) {
        static cudaEvent_t fork_ev = nullptr;
        if (!fork_ev) { CUDA_CHECK(cudaEventCreateWithFlags(&fork_ev, cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&g_cx_join_ev, cudaEventDisableTiming)); }
        CUDA_CHECK(cudaEventRecord(fork_ev, stream));
        stream = ctx.stream(ctx.device, 1);
        CUDA_CHECK(cudaStreamWaitEvent(stream, fork_ev, 0));
    }
    if (bprof) CUDA_CHECK(cudaEventRecord(ev[1], stream));
    cx_block_gateup<<<dim3((n_ff + CX_ROWS_BLOCK - 1)/CX_ROWS_BLOCK, k, n_tok), 8*CX_ROWS_BLOCK, 0, stream>>>(
        g1, g2, ids_p, ids_nb0, ids_nb1, qy_p, col_bytes, n_embd/QK_K, n_ff, k, h_p);
    if (bprof) CUDA_CHECK(cudaEventRecord(ev[2], stream));
    cx_block_down<<<dim3((n_embd + CX_ROWS_BLOCK - 1)/CX_ROWS_BLOCK, k, n_tok), 8*CX_ROWS_BLOCK, 0, stream>>>(
        g1, g2, ids_p, ids_nb0, ids_nb1, h_p, n_ff, n_embd, k, (char *) dst->data, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    if (side) {
        CUDA_CHECK(cudaEventRecord(g_cx_join_ev, stream));
        g_cx_join_main = ctx.stream();
    }
    if (bprof) {
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        CUDA_CHECK(cudaEventSynchronize(ev[3]));
        float a, b, c; CUDA_CHECK(hipEventElapsedTime(&a, ev[0], ev[1])); CUDA_CHECK(hipEventElapsedTime(&b, ev[1], ev[2])); CUDA_CHECK(hipEventElapsedTime(&c, ev[2], ev[3]));
        acc[0] += a; acc[1] += b; acc[2] += c;
        if (++calls % (48*64) == 0) { fprintf(stderr, "[cx-block] per token: quant %.2f ms, gate/up %.2f ms, down %.2f ms\n", acc[0]/64, acc[1]/64, acc[2]/64); acc[0] = acc[1] = acc[2] = 0; }
    }
}
