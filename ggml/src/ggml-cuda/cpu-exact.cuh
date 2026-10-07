#pragma once
// (fork) GPU kernels that reproduce, bit for bit, what the x86 AVX2 CPU backend computes for MoE experts
// (GGML_TENSOR_FLAG_CPU_EXACT): q8_K / q8_0 activation quantization, the IQ3_XXS, IQ2_XXS, Q4_K (x q8_K) and
// IQ4_NL (x q8_0) dot products with the same 8 int32 lanes, fma order and horizontal sum, and the vectorized SiLU
// of ggml_vec_swiglu_f32. Lets experts cached in VRAM give exactly the results of the CPU experts they replace.

#include "common.cuh"

bool ggml_cuda_cpu_exact_supported(const ggml_tensor * dst);

// MUL_MAT_ID with a CPU_EXACT src0
void ggml_cuda_mul_mat_id_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GLU (SWIGLU, split or not) flagged CPU_EXACT
void ggml_cuda_swiglu_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// MUL_MAT_ID gate + MUL_MAT_ID up + SWIGLU (all CPU_EXACT) in one kernel; false if the pattern does not fit
bool ggml_cuda_moe_gate_up_swiglu_cpu_exact(ggml_backend_cuda_context & ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu);
