#pragma once
#include "common.cuh"

// (fork) LLAMA_FA_GQA=<G>: decode FlashAttention (vector kernel, D 256, q8_0 K/V) with G query heads per block,
// bit-identical to the original; false when not applicable (the caller then runs the original)
bool ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
