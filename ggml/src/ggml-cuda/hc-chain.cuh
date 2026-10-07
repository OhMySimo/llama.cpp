#pragma once
#include "common.cuh"

// (fork) [HC_POST ->] RMS_NORM -> MUL -> MUL_MAT -> SCALE -> SILU -> MUL_MAT -> HC_PRE in one persistent kernel
// (hc-chain.cu); returns the number of extra nodes consumed, 0 if the pattern does not match (nothing launched)
int ggml_cuda_try_hc_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
