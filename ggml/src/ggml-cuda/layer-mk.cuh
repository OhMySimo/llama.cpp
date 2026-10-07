#pragma once
#include "common.cuh"

// (fork) runs of single-token nodes (hyper-connection chains, linear attention) as stages of one persistent kernel
// (layer-mk.cu); returns the number of extra nodes consumed, 0 if nothing matched (nothing launched)
int ggml_cuda_try_layer_mk(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
