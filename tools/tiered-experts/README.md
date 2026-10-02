# Tiered experts (per-expert mixed precision) for qwen4exp

Fork feature: every MoE layer can hold its routed experts in **two groups with different quantization types**
(`ffn_{gate,up,down}_exps` = group 1, `ffn_{gate,up,down}_exps_t2` = group 2). Routing is unchanged: softmax and
top-k over all experts, weights renormalised as in `build_moe_ffn`; then one `mul_mat_id` per group, where slots routed
to the other group point at that group's all-zero dummy expert (last index), so they contribute nothing.

1. Decide group 1 per layer (the experts whose precision matters, measured e.g. with `patch_experts.py` + a quiz).
2. `python split_tiers.py --src model-bf16.gguf --tiers tiers.json --out model-tier.gguf --imatrix imatrix.gguf --imatrix-out imatrix-tier.gguf`
3. `llama-quantize --imatrix imatrix-tier.gguf --tensor-type "ffn_(gate|up|down)_exps_t2=iq2_xxs" --tensor-type "ffn_(gate|up|down)_exps=q4_K" model-tier.gguf out.gguf Q4_K_M`
   (the `_t2` rule must come first: the first matching rule wins).

Checks done: an unquantized split model gives KLD 0 / 100 % same top token vs the original; a corrupted router order
is detected (98.8 %). Only the qwen4exp graph uses it for now. Files without `_t2` tensors behave exactly as upstream.
