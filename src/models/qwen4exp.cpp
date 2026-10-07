#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include "ggml-backend.h"
#include "gguf.h"
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
    hparams.indexer_kpool = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (hparams.indexer_kpool != 0 && r != hparams.indexer_kpool) {
            throw std::runtime_error(format("QSA layers must share one compress ratio, got %u and %u", hparams.indexer_kpool, r));
        }
        hparams.indexer_kpool = r;
    }
    if (hparams.indexer_kpool == 1 || (hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool != 0)) {
        throw std::runtime_error(format("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                        hparams.indexer_kpool, hparams.indexer_top_k));
    }
    // the reference groups the visible tokens in cache order and always keeps the tail
    hparams.indexer_kpool_row         = 2; // raw key | pooled key
    hparams.indexer_kpool_by_order    = true;
    hparams.indexer_kpool_select_tail = true;

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

static std::string g_qwen4exp_model_path;   // (fork) for the expert cache: experts are read from the file

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    g_qwen4exp_model_path = ml.fname_main;
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // an MTP-only file carries the MTP block, the embeddings and the LM head, but no trunk
    const bool mtp_only    = n_layer_nextn > 0 && ml.get_weight(tn(LLM_TENSOR_HC_ATTN_NORM, "weight", 0).str().c_str()) == nullptr;
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int  mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    auto load_block = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        // tiered experts (fork): two expert tensors per projection with different quant types; each group ends with one
        // all-zero dummy expert, router rows are ordered [group 1 | group 2] (see tools: split_tiers.py)
        const ggml_tensor * t2meta = (flags & TENSOR_SKIP) ? nullptr : ml.get_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXPS_T2, "weight", il).str().c_str());
        if (t2meta) {
            const ggml_tensor * t1meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il).str().c_str());
            GGML_ASSERT(t1meta && t1meta->ne[2] + t2meta->ne[2] - 2 == n_expert);
            const int64_t n1 = t1meta->ne[2], n2 = t2meta->ne[2];
            layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", il), { n_ff_exp, n_embd, n1 }, flags);
            layer.ffn_gate_exps    = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,    "weight", il), { n_embd, n_ff_exp, n1 }, flags);
            layer.ffn_up_exps      = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,      "weight", il), { n_embd, n_ff_exp, n1 }, flags);
            layer.ffn_down_exps_t2 = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS_T2, "weight", il), { n_ff_exp, n_embd, n2 }, flags);
            layer.ffn_gate_exps_t2 = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS_T2, "weight", il), { n_embd, n_ff_exp, n2 }, flags);
            layer.ffn_up_exps_t2   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS_T2,   "weight", il), { n_embd, n_ff_exp, n2 }, flags);
            // the slots routed to the other group point at this group's zero dummy: backends may skip it
            for (ggml_tensor * t : { layer.ffn_down_exps, layer.ffn_gate_exps, layer.ffn_up_exps,
                                     layer.ffn_down_exps_t2, layer.ffn_gate_exps_t2, layer.ffn_up_exps_t2 }) {
                // flags are set at load: the scheduler's copies of these tensors inherit them
                if (t) t->flags |= GGML_TENSOR_FLAG_ZERO_LAST_EXPERT | (getenv("LLAMA_EXPERT_CACHE") ? GGML_TENSOR_FLAG_EXPERT_CACHE_SKIP : 0);
            }
        } else {
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        }

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block(il, trunk_flags);
    }

    // the MTP block: one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream
    for (int il = n_layer; il < n_layer_all; ++il) {
        load_block(il, mtp_flags);

        auto & nextn = layers[il].nextn;
        nextn.eh_proj      = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,      "weight", il), { 2*n_embd, n_embd }, mtp_flags);
        nextn.enorm        = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,        "weight", il), { n_embd }, mtp_flags);
        // RMS per hc stream of the trunk residual, so the gammas load as [n_embd, hc] like the mixer norms
        nextn.hnorm        = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,        "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, mtp_flags);
        nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

// (fork) GPU work that does not depend on the CPU expert products, expanded right after them so that the scheduler
// runs it on the GPU while the CPU computes the experts (LLAMA_SCHED_OVERLAP): the FFN combine weights of the hyper
// connection, computed early with the same ops (LLAMA_HC_PREFIX=0 disables)
static ggml_tensor * g_hc_w_pre        = nullptr;
static ggml_tensor * g_hc_w_pre_inject = nullptr;
static ggml_tensor * g_ffn_prefix      = nullptr;
static bool hc_prefix_on() { static const bool on = !getenv("LLAMA_HC_PREFIX") || atoi(getenv("LLAMA_HC_PREFIX")) != 0; return on; }

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w;
    if (g_hc_w_pre && g_hc_w_pre_inject == inject) {
        w = g_hc_w_pre;
        g_hc_w_pre = g_hc_w_pre_inject = nullptr;
    } else {
        w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx0, w, 2.0f);
    }

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    // the QSA layers share one set of k-pool inputs
    // the CUDA lightning indexer takes 32 or 64 heads, QSA has a few, so it scores with plain ops
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_idx && hparams.indexer_kpool > 0) {
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
    // make sure hc_init is in the same graph split as the first layer (-sm tensor)
    ggml_build_forward_expand(gf, res_hc);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        if (hc_prefix_on() && model.layers[il].ffn_down_exps_t2) {
            g_hc_w_pre = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc)), 2.0f);
            g_hc_w_pre_inject = inject;
            g_ffn_prefix = g_hc_w_pre;
        }
        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // the MTP head reads the hc-wide residual, before the final mixer
    if (cparams.embeddings_nextn) {
        res->t_h_nextn = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        cb(res->t_h_nextn, "h_nextn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
    }

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP has a single block");
    GGML_ASSERT(ubatch.token && "qwen4exp MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "qwen4exp MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.hc_head_norm &&
            "MTP block missing, load the model with MTP enabled");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h = inp->h;

    res->add_input(std::move(inp));

    auto * inp_hyb = build_inp_mem_hybrid();
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);

    // the draft memory has no recurrent layer, but its input still has to be allocated
    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0) {
        GGML_ASSERT(mctx_hyb->get_idx()->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * h_norm = build_norm(ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens), layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0)); // [n_embd, hc, n_tokens]
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
    cur    = build_layer_attn(inp_hyb->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    cur    = build_hc_mix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
    cur    = build_layer_ffn(cur, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    // the next draft step reads this residual as its h
    ggml_tensor * flat     = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
    res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    cb(res->t_h_nextn, "h_nextn", il);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    cur = build_hc_mix(ggml_reshape_3d(ctx0, flat_out, n_embd, hc, flat_out->ne[1]),
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, il);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order, see llama_memory_hybrid_idx
class llama_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, false, new_pool_idxs, new_pool_rep,
                              ubatch, new_pool_pos);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // the scatter mask shape follows n_kv
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == mctx->get_n_kpool_new();
        res &= cache_safe        == mctx->get_kpool_cache_safe();

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]         cell caching each block's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for the padded blocks
    ggml_tensor * pool_mask     = nullptr; // F32 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]          cell to write each new pooled key into
    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0; // padded to a stable bound, never below 1
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
    bool cache_safe = true;
};

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pool, n_tokens);
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    // set_input fills them all, so keep them allocated even when no op reads them
    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv       = mctx_idx->get_n_kv();
    inp->n_new      = mctx_hyb->get_n_kpool_new();
    inp->cache_safe = mctx_hyb->get_kpool_cache_safe();
    // the top blocks plus the tail
    inp->n_sel      = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    if (inp->cache_safe) {
        inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
        ggml_set_input(inp->new_pool_rep);
    }
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_pos);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_sel(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    const int64_t n_new   = inp_kpool->n_new;

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == kpool);

    // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    cb(k_raw, "indexer_k_raw", il);

    ggml_tensor * pzero  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2*idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    // the raw keys and the persistent pooled slots, see llama_memory_hybrid_idx::mem_idx_stale
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    rows = ggml_reshape_3d(ctx0, rows, idx_dim, kpool, n_new);

    // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled_new = nullptr;
    for (int64_t i = 0; i < kpool; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx0, rows, idx_dim, n_new, rows->nb[2], i*rows->nb[1]);
        pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
    }
    pooled_new = ggml_scale(ctx0, pooled_new, 1.0f/(float) kpool);
    pooled_new = build_norm(pooled_new, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    pooled_new = ggml_reshape_3d(ctx0, pooled_new, idx_dim, 1, n_new);
    pooled_new = ggml_rope_multi(ctx0, pooled_new, inp_kpool->new_pool_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, idx_dim, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        // write before the pool gather
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    } else {
        // shared cells re-pool every pool, in layout order
        GGML_ASSERT(n_new < n_pool);
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_pool - n_new), 0.0f);
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim)
    // one product for all heads, then the heads are summed as slices, so nothing is transposed
    ggml_tensor * kq = ggml_mul_mat(ctx0,
            ggml_reshape_2d(ctx0, pooled, idx_dim, n_pool),
            ggml_reshape_2d(ctx0, q, idx_dim, n_idx_h*n_tokens)); // [n_pool, n_idx_h*n_tokens]
    kq = ggml_relu(ctx0, ggml_reshape_3d(ctx0, kq, n_pool, n_idx_h, n_tokens));

    ggml_tensor * score = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_2d(ctx0, kq, n_pool, n_tokens, kq->nb[2], h*kq->nb[1]);
        score = score ? ggml_add(ctx0, score, slice) : ggml_cont(ctx0, slice);
    }
    score = ggml_scale(ctx0, score, 1.0f/sqrtf((float) idx_dim));
    score = ggml_add(ctx0, score, inp_kpool->pool_mask); // [n_pool, n_tokens]
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], unordered
    cb(top_k, "indexer_top_k", il);

    // the top blocks, then the incomplete tail with n_kv for missing cells
    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens)); // [kpool, n_top_pool*n_tokens]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);
    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == inp_kpool->n_sel);

    ggml_build_forward_expand(gf, sel_idx);

    // TODO: figure out to reduce the large copmute buffer that this creates

    // scatter zeros for the selected cells into an all -inf row, each dead slot into its own dump row n_kv + slot
    // seeding from sel_idx ties the scatter storage lifetime to this layer
    const int64_t n_kv = inp_kpool->n_kv;

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and
    // top_k fills a short selection with invisible pools that can overlap the tail, so the scatter would write
    // some cells from several threads: map every dead slot to its own dump row, idx = dump + live*(idx - dump)
    // a picked pool is live when visible: a visible score is a rectified sum >= 0, an invisible one is -inf
    ggml_tensor * top_score = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
    ggml_tensor * live_pool = ggml_clamp(ctx0, ggml_scale_bias(ctx0, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, live_pool, kpool, n_top_pool, n_tokens, 1), kpool*n_top_pool, n_tokens);
    // a tail cell is live unless it is the n_kv sentinel
    ggml_tensor * live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);
    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]

    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source
    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
            row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
    sel = ggml_add(ctx0, sel, kq_mask);
    cb(sel, "indexer_sel", il);

    return sel;
}

// Dense GQA self-attention over the cells that the QSA mask keeps.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             sel,
        int64_t                   n_sel,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_sel, so the selection is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // the selection mask already carries the causal mask
    ggml_tensor * kq_mask = inp->get_kq_mask();
    ggml_tensor * mask    = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
    cb(mask, "kq_mask_qsa", il);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, mask, nullptr, nullptr, n_sel, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *   inp_kpool,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * sel = qsa ? build_qsa_sel(mctx_hyb, inp_kpool, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (sel) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, sel, inp_kpool->n_sel, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

// ---- (fork) adaptive expert cache for tiered experts -------------------------------------------------------------
// The most used experts of each (layer, group) are copied to VRAM and computed there, in parallel with the CPU, which
// skips them (GGML_TENSOR_FLAG_EXPERT_CACHE_SKIP + ggml_cpu_set_mmid_hook) and counts every expert it is handed.
// Between generated tokens, cold cached experts are swapped for hot uncached ones (same slots, same group). The model
// file is not changed: same experts, same types. Env:
//   LLAMA_EXPERT_CACHE=<profile> (tools/expert_cache_profile.py)   LLAMA_EXPERT_CACHE_MB (2048)
//   LLAMA_EXPERT_CACHE_EVERY (tokens between adaptations, 32)       LLAMA_EXPERT_CACHE_SWAPS (max swaps each, 24)
// ponytail: one model per process (static state); slots per (layer, group) fixed by the profile.
namespace {
typedef bool (*ggml_cpu_mmid_hook_fn)(const ggml_tensor * src0, int32_t expert, int64_t n_rows, bool count, void * ud);
struct ecache_group {
    ggml_tensor * base[3]  = {nullptr, nullptr, nullptr};  // gate, up, down (CPU)
    ggml_tensor * cache[3] = {nullptr, nullptr, nullptr};  // [ne0, ne1, cap + 1] on the GPU, slot cap = zero dummy
    ggml_tensor * map = nullptr;                            // F32 [1, n_expert]: global id -> slot, or cap
    int lo = 0, n = 0, cap = 0;                             // this group's experts are global ids lo .. lo + n - 1
    std::vector<int>   slot_of;                             // local expert -> slot, -1 = not cached
    std::vector<int>   expert_in;                           // slot -> local expert
    std::vector<float> score;                               // decayed uses per local expert
    std::vector<float> mapv;
};
struct tier_maps {     // (fork) per tiered layer: global expert id -> group-local id, or the group's zero dummy
    const llama_model * model = nullptr;
    ggml_context * ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
    std::vector<std::array<ggml_tensor *, 2>> m;
};
tier_maps g_tm;

void tier_maps_init(const llama_model & model) {
    for (const auto & l : model.layers) if (l.ffn_down_exps_t2 && !l.ffn_down_exps_t2->buffer) return;   // fit probe
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) return;
    if (g_tm.buf) ggml_backend_buffer_free(g_tm.buf);
    if (g_tm.ctx) ggml_free(g_tm.ctx);
    g_tm = tier_maps();
    g_tm.model = &model;
    const int n_layer = (int) model.layers.size();
    ggml_init_params ip = { (size_t) 2 * n_layer * ggml_tensor_overhead(), nullptr, true };
    g_tm.ctx = ggml_init(ip);
    g_tm.m.assign(n_layer, {nullptr, nullptr});
    for (int il = 0; il < n_layer; il++) {
        if (!model.layers[il].ffn_down_exps_t2) continue;
        for (int gi = 0; gi < 2; gi++) g_tm.m[il][gi] = ggml_new_tensor_2d(g_tm.ctx, GGML_TYPE_F32, 1, model.hparams.n_expert);
    }
    // (fork) LLAMA_TIER_MAP_CPU=1 (default): maps in RAM, so the id lookups run on the CPU, inside the expert split:
    // integer-valued ops, same results on any device, and a few kernels less on the GPU's critical path per layer
    static const bool map_cpu = !getenv("LLAMA_TIER_MAP_CPU") || atoi(getenv("LLAMA_TIER_MAP_CPU")) != 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    g_tm.buf = ggml_backend_alloc_ctx_tensors_from_buft(g_tm.ctx, ggml_backend_dev_buffer_type(map_cpu && cpu ? cpu : gpu));
    if (!g_tm.buf) { g_tm.model = nullptr; return; }
    ggml_backend_buffer_set_usage(g_tm.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> v(model.hparams.n_expert);
    for (int il = 0; il < n_layer; il++) {
        const auto & l = model.layers[il];
        if (!l.ffn_down_exps_t2) continue;
        const int n1 = (int) l.ffn_down_exps->ne[2] - 1, n2 = (int) l.ffn_down_exps_t2->ne[2] - 1;
        for (int gi = 0; gi < 2; gi++) {
            const int lo = gi ? n1 : 0, n = gi ? n2 : n1;
            for (int e = 0; e < (int) v.size(); e++) v[e] = (e >= lo && e < lo + n) ? (float) (e - lo) : (float) n;
            ggml_backend_tensor_set(g_tm.m[il][gi], v.data(), 0, v.size() * sizeof(float));
        }
    }
}

struct ecache_state {
    const llama_model * model = nullptr;              // the model the state belongs to (re-initialized for another)
    bool tried = false, on = false, active = false;   // active: the graph about to run computes the cache on the GPU
    std::vector<std::array<ecache_group, 2>> L;
    std::unordered_map<const ggml_tensor *, std::pair<int, int>> by_base;   // model tensors and the scheduler's copies
    std::unordered_map<std::string, std::pair<int, int>> by_name;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int64_t tokens = 0, swaps_total = 0, uses = 0, hits = 0;
    int every = 32, max_swaps = 24;
    gguf_context * gctx = nullptr; int fd = -1; size_t data_off = 0;   // source bytes when the CPU copy is repacked
    std::vector<uint8_t> tmp;
    const void * sig = nullptr;   // data pointer of a weight of the model: a probe model freed before the real one
                                  // was loaded can leave the same llama_model address
};
static const void * ecache_sig(const llama_model & model) {
    for (const auto & l : model.layers) if (l.ffn_down_exps_t2) return l.ffn_down_exps_t2->data;
    return nullptr;
}
ecache_state g_ec;

bool ecache_hook(const ggml_tensor * src0, int32_t e, int64_t n_rows, bool count, void *) {
    // src0 is a model tensor or a scheduler copy of it (same name). New copies are registered by the counting call,
    // which runs on one thread before the op's barrier; the skip queries of all threads come after it.
    auto it = g_ec.by_base.find(src0);
    if (it == g_ec.by_base.end()) {
        if (!count) return false;
        auto nt = g_ec.by_name.find(src0->name);
        if (nt == g_ec.by_name.end()) return false;
        it = g_ec.by_base.emplace(src0, nt->second).first;
    }
    ecache_group & g = g_ec.L[it->second.first][it->second.second];
    if (e < 0 || e >= g.n) return false;
    if (count) {
        if (strstr(src0->name, "ffn_gate_exps")) {   // gate only: one count per routed slot
            g.score[e] += (float) n_rows;
            g_ec.uses += n_rows; if (g.slot_of[e] >= 0) g_ec.hits += n_rows;
        }
        return false;
    }
    return g_ec.active && g.slot_of[e] >= 0;
}

size_t ecache_expert_bytes(const ggml_tensor * t) { return ggml_row_size(t->type, t->ne[0]) * t->ne[1]; }

// bytes of expert e of tensor t, read from the model file in the GGUF's own layout (the in-memory copy may be repacked
// or pinned elsewhere). The file is the .gguf this process maps, or LLAMA_EXPERT_CACHE_MODEL.
const uint8_t * ecache_src(const ggml_tensor * t, int e) {
    const size_t nb = ecache_expert_bytes(t);
    if (!g_ec.gctx) {
        std::string path = getenv("LLAMA_EXPERT_CACHE_MODEL") ? getenv("LLAMA_EXPERT_CACHE_MODEL") : g_qwen4exp_model_path;
        std::ifstream maps("/proc/self/maps"); std::string line;
        while (path.empty() && std::getline(maps, line)) {
            const size_t p = line.find('/');
            if (p != std::string::npos && line.size() > 5 && line.compare(line.size() - 5, 5, ".gguf") == 0) path = line.substr(p);
        }
        for (int fd = 0; path.empty() && fd < 1024; fd++) {     // or a .gguf this process still has open
            char link[64], buf[4096];
            snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
            const ssize_t n = readlink(link, buf, sizeof(buf) - 1);
            if (n > 5) { buf[n] = 0; if (strcmp(buf + n - 5, ".gguf") == 0) path = buf; }
        }
        if (path.empty()) { fprintf(stderr, "%s: model file not found: set LLAMA_EXPERT_CACHE_MODEL\n", __func__); return nullptr; }
        gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
        g_ec.gctx = gguf_init_from_file(path.c_str(), ip);
        g_ec.fd = open(path.c_str(), O_RDONLY);
        if (!g_ec.gctx || g_ec.fd < 0) return nullptr;
        g_ec.data_off = gguf_get_data_offset(g_ec.gctx);
    }
    const int64_t ti = gguf_find_tensor(g_ec.gctx, t->name);
    if (ti < 0) return nullptr;
    g_ec.tmp.resize(nb);
    const off_t off = (off_t) (g_ec.data_off + gguf_get_tensor_offset(g_ec.gctx, ti) + (size_t) e * nb);
    return pread(g_ec.fd, g_ec.tmp.data(), nb, off) == (ssize_t) nb ? g_ec.tmp.data() : nullptr;
}

bool ecache_load_expert(ecache_group & g, int slot, int e) {
    for (int r = 0; r < 3; r++) {
        const uint8_t * src = ecache_src(g.base[r], e);
        if (!src) return false;
        ggml_backend_tensor_set(g.cache[r], src, (size_t) slot * g.cache[r]->nb[2], ecache_expert_bytes(g.base[r]));
    }
    return true;
}

// LLAMA_EXPERT_CACHE_EXACT (default 1): the cached experts run with the CUDA kernels that reproduce the CPU's
// arithmetic (GGML_TENSOR_FLAG_CPU_EXACT), only for single-token steps (the CPU's own path for those): outputs are
// identical to an all-CPU run. 0: llama.cpp's GPU kernels, faster to write but numerically different
static bool ecache_exact() { static const bool on = !getenv("LLAMA_EXPERT_CACHE_EXACT") || atoi(getenv("LLAMA_EXPERT_CACHE_EXACT")) != 0; return on; }

void ecache_init(const llama_model & model) {
    const char * prof = getenv("LLAMA_EXPERT_CACHE");
    if (!prof) { g_ec.tried = true; return; }
    for (const auto & l : model.layers) {           // a memory-fit probe model has no weights: wait for the real one
        if (l.ffn_down_exps_t2 && (!l.ffn_down_exps_t2->buffer || !l.ffn_down_exps_t2->data)) return;
    }
    if (g_ec.buf) ggml_backend_buffer_free(g_ec.buf);
    if (g_ec.ctx) ggml_free(g_ec.ctx);
    if (g_ec.gctx) gguf_free(g_ec.gctx);
    if (g_ec.fd >= 0) close(g_ec.fd);
    g_ec = ecache_state();
    g_ec.model = &model;
    g_ec.sig   = ecache_sig(model);
    g_ec.tried = true;
    const double budget = (getenv("LLAMA_EXPERT_CACHE_MB") ? atof(getenv("LLAMA_EXPERT_CACHE_MB")) : 2048.0) * 1048576.0;
    if (getenv("LLAMA_EXPERT_CACHE_EVERY")) g_ec.every     = std::max(1, atoi(getenv("LLAMA_EXPERT_CACHE_EVERY")));
    if (getenv("LLAMA_EXPERT_CACHE_SWAPS")) g_ec.max_swaps = std::max(0, atoi(getenv("LLAMA_EXPERT_CACHE_SWAPS")));
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!gpu || !cpu) { fprintf(stderr, "%s: no GPU or CPU device, expert cache off\n", __func__); return; }

    const int n_layer = (int) model.layers.size();
    g_ec.L.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        const auto & l = model.layers[il];
        if (!l.ffn_down_exps_t2) continue;
        ggml_tensor * t[2][3] = {{l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps}, {l.ffn_gate_exps_t2, l.ffn_up_exps_t2, l.ffn_down_exps_t2}};
        int lo = 0;
        for (int gi = 0; gi < 2; gi++) {
            ecache_group & g = g_ec.L[il][gi];
            for (int r = 0; r < 3; r++) g.base[r] = t[gi][r];
            g.lo = lo; g.n = (int) t[gi][2]->ne[2] - 1; lo += g.n;
            g.slot_of.assign(g.n, -1); g.score.assign(g.n, 0.0f);
            for (int r = 0; r < 3; r++) { g_ec.by_base[t[gi][r]] = {il, gi}; g_ec.by_name[t[gi][r]->name] = {il, gi}; }
        }
    }
    // profile: most valuable first = uses per byte; slots per (layer, group) = what fits in the budget
    struct cand { int il, gi, e; double uses, bytes; };
    std::vector<cand> cs;
    std::ifstream f(prof); std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        int il, gr, e; double u;
        if (sscanf(line.c_str(), "%d %d %d %lf", &il, &gr, &e, &u) != 4) continue;
        if (il < 0 || il >= n_layer || gr < 1 || gr > 2 || !g_ec.L[il][gr - 1].base[0]) continue;
        ecache_group & g = g_ec.L[il][gr - 1];
        if (e < 0 || e >= g.n) continue;
        const double b = (double) ecache_expert_bytes(g.base[0]) + ecache_expert_bytes(g.base[1]) + ecache_expert_bytes(g.base[2]);
        cs.push_back({il, gr - 1, e, u, b});
        g.score[e] = (float) (u * 1e-3);       // a weak prior: runtime counts take over quickly
    }
    std::sort(cs.begin(), cs.end(), [](const cand & a, const cand & b) { return a.uses / a.bytes > b.uses / b.bytes; });
    double used = 0;
    std::vector<cand> pick;
    for (const cand & c : cs) { if (used + c.bytes > budget) continue; used += c.bytes; pick.push_back(c); }
    for (const cand & c : pick) g_ec.L[c.il][c.gi].cap++;

    int n_tensors = 0;
    for (auto & lg : g_ec.L) for (auto & g : lg) if (g.cap > 0) n_tensors += 4;
    if (n_tensors == 0) { fprintf(stderr, "%s: empty expert cache (budget %.0f MB)\n", __func__, budget / 1048576.0); return; }
    ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead(), nullptr, true };
    g_ec.ctx = ggml_init(ip);
    for (auto & lg : g_ec.L) for (auto & g : lg) {
        if (g.cap == 0) continue;
        for (int r = 0; r < 3; r++) {
            g.cache[r] = ggml_new_tensor_3d(g_ec.ctx, g.base[r]->type, g.base[r]->ne[0], g.base[r]->ne[1], g.cap + 1);
            g.cache[r]->flags |= GGML_TENSOR_FLAG_ZERO_LAST_EXPERT;
            if (ecache_exact()) g.cache[r]->flags |= GGML_TENSOR_FLAG_CPU_EXACT;   // GPU results = CPU results, bit for bit
        }
        g.map = ggml_new_tensor_2d(g_ec.ctx, GGML_TYPE_F32, 1, model.hparams.n_expert);
    }
    g_ec.buf = ggml_backend_alloc_ctx_tensors_from_buft(g_ec.ctx, ggml_backend_dev_buffer_type(gpu));
    if (!g_ec.buf) { fprintf(stderr, "%s: could not allocate %.0f MB of VRAM, expert cache off\n", __func__, used / 1048576.0); return; }
    ggml_backend_buffer_set_usage(g_ec.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & lg : g_ec.L) for (auto & g : lg) if (g.cap > 0) g.expert_in.assign(g.cap, -1);
    for (const cand & c : pick) {
        ecache_group & g = g_ec.L[c.il][c.gi];
        int slot = 0; while (g.expert_in[slot] >= 0) slot++;
        if (!ecache_load_expert(g, slot, c.e)) { fprintf(stderr, "%s: cannot read expert bytes, expert cache off\n", __func__); return; }
        g.expert_in[slot] = c.e; g.slot_of[c.e] = slot;
    }
    for (auto & lg : g_ec.L) for (auto & g : lg) {
        if (g.cap == 0) continue;
        for (int r = 0; r < 3; r++) ggml_backend_tensor_memset(g.cache[r], 0, (size_t) g.cap * g.cache[r]->nb[2], g.cache[r]->nb[2]);
        g.mapv.assign(model.hparams.n_expert, (float) g.cap);
        for (int e = 0; e < g.n; e++) if (g.slot_of[e] >= 0) g.mapv[g.lo + e] = (float) g.slot_of[e];
        ggml_backend_tensor_set(g.map, g.mapv.data(), 0, g.mapv.size() * sizeof(float));
    }
    auto set_hook = (void (*)(ggml_cpu_mmid_hook_fn, void *)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu), "ggml_cpu_set_mmid_hook");
    if (!set_hook) { fprintf(stderr, "%s: the CPU backend has no mmid hook, expert cache off\n", __func__); return; }
    set_hook(ecache_hook, nullptr);
    for (auto & lg : g_ec.L) for (auto & g : lg) for (int r = 0; r < 3; r++) if (g.base[r]) g.base[r]->flags |= GGML_TENSOR_FLAG_EXPERT_CACHE_SKIP;
    if (getenv("LLAMA_EC_DBG")) fprintf(stderr, "[ec-init] %s %p flags %d\n", g_ec.L[0][0].base[0]->name, (void *) g_ec.L[0][0].base[0], g_ec.L[0][0].base[0]->flags);
    g_ec.on = true;
    fprintf(stderr, "%s: %zu experts (%.0f MB) cached in VRAM from %s; adapting every %d tokens, up to %d swaps\n",
                   __func__, pick.size(), used / 1048576.0, prof, g_ec.every, g_ec.max_swaps);
}

// between generated tokens (nothing is computing): swap cold cached experts for hot uncached ones
void ecache_adapt() {
    if (!g_ec.on) return;
    if (++g_ec.tokens % g_ec.every) return;
    struct sw { float gain; int il, gi, hot, slot; };
    std::vector<sw> sws;
    for (int il = 0; il < (int) g_ec.L.size(); il++) for (int gi = 0; gi < 2; gi++) {
        ecache_group & g = g_ec.L[il][gi];
        if (g.cap == 0) continue;
        std::vector<int> hot, cold;
        for (int e = 0; e < g.n; e++) (g.slot_of[e] >= 0 ? cold : hot).push_back(e);
        std::sort(hot.begin(),  hot.end(),  [&](int a, int b) { return g.score[a] > g.score[b]; });
        std::sort(cold.begin(), cold.end(), [&](int a, int b) { return g.score[a] < g.score[b]; });
        for (size_t i = 0; i < hot.size() && i < cold.size(); i++) {
            const float h = g.score[hot[i]], c = g.score[cold[i]];
            if (h < 1.25f * c + 1.0f) break;     // hysteresis: only clearly hotter experts move in
            sws.push_back({h - c, il, gi, hot[i], g.slot_of[cold[i]]});
        }
    }
    std::sort(sws.begin(), sws.end(), [](const sw & a, const sw & b) { return a.gain > b.gain; });
    if ((int) sws.size() > g_ec.max_swaps) sws.resize(g_ec.max_swaps);
    std::vector<ecache_group *> dirty;
    for (const sw & s : sws) {
        ecache_group & g = g_ec.L[s.il][s.gi];
        const int old = g.expert_in[s.slot];
        if (!ecache_load_expert(g, s.slot, s.hot)) continue;
        g.slot_of[old] = -1; g.mapv[g.lo + old] = (float) g.cap;
        g.slot_of[s.hot] = s.slot; g.expert_in[s.slot] = s.hot; g.mapv[g.lo + s.hot] = (float) s.slot;
        dirty.push_back(&g);
        g_ec.swaps_total++;
    }
    for (ecache_group * g : dirty) ggml_backend_tensor_set(g->map, g->mapv.data(), 0, g->mapv.size() * sizeof(float));
    for (auto & lg : g_ec.L) for (auto & g : lg) for (float & v : g.score) v *= 0.9f;   // forget slowly
    if (getenv("LLAMA_EXPERT_CACHE_LOG") && g_ec.tokens % (g_ec.every * 8) == 0) {
        fprintf(stderr, "%s: %lld tokens, hit rate %.1f%%, %lld swaps so far\n", __func__, (long long) g_ec.tokens,
                       g_ec.uses ? 100.0 * g_ec.hits / g_ec.uses : 0.0, (long long) g_ec.swaps_total);
        g_ec.uses = g_ec.hits = 0;
    }
}
} // namespace

// runs ecache_adapt before each single-token step (the previous step's results have been read: nothing in flight)
class llm_graph_input_ecache : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override {
        g_ec.active = small && !getenv("LLAMA_EC_NOSKIP");
        static int dbg = getenv("LLAMA_EC_DBG") ? 0 : 1000; if (dbg < 3) { dbg++; fprintf(stderr, "[ec-input] n_tokens %d small %d\n", (int) ubatch->n_tokens, (int) small); }                    // the CPU skips cached experts only when this graph computes them on the GPU
        if (small && ubatch->n_tokens == 1) ecache_adapt();
    }
    bool can_reuse(const llm_graph_params & params) override { return (params.ubatch.n_tokens < 32) == small; }
    bool small = true;
};

// tiered experts (fork): same routing as build_moe_ffn (softmax, top-k over ALL experts, renormalised weights), then
// one mul_mat_id per group; a slot routed to the other group points at that group's all-zero dummy expert
ggml_tensor * llama_model_qwen4exp::graph::build_moe_tiered(ggml_tensor * cur, const int il) {
    const auto & L = model.layers[il];
    const int64_t n_tok = cur->ne[1];
    const int64_t k     = n_expert_used;
    const int64_t n1    = L.ffn_down_exps->ne[2] - 1;
    const int64_t n2    = L.ffn_down_exps_t2->ne[2] - 1;

    ggml_tensor * probs = ggml_soft_max(ctx0, build_lora_mm(L.ffn_gate_inp, cur));                      // [n_expert, n_tok]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx0, probs, k);                                             // [k, n_tok]
    ggml_tensor * w     = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tok), sel);     // [1, k, n_tok]
    w = ggml_reshape_2d(ctx0, w, k, n_tok);
    w = ggml_div(ctx0, w, ggml_clamp(ctx0, ggml_sum_rows(ctx0, w), 6.103515625e-5, INFINITY));
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
    }
    ggml_tensor * self = ggml_cast(ctx0, sel, GGML_TYPE_F32);
    ggml_tensor * x    = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tok);
    if (!g_ec.tried || g_ec.model != &model || g_ec.sig != ecache_sig(model)) ecache_init(model);
    if (g_tm.model != &model) tier_maps_init(model);
    const bool lookup = g_tm.model == &model && !getenv("LLAMA_TIER_ARITH");   // 2 ops per group instead of ~10
    ggml_tensor * sel_flat0 = lookup ? ggml_reshape_1d(ctx0, ggml_cont(ctx0, sel), k * n_tok) : nullptr;
    // bigger batches: llama.cpp streams all the experts to the GPU anyway. Exact mode: single tokens only (with more
    // tokens the CPU may take its tiled path for an expert with several rows, which the exact kernels do not mirror)
    const bool use_cache = g_ec.on && (ecache_exact() ? n_tok == 1 : n_tok < 32);
    if (g_ec.on && il == 0) {
        auto inp = std::make_unique<llm_graph_input_ecache>();
        inp->small = use_cache;
        res->add_input(std::move(inp));
    }

    auto prod = [&](ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, ggml_tensor * ids) {
        ggml_tensor * g = build_lora_mm_id(gate, x, ids, nullptr);
        ggml_tensor * u = build_lora_mm_id(up,   x, ids, nullptr);
        return build_lora_mm_id(down, ggml_swiglu_split(ctx0, g, u), ids, nullptr);
    };
    auto tier = [&](ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, float lo, int64_t n) {
        if (lookup) {
            ggml_tensor * map = g_tm.m[il][lo == 0.0f ? 0 : 1];
            ggml_tensor * ids = ggml_cast(ctx0, ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, map, sel_flat0), k, n_tok), GGML_TYPE_I32);
            ggml_tensor * g = build_lora_mm_id(gate, x, ids, nullptr);
            ggml_tensor * u = build_lora_mm_id(up,   x, ids, nullptr);
            return build_lora_mm_id(down, ggml_swiglu_split(ctx0, g, u), ids, nullptr);
        }
        // m = 1 if lo <= sel < lo + n ; local id = sel - lo inside the group, n (= the zero dummy) outside
        ggml_tensor * m   = ggml_mul(ctx0, ggml_step(ctx0, ggml_scale_bias(ctx0, self,  1.0f, 0.5f - lo)),
                                           ggml_step(ctx0, ggml_scale_bias(ctx0, self, -1.0f, lo + (float) n - 0.5f)));
        ggml_tensor * idf = ggml_add(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, self, 1.0f, -lo), m),
                                           ggml_scale_bias(ctx0, m, -(float) n, (float) n));
        ggml_tensor * ids = ggml_cast(ctx0, idf, GGML_TYPE_I32);
        ggml_tensor * g = build_lora_mm_id(gate, x, ids, nullptr);   // [n_ff, k, n_tok]
        ggml_tensor * u = build_lora_mm_id(up,   x, ids, nullptr);
        return build_lora_mm_id(down, ggml_swiglu_split(ctx0, g, u), ids, nullptr);  // [n_embd, k, n_tok]
    };
    const bool lc = use_cache && il < (int) g_ec.L.size() && g_ec.L[il][0].base[0];
    ggml_tensor * e1;
    ggml_tensor * e2;
    if (lookup && !getenv("LLAMA_TIER_IDS_LATE")) {
        // both groups' expert ids (GPU lookups) before any expert product (CPU): one CPU split per layer instead of
        // two, i.e. one GPU<->CPU round trip less per layer. Same operations, only the graph order changes.
        ggml_tensor * ids1 = ggml_cast(ctx0, ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, g_tm.m[il][0], sel_flat0), k, n_tok), GGML_TYPE_I32);
        ggml_tensor * ids2 = ggml_cast(ctx0, ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, g_tm.m[il][1], sel_flat0), k, n_tok), GGML_TYPE_I32);
        ggml_build_forward_expand(gf, ids1);
        ggml_build_forward_expand(gf, ids2);
        e1 = prod(L.ffn_gate_exps,    L.ffn_up_exps,    L.ffn_down_exps,    ids1);
        e2 = prod(L.ffn_gate_exps_t2, L.ffn_up_exps_t2, L.ffn_down_exps_t2, ids2);
    } else {
        e1 = tier(L.ffn_gate_exps,    L.ffn_up_exps,    L.ffn_down_exps,    0.0f,        n1);
        e2 = tier(L.ffn_gate_exps_t2, L.ffn_up_exps_t2, L.ffn_down_exps_t2, (float) n1, n2);
    }
    // graph order: both groups' expert products first, so the caller can place the shared expert right after them
    // (the scheduler then runs it on the GPU while the CPU computes these products: LLAMA_SCHED_OVERLAP)
    ggml_build_forward_expand(gf, e1);
    ggml_build_forward_expand(gf, e2);
    ggml_tensor * e = ggml_add(ctx0, e1, e2);
    if (n_tok < 32) {
        cb(e, "ffn_moe_e12_cpu", il);   // on the CPU (graph callback): one tensor to send to the GPU instead of two
        ggml_build_forward_expand(gf, e);
    }
    ggml_tensor * eg = nullptr;
    if (lc && !getenv("LLAMA_EC_NOGPU")) {
        // cached experts on the GPU, expanded here: the head of the GPU split that follows the CPU products, so the
        // scheduler launches them before the CPU work (overlap). The CPU writes zeros for these slots and the GPU
        // writes zeros for the others, so eg + e gives every slot its single value (x + 0 = x)
        ggml_tensor * sel_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, sel), k * n_tok);
        for (int gi = 0; gi < 2; gi++) {
            const ecache_group & g = g_ec.L[il][gi];
            if (g.cap == 0) continue;
            ggml_tensor * idg = ggml_cast(ctx0, ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, g.map, sel_flat), k, n_tok), GGML_TYPE_I32);
            ggml_tensor * gg = build_lora_mm_id(g.cache[0], x, idg, nullptr);
            ggml_tensor * uu = build_lora_mm_id(g.cache[1], x, idg, nullptr);
            ggml_tensor * hh = ggml_swiglu_split(ctx0, gg, uu);
            if (ecache_exact()) hh->flags |= GGML_TENSOR_FLAG_CPU_EXACT;
            ggml_tensor * pg = build_lora_mm_id(g.cache[2], hh, idg, nullptr);
            eg = eg ? ggml_add(ctx0, eg, pg) : pg;
        }
        if (eg) ggml_build_forward_expand(gf, eg);
    }
    if (hc_prefix_on()) {
        ggml_build_forward_expand(gf, w);
        if (g_ffn_prefix) {
            ggml_build_forward_expand(gf, g_ffn_prefix);
            g_ffn_prefix = nullptr;
        }
    }
    if (eg) e = ggml_add(ctx0, eg, e);
    e = ggml_mul(ctx0, e, ggml_reshape_3d(ctx0, w, 1, k, n_tok));
    if (getenv("LLAMA_TIER_SUMROWS")) {   // opt-in: one reduction instead of k-1 adds (NOT bit-identical: sum order)
        ggml_tensor * t = ggml_cont(ctx0, ggml_permute(ctx0, e, 1, 0, 2, 3));      // [k, n_embd, n_tok]
        return ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, t), n_embd, n_tok);
    }
    ggml_tensor * out = ggml_view_2d(ctx0, e, n_embd, n_tok, e->nb[2], 0);
    for (int64_t i = 1; i < k; ++i) {
        out = ggml_add(ctx0, out, ggml_view_2d(ctx0, e, n_embd, n_tok, e->nb[2], i*e->nb[1]));
    }
    return out;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out = model.layers[il].ffn_down_exps_t2 ? build_moe_tiered(cur, il) :
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        ggml_build_forward_expand(gf, ffn_shexp);  // before the routed experts' combination (see build_moe_tiered)

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
    llm_graph_input_qwen4exp_ple(const llama_model & model,
                        const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}
    virtual ~llm_graph_input_qwen4exp_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) model.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_qwen4exp_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hparams = model.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hparams.ple_image_token_id != 0
        ? (llama_token) hparams.ple_image_token_id
        : (llama_token) hparams.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t n_heads  = hparams.ple_n_heads;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
            }
        }
    }

    {
        ggml_tensor * ple = model.per_layer_tok_embd;

        const bool prefetch = model.can_prefetch.count(ple);
        if (prefetch) {
            llama_prefetch_rows(ple, idx.data(), idx.size());
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_qwen4exp_ple>(model, mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
