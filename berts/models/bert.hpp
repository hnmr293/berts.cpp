#pragma once

#include <vector>
#include "berts/models/internal.hpp"

namespace berts::bert {

struct transformer_block {
    // attn
    ggml_tensor *q_w = nullptr;
    ggml_tensor *q_b = nullptr;

    ggml_tensor *k_w = nullptr;
    ggml_tensor *k_b = nullptr;

    ggml_tensor *v_w = nullptr;
    ggml_tensor *v_b = nullptr;

    // attn ff
    ggml_tensor *ff_w = nullptr;
    ggml_tensor *ff_b = nullptr;

    ggml_tensor *ln_ff_w = nullptr;
    ggml_tensor *ln_ff_b = nullptr;

    // intermediate
    ggml_tensor *i_w = nullptr;
    ggml_tensor *i_b = nullptr;

    // output
    ggml_tensor *o_w = nullptr;
    ggml_tensor *o_b = nullptr;

    ggml_tensor *ln_out_w = nullptr;
    ggml_tensor *ln_out_b = nullptr;
};

struct model : public internal::model {
    ggml_tensor *token_embedding = nullptr;
    ggml_tensor *segment_embedding = nullptr;
    ggml_tensor *position_embedding = nullptr;
    ggml_tensor *ln_w = nullptr;
    ggml_tensor *ln_b = nullptr;
    std::vector<transformer_block> layers;
    
    model(ggml_type type)
        : berts::internal::model(type) {}
    
    bool init_weight(berts_context *ctx) override;
    
    bool load_vocab(berts_context *ctx) override;
    
    ggml_tensor *eval(const berts_context *ctx, const std::vector<bert_token_t> &tokens, const std::vector<bert_segment_t> &segments) const override;
};

} // namespace berts::bert
