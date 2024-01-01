#pragma once

#include <vector>
#include <memory>
#include "berts/models/internal.hpp"

namespace berts::bert {

struct vocab;

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
    // weights
    ggml_tensor *token_embedding = nullptr;
    ggml_tensor *segment_embedding = nullptr;
    ggml_tensor *position_embedding = nullptr;
    ggml_tensor *ln_w = nullptr;
    ggml_tensor *ln_b = nullptr;
    std::vector<transformer_block> layers;
    ggml_tensor *pool_w = nullptr;
    ggml_tensor *pool_b = nullptr;
    
    // tokenizer
    std::unique_ptr<vocab> vocab;

    model(ggml_type type);
    
    ~model() override;

    bool init_vocab(berts_context *ctx) override;

    bool init_weight(berts_context *ctx) override;

    std::string id_to_token(bert_token_t token_id) const noexcept override;

    bert_token_t token_to_id(const std::string &token) const noexcept override;

    bool add_token(const std::string &token) override;

    bool has_token(const std::string &token) const noexcept override;

    bert_token_t cls_id() const noexcept override;

    bert_token_t mask_id() const noexcept override;

    bert_token_t pad_id() const noexcept override;

    bert_token_t sep_id() const noexcept override;

    bert_token_t unk_id() const noexcept override;
    
    bool tokenize(const berts_context *ctx,
                  const std::string &text,
                  std::vector<bert_token_t> &out) const override;

    bool eval(berts_context *ctx,
              const std::vector<bert_token_t> &tokens,
              const std::vector<bert_segment_t> &segments,
              const berts_eval_info &cond,
              float *out,
              size_t &out_count) const override;
};

} // namespace berts::bert
