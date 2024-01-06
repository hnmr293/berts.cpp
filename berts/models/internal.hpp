#pragma once

#include <string>
#include <vector>
#include "berts/berts.h"
#include "berts/models/log.hpp"

namespace berts::internal {

//
// models
//

enum hidden_act {
    BERTS_HIDDEN_ACT_GELU,
    BERTS_HIDDEN_ACT_RELU,
    BERTS_HIDDEN_ACT_SILU,
    BERTS_HIDDEN_ACT_GELU_NEW,
};

struct hparams {
    bert_type architecture;
    bert_int vocab_size;
    bert_int hidden_dim;
    bert_int n_layers;
    bert_int attn_heads;
    bert_int max_tokens;
    bert_int intermediate_dim;
    bert_int segment_count;
    enum hidden_act hidden_act;
    double eps;
    double initializer_range;
};

struct model {
    ggml_type type;

    model(ggml_type type)
        : type(type) {}

    virtual ~model();

    //
    // initialize
    //

    virtual bool init_vocab(berts_context *ctx) = 0;

    virtual bool init_weight(berts_context *ctx) = 0;

    //
    // tokenizer
    //

    virtual std::string id_to_token(bert_token_t token_id) const noexcept = 0;

    virtual bert_token_t token_to_id(const std::string &token) const noexcept = 0;

    virtual bool add_token(const std::string &token) = 0;

    virtual bool has_token(const std::string &token) const noexcept = 0;

    virtual bert_token_t cls_id() const noexcept = 0;

    virtual bert_token_t mask_id() const noexcept = 0;

    virtual bert_token_t pad_id() const noexcept = 0;

    virtual bert_token_t sep_id() const noexcept = 0;

    virtual bert_token_t unk_id() const noexcept = 0;

    virtual bert_token_t bos_id() const noexcept = 0;

    virtual bert_token_t eos_id() const noexcept = 0;

    virtual size_t vocab_count() const noexcept = 0;

    virtual bool tokenize(const berts_context *ctx,
                          const std::string &text,
                          std::vector<bert_token_t> &out) const = 0;

    bool eval(berts_context *ctx,
              const std::vector<bert_token_t> &tokens,
              const berts_eval_info &cond,
              float *out,
              size_t &out_count) const;

    virtual bool eval(berts_context *ctx,
                      const std::vector<bert_token_t> &tokens,
                      const std::vector<bert_segment_t> &segments,
                      const berts_eval_info &cond,
                      float *out,
                      size_t &out_count) const = 0;
};

/// @brief create new `berts_context`
/// @param hparams hyper parameters
/// @param model model (invalidated if function call is failed)
/// @param gguf gguf context (invalidated if function call is failed)
/// @param ctx ggml context (invalidated if function call is failed)
/// @return a pointer to new `berts_context` or `nullptr` if function call is failed
berts_context *new_context(const hparams &hparams, model *model, gguf_context *gguf, ggml_context *ctx);

void free_context(berts_context *ctx);

gguf_context *get_gguf_context(berts_context *ctx);

ggml_context *get_ggml_context(berts_context *ctx);

bool get_hparams(const berts_context *ctx, hparams *params);

bool is_model_loaded(const berts_context *ctx);

model &get_model(berts_context *ctx);

const model &get_model(const berts_context *ctx);

//
// utilities
//

inline bool check_ctx(const berts_context *ctx) {
    if (!ctx) {
        log::warn("ctx=nullptr");
        return false;
    } else {
        return true;
    }
}

inline bool check_model(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (!is_model_loaded(ctx)) {
        log::error("model is not loaded");
        return false;
    }

    return true;
}

} // namespace berts::internal
