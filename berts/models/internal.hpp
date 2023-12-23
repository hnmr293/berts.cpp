#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "berts/berts.h"
#include "berts/models/log.hpp"

namespace berts::internal {

struct hparams {
    bert_type architecture;
    bert_int vocab_size;
    bert_int hidden_dim;
    bert_int n_layers;
    bert_int attn_heads;
    bert_int max_tokens;
    bert_int intermediate_dim;
    enum hidden_act hidden_act;
};

struct model {
    ggml_type type;

    model(ggml_type type)
        : type(type) {}

    virtual bool init_weight(berts_context *ctx) = 0;

    virtual bool load_vocab(berts_context *ctx) = 0;

    ggml_tensor *eval(berts_context *ctx, const std::vector<bert_token_t> &tokens);

    virtual ggml_tensor *eval(berts_context *ctx, const std::vector<bert_token_t> &tokens, const std::vector<bert_segment_t> &segments) = 0;
};

berts_context *new_context(const hparams &hparams, model *model, gguf_context *gguf, ggml_context *ctx);

void free_context(berts_context *ctx);

gguf_context *get_gguf_context(berts_context *ctx);

ggml_context *get_ggml_context(berts_context *ctx);

bool get_hparams(berts_context *ctx, hparams *params);

std::string id_to_token(berts_context *ctx, bert_token_t token_id);

bert_token_t token_to_id(berts_context *ctx, const std::string &token);

bool add_token(berts_context *ctx, const std::string &token);

bool has_token(berts_context *ctx, const std::string &token);

bool is_model_loaded(berts_context *ctx);

double get_eps(berts_context *ctx);

double set_eps(berts_context *ctx, double new_val);

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens);

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments);

//
// utilities
//

inline bool check_ctx(berts_context *ctx) {
    if (!ctx) {
        log::warn("ctx=nullptr");
        return false;
    } else {
        return true;
    }
}

inline bool check_model(berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (!is_model_loaded(ctx)) {
        log::error("model is not loaded");
        return false;
    }

    return true;
}

inline ggml_tensor *bert_dense(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
    return ggml_add(ctx,
                    ggml_mul_mat(ctx, w, x),
                    ggml_repeat(ctx, b, x));
}

inline ggml_tensor *bert_layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *ln_w, ggml_tensor *ln_b, float eps) {
    x = ggml_norm(ctx, x, eps);
    return ggml_add(ctx,
                    ggml_mul(ctx, ggml_repeat(ctx, ln_w, x), x),
                    ggml_repeat(ctx, ln_b, x));
}

} // namespace berts::internal
