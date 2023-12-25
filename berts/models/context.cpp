#include "berts/models/context.hpp"
#include <cmath>
#include <memory>
#include "berts/common/log.hpp"
#include "berts/models/utils.hpp"

using namespace berts;
using namespace berts::models;

struct berts_context {
    hparams hparams;
    std::unique_ptr<model> model;
    gguf_context *gguf;
    ggml_context *ctx;
    bool initialized_success;

    berts_context(const struct hparams &hparams, struct model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , model(model)
        , gguf(gguf)
        , ctx(ctx)
        , initialized_success(false) {
        if (!model->init_weight(this)) {
            log::error("fail to load weights");
            return;
        }
        this->initialized_success = true;
    }
};

namespace berts::models {

berts_context *new_context(const hparams &hparams, model *model, gguf_context *gguf, ggml_context *ctx) {
    auto bert = new berts_context{hparams, model, gguf, ctx};
    if (!bert->initialized_success) {
        delete bert;
        bert = nullptr;
    }
    return bert;
}

void free_context(berts_context *ctx) {
    delete ctx;
}

gguf_context *get_gguf_context(berts_context *ctx) {
    return ctx->gguf;
}

ggml_context *get_ggml_context(berts_context *ctx) {
    return ctx->ctx;
}

bool get_hparams(berts_context *ctx, hparams *params) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (params) {
        *params = ctx->hparams;
    }

    return true;
}

bool is_model_loaded(berts_context *ctx) {
    return ctx && ctx->model;
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens);
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens, segments);
}

ggml_tensor *model::eval(berts_context *ctx, const std::vector<bert_token_t> &tokens) {
    std::vector<bert_segment_t> segments(tokens.size());
    return this->eval(ctx, tokens, segments);
}

} // namespace berts::models
