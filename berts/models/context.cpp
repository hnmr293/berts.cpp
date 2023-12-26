#include "berts/models/context.hpp"
#include <cmath>
#include <memory>
#include "berts/common/log.hpp"
#include "berts/models/utils.hpp"

using namespace berts;
using namespace berts::models;

namespace berts::models {

struct context {
    hparams hparams;
    std::unique_ptr<model> model;
    gguf_ctx gguf;
    ggml_ctx ctx;

    context(const struct hparams &hparams, struct model *model, gguf_context *gguf, ggml_context *ggml)
        : hparams(hparams)
        , model(model)
        , gguf(gguf)
        , ctx(ggml) {}

    context(const context &) = delete;

    context(context &&other) noexcept
        : hparams(std::move(other.hparams))
        , model(std::move(other.model))
        , gguf(std::move(other.gguf))
        , ctx(std::move(other.ctx)) {}

    ~context() {}

    context &operator=(const context &) = delete;

    context &operator=(context &&other) noexcept {
        if (this != &other) {
            hparams = other.hparams;
            model = std::move(other.model);
            gguf = std::move(other.gguf);
            ctx = std::move(other.ctx);
        }
        return *this;
    }
};

} // namespace berts::models

struct berts_context {
    models::context model;
    tokenizer_ctx tokenizer;
};

namespace berts::models {

berts_context *new_context(const hparams &hparams, model *model, gguf_context *gguf, ggml_context *ctx, tokenizers::context *tokenizer) {
    auto bert = new berts_context{
        {hparams, model, gguf, ctx},
        tokenizer
    };
    if (!bert->model.model->init_weight(bert)) {
        log::error("fail to load weights");
        delete bert;
        bert = nullptr;
    }
    return bert;
}

void free_context(berts_context *ctx) {
    delete ctx;
}

gguf_context *get_gguf_context(berts_context *ctx) {
    return ctx->model.gguf;
}

ggml_context *get_ggml_context(berts_context *ctx) {
    return ctx->model.ctx;
}

bool get_hparams(berts_context *ctx, hparams *params) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (params) {
        *params = ctx->model.hparams;
    }

    return true;
}

bool is_model_loaded(berts_context *ctx) {
    return ctx && ctx->model.model;
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model.model->eval(ctx, tokens);
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model.model->eval(ctx, tokens, segments);
}

ggml_tensor *model::eval(berts_context *ctx, const std::vector<bert_token_t> &tokens) {
    std::vector<bert_segment_t> segments(tokens.size());
    return this->eval(ctx, tokens, segments);
}

} // namespace berts::models
