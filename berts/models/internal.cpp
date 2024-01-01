#include "berts/models/internal.hpp"

#include <memory>
#include "berts/models/log.hpp"

using namespace berts;

//
// models
//

struct berts_context {
    internal::hparams hparams;
    std::unique_ptr<internal::model> model;
    gguf_context *gguf;
    ggml_context *ctx;
    bool initialized_success;

    berts_context(const internal::hparams &hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , model(model)
        , gguf(gguf)
        , ctx(ctx)
        , initialized_success(false) {
        if (!model) {
            log::error("model is empty");
            return;
        }

        if (!model->init_vocab(this)) {
            log::error("fail to load vocab");
            return;
        }

        if (!model->init_weight(this)) {
            log::error("fail to load weights");
            return;
        }

        this->initialized_success = true;
    }
};

namespace berts::internal {

model::~model() = default;

bool model::eval(berts_context *ctx,
                 const std::vector<bert_token_t> &tokens,
                 const berts_eval_info &cond,
                 float *out,
                 size_t &out_count) const {
    std::vector<bert_segment_t> segments(tokens.size());
    // ^ already initializd by 0
    return this->eval(ctx, tokens, segments, cond, out, out_count);
}

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

bool get_hparams(const berts_context *ctx, hparams *params) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (params) {
        *params = ctx->hparams;
    }

    return true;
}

bool is_model_loaded(const berts_context *ctx) {
    return ctx && ctx->model;
}

model &get_model(berts_context *ctx) {
    return *ctx->model;
}

const model &get_model(const berts_context *ctx) {
    return *ctx->model;
}

} // namespace berts::internal
