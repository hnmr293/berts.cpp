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

    berts_context(const internal::hparams &hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , model(model)
        , gguf(gguf)
        , ctx(ctx) {}

    static berts_context *create(const internal::hparams &hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx) {
        if (!model) {
            log::error("model is empty");
            return nullptr;
        }

        berts_context *berts = new berts_context{hparams, model, gguf, ctx};

        if (!model->init_vocab(berts)) {
            log::error("fail to load vocab");
            delete berts;
            return nullptr;
        }

        if (!model->init_weight(berts)) {
            log::error("fail to load weights");
            delete berts;
            return nullptr;
        }

        return berts;
    }

    static void free(berts_context *berts) {
        delete berts;
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
    return berts_context::create(hparams, model, gguf, ctx);
}

void free_context(berts_context *ctx) {
    berts_context::free(ctx);
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
