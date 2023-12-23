#include "berts/models/internal.hpp"
#include <cmath>
#include <format>
#include <memory>

namespace internal = berts::internal;

struct vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, bert_token_t> token_to_id;
    vocab() {}
    vocab(size_t n) {
        this->id_to_token.reserve(n);
        this->token_to_id.reserve(n);
    }
};

struct berts_context {
    internal::hparams hparams;
    struct vocab vocab;
    std::unique_ptr<internal::model> model;
    double eps;
    gguf_context *gguf;
    ggml_context *ctx;
    berts_context(internal::hparams hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , vocab()
        , model(model)
        , gguf(gguf)
        , ctx(ctx) {
        if (model) {
            bool ok;
            ok = model->load_vocab(this);
            GGML_ASSERT(ok && "fail to load vocab");
            ok = model->init_weight(this);
            GGML_ASSERT(ok && "fail to load weights");
        }
    }
};

namespace berts::internal {

berts_context *new_context(hparams hparams, model *model, gguf_context *gguf, ggml_context *ctx) {
    return new berts_context{hparams, model, gguf, ctx};
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

std::string id_to_token(berts_context *ctx, bert_token_t token_id) {
    if (!check_ctx(ctx)) {
        return "";
    }

    if (ctx->vocab.id_to_token.size() <= token_id) {
        log::error(std::format("token id {} is not found (max={})", token_id, ctx->vocab.id_to_token.size()));
        return "";
    }

    return ctx->vocab.id_to_token[token_id];
}

bert_token_t token_to_id(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return (bert_token_t)-1;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    if (p == ctx->vocab.token_to_id.end()) {
        log::error(std::format("token {} is not found", token));
        return (bert_token_t)-1;
    }

    return p->second;
}

bool add_token(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (has_token(ctx, token)) {
        log::warn(std::format("token {} already exists", token));
        return false;
    }

    const auto next_id = static_cast<bert_token_t>(ctx->vocab.id_to_token.size());
    ctx->vocab.id_to_token.push_back(token);
    ctx->vocab.token_to_id[token] = next_id;

    return true;
}

bool has_token(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    return p != ctx->vocab.token_to_id.end();
}

bool is_model_loaded(berts_context *ctx) {
    return ctx && ctx->model;
}

double get_eps(berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return std::nan("");
    }

    return ctx->eps;
}

double set_eps(berts_context *ctx, double new_val) {
    if (!check_ctx(ctx)) {
        return std::nan("");
    }

    auto old = ctx->eps;
    ctx->eps = new_val;
    return old;
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

} // namespace berts::internal
