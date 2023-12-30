#include "berts/models/internal.hpp"
#include <cmath>
#include <memory>
#include "berts/models/utils.hpp"
#include "berts/models/log.hpp"

using namespace berts;

//
// models
//

struct vocab {
    berts_tokenizer_info cond;
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
    gguf_context *gguf;
    ggml_context *ctx;
    bool initialized_success;

    berts_context(const internal::hparams &hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , vocab()
        , model(model)
        , gguf(gguf)
        , ctx(ctx)
        , initialized_success(false) {
        if (model) {
            if (!model->load_vocab(this)) {
                log::error("fail to load vocab");
                return;
            }
            if (!model->init_weight(this)) {
                log::error("fail to load weights");
                return;
            }
        }
        this->initialized_success = true;
    }
};

namespace berts::internal {

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

ggml_tensor *eval(const berts_context *ctx,
                  const std::vector<bert_token_t> &tokens) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens);
}

ggml_tensor *eval(const berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens, segments);
}

ggml_tensor *model::eval(const berts_context *ctx, const std::vector<bert_token_t> &tokens) const {
    std::vector<bert_segment_t> segments(tokens.size());
    return this->eval(ctx, tokens, segments);
}

//
// tokenizers
//

std::string id_to_token(const berts_context *ctx, bert_token_t token_id) {
    if (!check_ctx(ctx)) {
        return "";
    }

    if (ctx->vocab.id_to_token.size() <= token_id) {
        log::error("token id {} is not found (max={})", token_id, ctx->vocab.id_to_token.size());
        return "";
    }

    return ctx->vocab.id_to_token[token_id];
}

bert_token_t token_to_id(const berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    if (p == ctx->vocab.token_to_id.end()) {
        log::error("token {} is not found", token);
        return BERTS_INVALID_TOKEN_ID;
    }

    return p->second;
}

bool add_token(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (has_token(ctx, token)) {
        log::warn("token {} already exists", token);
        return false;
    }

    const auto next_id = static_cast<bert_token_t>(ctx->vocab.id_to_token.size());
    ctx->vocab.id_to_token.push_back(token);
    ctx->vocab.token_to_id[token] = next_id;

    return true;
}

bool has_token(const berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    return p != ctx->vocab.token_to_id.end();
}

void get_tokenizer_info(const berts_context *ctx, berts_tokenizer_info &cond) {
    if (!check_ctx(ctx)) {
        return;
    }

    cond = ctx->vocab.cond;
}

void set_tokenizer_info(berts_context *ctx, const berts_tokenizer_info &cond) {
    if (!check_ctx(ctx)) {
        return;
    }

    ctx->vocab.cond = cond;
}

void init_tokenizer_info_default(berts_tokenizer_info &cond) {
    cond.normalize = true;
    cond.remove_replacement_char = true;
    cond.remove_null_char = true;
    cond.remove_control_char = true;
    cond.normalize_whitespaces = true;
    cond.add_space_around_cjk_char = true;
    cond.do_lower_case = true;
    cond.strip_accents = true;
    cond.split_on_punc = true;
    //cond.unknown_token_id = unknown_token_id;
}

void init_tokenizer_info_no_basic(berts_tokenizer_info &cond) {
    cond.normalize = true;
    cond.remove_replacement_char = false;
    cond.remove_null_char = false;
    cond.remove_control_char = false;
    cond.normalize_whitespaces = true;
    cond.add_space_around_cjk_char = false;
    cond.do_lower_case = false;
    cond.strip_accents = false;
    cond.split_on_punc = false;
    //cond.unknown_token_id = unknown_token_id;
}

bert_token_t get_cls_id(const berts_context *ctx) {
    if (check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return BERTS_INVALID_TOKEN_ID;
}

bert_token_t get_mask_id(const berts_context *ctx) {
    if (check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return BERTS_INVALID_TOKEN_ID;
}

bert_token_t get_pad_id(const berts_context *ctx) {
    if (check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return BERTS_INVALID_TOKEN_ID;
}

bert_token_t get_sep_id(const berts_context *ctx) {
    if (check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return BERTS_INVALID_TOKEN_ID;
}

bert_token_t get_unk_id(const berts_context *ctx) {
    if (check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return BERTS_INVALID_TOKEN_ID;
}

bool tokenize(const berts_context *ctx, const std::string &text, std::vector<bert_token_t> &out) {
    return false;
}

} // namespace berts::internal
