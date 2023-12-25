#include "berts/models/internal.hpp"
#include <cmath>
#include <memory>
#include "berts/models/log.hpp"
#include "berts/models/utils.hpp"
#include "berts/models/vocab.hpp"

using namespace berts;

struct vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, bert_token_t> token_to_id;
    bert_token_t cls_id;
    bert_token_t mask_id;
    bert_token_t pad_id;
    bert_token_t sep_id;
    bert_token_t unk_id;
    berts::vocab::trie *trie;

    vocab()
        : id_to_token()
        , token_to_id()
        , cls_id((bert_token_t)-1)
        , mask_id((bert_token_t)-1)
        , pad_id((bert_token_t)-1)
        , sep_id((bert_token_t)-1)
        , unk_id((bert_token_t)-1)
        , trie(nullptr) {}

    vocab(size_t n)
        : vocab() {
        this->id_to_token.reserve(n);
        this->token_to_id.reserve(n);
    }

    vocab(const vocab &) = delete;

    vocab(vocab &&other) noexcept
        : id_to_token(std::move(other.id_to_token))
        , token_to_id(std::move(other.token_to_id))
        , trie(other.trie) {}

    ~vocab() {
        this->dispose();
    }

    vocab &operator=(const vocab &) = delete;

    vocab &operator=(vocab &&other) noexcept {
        if (this != &other) {
            this->dispose();
            this->id_to_token = std::move(other.id_to_token);
            this->token_to_id = std::move(other.token_to_id);
            this->trie = other.trie;
            other.trie = nullptr;
        }
        return *this;
    }

    bool build_trie() {
        log::info("building vocab");
        this->dispose();
        this->trie = berts::vocab::build_trie(this->id_to_token);
        return !!this->trie;
    }

    void dispose() {
        if (this->trie) berts::vocab::free_trie(trie);
    }

    void init_ids() {
        using ustr = unicode::ustr;

        if (this->cls_id == (bert_token_t)-1) {
            this->cls_id = berts::vocab::search_trie(this->trie, ustr{"[CLS]"});
        }
        
        if (this->mask_id == (bert_token_t)-1) {
            this->mask_id = berts::vocab::search_trie(this->trie, ustr{"[MASK]"});
        }
        
        if (this->sep_id == (bert_token_t)-1) {
            this->sep_id = berts::vocab::search_trie(this->trie, ustr{"[SEP]"});
        }
        
        if (this->pad_id == (bert_token_t)-1) {
            this->pad_id = berts::vocab::search_trie(this->trie, ustr{"[PAD]"});
        }
        
        if (this->unk_id == (bert_token_t)-1) {
            this->unk_id = berts::vocab::search_trie(this->trie, ustr{"[UNK]"});
        }
    }
};

struct berts_context {
    internal::hparams hparams;
    struct vocab vocab;
    std::unique_ptr<internal::model> model;
    double eps;
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

            if (!this->vocab.build_trie()) {
                log::error("fail to build vocab");
                return;
            }

            vocab.init_ids();

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

bool get_hparams(berts_context *ctx, hparams *params) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (params) {
        *params = ctx->hparams;
    }

    return true;
}

bert_token_t get_cls_id(berts_context *ctx) {
    return ctx->vocab.cls_id;
}

void set_cls_id(berts_context *ctx, bert_token_t id) {
    ctx->vocab.cls_id = id;
}

bert_token_t get_mask_id(berts_context *ctx) {
    return ctx->vocab.mask_id;
}

void set_mask_id(berts_context *ctx, bert_token_t id) {
    ctx->vocab.mask_id = id;
}

bert_token_t get_pad_id(berts_context *ctx) {
    return ctx->vocab.pad_id;
}
void set_pad_id(berts_context *ctx, bert_token_t id) {
    ctx->vocab.pad_id = id;
}

bert_token_t get_sep_id(berts_context *ctx) {
    return ctx->vocab.sep_id;
}

void set_sep_id(berts_context *ctx, bert_token_t id) {
    ctx->vocab.sep_id = id;
}

bert_token_t get_unk_id(berts_context *ctx) {
    return ctx->vocab.unk_id;
}

void set_unk_id(berts_context *ctx, bert_token_t id) {
    ctx->vocab.unk_id = id;
}

std::string id_to_token(berts_context *ctx, bert_token_t token_id) {
    if (!check_ctx(ctx)) {
        return "";
    }

    if (ctx->vocab.id_to_token.size() <= token_id) {
        log::error(berts::fmt("token id {} is not found (max={})", token_id, ctx->vocab.id_to_token.size()));
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
        log::error(berts::fmt("token {} is not found", token));
        return (bert_token_t)-1;
    }

    return p->second;
}

bool add_token(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (has_token(ctx, token)) {
        log::warn(berts::fmt("token {} already exists", token));
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
