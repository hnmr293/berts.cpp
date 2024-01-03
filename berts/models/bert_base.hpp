#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "berts/models/gguf.hpp"
#include "berts/models/internal.hpp"
#include "berts/models/keys.h"
#include "berts/models/log.hpp"

namespace berts::bert {

template <typename T>
concept Vocab = requires(T &obj, berts_context *ctx, ggml_context *ggml, gguf_context *gguf, const std::string &str) {
    typename T::self_type;

    new T{};

    { obj.cls_id() } -> std::convertible_to<bert_token_t>;
    { obj.mask_id() } -> std::convertible_to<bert_token_t>;
    { obj.pad_id() } -> std::convertible_to<bert_token_t>;
    { obj.sep_id() } -> std::convertible_to<bert_token_t>;
    { obj.unk_id() } -> std::convertible_to<bert_token_t>;
    { obj.bos_id() } -> std::convertible_to<bert_token_t>;
    { obj.eos_id() } -> std::convertible_to<bert_token_t>;

    { obj.id_to_token(bert_token_t{}) } -> std::convertible_to<std::string>;
    { obj.token_to_id(str) } -> std::convertible_to<bert_token_t>;

    { obj.add_token(str) } -> std::convertible_to<bool>;
    { obj.has_token(str) } -> std::convertible_to<bool>;

    // called from class `base` after tokens have been added
    { obj.init(ctx, ggml, gguf) } -> std::convertible_to<bool>;
    obj.clear();

    // implemented in vocab_base
    { obj.cls_token() } -> std::convertible_to<std::string>;
    { obj.mask_token() } -> std::convertible_to<std::string>;
    { obj.pad_token() } -> std::convertible_to<std::string>;
    { obj.sep_token() } -> std::convertible_to<std::string>;
    { obj.unk_token() } -> std::convertible_to<std::string>;
    { obj.bos_token() } -> std::convertible_to<std::string>;
    { obj.eos_token() } -> std::convertible_to<std::string>;
};

template <typename Self>
struct vocab_base {
    using inherited = vocab_base<Self>;

    bert_token_t get_token_id(const gguf_context *gguf, const char *key, const char *alternate1, const char *alternate2 = nullptr) {
        const char *failed_key = nullptr;

        auto id = gguf::gguf_u32(gguf, key, BERTS_INVALID_TOKEN_ID);
        if (id == BERTS_INVALID_TOKEN_ID) {
            if (!alternate1) {
                failed_key = key;
                goto FAIL;
            }

            log::warn("{} is not defined; use {} instead", key, alternate1);
            id = as_self()->token_to_id(alternate1);

            if (id == BERTS_INVALID_TOKEN_ID) {
                if (!alternate2) {
                    failed_key = alternate1;
                    goto FAIL;
                }

                log::warn("{} is not defined; use {} instead", alternate1, alternate2);
                id = as_self()->token_to_id(alternate2);

                if (id == BERTS_INVALID_TOKEN_ID) {
                    failed_key = alternate2;
                    goto FAIL;
                }
            }
        }

        return id;

    FAIL:
        if (failed_key) {
            log::error("{} does not exist in vocab", failed_key);
        }
        return BERTS_INVALID_TOKEN_ID;
    }

    std::string cls_token() const noexcept { return as_self()->id_to_token(as_self()->cls_id()); };
    std::string mask_token() const noexcept { return as_self()->id_to_token(as_self()->mask_id()); };
    std::string pad_token() const noexcept { return as_self()->id_to_token(as_self()->pad_id()); };
    std::string sep_token() const noexcept { return as_self()->id_to_token(as_self()->sep_id()); };
    std::string unk_token() const noexcept { return as_self()->id_to_token(as_self()->unk_id()); };
    std::string bos_token() const noexcept { return as_self()->id_to_token(as_self()->bos_id()); };
    std::string eos_token() const noexcept { return as_self()->id_to_token(as_self()->eos_id()); };

  protected:
    auto as_self() noexcept { return static_cast<typename Self::self_type *>(this); }
    const auto as_self() const noexcept { return static_cast<const typename Self::self_type *>(this); }
};

template <typename Self>
struct vocab_base2 : public vocab_base<vocab_base2<Self>> {
    using self_type = Self;
    using inherited = vocab_base2<Self>;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, bert_token_t> token_to_id_;

    vocab_base2() = default;

    std::string id_to_token(bert_token_t token_id) const noexcept {
        if (id_to_token_.size() <= token_id) {
            log::error("token id {} is not found (max={})", token_id, id_to_token_.size());
            return "";
        }
        return id_to_token_[token_id];
    }

    bert_token_t token_to_id(const std::string &token) const noexcept {
        const auto p = token_to_id_.find(token);
        if (p == token_to_id_.end()) {
            log::error("token {} is not found", token);
            return BERTS_INVALID_TOKEN_ID;
        }
        return p->second;
    }

    bool add_token(const std::string &token) {
        if (has_token(token)) {
            log::warn("  token {} already exists", token);
            return false;
        }

        const auto next_id = static_cast<bert_token_t>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_[token] = next_id;
        // log::debug("  token {}: {}", next_id, token);
        return true;
    }

    bool has_token(const std::string &token) const noexcept {
        const auto p = token_to_id_.find(token);
        return p != token_to_id_.end();
    }

    void clear() {
        id_to_token_.clear();
        token_to_id_.clear();
    }
};

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

template <Vocab VocabType>
struct base : public internal::model {
    using vocab_t = VocabType;

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
    std::unique_ptr<vocab_t> vocab;

    base(ggml_type type)
        : internal::model(type)
        , vocab(new vocab_t{}) {}

    ~base() override = default;

    bool init_vocab(berts_context *ctx) override {
        using namespace berts::internal;

        log::info("loading vocab");

        if (!check_ctx(ctx)) {
            return false;
        }

        auto ggml = get_ggml_context(ctx);
        auto gguf = get_gguf_context(ctx);

        auto vocab_size = ggml_get_tensor(ggml, BERTS_KEY_ALL_VOCAB_SIZE);
        auto vocab_data = ggml_get_tensor(ggml, BERTS_KEY_ALL_VOCAB_DATA);

        if (!vocab_size) {
            log::error("key {} is not found", BERTS_KEY_ALL_VOCAB_SIZE);
            return false;
        }

        if (!vocab_data) {
            log::error("key {} is not found", BERTS_KEY_ALL_VOCAB_DATA);
            return false;
        }

        if (vocab_size->n_dims != 1 || vocab_data->n_dims != 1) {
            log::error("invalid shape: vocab_size={}, vocab_data={}", vocab_size->n_dims, vocab_data->n_dims);
            return false;
        }

        if (vocab_size->type != GGML_TYPE_I8) {
            log::error("invalid type of vocab_size: {}", (int)vocab_size->type);
            return false;
        }

        if (vocab_data->type != GGML_TYPE_I8) {
            log::error("invalid type of vocab_data: {}", (int)vocab_data->type);
            return false;
        }

        log::debug("  vocab count: {}", vocab_size->ne[0]);

        const int64_t vocab_count = vocab_size->ne[0];
        auto token_lengths = static_cast<const uint8_t *>(vocab_size->data);
        const auto data = static_cast<const char *>(vocab_data->data);
        ptrdiff_t p = 0;
        for (int64_t token_id = 0; token_id < vocab_count; ++token_id) {
            size_t token_len = (size_t)token_lengths[token_id];
            if (token_len == 0) {
                token_len = 256;
            }
            std::string token{&data[p], token_len};
            p += token_len;

            if (!add_token(token)) {
                log::error("failed to add token: {}", token);
                vocab->clear();
                return false;
            }
        }

        if (p != vocab_data->ne[0]) {
            log::error("something wrong");
            vocab->clear();
            return false;
        }

        if (!vocab->init(ctx, ggml, gguf)) {
            log::error("fail to build vocab");
            vocab->clear();
            return false;
        }

        log::info("finish loading vocab");
        return true;
    }

    bool init_weight(berts_context *ctx) override {
        using namespace berts::internal;

        log::info("initializing weights");

        if (!check_ctx(ctx)) {
            return false;
        }

        auto ggml = get_ggml_context(ctx);
        auto gguf = get_gguf_context(ctx);

        if (!init_weight(ctx, ggml, gguf)) {
            return false;
        }

        log::info("finish loading vocab");
        return true;
    }

    // called from init_weight(berts_context *ctx)
    virtual bool init_weight(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) = 0;

    // delegate to vocab
    std::string id_to_token(bert_token_t token_id) const noexcept override {
        return vocab->id_to_token(token_id);
    }

    bert_token_t token_to_id(const std::string &token) const noexcept override {
        return vocab->token_to_id(token);
    }

    bool add_token(const std::string &token) override {
        return vocab->add_token(token);
    }

    bool has_token(const std::string &token) const noexcept override {
        return vocab->has_token(token);
    }

    bert_token_t cls_id() const noexcept override {
        return vocab->cls_id();
    }

    bert_token_t mask_id() const noexcept override {
        return vocab->mask_id();
    }

    bert_token_t pad_id() const noexcept override {
        return vocab->pad_id();
    }

    bert_token_t sep_id() const noexcept override {
        return vocab->sep_id();
    }

    bert_token_t unk_id() const noexcept override {
        return vocab->unk_id();
    }

    bert_token_t bos_id() const noexcept override {
        return vocab->bos_id();
    }

    bert_token_t eos_id() const noexcept override {
        return vocab->eos_id();
    }

    virtual bool tokenize(const berts_context *ctx,
                          const std::string &text,
                          std::vector<bert_token_t> &out) const override = 0;

    virtual bool eval(berts_context *ctx,
                      const std::vector<bert_token_t> &tokens,
                      const std::vector<bert_segment_t> &segments,
                      const berts_eval_info &cond,
                      float *out,
                      size_t &out_count) const override = 0;
};

} // namespace berts::bert
