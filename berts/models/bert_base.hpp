#pragma once

#include <memory>
#include <vector>
#include "berts/models/internal.hpp"

namespace berts::bert {

struct vocab_base {
    vocab_base() = default;
    virtual ~vocab_base() = default;

    // called from class `base` after tokens have been added
    virtual bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) = 0;

    virtual std::string id_to_token(bert_token_t id) const noexcept = 0;
    virtual bert_token_t token_to_id(const std::string &token) const noexcept = 0;
    virtual bool add_token(const std::string &token) = 0;
    virtual bool has_token(const std::string &token) const noexcept = 0;
    virtual void clear() = 0;

    virtual bert_token_t cls_id() const noexcept = 0;
    virtual bert_token_t mask_id() const noexcept = 0;
    virtual bert_token_t pad_id() const noexcept = 0;
    virtual bert_token_t sep_id() const noexcept = 0;
    virtual bert_token_t unk_id() const noexcept = 0;
    virtual bert_token_t bos_id() const noexcept = 0;
    virtual bert_token_t eos_id() const noexcept = 0;
    bert_token_t get_token_id(const gguf_context *gguf, const char *key, const char *alternate1, const char *alternate2 = nullptr);

    std::string cls_token() const noexcept { return id_to_token(cls_id()); };
    std::string mask_token() const noexcept { return id_to_token(mask_id()); };
    std::string pad_token() const noexcept { return id_to_token(pad_id()); };
    std::string sep_token() const noexcept { return id_to_token(sep_id()); };
    std::string unk_token() const noexcept { return id_to_token(unk_id()); };
    std::string bos_token() const noexcept { return id_to_token(bos_id()); };
    std::string eos_token() const noexcept { return id_to_token(eos_id()); };
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

struct base : public internal::model {
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
    std::unique_ptr<vocab_base> vocab;

    base(ggml_type type, vocab_base *&&vocab)
        : internal::model(type)
        , vocab(vocab) {}

    ~base() override;

    bool init_vocab(berts_context *ctx) override;

    bool init_weight(berts_context *ctx) override;

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
