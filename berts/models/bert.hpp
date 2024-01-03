#pragma once

#include <memory>
#include <vector>
#include "berts/models/bert_base.hpp"
#include "berts/models/trie.hpp"

// std::unique_resource<trie::trie>
namespace std {

template <>
struct default_delete<berts::trie::trie> {
    void operator()(berts::trie::trie *trie) const noexcept {
        if (trie) berts::trie::free_trie(trie);
    }
};

} // namespace std

namespace berts::bert {

struct tokenizer_info {
    // ignored, always normalized with NFC
    bool normalize;

    // remove U+FFFD
    bool remove_replacement_char;

    // remove U+0000
    bool remove_null_char;

    // remove control chars (category C*)
    bool remove_control_char;

    // convert all whitespaces to a normal space (U+0020)
    bool normalize_whitespaces;

    // add space around all CJK characters
    bool add_space_around_cjk_char;

    // force input to be lowercase letters
    bool do_lower_case;

    // remove all accent chars
    bool strip_accents;

    // split words at a punctuation
    bool split_on_punc;
};

struct special_tokens {
    bert_token_t cls;
    bert_token_t mask;
    bert_token_t pad;
    bert_token_t sep;
    bert_token_t unk;
};

struct vocab : public vocab_base<vocab> {
    tokenizer_info cond;
    special_tokens special;
    std::unique_ptr<berts::trie::trie> trie;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, bert_token_t> token_to_id_;

    vocab();
    vocab(size_t n);

    ~vocab();

    bool build_trie();

    bert_token_t cls_id() const noexcept;
    bert_token_t mask_id() const noexcept;
    bert_token_t pad_id() const noexcept;
    bert_token_t sep_id() const noexcept;
    bert_token_t unk_id() const noexcept;
    bert_token_t bos_id() const noexcept;
    bert_token_t eos_id() const noexcept;

    std::string id_to_token(bert_token_t token_id) const noexcept;
    bert_token_t token_to_id(const std::string &token) const noexcept;

    bool add_token(const std::string &token);
    bool has_token(const std::string &token) const noexcept;

    bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf);

    void clear();
};

static_assert(Vocab<vocab>);

struct model : public base<vocab> {

    model(ggml_type type);

    bool init_weight(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) override;

    bool tokenize(const berts_context *ctx,
                  const std::string &text,
                  std::vector<bert_token_t> &out) const override;

    bool eval(berts_context *ctx,
              const std::vector<bert_token_t> &tokens,
              const std::vector<bert_segment_t> &segments,
              const berts_eval_info &cond,
              float *out,
              size_t &out_count) const override;
};

} // namespace berts::bert
