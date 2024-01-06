#pragma once

#include <memory>
#include <vector>
#include "berts/models/model_berts.hpp"
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

struct vocab : public internal::vocab_base2<vocab> {
    tokenizer_info cond;
    special_tokens special;
    std::unique_ptr<berts::trie::trie> trie;

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

    bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf);

    void clear();
};

static_assert(internal::Vocab<vocab>);

struct weights {
    using self_type = weights;

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

    // bert weights
    ggml_tensor *token_embedding = nullptr;
    ggml_tensor *segment_embedding = nullptr;
    ggml_tensor *position_embedding = nullptr;
    ggml_tensor *ln_w = nullptr;
    ggml_tensor *ln_b = nullptr;
    std::vector<transformer_block> layers;
    ggml_tensor *pool_w = nullptr;
    ggml_tensor *pool_b = nullptr;

    // lm head
    ggml_tensor *lm_dense_w = nullptr; // hidden_dim -> hidden_dim
    ggml_tensor *lm_dense_b = nullptr;
    ggml_tensor *lm_ln_w = nullptr;
    ggml_tensor *lm_ln_b = nullptr;
    ggml_tensor *lm_decoder_w = nullptr; // hidden_dim -> vocab_size
    ggml_tensor *lm_decoder_b = nullptr;

    bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf);
};

struct model : public internal::model_berts<vocab, weights> {
    using inherited = internal::model_berts<struct vocab, struct weights>;
    using inherited::inherited;

    std::string model_name() const override {
        return "BERT";
    }

    bool tokenize(const berts_context *ctx,
                  const std::string &text,
                  std::vector<bert_token_t> &out) const override;

    internal::ggml_size_info get_context_buffer_size(
        size_t token_count,
        const internal::hparams &hparams,
        const berts_eval_info &cond) const override;

    internal::ggml_size_info get_context_buffer_size_for_lm(
        size_t input_token_count,
        size_t output_token_count,
        const internal::hparams &hparams,
        const berts_eval_lm_info &cond) const override;

    bool build_graph(ggml_ctx &ctx,
                     const internal::hparams &hparams,
                     const berts_eval_info &cond,
                     const std::vector<bert_token_t> &tokens,
                     const std::vector<bert_segment_t> &segments) const override;

    bool build_lm_graph(ggml_ctx &ctx,
                        const internal::hparams &hparams,
                        const berts_eval_lm_info &cond,
                        const float *hidden_states,
                        size_t hidden_states_count) const override;
};

} // namespace berts::bert
