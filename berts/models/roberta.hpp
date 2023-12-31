#pragma once

#include <memory>
#include <vector>
#include "berts/models/bert.hpp"
#include "berts/models/bpe.hpp"

namespace berts::roberta {

struct special_tokens {
    bert_token_t bos;
    bert_token_t eos;
    bert_token_t cls;
    bert_token_t mask;
    bert_token_t pad;
    bert_token_t sep;
    bert_token_t unk;
};

struct vocab : public internal::vocab_base2<vocab> {
    special_tokens special;
    std::unique_ptr<bpe> bpe;

    vocab();
    vocab(size_t n);

    bert_token_t cls_id() const noexcept;
    bert_token_t mask_id() const noexcept;
    bert_token_t pad_id() const noexcept;
    bert_token_t sep_id() const noexcept;
    bert_token_t unk_id() const noexcept;
    bert_token_t bos_id() const noexcept;
    bert_token_t eos_id() const noexcept;

    std::string id_to_token(bert_token_t token_id) const noexcept;
    
    bert_token_t token_to_id(const std::string &token) const noexcept;

    std::string id_to_token_internal(bert_token_t token_id) const noexcept {
        return inherited::id_to_token(token_id);
    }
    
    bert_token_t token_to_id_internal(const std::string &token) const noexcept {
        return inherited::token_to_id(token);
    }

    bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf);
};

struct model : public internal::model_berts<vocab, bert::weights> {
    using inherited = internal::model_berts<struct vocab, bert::weights>;
    using inherited::inherited;

    std::string model_name() const override {
        return "RoBERTa";
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

} // namespace berts::roberta
