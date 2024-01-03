#pragma once

#include <memory>
#include <vector>
#include "berts/models/bert.hpp"

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

struct vocab : public bert::vocab_base2<vocab> {
    special_tokens special;

    vocab();
    vocab(size_t n);

    bert_token_t cls_id() const noexcept;
    bert_token_t mask_id() const noexcept;
    bert_token_t pad_id() const noexcept;
    bert_token_t sep_id() const noexcept;
    bert_token_t unk_id() const noexcept;
    bert_token_t bos_id() const noexcept;
    bert_token_t eos_id() const noexcept;

    bool init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf);
};

struct model : public bert::base<vocab, bert::weights> {
    using inherited::inherited;

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

} // namespace berts::roberta
