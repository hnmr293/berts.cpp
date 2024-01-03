#pragma once

#include <memory>
#include <vector>
#include "berts/models/bert_base.hpp"

namespace berts::bert {

struct model : public base {

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
