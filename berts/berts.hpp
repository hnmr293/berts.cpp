#pragma once

//
// berts.cpp
//   C++ API
//

#include <string>
#include <vector>
#include "berts/berts.h"

namespace berts {

//
// context
//

// berts_context *load_from_stream(std::istream &stream);

//
// inference
//

bool eval(berts_context *ctx,
          const std::vector<bert_token_t> &tokens,
          const berts_eval_info &cond,
          float *out,
          size_t &out_count);

bool eval(berts_context *ctx,
          const std::vector<bert_token_t> &tokens,
          const std::vector<bert_segment_t> &segments,
          const berts_eval_info &cond,
          float *out,
          size_t &out_count);

//
// quantization
//

bool model_quantize(const std::string &input_path,
                    const std::string &output_path,
                    ggml_type qtype);

} // namespace berts
