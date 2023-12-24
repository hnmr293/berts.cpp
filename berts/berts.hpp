#pragma once

//
// berts.cpp
//   C++ API
//

#include <vector>
#include <string>
#include "berts/berts.h"

namespace berts {

//
// context
//

// berts_context *load_from_stream(std::istream &stream);

//
// inference
//

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens);

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments);

//
// quantization
//

bool berts_model_quantize(const std::string &input_path,
                          const std::string &output_path,
                          ggml_type qtype);

} // namespace berts
