#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include "berts/models/internal.hpp"

namespace berts::gguf {

berts_context *load_from_file(const std::string &path);

// berts_context *load_from_memory(const uint8_t *data, size_t data_len);

// berts_context *load_from_stream(std::istream &stream);

//
// utilities
//

std::string type_to_str(ggml_type type);

} // namespace berts::gguf
