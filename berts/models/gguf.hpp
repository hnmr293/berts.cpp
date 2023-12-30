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

/**
 * gguf accessors
 * 
 * ---
 * gguf::safe_index
 *     get index of given key, or fail fast
 * gguf::index
 *     get index of given key (alias of gguf_find_key)
 * gguf::gguf_u8
 * gguf::gguf_i8
 * gguf::gguf_u16
 * gguf::gguf_i16
 * gguf::gguf_u32
 * gguf::gguf_i32
 * gguf::gguf_u64
 * gguf::gguf_i64
 * gguf::gguf_f32
 * gguf::gguf_f64
 * gguf::gguf_bool
 * gguf::gguf_str
 * gguf::gguf_data
 *     get the value with specified type
 *     if default value is not given, fail fast when key is not found
*/

static inline size_t safe_index(const struct gguf_context *ctx, const char *key) {
    auto idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        log::error("key {0} is not found in gguf", key);
        GGML_ASSERT(false && "key is not found in gguf");
    }

    return (size_t)idx;
}

static inline size_t safe_index(const struct gguf_context *ctx, const std::string &key) {
    return safe_index(ctx, key.c_str());
}

static inline int index(const struct gguf_context *ctx, const char *key) {
    return gguf_find_key(ctx, key);
}

static inline int index(const struct gguf_context *ctx, const std::string &key) {
    return gguf_find_key(ctx, key.c_str());
}

#define DEFINE_GGUF_VALUE_GET(gguf_type, c_type)                                                    \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const char *key) {        \
        return gguf_get_val_##gguf_type(ctx, safe_index(ctx, key));                                 \
    }                                                                                               \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const std::string &key) { \
        return gguf_get_val_##gguf_type(ctx, safe_index(ctx, key));                                 \
    }

#define DEFINE_GGUF_VALUE_GET_OR(gguf_type, c_type)                                                                    \
    template <typename T>                                                                                              \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const char *key, const T &default_) {        \
        auto idx = index(ctx, key);                                                                                    \
        return idx < 0 ? default_ : gguf_get_val_##gguf_type(ctx, idx);                                                \
    }                                                                                                                  \
    template <typename T>                                                                                              \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const std::string &key, const T &default_) { \
        auto idx = index(ctx, key);                                                                                    \
        return idx < 0 ? default_ : gguf_get_val_##gguf_type(ctx, idx);                                                \
    }

#define DEFINE_GGUF_VALUE(gguf_type, c_type) \
    DEFINE_GGUF_VALUE_GET(gguf_type, c_type) \
    DEFINE_GGUF_VALUE_GET_OR(gguf_type, c_type)

DEFINE_GGUF_VALUE(u8, uint8_t)
DEFINE_GGUF_VALUE(i8, int8_t)
DEFINE_GGUF_VALUE(u16, uint16_t)
DEFINE_GGUF_VALUE(i16, int16_t)
DEFINE_GGUF_VALUE(u32, uint32_t)
DEFINE_GGUF_VALUE(i32, int32_t)
DEFINE_GGUF_VALUE(f32, float)
DEFINE_GGUF_VALUE(u64, uint64_t)
DEFINE_GGUF_VALUE(i64, int64_t)
DEFINE_GGUF_VALUE(f64, double)
DEFINE_GGUF_VALUE(bool, bool)
DEFINE_GGUF_VALUE(str, const char *)
DEFINE_GGUF_VALUE(data, const void *)

#undef DEFINE_GGUF_VALUE_GET
#undef DEFINE_GGUF_VALUE_GET_OR
#undef DEFINE_GGUF_VALUE

} // namespace berts::gguf
