#include "berts/berts.h"
#include "berts/berts.hpp"
#include "berts/models/gguf.hpp"
#include "berts/models/internal.hpp"
#include "berts/models/log.hpp"

namespace internal = berts::internal;
namespace gguf = berts::gguf;
namespace log = berts::log;

#define BERTS_VERSION_MAJOR 0
#define BERTS_VERSION_MINOR 2
#define BERTS_VERSION_PATCH 0
#define STRINGIFY(v) #v
#define CREATE_VERSION(i, j, k) STRINGIFY(i) "." STRINGIFY(j) "." STRINGIFY(k)

static const char *BERTS_VERSION = CREATE_VERSION(BERTS_VERSION_MAJOR, BERTS_VERSION_MINOR, BERTS_VERSION_PATCH);

const char *berts_version(void) {
    return BERTS_VERSION;
}

int berts_version_major(void) {
    return BERTS_VERSION_MAJOR;
}

int berts_version_minor(void) {
    return BERTS_VERSION_MINOR;
}

int berts_version_patch(void) {
    return BERTS_VERSION_PATCH;
}

berts_log_level berts_get_log_level(void) {
    return (berts_log_level)(int)log::get_log_level();
}

void berts_set_log_level(berts_log_level level) {
    log::set_log_level(level);
}

FILE *berts_get_log_file(void) {
    return log::get_log_file();
}

void berts_set_log_file(FILE *file) {
    log::set_log_file(file);
}

void berts_free(berts_context *ctx) {
    internal::free_context(ctx);
}

berts_context *berts_load_from_file(const char *path) {
    return gguf::load_from_file(path);
}

// berts_context *berts_load_from_memory(const uint8_t *data, size_t data_len) {
//     return gguf::load_from_memory(const uint8_t *data, size_t data_len);
// }

//
// tokenizer
//

void berts_init_tokenizer_info(berts_tokenizer_info *cond) {
    if (cond) internal::init_tokenizer_info_default(*cond);
}

void berts_init_tokenizer_info_no_basic(berts_tokenizer_info *cond) {
    if (cond) internal::init_tokenizer_info_no_basic(*cond);
}

void berts_get_tokenizer_info(const berts_context *ctx, berts_tokenizer_info *cond) {
    if (cond) internal::get_tokenizer_info(ctx, *cond);
}

void berts_set_tokenizer_info(berts_context *ctx, const berts_tokenizer_info *cond) {
    if (cond) internal::set_tokenizer_info(ctx, *cond);
}

bert_token_t berts_cls_id(const berts_context *ctx) {
    return internal::get_cls_id(ctx);
}

bert_token_t berts_mask_id(const berts_context *ctx) {
    return internal::get_mask_id(ctx);
}

bert_token_t berts_pad_id(const berts_context *ctx) {
    return internal::get_pad_id(ctx);
}

bert_token_t berts_sep_id(const berts_context *ctx) {
    return internal::get_sep_id(ctx);
}

bert_token_t berts_unk_id(const berts_context *ctx) {
    return internal::get_unk_id(ctx);
}

bool berts_id_to_token(const berts_context *ctx,
                       bert_token_t id,
                       char *out,
                       size_t *out_len) {
    const auto token = internal::id_to_token(ctx, id);
    if (out_len) {
        size_t out_len_ = std::min(*out_len, token.size());
        *out_len = token.size();
        if (out) std::copy(token.begin(), token.begin() + out_len_, out);
    }
    return token.size() != 0;
}

bert_token_t berts_token_to_id(const berts_context *ctx, const char *token) {
    return internal::token_to_id(ctx, token);
}

bool berts_tokenize(const berts_context *ctx,
                    const char *text,
                    bert_token_t *out,
                    size_t *out_len) {
    std::vector<bert_token_t> ids;
    
    auto cls_id = internal::get_cls_id(ctx);
    auto sep_id = internal::get_sep_id(ctx);

    if (cls_id == BERTS_INVALID_TOKEN_ID) {
        log::error("cls_id is not found");
        return false;
    }
    
    if (sep_id == BERTS_INVALID_TOKEN_ID) {
        log::error("sep_id is not found");
        return false;
    }
    
    ids.reserve(std::strlen(text) + 2 /* cls, sep */);
    ids.push_back(cls_id);
    
    bool ok = internal::tokenize(ctx, text, ids);
    
    if (ok) {
        ids.push_back(sep_id);
        if (out_len) {
            size_t out_len_ = std::min(*out_len, ids.size());
            *out_len = ids.size();
            if (out) std::copy(ids.begin(), ids.begin() + out_len_, out);
        }
    }
    return ok;
}

//
// inference
//

ggml_tensor *berts_eval(berts_context *ctx,
                        const bert_token_t *tokens,
                        const bert_segment_t *segments,
                        size_t token_count) {
    std::vector<bert_token_t> token_vec(token_count);
    std::copy(tokens, tokens + token_count, token_vec.data());

    std::vector<bert_segment_t> segm_vec(token_count);
    std::copy(segments, segments + token_count, segm_vec.data());

    return internal::eval(ctx, token_vec, segm_vec);
}

namespace berts {

// berts_context *load_from_stream(std::istream &stream) {
//     return gguf::load_from_stream(stream);
// }

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens) {
    return internal::eval(ctx, tokens);
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments) {
    return internal::eval(ctx, tokens, segments);
}

bool model_quantize(const std::string &input_path,
                    const std::string &output_path,
                    ggml_type qtype) {
    return berts_model_quantize(input_path.c_str(), output_path.c_str(), qtype);
}

} // namespace berts
