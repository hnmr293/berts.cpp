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

void berts_set_eps(berts_context *ctx, double eps) {
    internal::set_eps(ctx, eps);
}

double berts_get_eps(berts_context *ctx) {
    return internal::get_eps(ctx);
}

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
