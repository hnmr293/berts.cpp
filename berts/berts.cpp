#include "berts/berts.h"

#include <cstring>
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

bert_type berts_arch(const berts_context *ctx) {
    internal::hparams hparams{};
    internal::get_hparams(ctx, &hparams);
    return hparams.architecture;
}

//
// tokenizer
//

static inline bool check_model(const berts_context *ctx) {
    if (!internal::check_ctx(ctx)) {
        return false;
    }
    if (!internal::check_model(ctx)) {
        return false;
    }
    return true;
}

#define BERTS_CHECK_MODEL_OR(default_) \
    if (!check_model(ctx)) {           \
        return (default_);             \
    }                                  \
    auto &model = internal::get_model(ctx);

bert_token_t berts_cls_id(const berts_context *ctx) {
    BERTS_CHECK_MODEL_OR(BERTS_INVALID_TOKEN_ID);
    return model.cls_id();
}

bert_token_t berts_mask_id(const berts_context *ctx) {
    BERTS_CHECK_MODEL_OR(BERTS_INVALID_TOKEN_ID);
    return model.mask_id();
}

bert_token_t berts_pad_id(const berts_context *ctx) {
    BERTS_CHECK_MODEL_OR(BERTS_INVALID_TOKEN_ID);
    return model.pad_id();
}

bert_token_t berts_sep_id(const berts_context *ctx) {
    BERTS_CHECK_MODEL_OR(BERTS_INVALID_TOKEN_ID);
    return model.sep_id();
}

bert_token_t berts_unk_id(const berts_context *ctx) {
    BERTS_CHECK_MODEL_OR(BERTS_INVALID_TOKEN_ID);
    return model.unk_id();
}

bool berts_id_to_token(const berts_context *ctx,
                       bert_token_t id,
                       char *out,
                       size_t *out_len) {
    BERTS_CHECK_MODEL_OR(false);
    const auto token = model.id_to_token(id);
    if (out_len) {
        size_t out_len_ = std::min(*out_len, token.size());
        *out_len = token.size();
        if (out) std::copy(token.begin(), token.begin() + out_len_, out);
    }
    return token.size() != 0;
}

bert_token_t berts_token_to_id(const berts_context *ctx, const char *token) {
    BERTS_CHECK_MODEL_OR(false);
    return model.token_to_id(token);
}

bool berts_tokenize(const berts_context *ctx,
                    const char *text,
                    bert_token_t *out,
                    size_t *out_len) {
    BERTS_CHECK_MODEL_OR(false);

    std::vector<bert_token_t> ids;

    auto cls_id = model.cls_id();
    auto sep_id = model.sep_id();

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

    bool ok = model.tokenize(ctx, text, ids);

    if (ok) {
        ids.push_back(sep_id);
        if (out_len) {
            size_t out_len_ = std::min(*out_len, ids.size());
            *out_len = ids.size();
            if (out) std::copy(ids.begin(), ids.begin() + out_len_, out);
        }
    }

    internal::hparams hparams{};
    internal::get_hparams(ctx, &hparams);
    if ((size_t)hparams.max_tokens < ids.size()) {
        log::warn(
            "Token count ({}) is larger than the max_position_embeddings ({}). "
            "Calling eval() with this sequence will cause a failure.",
            ids.size(),
            hparams.max_tokens);
    }

    return ok;
}

//
// inference
//

void berts_init_eval_info(berts_eval_info *cond) {
    if (cond) {
        cond->output_layer = -1;
        cond->pool_type = BERTS_POOL_CLS;
        // cond->output_all_layers = false;
        cond->n_threads = -1;
    }
}

bool berts_eval(berts_context *ctx,
                const bert_token_t *tokens,
                const bert_segment_t *segments,
                size_t token_count,
                const berts_eval_info *cond,
                float *out,
                size_t *out_count) {
    BERTS_CHECK_MODEL_OR(false);

    if (!cond) {
        return false;
    }

    if (!out_count) {
        return false;
    }

    std::vector<bert_token_t> token_vec(token_count);
    std::copy(tokens, tokens + token_count, token_vec.data());

    if (segments) {
        std::vector<bert_segment_t> segm_vec(token_count);
        std::copy(segments, segments + token_count, segm_vec.data());
        return model.eval(ctx, token_vec, segm_vec, *cond, out, *out_count);
    } else {
        return model.eval(ctx, token_vec, *cond, out, *out_count);
    }
}

namespace berts {

// berts_context *load_from_stream(std::istream &stream) {
//     return gguf::load_from_stream(stream);
// }

bool eval(berts_context *ctx,
          const std::vector<bert_token_t> &tokens,
          const berts_eval_info &cond,
          float *out,
          size_t &out_count) {
    BERTS_CHECK_MODEL_OR(false);
    return model.eval(ctx, tokens, cond, out, out_count);
}

bool eval(berts_context *ctx,
          const std::vector<bert_token_t> &tokens,
          const std::vector<bert_segment_t> &segments,
          const berts_eval_info &cond,
          float *out,
          size_t &out_count) {
    BERTS_CHECK_MODEL_OR(false);
    return model.eval(ctx, tokens, segments, cond, out, out_count);
}

bool model_quantize(const std::string &input_path,
                    const std::string &output_path,
                    ggml_type qtype) {
    return berts_model_quantize(input_path.c_str(), output_path.c_str(), qtype);
}

} // namespace berts
