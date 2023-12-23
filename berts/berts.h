#pragma once

//
// berts.cpp
//   C API
//

#include <stdint.h>
#include <stdio.h>
#include "ggml/ggml.h"

#ifdef BERTS_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef BERTS_BUILD
#define BERTS_API __declspec(dllexport)
#else
#define BERTS_API __declspec(dllimport)
#endif
#else
#define BERTS_API __attribute__((visibility("default")))
#endif
#else
#define BERTS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// typedefs
//

typedef int64_t bert_int;
typedef uint32_t bert_token_t;
typedef uint32_t bert_segment_t;

//
// default hparams
//

const bert_int DEFAULT_HIDDEN_SIZE = 768;
const bert_int DEFAULT_LAYERS = 12;
const bert_int DEFAULT_ATTN_HEADS = 12;

enum bert_type {
    BERTS_TYPE_BERT,
    // BERTS_TYPE_DEBERTA,
};

enum hidden_act {
    BERTS_HIDDEN_ACT_GELU,
};

//
// general
//

const char *berts_version(void);

int berts_version_major(void);

int berts_version_minor(void);

int berts_version_patch(void);

//
// logging
//

enum berts_log_level {
    BERTS_LOG_ALL = 0,
    BERTS_LOG_DEBUG = 0,
    BERTS_LOG_INFO = 1,
    BERTS_LOG_WARN = 2,
    BERTS_LOG_DEFAULT = 2,
    BERTS_LOG_ERROR = 3,
    BERTS_LOG_QUIET = 10,
};

berts_log_level berts_get_log_level(void);

void berts_set_log_level(berts_log_level level);

FILE *berts_get_log_file(void);

void berts_set_log_file(FILE *file);

//
// context
//

struct berts_context;

BERTS_API void berts_free(berts_context *ctx);

BERTS_API berts_context *berts_load_from_file(const char *path);

// BERTS_API berts_context *berts_load_from_memory(const uint8_t *data, size_t data_len);

BERTS_API void berts_set_eps(berts_context *ctx, double eps);

BERTS_API double berts_get_eps(berts_context *ctx);

//
// inference
//

ggml_tensor *berts_eval(berts_context *ctx,
                        const bert_token_t *tokens,
                        const bert_segment_t *segments,
                        size_t token_count);

//
// quantization
//

/// @brief quantize {input_path} model to {output_path}
/// @param input_path 
/// @param output_path 
/// @param qtype 
/// @return 0 if succeeded, otherwize non-0 value
int berts_model_quantize(const char *input_path,
                         const char *output_path,
                         ggml_type qtype);

#ifdef __cplusplus
}
#endif
