#pragma once

//
// berts.cpp
//   C API
//

#include <stdbool.h>
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
// general
//

BERTS_API const char *berts_version(void);

BERTS_API int berts_version_major(void);

BERTS_API int berts_version_minor(void);

BERTS_API int berts_version_patch(void);

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

BERTS_API berts_log_level berts_get_log_level(void);

BERTS_API void berts_set_log_level(berts_log_level level);

BERTS_API FILE *berts_get_log_file(void);

BERTS_API void berts_set_log_file(FILE *file);

//
// hparams
//

enum bert_type {
    BERTS_TYPE_BERT,
    // BERTS_TYPE_DEBERTA,
};

enum hidden_act {
    BERTS_HIDDEN_ACT_GELU,
};

//
// context
//

struct berts_context;

BERTS_API void berts_free(berts_context *ctx);

BERTS_API berts_context *berts_load_from_file(const char *path);

// BERTS_API berts_context *berts_load_from_memory(const uint8_t *data, size_t data_len);

//
// inference
//

BERTS_API ggml_tensor *berts_eval(berts_context *ctx,
                                  const bert_token_t *tokens,
                                  const bert_segment_t *segments,
                                  size_t token_count);

//
// quantization
//

/// @brief quantize {input_path} model to {output_path}
BERTS_API bool berts_model_quantize(const char *input_path,
                                    const char *output_path,
                                    ggml_type qtype);

#ifdef __cplusplus
}
#endif
