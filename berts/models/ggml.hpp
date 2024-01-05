#pragma once

/**
 * ggml utilities
 */

#include <string>
#include "berts/berts.h"
#include "berts/models/log.hpp"
#include "ggml/ggml.h"

namespace berts::internal {

struct ggml_size_info {
    size_t emb;
    size_t layer;
    size_t pooler;
    size_t graph;

    size_t layers(size_t n) const noexcept {
        return layer * n;
    }

    size_t calc(size_t layers) const noexcept {
        return emb + this->layers(layers) + pooler + graph;
    }
};

struct ggml_context_for_debug {
    size_t mem_size;
    void *mem_buffer;
    bool mem_buffer_owned;
    bool no_alloc;
    bool no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int n_objects;

    struct ggml_object *objects_begin;
    struct ggml_object *objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;

    static const ggml_context_for_debug &from(const ggml_context *ctx) {
        return *(const ggml_context_for_debug *)ctx;
    }

    size_t current() const noexcept {
        const size_t cur_offs = objects_end ? objects_end->offs : 0;
        const size_t cur_size = objects_end ? objects_end->size : 0;
        const size_t cur_end = cur_offs + cur_size;
        return cur_end;
    }

    void check(size_t expected, const std::string &msg) const {
        const size_t cur = current();
        if (cur != expected) {
            log::error("size mismatch ({}): expected = {}, but {}", msg, expected, cur);
            GGML_ASSERT(false && "size mismatch");
        }
    }
};

static inline std::string
pool_type_str(berts_pool_type type) {
    switch (type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE: return "none";
    case BERTS_POOL_CLS: return "cls";
    case BERTS_POOL_AVG: return "avg";
    case BERTS_POOL_MAX: return "max";
    default: return "";
    }
}

static inline size_t get_data_size(ggml_type type, size_t ne0, size_t ne1 = 1, size_t ne2 = 1, size_t ne3 = 1) {
    size_t data_size = ggml_type_size(type) * (ne0 / ggml_blck_size(type));
    data_size *= ne1;
    data_size *= ne2;
    data_size *= ne3;
    return data_size;
}

static inline size_t get_tensor_size(ggml_type type, size_t ne0, size_t ne1 = 1, size_t ne2 = 1, size_t ne3 = 1) {
    size_t size = get_data_size(type, ne0, ne1, ne2, ne3);
    size += GGML_TENSOR_SIZE;
    size = GGML_PAD(size, GGML_MEM_ALIGN);
    size += GGML_OBJECT_SIZE;
    return size;
}

static inline ggml_tensor *bert_dense(ggml_context *ctx, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
    x = ggml_mul_mat(ctx, w, x);
    x = ggml_add(ctx, x, ggml_repeat(ctx, b, x));
    return x;
}

static inline ggml_tensor *bert_layer_norm(ggml_context *ctx, ggml_tensor *x, ggml_tensor *ln_w, ggml_tensor *ln_b, float eps) {
    x = ggml_norm(ctx, x, eps);
    return ggml_add(ctx,
                    ggml_mul(ctx, ggml_repeat(ctx, ln_w, x), x),
                    ggml_repeat(ctx, ln_b, x));
}

} // namespace berts::internal
