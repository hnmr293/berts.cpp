#pragma once

#include "berts/models/log.hpp"
#include "ggml/ggml.h"

namespace berts {

template <typename T, typename Disposer>
struct unique_ctx {
    using ctx_type = T;
    using self_type = unique_ctx<T, Disposer>;

    ctx_type *ctx;

    unique_ctx()
        : unique_ctx(nullptr) {}

    unique_ctx(ctx_type *ctx)
        : ctx(ctx) {}

    unique_ctx(const self_type &) = delete;

    unique_ctx(self_type &&other)
        : ctx(other.ctx) {
        other.ctx = nullptr;
    }

    self_type &operator=(const self_type &) = delete;

    self_type &operator=(self_type &&other) {
        if (this != &other) {
            this->dispose();
            this->ctx = other.ctx;
            other.ctx = nullptr;
        }
    }

    virtual ~unique_ctx() {
        this->dispose();
    }

    void dispose() {
        if (this->ctx) {
            Disposer::dispose(this->ctx);
            this->ctx = nullptr;
        }
    }

    operator ctx_type *() { return this->ctx; }

    operator const ctx_type *() const { return this->ctx; }

    ctx_type **ptr() { return &this->ctx; }

    const ctx_type *const *ptr() const { return &this->ctx; }

    ctx_type *release() {
        auto ctx = this->ctx;
        this->ctx = nullptr;
        return ctx;
    }

    operator bool() const { return !!this->ctx; }
};

struct ggml_context_disposer {
    static void dispose(ggml_context *ctx) {
        if (ctx)
            ggml_free(ctx);
    }
};

struct gguf_context_disposer {
    static void dispose(gguf_context *ctx) {
        if (ctx)
            gguf_free(ctx);
    }
};

/// @brief RAII class for ggml_context
struct ggml_ctx : public unique_ctx<ggml_context, ggml_context_disposer> {
    ggml_ctx()
        : ggml_ctx(nullptr) {}

    ggml_ctx(ggml_context *ctx)
        : unique_ctx(ctx) {}

    ggml_ctx(ggml_init_params params)
        : unique_ctx(ggml_init(params)) { log::debug("ggml_init"); }

    ~ggml_ctx() { log::debug("ggml_free"); }
};

/// @brief RAII class for gguf_context
struct gguf_ctx : public unique_ctx<gguf_context, gguf_context_disposer> {
    gguf_ctx()
        : gguf_ctx(nullptr) {}

    gguf_ctx(gguf_context *ctx)
        : unique_ctx(ctx) {}

    gguf_ctx(const std::string &path, gguf_init_params params)
        : unique_ctx(gguf_init_from_file(path.c_str(), params)) { log::debug("gguf_init"); }

    gguf_ctx(const std::string &path, bool no_alloc, ggml_context **ctx)
        : gguf_ctx(path, {.no_alloc = no_alloc, .ctx = ctx}) {}

    ~gguf_ctx() { log::debug("gguf_free"); }
};

/// @brief RAII class for gguf_context AND ggml_context
struct gg_ctx {
    ggml_ctx ggml;
    gguf_ctx gguf;

    gg_ctx()
        : ggml()
        , gguf() {}

    gg_ctx(const std::string &path, bool no_alloc)
        : ggml()
        , gguf(path, no_alloc, ggml.ptr()) {}

    gg_ctx(const gg_ctx &) = delete;

    gg_ctx(gg_ctx &&other)
        : ggml(other.ggml.release())
        , gguf(other.gguf.release()) {}

    gg_ctx &operator=(const gg_ctx &) = delete;

    gg_ctx &operator=(gg_ctx &&other) {
        if (this != &other) {
            this->dispose();
            this->ggml.ctx = other.ggml.release();
            this->gguf.ctx = other.gguf.release();
        }
        return *this;
    }

    ~gg_ctx() {
        this->dispose();
    }

    void dispose() {
        this->ggml.dispose();
        this->gguf.dispose();
    }

    operator bool() const {
        return gguf.operator bool() && ggml.operator bool();
    }
};

} // namespace berts
