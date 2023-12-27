#pragma once

#include <iterator>
#include <string>
#include "berts/berts.h"
#include "berts/models/log.hpp"
#include "ggml/ggml.h"

namespace berts {

//
// resources
//

template <typename T>
using disposer_func_t = void (*)(T *ctx);

template <typename T, disposer_func_t<T> free_func>
struct context_disposer {
    using ctx_type = T;
    static void dispose(ctx_type *ctx) {
        if (ctx) {
            free_func(ctx);
        }
    }
};

template <typename Self, typename Disposer>
struct unique_ctx {
    using ctx_type = Disposer::ctx_type;
    using self_type = Self;
    using inherited = unique_ctx<Self, Disposer>;

    ctx_type *ctx;

    unique_ctx()
        : unique_ctx(nullptr) {}

    unique_ctx(ctx_type *ctx)
        : ctx(ctx) {
        if (ctx) {
            reinterpret_cast<self_type*>(this)->on_init(ctx);
        }
    }

    unique_ctx(const inherited &) = delete;

    unique_ctx(inherited &&other) noexcept
        : inherited(other.ctx) {
        other.ctx = nullptr;
    }

    self_type &operator=(const inherited &) = delete;

    self_type &operator=(inherited &&other) noexcept {
        if (this != &other) {
            this->dispose();
            this->ctx = other.ctx;
            other.ctx = nullptr;
        }
        return *reinterpret_cast<self_type*>(this);
    }

    ~unique_ctx() {
        this->dispose();
    }

    void dispose() {
        if (this->ctx) {
            reinterpret_cast<self_type*>(this)->on_dispose(this->ctx);
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

//
// berts_ctx
//

using berts_disposer = context_disposer<berts_context, berts_free>;

/// @brief RAII class for berts_context
struct berts_ctx : public unique_ctx<berts_ctx, berts_disposer> {
    using inherited::inherited;
    using inherited::operator=;

    void on_init(const ctx_type *ctx) {
        log::debug("berts_init @ {:016x}", (intptr_t)ctx);
    }

    void on_dispose(const ctx_type *ctx) {
        log::debug("berts_free @ {:016x}", (intptr_t)ctx);
    }
};

//
// ggml_ctx
//

using ggml_disposer = context_disposer<ggml_context, ggml_free>;

/// @brief RAII class for ggml_context
struct ggml_ctx : public unique_ctx<ggml_ctx, ggml_disposer> {
    using inherited::inherited;
    using inherited::operator=;

    ggml_ctx(ggml_init_params params)
        : inherited(ggml_init(params)) {}

    void on_init(const ctx_type *ctx) {
        log::debug("ggml_init @ {:016x}", (intptr_t)ctx);
    }

    void on_dispose(const ctx_type *ctx) {
        log::debug("ggml_free @ {:016x}", (intptr_t)ctx);
    }
};

//
// gguf_ctx
//

using gguf_disposer = context_disposer<gguf_context, gguf_free>;

/// @brief RAII class for gguf_context
struct gguf_ctx : public unique_ctx<gguf_ctx, gguf_disposer> {
    using inherited::inherited;
    using inherited::operator=;

    gguf_ctx(const std::string &path, gguf_init_params params)
        : inherited(gguf_init_from_file(path.c_str(), params)) {}

    gguf_ctx(const std::string &path, bool no_alloc, ggml_context **ctx)
        : self_type(path, {.no_alloc = no_alloc, .ctx = ctx}) {}

    void on_init(const ctx_type *ctx) {
        log::debug("gguf_init @ {:016x}", (intptr_t)ctx);
    }

    void on_dispose(const ctx_type *ctx) {
        log::debug("gguf_free @ {:016x}", (intptr_t)ctx);
    }
};

//
// gg_ctx
//

/// @brief RAII class for gguf_context AND ggml_context
struct gg_ctx : public std::pair<ggml_ctx, gguf_ctx> {
    using inherited = std::pair<ggml_ctx, gguf_ctx>;
    using inherited::inherited;

    gg_ctx(const std::string &path, bool no_alloc)
        : inherited() {
        second = gguf_ctx{path, no_alloc, first.ptr()};
    }

    gguf_ctx &gguf() {
        return second;
    }

    ggml_ctx &ggml() {
        return first;
    }

    operator bool() const {
        return first && second;
    }
};

} // namespace berts
