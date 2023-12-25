#pragma once

#include <iterator>
#include <string>
#include "berts/berts.h"
#include "berts/common/log.hpp"
#include "ggml/ggml.h"

namespace berts {

//
// resources
//

template <typename T>
struct noncopyable {
    noncopyable(const noncopyable &) = delete;
    T &operator=(const T &) = delete;

  protected:
    noncopyable() = default;
    ~noncopyable() = default;
};

template <typename T, typename Disposer>
struct unique_ctx : private noncopyable<T> {
    using ctx_type = Disposer::type;
    using self_type = T;
    using inherited = unique_ctx<T, Disposer>;

  protected:
    ctx_type *ctx;

  public:
    unique_ctx()
        : unique_ctx(nullptr) {}

    unique_ctx(ctx_type *ctx)
        : ctx(ctx) {}

    unique_ctx(self_type &&other) noexcept
        : ctx(other.ctx) {
        other.ctx = nullptr;
    }

    self_type &operator=(self_type &&other) noexcept {
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

struct berts_context_disposer {
    using type = berts_context;
    static void dispose(type *ctx) {
        if (ctx)
            berts_free(ctx);
    }
};

struct ggml_context_disposer {
    using type = ggml_context;
    static void dispose(type *ctx) {
        if (ctx)
            ggml_free(ctx);
    }
};

struct gguf_context_disposer {
    using type = gguf_context;
    static void dispose(type *ctx) {
        if (ctx)
            gguf_free(ctx);
    }
};

/// @brief RAII class for berts_context
struct berts_ctx : public unique_ctx<berts_ctx, berts_context_disposer> {
    berts_ctx()
        : berts_ctx(nullptr) {}

    berts_ctx(berts_context *ctx)
        : inherited(ctx) {
        if (ctx) log::debug("berts_init @ {:016x}", (intptr_t)ctx);
    }

    berts_ctx(self_type &&ctx)
        : inherited(std::move(ctx)) {}
    
    self_type &operator=(self_type&&ctx) {
        return inherited::operator=(std::move(ctx));
    }

    ~berts_ctx() {
        if (ctx) log::debug("berts_free @ {:016x}", (intptr_t)ctx);
    }
};

/// @brief RAII class for ggml_context
struct ggml_ctx : public unique_ctx<ggml_ctx, ggml_context_disposer> {
    ggml_ctx()
        : ggml_ctx(nullptr) {}

    ggml_ctx(ggml_context *ctx)
        : unique_ctx(ctx) {}

    ggml_ctx(ggml_init_params params)
        : unique_ctx(ggml_init(params)) {
        if (ctx) log::debug("ggml_init @ {:016x}", (intptr_t)ctx);
    }

    ggml_ctx(self_type &&ctx)
        : inherited(std::move(ctx)) {}
    
    self_type &operator=(self_type&&ctx) {
        return inherited::operator=(std::move(ctx));
    }

    ~ggml_ctx() {
        if (ctx) log::debug("ggml_free @ {:016x}", (intptr_t)ctx);
    }
};

/// @brief RAII class for gguf_context
struct gguf_ctx : public unique_ctx<gguf_ctx, gguf_context_disposer> {
    gguf_ctx()
        : gguf_ctx(nullptr) {}

    gguf_ctx(gguf_context *ctx)
        : unique_ctx(ctx) {}

    gguf_ctx(const std::string &path, gguf_init_params params)
        : unique_ctx(gguf_init_from_file(path.c_str(), params)) {
        if (ctx) log::debug("gguf_init @ {:016x}", (intptr_t)ctx);
    }

    gguf_ctx(const std::string &path, bool no_alloc, ggml_context **ctx)
        : gguf_ctx(path, {.no_alloc = no_alloc, .ctx = ctx}) {}

    gguf_ctx(self_type &&ctx)
        : inherited(std::move(ctx)) {}
    
    self_type &operator=(self_type&&ctx) {
        return inherited::operator=(std::move(ctx));
    }

    ~gguf_ctx() {
        if (ctx) log::debug("gguf_free @ {:016x}", (intptr_t)ctx);
    }
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

    gg_ctx(gg_ctx &&other) noexcept
        : ggml(other.ggml.release())
        , gguf(other.gguf.release()) {}

    gg_ctx &operator=(const gg_ctx &) = delete;

    gg_ctx &operator=(gg_ctx &&other) noexcept {
        if (this != &other) {
            this->ggml = std::move(other.ggml);
            this->gguf = std::move(other.gguf);
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
