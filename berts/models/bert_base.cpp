#include "berts/models/bert_base.hpp"

#include "berts/models/internal.hpp"
#include "berts/models/keys.h"
#include "berts/models/log.hpp"

using namespace berts::internal;

namespace berts::bert {

//
// base::~base
//

base::~base() = default;

//
// base::init_vocab
//

bool base::init_vocab(berts_context *ctx) {
    log::info("loading vocab");

    if (!check_ctx(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);
    auto gguf = get_gguf_context(ctx);

    auto vocab_size = ggml_get_tensor(ggml, BERTS_KEY_ALL_VOCAB_SIZE);
    auto vocab_data = ggml_get_tensor(ggml, BERTS_KEY_ALL_VOCAB_DATA);

    if (!vocab_size) {
        log::error("key {} is not found", BERTS_KEY_ALL_VOCAB_SIZE);
        return false;
    }

    if (!vocab_data) {
        log::error("key {} is not found", BERTS_KEY_ALL_VOCAB_DATA);
        return false;
    }

    if (vocab_size->n_dims != 1 || vocab_data->n_dims != 1) {
        log::error("invalid shape: vocab_size={}, vocab_data={}", vocab_size->n_dims, vocab_data->n_dims);
        return false;
    }

    if (vocab_size->type != GGML_TYPE_I8) {
        log::error("invalid type of vocab_size: {}", (int)vocab_size->type);
        return false;
    }

    if (vocab_data->type != GGML_TYPE_I8) {
        log::error("invalid type of vocab_data: {}", (int)vocab_data->type);
        return false;
    }

    log::debug("  vocab count: {}", vocab_size->ne[0]);

    const int64_t vocab_count = vocab_size->ne[0];
    auto token_lengths = static_cast<const uint8_t *>(vocab_size->data);
    const auto data = static_cast<const char *>(vocab_data->data);
    ptrdiff_t p = 0;
    for (int64_t token_id = 0; token_id < vocab_count; ++token_id) {
        size_t token_len = (size_t)token_lengths[token_id];
        if (token_len == 0) {
            token_len = 256;
        }
        std::string token{&data[p], token_len};
        p += token_len;

        add_token(token);
    }

    if (p != vocab_data->ne[0]) {
        log::error("something wrong");
        vocab->clear();
        return false;
    }

    if (!vocab->init(ctx, ggml, gguf)) {
        log::error("fail to build vocab");
        vocab->clear();
        return false;
    }

    log::info("finish loading vocab");
    return true;
}

//
// base::init_weight
//

bool base::init_weight(berts_context *ctx) {
    log::info("initializing weights");

    if (!check_ctx(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);
    auto gguf = get_gguf_context(ctx);
    
    if (!init_weight(ctx, ggml, gguf)) {
        return false;
    }

    log::info("finish loading vocab");
    return true;
}

} // namespace berts::bert
