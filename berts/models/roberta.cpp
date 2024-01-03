#include "berts/models/roberta.hpp"

#include <unordered_map>
#include "berts/models/keys.h"

namespace berts::roberta {

vocab::vocab()
    : inherited() {
    special.bos = BERTS_INVALID_TOKEN_ID;
    special.eos = BERTS_INVALID_TOKEN_ID;
    special.cls = BERTS_INVALID_TOKEN_ID;
    special.mask = BERTS_INVALID_TOKEN_ID;
    special.pad = BERTS_INVALID_TOKEN_ID;
    special.sep = BERTS_INVALID_TOKEN_ID;
    special.unk = BERTS_INVALID_TOKEN_ID;
}

vocab::vocab(size_t n)
    : vocab() {
    id_to_token_.reserve(n);
    token_to_id_.reserve(n);
}

bert_token_t vocab::cls_id() const noexcept {
    return special.cls;
}

bert_token_t vocab::mask_id() const noexcept {
    return special.mask;
}

bert_token_t vocab::pad_id() const noexcept {
    return special.pad;
}

bert_token_t vocab::sep_id() const noexcept {
    return special.sep;
}

bert_token_t vocab::unk_id() const noexcept {
    return special.unk;
}

bert_token_t vocab::bos_id() const noexcept {
    return special.bos;
}

bert_token_t vocab::eos_id() const noexcept {
    return special.eos;
}

bool vocab::init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) {
    (void)ctx;
    (void)ggml;

    auto bos_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_CLS_ID, "<s>");
    auto eos_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_CLS_ID, "</s>");
    auto cls_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_CLS_ID, "<s>");
    auto mask_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_MASK_ID, "<mask>");
    auto pad_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_PAD_ID, "<pad>");
    auto sep_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_SEP_ID, "</s>");
    auto unk_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_UNK_ID, "<unk>");

    if (bos_id == BERTS_INVALID_TOKEN_ID) {
        bos_id = cls_id;
    }

    if (eos_id == BERTS_INVALID_TOKEN_ID) {
        eos_id = sep_id;
    }

    if (cls_id == BERTS_INVALID_TOKEN_ID) {
        cls_id = bos_id;
    }

    if (sep_id == BERTS_INVALID_TOKEN_ID) {
        sep_id = eos_id;
    }

    log::when(BERTS_LOG_INFO, [=, this]() {
        auto bos = id_to_token(bos_id);
        auto eos = id_to_token(eos_id);
        auto cls = id_to_token(cls_id);
        auto mask = id_to_token(mask_id);
        auto pad = id_to_token(pad_id);
        auto sep = id_to_token(sep_id);
        auto unk = id_to_token(unk_id);
        log::info("  bos_id:  {} ({})", bos_id, bos);
        log::info("  eos_id:  {} ({})", eos_id, eos);
        log::info("  cls_id:  {} ({})", cls_id, cls);
        log::info("  mask_id: {} ({})", mask_id, mask);
        log::info("  pad_id:  {} ({})", pad_id, pad);
        log::info("  sep_id:  {} ({})", sep_id, sep);
        log::info("  unk_id:  {} ({})", unk_id, unk);
    });

    if (bos_id == BERTS_INVALID_TOKEN_ID ||
        eos_id == BERTS_INVALID_TOKEN_ID ||
        cls_id == BERTS_INVALID_TOKEN_ID ||
        mask_id == BERTS_INVALID_TOKEN_ID ||
        pad_id == BERTS_INVALID_TOKEN_ID ||
        sep_id == BERTS_INVALID_TOKEN_ID ||
        unk_id == BERTS_INVALID_TOKEN_ID) {
        return false;
    }

    special.bos = eos_id;
    special.eos = bos_id;
    special.cls = cls_id;
    special.mask = mask_id;
    special.pad = pad_id;
    special.sep = sep_id;
    special.unk = unk_id;

    return true;
}

//
// model::model
//

model::model(ggml_type type)
    : base(type) {}

//
// model::init_weight
//

bool model::init_weight(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) {
    return false;
}

bool model::tokenize(const berts_context *ctx,
                     const std::string &text,
                     std::vector<bert_token_t> &out) const {
    return false;
}

bool model::eval(berts_context *ctx,
                 const std::vector<bert_token_t> &tokens,
                 const std::vector<bert_segment_t> &segments,
                 const berts_eval_info &cond,
                 float *out,
                 size_t &out_count) const {
    return false;
}

} // namespace berts::roberta
