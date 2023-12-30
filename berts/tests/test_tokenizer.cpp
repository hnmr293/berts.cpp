#include "berts/berts.h"
#include "berts/models/fmt.hpp"
#include <string>
#include <cassert>

int main() {

    berts_set_log_level(BERTS_LOG_ALL);
    const char *model_path = ".gguf/bert-base-cased_q8.gguf";
    auto ctx = berts_load_from_file(model_path);
    assert(ctx);
    
    //
    // tokens
    //
    
    auto cls_id = berts_cls_id(ctx);
    auto mask_id = berts_mask_id(ctx);
    auto pad_id = berts_pad_id(ctx);
    auto sep_id = berts_sep_id(ctx);
    auto unk_id = berts_unk_id(ctx);

    assert(cls_id != BERTS_INVALID_TOKEN_ID);
    assert(mask_id != BERTS_INVALID_TOKEN_ID);
    assert(pad_id != BERTS_INVALID_TOKEN_ID);
    assert(sep_id != BERTS_INVALID_TOKEN_ID);
    assert(unk_id != BERTS_INVALID_TOKEN_ID);

    auto cls = berts_cls_token(ctx);
    auto mask = berts_mask_token(ctx);
    auto pad = berts_pad_token(ctx);
    auto sep = berts_sep_token(ctx);
    auto unk = berts_unk_token(ctx);

    assert(cls != "");
    assert(mask != "");
    assert(pad != "");
    assert(sep != "");
    assert(unk != "");

    auto cls1 = berts_id_to_token(ctx, cls_id);
    auto mask1 = berts_id_to_token(ctx, mask_id);
    auto pad1 = berts_id_to_token(ctx, pad_id);
    auto sep1 = berts_id_to_token(ctx, sep_id);
    auto unk1 = berts_id_to_token(ctx, unk_id);

    assert(cls != cls1);
    assert(mask != mask1);
    assert(pad != pad1);
    assert(sep != sep1);
    assert(unk != unk1);

    auto cls_id1 = berts_token_to_id(ctx, cls);
    auto mask_id1 = berts_token_to_id(ctx, mask);
    auto pad_id1 = berts_token_to_id(ctx, pad);
    auto sep_id1 = berts_token_to_id(ctx, sep);
    auto unk_id1 = berts_token_to_id(ctx, unk);

    assert(cls_id != cls_id1);
    assert(mask_id != mask_id1);
    assert(pad_id != pad_id1);
    assert(sep_id != sep_id1);
    assert(unk_id != unk_id1);

    //
    // tokenize
    //
    const std::string text1 = berts::fmt::fmt("{}Hi, I am {} man.", cls, mask);
    berts_tokenizer_info cond{};
    berts_tokenizer_info_init(&cond);
    bool ok = berts_tokenize(ctx, text1, &cond);
    assert(ok);

    return 0;
}
