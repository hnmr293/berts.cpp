#include <array>
#include <cassert>
#include <memory>
#include <string>
#include "berts/berts.h"
#include "berts/models/fmt.hpp"

int main() {

    berts_set_log_level(BERTS_LOG_ALL);
    const char *model_path = ".gguf/bert-base-cased-f32.gguf";
    auto ctx = berts_load_from_file(model_path);
    assert(ctx);

    berts_set_log_level(BERTS_LOG_ALL);

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

    std::string cls, mask, pad, sep, unk;
    {
        const size_t N = 256;
        bool ok;
        size_t size;

        std::array<char, N> s{};

        size = s.size();
        ok = berts_id_to_token(ctx, cls_id, s.data(), &size);
        assert(ok);
        cls = std::string{s.data(), size};

        size = s.size();
        ok = berts_id_to_token(ctx, mask_id, s.data(), &size);
        assert(ok);
        mask = std::string{s.data(), size};

        size = s.size();
        ok = berts_id_to_token(ctx, pad_id, s.data(), &size);
        assert(ok);
        pad = std::string{s.data(), size};

        size = s.size();
        ok = berts_id_to_token(ctx, sep_id, s.data(), &size);
        assert(ok);
        sep = std::string{s.data(), size};

        size = s.size();
        ok = berts_id_to_token(ctx, unk_id, s.data(), &size);
        assert(ok);
        unk = std::string{s.data(), size};
    }

    assert(cls != "");
    assert(mask != "");
    assert(pad != "");
    assert(sep != "");
    assert(unk != "");

    assert(cls == "[CLS]");
    assert(mask == "[MASK]");
    assert(pad == "[PAD]");
    assert(sep == "[SEP]");
    assert(unk == "[UNK]");

    auto cls_id1 = berts_token_to_id(ctx, cls.c_str());
    auto mask_id1 = berts_token_to_id(ctx, mask.c_str());
    auto pad_id1 = berts_token_to_id(ctx, pad.c_str());
    auto sep_id1 = berts_token_to_id(ctx, sep.c_str());
    auto unk_id1 = berts_token_to_id(ctx, unk.c_str());

    assert(cls_id == cls_id1);
    assert(mask_id == mask_id1);
    assert(pad_id == pad_id1);
    assert(sep_id == sep_id1);
    assert(unk_id == unk_id1);

    //
    // tokenize
    //

    {
        const std::string text1 = berts::fmt::fmt("Hi, I am {} man.", mask);
        size_t size = text1.size();
        std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};
        bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
        assert(ok);
        assert(size == 9);
        assert(tokens[0] == 101);  // [CLS]
        assert(tokens[1] == 8790); // Hi
        assert(tokens[2] == 117);  // ,
        assert(tokens[3] == 146);  // I
        assert(tokens[4] == 1821); // am
        assert(tokens[5] == 103);  // [MASK]
        assert(tokens[6] == 1299); // man
        assert(tokens[7] == 119);  // .
        assert(tokens[8] == 102);  // [SEP]
    }

    {
        const std::string text1 = berts::fmt::fmt("Hi, I am {} man. How are you?", mask);
        size_t size = text1.size();
        std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};
        bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
        assert(ok);
        assert(size == 13);
        assert(tokens[0] == 101);   // [CLS]
        assert(tokens[1] == 8790);  // Hi
        assert(tokens[2] == 117);   // ,
        assert(tokens[3] == 146);   // I
        assert(tokens[4] == 1821);  // am
        assert(tokens[5] == 103);   // [MASK]
        assert(tokens[6] == 1299);  // man
        assert(tokens[7] == 119);   // .
        assert(tokens[8] == 1731);  // How
        assert(tokens[9] == 1132);  // are
        assert(tokens[10] == 1128); // you
        assert(tokens[11] == 136);  // ?
        assert(tokens[12] == 102);  // [SEP]
    }

    return 0;
}
