#include <array>
#include <cassert>
#include <memory>
#include <string>
#include "berts/berts.h"
#include "berts/models/fmt.hpp"

int main() {

    berts_set_log_level(BERTS_LOG_ALL);
    const char *model_path = ".gguf/roberta-base-f32.gguf";
    auto ctx = berts_load_from_file(model_path);
    assert(ctx);
    assert(berts_arch(ctx) == BERTS_TYPE_ROBERTA);

    berts_set_log_level(BERTS_LOG_ALL);

    //
    // tokens
    //

    auto cls_id = berts_cls_id(ctx);
    auto mask_id = berts_mask_id(ctx);
    auto pad_id = berts_pad_id(ctx);
    auto sep_id = berts_sep_id(ctx);
    auto unk_id = berts_unk_id(ctx);
    auto bos_id = berts_bos_id(ctx);
    auto eos_id = berts_eos_id(ctx);

    assert(cls_id != BERTS_INVALID_TOKEN_ID);
    assert(mask_id != BERTS_INVALID_TOKEN_ID);
    assert(pad_id != BERTS_INVALID_TOKEN_ID);
    assert(sep_id != BERTS_INVALID_TOKEN_ID);
    assert(unk_id != BERTS_INVALID_TOKEN_ID);
    assert(bos_id != BERTS_INVALID_TOKEN_ID);
    assert(eos_id != BERTS_INVALID_TOKEN_ID);

    std::string cls, mask, pad, sep, unk, bos, eos;
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

        size = s.size();
        ok = berts_id_to_token(ctx, bos_id, s.data(), &size);
        assert(ok);
        bos = std::string{s.data(), size};

        size = s.size();
        ok = berts_id_to_token(ctx, eos_id, s.data(), &size);
        assert(ok);
        eos = std::string{s.data(), size};
    }

    assert(cls != "");
    assert(mask != "");
    assert(pad != "");
    assert(sep != "");
    assert(unk != "");
    assert(bos != "");
    assert(eos != "");

    assert(cls == "<s>");
    assert(mask == "<mask>");
    assert(pad == "<pad>");
    assert(sep == "</s>");
    assert(unk == "<unk>");
    assert(bos == "<s>");
    assert(eos == "</s>");

    auto cls_id1 = berts_token_to_id(ctx, cls.c_str());
    auto mask_id1 = berts_token_to_id(ctx, mask.c_str());
    auto pad_id1 = berts_token_to_id(ctx, pad.c_str());
    auto sep_id1 = berts_token_to_id(ctx, sep.c_str());
    auto unk_id1 = berts_token_to_id(ctx, unk.c_str());
    auto bos_id1 = berts_token_to_id(ctx, bos.c_str());
    auto eos_id1 = berts_token_to_id(ctx, eos.c_str());

    assert(cls_id == cls_id1);
    assert(mask_id == mask_id1);
    assert(pad_id == pad_id1);
    assert(sep_id == sep_id1);
    assert(unk_id == unk_id1);
    assert(bos_id == bos_id1);
    assert(eos_id == eos_id1);

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
        assert(tokens[0] == 0);     // "<s>"
        assert(tokens[1] == 30086); // "Hi"
        assert(tokens[2] == 6);     // ","
        assert(tokens[3] == 38);    // " I"
        assert(tokens[4] == 524);   // " am"
        assert(tokens[5] == 50264); // "<mask>"
        assert(tokens[6] == 313);   // " man"
        assert(tokens[7] == 4);     // "."
        assert(tokens[8] == 2);     // "</s>"
    }

    {
        const std::string text1 = berts::fmt::fmt("Hi, I am {} man. How are you?", mask);
        size_t size = text1.size();
        std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};
        bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
        assert(ok);
        assert(size == 13);
        assert(tokens[0] == 0);     // "<s>"
        assert(tokens[1] == 30086); // "Hi"
        assert(tokens[2] == 6);     // ","
        assert(tokens[3] == 38);    // " I"
        assert(tokens[4] == 524);   // " am"
        assert(tokens[5] == 50264); // "<mask>"
        assert(tokens[6] == 313);   // " man"
        assert(tokens[7] == 4);     // "."
        assert(tokens[8] == 1336);  // " How"
        assert(tokens[9] == 32);    // " are"
        assert(tokens[10] == 47);   // " you"
        assert(tokens[11] == 116);  // "?"
        assert(tokens[12] == 2);    // "</s>"
    }

    return 0;
}
