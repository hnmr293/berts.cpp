#include <array>
#include <cassert>
#include <memory>
#include <string>
#include "berts/berts.h"
#include "berts/models/fmt.hpp"

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

    assert(cls_id != cls_id1);
    assert(mask_id != mask_id1);
    assert(pad_id != pad_id1);
    assert(sep_id != sep_id1);
    assert(unk_id != unk_id1);

    //
    // tokenizer conditions
    //

    {
        berts_tokenizer_info cond1{};
        berts_init_tokenizer_info(&cond1);
        assert(cond1.normalize);
        assert(cond1.remove_replacement_char);
        assert(cond1.remove_null_char);
        assert(cond1.remove_control_char);
        assert(cond1.normalize_whitespaces);
        assert(cond1.add_space_around_cjk_char);
        assert(cond1.do_lower_case);
        assert(cond1.strip_accents);
        assert(cond1.split_on_punc);
    }

    {
        berts_tokenizer_info cond1{};
        berts_init_tokenizer_info_no_basic(&cond1);
        assert(cond1.normalize);
        assert(!cond1.remove_replacement_char);
        assert(!cond1.remove_null_char);
        assert(!cond1.remove_control_char);
        assert(cond1.normalize_whitespaces);
        assert(!cond1.add_space_around_cjk_char);
        assert(!cond1.do_lower_case);
        assert(!cond1.strip_accents);
        assert(!cond1.split_on_punc);
    }

    {
        berts_tokenizer_info cond{};
        berts_get_tokenizer_info(ctx, &cond);
        assert(cond.normalize);
        assert(cond.remove_replacement_char);
        assert(cond.remove_null_char);
        assert(cond.remove_control_char);
        assert(cond.normalize_whitespaces);
        assert(cond.add_space_around_cjk_char);
        assert(cond.do_lower_case);
        assert(cond.strip_accents);
        assert(cond.split_on_punc);
    }

    {
        berts_tokenizer_info cond1{};
        berts_get_tokenizer_info(ctx, &cond1);
        assert(cond1.normalize);
        assert(cond1.remove_replacement_char);
        assert(cond1.remove_null_char);
        assert(cond1.remove_control_char);
        assert(cond1.normalize_whitespaces);
        assert(cond1.add_space_around_cjk_char);
        assert(cond1.do_lower_case);
        assert(cond1.strip_accents);
        assert(cond1.split_on_punc);

        berts_tokenizer_info cond2{cond1};
        cond2.strip_accents = false;
        berts_set_tokenizer_info(ctx, &cond2);
        berts_tokenizer_info cond3{};
        berts_get_tokenizer_info(ctx, &cond3);
        assert(cond3.normalize);
        assert(cond3.remove_replacement_char);
        assert(cond3.remove_null_char);
        assert(cond3.remove_control_char);
        assert(cond3.normalize_whitespaces);
        assert(cond3.add_space_around_cjk_char);
        assert(cond3.do_lower_case);
        assert(!cond3.strip_accents);
        assert(cond3.split_on_punc);

        berts_set_tokenizer_info(ctx, &cond1);
        berts_get_tokenizer_info(ctx, &cond3);
        assert(cond3.normalize);
        assert(cond3.remove_replacement_char);
        assert(cond3.remove_null_char);
        assert(cond3.remove_control_char);
        assert(cond3.normalize_whitespaces);
        assert(cond3.add_space_around_cjk_char);
        assert(cond3.do_lower_case);
        assert(cond3.strip_accents);
        assert(cond3.split_on_punc);
    }

    //
    // tokenize
    //

    const std::string text1 = berts::fmt::fmt("{}Hi, I am {} man.", cls, mask);
    size_t size = text1.size();
    std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};
    bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
    assert(ok);

    return 0;
}
