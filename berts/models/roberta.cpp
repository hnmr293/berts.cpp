#include "berts/models/roberta.hpp"

#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include "berts/models/keys.h"
#include "berts/models/unicode.hpp"

namespace berts::roberta {

//
// vocab
//

vocab::vocab()
    : inherited()
    , bpe(nullptr) {
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

    auto bos_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_BOS_ID, "<s>");
    auto eos_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_EOS_ID, "</s>");
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

    special.bos = bos_id;
    special.eos = eos_id;
    special.cls = cls_id;
    special.mask = mask_id;
    special.pad = pad_id;
    special.sep = sep_id;
    special.unk = unk_id;

    //
    // bpe initialization
    //

    bpe.reset(new berts::bpe{unk_token()});

    // initialize bpe vocab from constructed vocab
    bpe::vocab_t bpe_vocab{};
    for (const auto &[id, token] : id_to_token_ | std::views::enumerate) {
        bpe_vocab.emplace(token, id);
    }

    // initialize merge vocab
    auto merge_data = ggml_get_tensor(ggml, BERTS_KEY_ALL_MERGE_DATA);

    if (!merge_data) {
        log::error("merge data ({}) is not found", BERTS_KEY_ALL_MERGE_DATA);
        return false;
    }

    if (merge_data->n_dims != 1) {
        log::error("invalid shape: merge_data={}", merge_data->n_dims);
        return false;
    }

    if (merge_data->type != GGML_TYPE_I32) {
        log::error("invalid type of merge_data: {}", (int)merge_data->type);
        return false;
    }

    if (merge_data->ne[0] % 3 != 0) {
        log::error("invalid size of merge_data: {}", merge_data->ne[0]);
        return false;
    }

    size_t merge_count = merge_data->ne[0] / 3;
    log::debug("  merge count: {}", merge_count);

    std::vector<bpe::token_id_pair> merges{};
    for (size_t i = 0; i < merge_count; ++i) {
        int32_t id0 = ggml_get_i32_1d(merge_data, (int)(i * 3 + 0));
        int32_t id1 = ggml_get_i32_1d(merge_data, (int)(i * 3 + 1));
        // int32_t rank = ggml_get_i32_1d(merge_data, (int)(i * 3 + 2));
        merges.emplace_back(id0, id1);
    }

    if (!bpe->load_vocab(bpe_vocab, merges)) {
        log::error("failed to load bpe vocab");
        return false;
    }

    return true;
}

//
// tokenize
//

static bool tokenize(const vocab &vocab,
                     const std::string &text,
                     const std::unordered_set<std::string> &never_split,
                     std::vector<bert_token_t> &result) {
    log::info("tokenization start");

    /**
     * import struct
     * from transformers import RobertaTokenizer
     * t = RobertaTokenizer.from_pretrained('roberta-base')
     * [ struct.unpack('<H', t.byte_encoder[x].encode('utf-16le')) for x in range(256) ]
     */
    static std::array<uint16_t, 256> byte_encoder{{
        // clang-format off
        // "Ā", "ā", "Ă",  "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č",  "č", "Ď", "ď",
        0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10a, 0x10b, 0x10c, 0x10d, 0x10e, 0x10f,
        // "Đ", "đ", "Ē",  "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ",  "ĝ", "Ğ", "ğ",
        0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x116, 0x117, 0x118, 0x119, 0x11a, 0x11b, 0x11c, 0x11d, 0x11e, 0x11f,
        // "Ġ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",",  "-", ".", "/",
        0x120, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
        // "0", "1", "2",  "3", "4", "5", "6", "7", "8", "9", ":", ";", "<",  "=", ">", "?",
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
        // "@", "A", "B",  "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",  "M", "N", "O",
        0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
        // "P", "Q", "R",  "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
        0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
        // "`", "a", "b",  "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",  "m", "n", "o",
        0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
        // "p", "q", "r",  "s", "t", "u", "v", "w", "x", "y", "z", "{", "|",  "}", "~", "ġ",
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x121,
        // "Ģ", "ģ", "Ĥ",  "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į",  "į", "İ", "ı",
        0x122, 0x123, 0x124, 0x125, 0x126, 0x127, 0x128, 0x129, 0x12a, 0x12b, 0x12c, 0x12d, 0x12e, 0x12f, 0x130, 0x131,
        // "Ĳ", "ĳ", "Ĵ",  "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ",  "Ŀ", "ŀ", "Ł",
        0x132, 0x133, 0x134, 0x135, 0x136, 0x137, 0x138, 0x139, 0x13a, 0x13b, 0x13c, 0x13d, 0x13e, 0x13f, 0x140, 0x141,
        // "ł", "¡", "¢",  "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬",  "Ń", "®", "¯",
        0x142, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0x143, 0xae, 0xaf,
        // "°", "±", "²",  "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼",  "½", "¾", "¿",
        0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
        // "À", "Á", "Â",  "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì",  "Í", "Î", "Ï",
        0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
        // "Ð", "Ñ", "Ò",  "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü",  "Ý", "Þ", "ß",
        0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
        // "à", "á", "â",  "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì",  "í", "î", "ï",
        0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
        // "ð", "ñ", "ò",  "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü",  "ý", "þ", "ÿ"
        0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff
        // clang-format on
    }};

    // split text into subtexts
    // "<s>abc <mask>def</s>"
    // -> "<s>", "abc ", "<mask>", "def", "</s>"
    std::vector<std::pair<bool, std::string>> subtexts{};

    {
        std::string rest = text;
        std::string tmp{};
        while (rest.size() != 0) {
            for (const auto &keep : never_split) {
                if (rest.starts_with(keep)) {
                    if (tmp.size() != 0) {
                        subtexts.emplace_back(false, tmp);
                        tmp.clear();
                    }
                    subtexts.emplace_back(true, keep);

                    rest = rest.substr(keep.size());
                    goto LOOP_END;
                }
            }
            tmp += rest[0];
            rest = rest.substr(1);
        LOOP_END:
        }

        if (tmp.size() != 0) {
            subtexts.emplace_back(false, tmp);
        }
    }

    log::when(BERTS_LOG_DEBUG, [&subtexts]() {
        log::debug("  subtexts:");
        for (const auto &[is_special, subtext] : subtexts) {
            log::debug("    \"{}\" {}", subtext, is_special ? "*" : "");
        }
    });

    bpe::cache_t bpe_cache{};
    unicode::regex re{R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};
    std::string mask_token = vocab.mask_token();

    // "<s>", "abc ", "<mask>", "def", "</s>"
    // -> "<s>", "abc", "<mask>", "def", "</s>"
    for (size_t i = 0; i < subtexts.size(); ++i) {
        auto &[is_special, subtext_] = subtexts[i];
        unicode::ustr subtext{subtext_};

        // Mask token behave like a normal word, i.e. include the space before it
        if (i != subtexts.size() - 1) {
            const auto &[is_special_next, subtext_next] = subtexts[i + 1];
            if (is_special_next && subtext_next == mask_token) {
                // use unicode whitespaces
                subtext = subtext.rstrip();
            }
        }

        if (is_special) {
            // special token should not contain whitespaces
            // so here `subtext_` is identical to `subtext`
            bert_token_t id = vocab.token_to_id(subtext_);
            if (id == BERTS_INVALID_TOKEN_ID) {
                log::error("unknown special token: \"{}\"", subtext.encode());
                return false;
            }
            log::when(BERTS_LOG_DEBUG, [&]() {
                log::debug("special token: {} ({})", subtext.encode(), id);
            });
            result.push_back(id);
            continue;
        }

        std::vector<unicode::ustr> ss{};
        re.split(subtext, ss);

        std::vector<unicode::ustr> bpe_tokens{};
        for (const auto &s : ss) {
            // replace control chars
            std::string token = s.encode();
            std::vector<unicode::unic_t> token_{};
            for (char c : token) {
                token_.push_back(byte_encoder[c]);
            }

            if (!vocab.bpe->tokenize(token_, bpe_tokens, bpe_cache)) {
                log::error("failed to tokenize: {}", s.encode());
                return false;
            }
        }

        for (const auto &token : bpe_tokens) {
            bert_token_t id = vocab.token_to_id(token.encode());
            if (id == BERTS_INVALID_TOKEN_ID) {
                log::error("failed to tokenize: {}", token.encode());
                return false;
            }
            result.push_back(id);
        }
    }

    log::debug("finish tokenization");
    return true;
}

bool model::tokenize(const berts_context *ctx,
                     const std::string &text,
                     std::vector<bert_token_t> &out) const {
    (void)ctx;

    std::string bos = vocab->bos_token();
    std::string eos = vocab->eos_token();
    std::string cls = vocab->cls_token();
    std::string mask = vocab->mask_token();
    std::string pad = vocab->pad_token();
    std::string sep = vocab->sep_token();
    std::string unk = vocab->unk_token();

    std::unordered_set<std::string> never_split{bos, eos, cls, mask, pad, sep, unk};

    return roberta::tokenize(*vocab, text, never_split, out);
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
