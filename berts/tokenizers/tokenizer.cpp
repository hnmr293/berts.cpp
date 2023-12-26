#include "berts/tokenizers/tokenizer.hpp"

#include <string>
#include <unordered_set>
#include <vector>
#include "berts/common/log.hpp"
#include "berts/models/utils.hpp"
#include "berts/tokenizers/unicode.hpp"

namespace uni = berts::unicode;
using ustr = uni::ustr;
using unic_t = uni::unic_t;
using unic32_t = uni::unic32_t;

namespace berts::tokenizers {

inline bool check_ctx(context *ctx) {
    if (!ctx) {
        log::warn("ctx=nullptr");
        return false;
    } else {
        return true;
    }
}

// id <-> tokens
struct vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, bert_token_t> token_to_id;
    berts::trie::trie *trie;

    vocab()
        : id_to_token()
        , token_to_id()
        , trie(nullptr) {}

    vocab(size_t n)
        : vocab() {
        this->id_to_token.reserve(n);
        this->token_to_id.reserve(n);
    }

    vocab(const vocab &) = delete;

    vocab(vocab &&other) noexcept
        : id_to_token(std::move(other.id_to_token))
        , token_to_id(std::move(other.token_to_id))
        , trie(other.trie) {
        other.trie = nullptr;
    }

    ~vocab() {
        this->dispose();
    }

    vocab &operator=(const vocab &) = delete;

    vocab &operator=(vocab &&other) noexcept {
        if (this != &other) {
            this->dispose();
            this->id_to_token = std::move(other.id_to_token);
            this->token_to_id = std::move(other.token_to_id);
            this->trie = other.trie;
            other.trie = nullptr;
        }
        return *this;
    }

    void dispose() {
        if (this->trie) berts::trie::free_trie(trie);
    }

    bool build() {
        this->dispose();
        this->trie = berts::trie::build_trie(this->id_to_token);
        return !!this->trie;
    }
};

const bert_token_t invalid_token = (bert_token_t)-1;

struct special_tokens {
    bert_token_t cls_id;
    bert_token_t mask_id;
    bert_token_t pad_id;
    bert_token_t sep_id;
    bert_token_t unk_id;
};

struct context {
    vocab vocab;
    special_tokens special_tokens;
    std::unordered_set<ustr> never_split;
    berts_tokenize_info cond;

    context(const berts_tokenize_info &cond)
        : vocab()
        , special_tokens({invalid_token, invalid_token, invalid_token, invalid_token, invalid_token})
        , never_split()
        , cond(cond) {}

    ~context() {
        dispose();
    }

    context(const context &) = delete;

    context(context &&other) noexcept
        : vocab(std::move(other.vocab))
        , special_tokens(std::move(other.special_tokens))
        , never_split(std::move(other.never_split))
        , cond(std::move(cond)) {}

    context &operator=(const context &) = delete;

    context &operator=(context &&other) noexcept {
        if (this != &other) {
            dispose();
            vocab = std::move(other.vocab);
            special_tokens = std::move(other.special_tokens);
            never_split = std::move(other.never_split);
            cond = std::move(other.cond);
        }
    }

    void dispose() {
    }
};

context *new_context(const berts_tokenize_info &cond) {
    return new context{cond};
}

void free_context(context *ctx) {
    delete ctx;
}

bert_token_t get_cls_id(context *ctx) {
    return check_ctx(ctx) ? ctx->special_tokens.cls_id : invalid_token;
}

void set_cls_id(context *ctx, bert_token_t id) {
    if (check_ctx(ctx)) ctx->special_tokens.cls_id = id;
}

bert_token_t get_mask_id(context *ctx) {
    return check_ctx(ctx) ? ctx->special_tokens.mask_id : invalid_token;
}

void set_mask_id(context *ctx, bert_token_t id) {
    if (check_ctx(ctx)) ctx->special_tokens.mask_id = id;
}

bert_token_t get_pad_id(context *ctx) {
    return check_ctx(ctx) ? ctx->special_tokens.pad_id : invalid_token;
}
void set_pad_id(context *ctx, bert_token_t id) {
    if (check_ctx(ctx)) ctx->special_tokens.pad_id = id;
}

bert_token_t get_sep_id(context *ctx) {
    return check_ctx(ctx) ? ctx->special_tokens.sep_id : invalid_token;
}

void set_sep_id(context *ctx, bert_token_t id) {
    if (check_ctx(ctx)) ctx->special_tokens.sep_id = id;
}

bert_token_t get_unk_id(context *ctx) {
    return check_ctx(ctx) ? ctx->special_tokens.unk_id : invalid_token;
}

void set_unk_id(context *ctx, bert_token_t id) {
    if (check_ctx(ctx)) ctx->special_tokens.unk_id = id;
}

std::string id_to_token(context *ctx, bert_token_t token_id) {
    if (!check_ctx(ctx)) {
        return "";
    }

    if (ctx->vocab.id_to_token.size() <= token_id) {
        log::error("token id {} is not found (max={})", token_id, ctx->vocab.id_to_token.size());
        return "";
    }

    return ctx->vocab.id_to_token[token_id];
}

bert_token_t token_to_id(context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return invalid_token;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    if (p == ctx->vocab.token_to_id.end()) {
        log::error("token {} is not found", token);
        return invalid_token;
    }

    return p->second;
}

bool add_token(context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (has_token(ctx, token)) {
        log::warn("token {} already exists", token);
        return false;
    }

    const auto next_id = static_cast<bert_token_t>(ctx->vocab.id_to_token.size());
    ctx->vocab.id_to_token.push_back(token);
    ctx->vocab.token_to_id[token] = next_id;

    return true;
}

bool has_token(context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    return p != ctx->vocab.token_to_id.end();
}

static inline bool build_trie(context *ctx) {
    log::info("building vocab");
    if (!check_ctx(ctx)) {
        return false;
    }
    return !!ctx->vocab.build();
}

// void init_ids() {
//     using ustr = unicode::ustr;
//
//     if (this->cls_id == (bert_token_t)-1) {
//         this->cls_id = berts::trie::search_trie(this->trie, ustr{"[CLS]"});
//     }
//
//     if (this->mask_id == (bert_token_t)-1) {
//         this->mask_id = berts::trie::search_trie(this->trie, ustr{"[MASK]"});
//     }
//
//     if (this->sep_id == (bert_token_t)-1) {
//         this->sep_id = berts::trie::search_trie(this->trie, ustr{"[SEP]"});
//     }
//
//     if (this->pad_id == (bert_token_t)-1) {
//         this->pad_id = berts::trie::search_trie(this->trie, ustr{"[PAD]"});
//     }
//
//     if (this->unk_id == (bert_token_t)-1) {
//         this->unk_id = berts::trie::search_trie(this->trie, ustr{"[UNK]"});
//     }
// }

// ref: transformers.BasicTokenizer
// ' ', '\t', '\n' and '\r' are control characters,
// but we treat them as whitespace here.
#define BERTS_UNICODE_IS_WS(c) ((c == ' ' || c == '\t' || c == '\n' || c == '\r') || uni::is_whitespace(c))
#define BERTS_UNICODE_IS_CTRL(c) (c != ' ' && c != '\t' && c != '\n' && c != '\r' && uni::is_control(c))
#define BERTS_UNICODE_IS_CJK(c) ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F))
// ^ from transformers.BasicTokenizer._is_chinese_char

static inline ustr safe_norm_nfc(const ustr &in) {
    ustr s{};
    if (!uni::normalize_nfc(in, s)) {
        s = in;
    }
    return s;
}

static inline ustr safe_norm_nfd(const ustr &in) {
    ustr s{};
    if (!uni::normalize_nfd(in, s)) {
        s = in;
    }
    return s;
}

static void clean_text_and_split(const ustr &in, std::vector<ustr> &out, const berts_tokenize_info &cond) {
    std::vector<ustr::cp> cleaned;
    cleaned.reserve(in.packsize());

    in.each_cp(false, [&cleaned, &cond](ustr::cp C) {
        auto c = C.c;

        if (c == 0) {
            if (cond.remove_null_char) {
                log::info("null character found in text");
            } else {
                cleaned.push_back(c);
            }
            return;
        }

        if (c == 0xfffd) {
            if (cond.remove_replacement_char) {
                log::when(BERTS_LOG_INFO, [&C]() {
                    std::string msg;
                    if (C.is_pair()) {
                        msg = berts::fmt::fmt(
                            "invalid sequence found: lone {} surrogate {:04x}",
                            C.hi ? "high" : "low",
                            C.hi ? (int)C.hi : (int)C.lo);
                    } else {
                        msg = "0xfffd found";
                    }
                    log::info(msg);
                });
            } else {
                cleaned.push_back(c);
            }
            return;
        }

        if (BERTS_UNICODE_IS_CTRL(c)) {
            if (cond.remove_control_char) {
                // do nothing
            } else {
                cleaned.push_back(c);
            }
            return;
        }

        if (BERTS_UNICODE_IS_WS(c)) {
            if (cond.normalize_whitespaces) {
                // treat all whitespaces to single space ' ' (U+0020)
                cleaned.push_back(' ');
            } else {
                cleaned.push_back(c);
            }
            return;
        }

        // normal character
        {
            bool add_space = BERTS_UNICODE_IS_CJK(c) && cond.add_space_around_cjk_char;
            if (add_space) cleaned.push_back(' ');
            cleaned.push_back(C);
            if (add_space) cleaned.push_back(' ');
        }
    });

    // strip
    std::vector<unic_t> current{};
    for (auto &&C : cleaned) {
        if (C.c == ' ') {
            // split words
            if (current.empty()) {
                // proceeding spaces
                // skipping...
            } else {
                out.emplace_back(current);
                current.clear();
            }
            continue;
        }

        if (C.is_pair()) {
            current.push_back(C.hi);
            current.push_back(C.lo);
        } else {
            current.push_back((unic_t)C.c);
        }
    }

    if (!current.empty()) {
        out.emplace_back(current);
    }
}

static bool basic_tokenize(const std::string &text,
                           const std::unordered_set<ustr> &never_split,
                           std::vector<ustr> &result,
                           const berts_tokenize_info &cond) {
    log::debug("start basic_tokenize");

    // NFC normalization
    ustr s = safe_norm_nfc({text});

    // clean text (invalid character removal and whitespace cleanup)
    // and add whitespaces around CJK chars
    // and strip preceeding and trailing spaces
    // and split words by spaces
    std::vector<ustr> words{};
    clean_text_and_split(s, words, cond);

    for (auto &&word : words) {
        if (never_split.contains(word)) {
            result.push_back(word);
            continue;
        }

        if (cond.do_lower_case) {
            ustr s1{std::move(word)};
            uni::to_lower(s1, word);
        }

        if (cond.strip_accents) {
            ustr s1 = safe_norm_nfd(word);
            std::vector<unic32_t> s2{};
            s1.each_cp(true, [&s2](ustr::cp C) {
                if (!uni::is_category(C.c, "Mn")) {
                    s2.push_back(C.c);
                }
            });
            word = safe_norm_nfc({s2});
        }

        if (!cond.split_on_punc) {
            result.push_back(word);
            continue;
        }

        // split at a puctuation

        std::vector<unic32_t> temp{};
        word.each_cp(true, [&result, &temp](ustr::cp C) {
            // .ab.cd.
            if (uni::is_punct(C.c)) {
                // .ab.cd.
                // ^  ^  ^
                if (!temp.empty()) {
                    // .ab.cd.
                    //     ~~^-C
                    //       ` temp
                    result.emplace_back(temp);
                    temp.clear();
                }
                result.emplace_back(&C.c, 1);
            } else {
                // .ab.cd.
                //  ^^ ^^
                temp.push_back(C.c);
            }
        });

        if (!temp.empty()) {
            result.emplace_back(temp);
        }
    }

    log::debug("end basic_tokenize");

    return true;
}

static bool wordpiece_tokenize(const std::vector<ustr> &words,
                               const trie::trie *vocab,
                               std::vector<bert_token_t> &result,
                               const berts_tokenize_info &cond) {
    log::debug("start wordpiece_tokenize");

    auto root_node = trie::trie_root(vocab);
    auto cont_node = trie::search_node(vocab, ustr{"##", 2});
    if (!cont_node) {
        log::error("corrupted vocab: \"##\" is not found");
        return false;
    }

    for (const auto &word : words) {
        ustr found{}, rest{word};
        while (!rest.empty()) {
            auto root = root_node;
            const auto id = trie::search_trie_substr(root, rest, found, rest);
            if (id != (bert_token_t)-1) {
                // found
                result.push_back(id);
                root = cont_node;
                log::when(BERTS_LOG_DEBUG, [&found, id] {
                    log::debug("  token: {} ({})", found.encode(), id);
                });
            } else {
                // not found
                result.push_back(cond.unknown_token_id);
                log::when(BERTS_LOG_WARN, [&rest] {
                    log::warn("  unknown token: {}", rest.encode());
                });
                break;
            }
        }
    }

    log::debug("end wordpiece_tokenize");

    return true;
}

bool tokenize(const std::string &text,
              const trie::trie *vocab,
              const std::unordered_set<std::string> &never_split,
              std::vector<bert_token_t> &result,
              const berts_tokenize_info &cond) {
    log::info("start tokenize");

    // usually never_split is small, so
    // i expect this does not cause
    // performance issue :)
    std::unordered_set<ustr> keep{};
    for (const auto &word : never_split) {
        keep.emplace(word);
    }

    std::vector<ustr> split_tokens{};
    basic_tokenize(text, keep, split_tokens, cond);

    wordpiece_tokenize(split_tokens, vocab, result, cond);

    log::info("end tokenize");

    return true;
}

} // namespace berts::tokenizers
