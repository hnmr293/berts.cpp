#include "berts/models/internal.hpp"
#include <cmath>
#include <memory>
#include <unordered_set>
#include "berts/models/log.hpp"
#include "berts/models/trie.hpp"
#include "berts/models/unicode.hpp"
#include "berts/models/utils.hpp"

using namespace berts;
namespace uni = berts::unicode;
using namespace berts::unicode;

// std::unique_resource<trie::trie>

namespace std {

template <>
struct default_delete<trie::trie> {
    void operator()(trie::trie *trie) const noexcept {
        if (trie) trie::free_trie(trie);
    }
};

} // namespace std

using trie_t = std::unique_ptr<trie::trie>;

//
// models
//

struct special_tokens {
    bert_token_t cls;
    bert_token_t mask;
    bert_token_t pad;
    bert_token_t sep;
    bert_token_t unk;
};

struct vocab {
    berts_tokenizer_info cond;
    special_tokens special;
    trie_t trie;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, bert_token_t> token_to_id;

    vocab()
        : trie(nullptr) {
        this->special.cls = BERTS_INVALID_TOKEN_ID;
        this->special.mask = BERTS_INVALID_TOKEN_ID;
        this->special.pad = BERTS_INVALID_TOKEN_ID;
        this->special.sep = BERTS_INVALID_TOKEN_ID;
        this->special.unk = BERTS_INVALID_TOKEN_ID;
    }

    vocab(size_t n)
        : vocab() {
        this->id_to_token.reserve(n);
        this->token_to_id.reserve(n);
    }

    void build_trie() {
        this->trie.reset(trie::build_trie(this->id_to_token));
    }

    operator bool() const noexcept {
        return trie && id_to_token.size();
    }
};

struct berts_context {
    internal::hparams hparams;
    struct vocab vocab;
    std::unique_ptr<internal::model> model;
    gguf_context *gguf;
    ggml_context *ctx;
    bool initialized_success;

    berts_context(const internal::hparams &hparams, internal::model *model, gguf_context *gguf, ggml_context *ctx)
        : hparams(hparams)
        , vocab()
        , model(model)
        , gguf(gguf)
        , ctx(ctx)
        , initialized_success(false) {
        if (!model) {
            log::error("model is empty");
            return;
        }

        if (!model->load_vocab(this)) {
            log::error("fail to load vocab");
            return;
        }

        vocab.build_trie();
        if (!vocab) {
            log::error("fail to build vocab");
            return;
        }

        if (!model->init_weight(this)) {
            log::error("fail to load weights");
            return;
        }

        this->initialized_success = true;
    }
};

namespace berts::internal {

berts_context *new_context(const hparams &hparams, model *model, gguf_context *gguf, ggml_context *ctx) {
    auto bert = new berts_context{hparams, model, gguf, ctx};
    if (!bert->initialized_success) {
        delete bert;
        bert = nullptr;
    }
    return bert;
}

void free_context(berts_context *ctx) {
    delete ctx;
}

gguf_context *get_gguf_context(berts_context *ctx) {
    return ctx->gguf;
}

ggml_context *get_ggml_context(berts_context *ctx) {
    return ctx->ctx;
}

bool get_hparams(const berts_context *ctx, hparams *params) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (params) {
        *params = ctx->hparams;
    }

    return true;
}

bool is_model_loaded(const berts_context *ctx) {
    return ctx && ctx->model;
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens);
}

ggml_tensor *eval(berts_context *ctx,
                  const std::vector<bert_token_t> &tokens,
                  const std::vector<bert_segment_t> &segments) {
    if (!check_ctx(ctx)) {
        return nullptr;
    }

    return ctx->model->eval(ctx, tokens, segments);
}

ggml_tensor *model::eval(berts_context *ctx, const std::vector<bert_token_t> &tokens) const {
    std::vector<bert_segment_t> segments(tokens.size());
    return this->eval(ctx, tokens, segments);
}

//
// tokenizers
//

std::string id_to_token(const berts_context *ctx, bert_token_t token_id) {
    if (!check_ctx(ctx)) {
        return "";
    }

    if (ctx->vocab.id_to_token.size() <= token_id) {
        log::error("token id {} is not found (max={})", token_id, ctx->vocab.id_to_token.size());
        return "";
    }

    return ctx->vocab.id_to_token[token_id];
}

bert_token_t token_to_id(const berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    if (p == ctx->vocab.token_to_id.end()) {
        log::error("token {} is not found", token);
        return BERTS_INVALID_TOKEN_ID;
    }

    return p->second;
}

bool add_token(berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    if (has_token(ctx, token)) {
        log::warn("  token {} already exists", token);
        return false;
    }

    const auto next_id = static_cast<bert_token_t>(ctx->vocab.id_to_token.size());
    ctx->vocab.id_to_token.push_back(token);
    ctx->vocab.token_to_id[token] = next_id;
    // log::debug("  token {}: {}", next_id, token);

    return true;
}

bool has_token(const berts_context *ctx, const std::string &token) {
    if (!check_ctx(ctx)) {
        return false;
    }

    const auto p = ctx->vocab.token_to_id.find(token);
    return p != ctx->vocab.token_to_id.end();
}

void get_tokenizer_info(const berts_context *ctx, berts_tokenizer_info &cond) {
    if (!check_ctx(ctx)) {
        return;
    }

    cond = ctx->vocab.cond;
}

void set_tokenizer_info(berts_context *ctx, const berts_tokenizer_info &cond) {
    if (!check_ctx(ctx)) {
        return;
    }

    ctx->vocab.cond = cond;
}

void init_tokenizer_info_default(berts_tokenizer_info &cond) {
    cond.normalize = true;
    cond.remove_replacement_char = true;
    cond.remove_null_char = true;
    cond.remove_control_char = true;
    cond.normalize_whitespaces = true;
    cond.add_space_around_cjk_char = true;
    cond.do_lower_case = true;
    cond.strip_accents = true;
    cond.split_on_punc = true;
}

void init_tokenizer_info_no_basic(berts_tokenizer_info &cond) {
    cond.normalize = true;
    cond.remove_replacement_char = false;
    cond.remove_null_char = false;
    cond.remove_control_char = false;
    cond.normalize_whitespaces = true;
    cond.add_space_around_cjk_char = false;
    cond.do_lower_case = false;
    cond.strip_accents = false;
    cond.split_on_punc = false;
}

bert_token_t get_cls_id(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return ctx->vocab.special.cls;
}

void set_cls_id(berts_context *ctx, bert_token_t id) {
    if (!check_ctx(ctx)) {
        return;
    }
    ctx->vocab.special.cls = id;
}

bert_token_t get_mask_id(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return ctx->vocab.special.mask;
}

void set_mask_id(berts_context *ctx, bert_token_t id) {
    if (!check_ctx(ctx)) {
        return;
    }
    ctx->vocab.special.mask = id;
}

bert_token_t get_pad_id(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return ctx->vocab.special.pad;
}

void set_pad_id(berts_context *ctx, bert_token_t id) {
    if (!check_ctx(ctx)) {
        return;
    }
    ctx->vocab.special.pad = id;
}

bert_token_t get_sep_id(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return ctx->vocab.special.sep;
}

void set_sep_id(berts_context *ctx, bert_token_t id) {
    if (!check_ctx(ctx)) {
        return;
    }
    ctx->vocab.special.sep = id;
}

bert_token_t get_unk_id(const berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return BERTS_INVALID_TOKEN_ID;
    }
    return ctx->vocab.special.unk;
}

void set_unk_id(berts_context *ctx, bert_token_t id) {
    if (!check_ctx(ctx)) {
        return;
    }
    ctx->vocab.special.unk = id;
}

// ref: transformers.BasicTokenizer
// ' ', '\t', '\n' and '\r' are control characters,
// but we treat them as whitespace here.
#define BERTS_UNICODE_IS_WS(c) ((c == ' ' || c == '\t' || c == '\n' || c == '\r') || uni::is_whitespace(c))
#define BERTS_UNICODE_IS_CTRL(c) (c != ' ' && c != '\t' && c != '\n' && c != '\r' && uni::is_control(c))
#define BERTS_UNICODE_IS_CJK(c) ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F))
// ^ from transformers.BasicTokenizer._is_chinese_char

static inline ustr safe_norm_nfc(const ustr &in) {
    ustr s{};
    if (!normalize_nfc(in, s)) {
        s = in;
    }
    return s;
}

static inline ustr safe_norm_nfd(const ustr &in) {
    ustr s{};
    if (!normalize_nfd(in, s)) {
        s = in;
    }
    return s;
}

static void clean_text_and_split(const ustr &in, std::vector<ustr> &out, const berts_tokenizer_info &cond) {
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
                           const berts_tokenizer_info &cond) {
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

static bool wordpiece_tokenize(const berts_context *ctx,
                               const std::vector<ustr> &words,
                               const trie::trie *vocab,
                               std::vector<bert_token_t> &result,
                               const berts_tokenizer_info &cond) {
    (void)cond;
    log::debug("start wordpiece_tokenize");

    auto root_node = trie::trie_root(vocab);
    auto cont_node = trie::search_node(vocab, ustr{"##", 2});
    if (!cont_node) {
        log::error("corrupted vocab: \"##\" is not found");
        return false;
    }

    auto unk = internal::get_unk_id(ctx);

    auto root = root_node;
    for (const auto &word : words) {
        ustr found{}, rest{word};
        while (!rest.empty()) {
            const auto id = trie::search_trie_substr(root, rest, found, rest);
            if (id != BERTS_INVALID_TOKEN_ID) {
                // found
                result.push_back(id);
                root = rest.empty() ? root_node : cont_node;
                log::when(BERTS_LOG_DEBUG, [&found, id] {
                    log::debug("  token: {} ({})", found.encode(), id);
                });
            } else {
                // not found
                result.push_back(unk);
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

static bool tokenize(const berts_context *ctx,
                     const std::string &text,
                     const trie::trie *vocab,
                     const std::unordered_set<std::string> &never_split,
                     std::vector<bert_token_t> &result,
                     const berts_tokenizer_info &cond) {
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

    wordpiece_tokenize(ctx, split_tokens, vocab, result, cond);

    log::info("end tokenize");

    return true;
}

bool tokenize(const berts_context *ctx, const std::string &text, std::vector<bert_token_t> &out) {
    if (!check_ctx(ctx)) {
        return false;
    }

    const auto &vocab = ctx->vocab;
    const auto &sp = vocab.special;
    std::string cls = vocab.id_to_token[sp.cls];
    std::string mask = vocab.id_to_token[sp.mask];
    std::string pad = vocab.id_to_token[sp.pad];
    std::string sep = vocab.id_to_token[sp.sep];
    std::string unk = vocab.id_to_token[sp.unk];

    std::unordered_set<std::string> never_split{cls, mask, pad, sep, unk};

    return tokenize(ctx, text, ctx->vocab.trie.get(), never_split, out, ctx->vocab.cond);
}

} // namespace berts::internal
