#include "berts/models/bert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include "berts/models/gguf.hpp"
#include "berts/models/keys.h"
#include "berts/models/unicode.hpp"
#include "berts/models/utils.hpp"

using namespace berts::internal;
using namespace berts::unicode;
namespace uni = berts::unicode;
using trie_t = berts::trie::trie;

namespace berts::bert {

//
// vocab
//

static inline tokenizer_info tokenizer_info_basic() {
    return {
        .normalize = true,
        .remove_replacement_char = true,
        .remove_null_char = true,
        .remove_control_char = true,
        .normalize_whitespaces = true,
        .add_space_around_cjk_char = true,
        .do_lower_case = true,
        .strip_accents = true,
        .split_on_punc = true,
    };
}

static inline tokenizer_info tokenizer_info_no_basic() {
    return {
        .normalize = true,
        .remove_replacement_char = false,
        .remove_null_char = false,
        .remove_control_char = false,
        .normalize_whitespaces = true,
        .add_space_around_cjk_char = false,
        .do_lower_case = false,
        .strip_accents = false,
        .split_on_punc = false,
    };
}

vocab::vocab()
    : inherited()
    , trie(nullptr) {
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

vocab::~vocab() = default;

bool vocab::build_trie() {
    trie.reset(trie::build_trie(id_to_token_));
    return trie && id_to_token_.size();
}

void vocab::clear() {
    id_to_token_.clear();
    token_to_id_.clear();
    trie.reset();
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
    return BERTS_INVALID_TOKEN_ID;
}

bert_token_t vocab::eos_id() const noexcept {
    return BERTS_INVALID_TOKEN_ID;
}

bool vocab::init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) {
    (void)ctx;
    (void)ggml;

    auto cls_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_CLS_ID, "[CLS]", "<s>");
    auto mask_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_MASK_ID, "[MASK]", "<mask>");
    auto pad_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_PAD_ID, "[PAD]", "<pad>");
    auto sep_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_SEP_ID, "[SEP]", "</s>");
    auto unk_id = get_token_id(gguf, BERTS_KEY_TOKENIZER_UNK_ID, "[UNK]", "<unk>");

    log::when(BERTS_LOG_INFO, [=, this]() {
        log::info("  cls_id:  {} ({})", cls_id, cls_token());
        log::info("  mask_id: {} ({})", mask_id, mask_token());
        log::info("  pad_id:  {} ({})", pad_id, pad_token());
        log::info("  sep_id:  {} ({})", sep_id, sep_token());
        log::info("  unk_id:  {} ({})", unk_id, unk_token());
    });

    if (cls_id == BERTS_INVALID_TOKEN_ID ||
        mask_id == BERTS_INVALID_TOKEN_ID ||
        pad_id == BERTS_INVALID_TOKEN_ID ||
        sep_id == BERTS_INVALID_TOKEN_ID ||
        unk_id == BERTS_INVALID_TOKEN_ID) {
        return false;
    }

    special.cls = cls_id;
    special.mask = mask_id;
    special.pad = pad_id;
    special.sep = sep_id;
    special.unk = unk_id;

    auto do_lower_case = gguf::gguf_bool(gguf, BERTS_KEY_TOKENIZER_DO_LOWER_CASE, true);
    auto do_basic_tokenize = gguf::gguf_bool(gguf, BERTS_KEY_TOKENIZER_DO_BASIC_TOKENIZE, true);
    // BERTS_KEY_TOKENIZER_NEVER_SPLIT
    auto tokenize_chinese_chars = gguf::gguf_bool(gguf, BERTS_KEY_TOKENIZER_CHINESE_CHARS, true);
    auto strip_accent = gguf::gguf_bool(gguf, BERTS_KEY_TOKENIZER_STRIP_ACCENT, do_lower_case);

    /**
    skip props
    {
        // ignored, always normalized with NFC
        bool normalize;
        // remove U+FFFD
        bool remove_replacement_char;
        // remove U+0000
        bool remove_null_char;
        // remove control chars (category C*)
        bool remove_control_char;
        // convert all whitespaces to a normal space (U+0020)
        bool normalize_whitespaces;
        // split words at a punctuation
        bool split_on_punc;
    }
    */

    cond = do_basic_tokenize
               ? tokenizer_info_basic()
               : tokenizer_info_no_basic();

    cond.do_lower_case = do_lower_case;
    cond.add_space_around_cjk_char = tokenize_chinese_chars;
    cond.strip_accents = strip_accent;

    log::when(BERTS_LOG_INFO, [this, do_basic_tokenize]() {
        log::info("  do_basic_tokenize = {}", do_basic_tokenize);
        log::info(
            "  tokenizer_info {{\n"
            "    bool normalize = {:<5s};                 // ignored, always normalized with NFC\n"
            "    bool remove_replacement_char = {:<5s};   // remove U+FFFD\n"
            "    bool remove_null_char = {:<5s};          // remove U+0000\n"
            "    bool remove_control_char = {:<5s};       // remove control chars (category C*)\n"
            "    bool normalize_whitespaces = {:<5s};     // convert all whitespaces to a normal space (U+0020)\n"
            "    bool add_space_around_cjk_char = {:<5s}; // add space around all CJK characters\n"
            "    bool do_lower_case = {:<5s};             // force input to be lowercase letters\n"
            "    bool strip_accents = {:<5s};             // remove all accent chars\n"
            "    bool split_on_punc = {:<5s};             // split words at a punctuation\n"
            "  }}",
            cond.normalize,
            cond.remove_replacement_char,
            cond.remove_null_char,
            cond.remove_control_char,
            cond.normalize_whitespaces,
            cond.add_space_around_cjk_char,
            cond.do_lower_case,
            cond.strip_accents,
            cond.split_on_punc);
    });

    if (!build_trie()) {
        log::error("fail to build vocab");
        clear();
        return false;
    }

    return true;
}

//
// weights::init
//

#define KEY_PREFIX "berts.bert."
#define KEY(s) KEY_PREFIX #s
#define KEY_N(pre, post) KEY_PREFIX #pre ".{}." #post

// embedding keys
const char *BERTS_KEY_BERT_EMB_TOKEN = KEY(embeddings.word_embeddings.weight);
const char *BERTS_KEY_BERT_EMB_SEGM = KEY(embeddings.token_type_embeddings.weight);
const char *BERTS_KEY_BERT_EMB_POS = KEY(embeddings.position_embeddings.weight);
const char *BERTS_KEY_BERT_LN_W = KEY(embeddings.LayerNorm.weight);
const char *BERTS_KEY_BERT_LN_B = KEY(embeddings.LayerNorm.bias);

// encoder keys
const char *BERTS_KEY_BERT_ENC_N_Q_W = KEY_N(encoder.layer, attention.self.query.weight);
const char *BERTS_KEY_BERT_ENC_N_Q_B = KEY_N(encoder.layer, attention.self.query.bias);
const char *BERTS_KEY_BERT_ENC_N_K_W = KEY_N(encoder.layer, attention.self.key.weight);
const char *BERTS_KEY_BERT_ENC_N_K_B = KEY_N(encoder.layer, attention.self.key.bias);
const char *BERTS_KEY_BERT_ENC_N_V_W = KEY_N(encoder.layer, attention.self.value.weight);
const char *BERTS_KEY_BERT_ENC_N_V_B = KEY_N(encoder.layer, attention.self.value.bias);
const char *BERTS_KEY_BERT_ENC_N_FF_W = KEY_N(encoder.layer, attention.output.dense.weight);
const char *BERTS_KEY_BERT_ENC_N_FF_B = KEY_N(encoder.layer, attention.output.dense.bias);
const char *BERTS_KEY_BERT_ENC_N_LN_FF_W = KEY_N(encoder.layer, attention.output.LayerNorm.weight);
const char *BERTS_KEY_BERT_ENC_N_LN_FF_B = KEY_N(encoder.layer, attention.output.LayerNorm.bias);
const char *BERTS_KEY_BERT_ENC_N_I_W = KEY_N(encoder.layer, intermediate.dense.weight);
const char *BERTS_KEY_BERT_ENC_N_I_B = KEY_N(encoder.layer, intermediate.dense.bias);
const char *BERTS_KEY_BERT_ENC_N_O_W = KEY_N(encoder.layer, output.dense.weight);
const char *BERTS_KEY_BERT_ENC_N_O_B = KEY_N(encoder.layer, output.dense.bias);
const char *BERTS_KEY_BERT_ENC_N_LN_OUT_W = KEY_N(encoder.layer, output.LayerNorm.weight);
const char *BERTS_KEY_BERT_ENC_N_LN_OUT_B = KEY_N(encoder.layer, output.LayerNorm.bias);

// pooler keys
const char *BERTS_KEY_BERT_POOL_W = KEY(pooler.dense.weight);
const char *BERTS_KEY_BERT_POOL_B = KEY(pooler.dense.bias);

static inline ggml_tensor *tensor(ggml_context *ctx, const char *key) {
    auto t = ggml_get_tensor(ctx, key);
    if (!t) {
        log::error("failed to read tensor: {}", key);
    }
    log::debug("  store {}", key);
    return t;
}

bool weights::init(berts_context *ctx, ggml_context *ggml, gguf_context *gguf) {
    std::vector<std::string> stored;

#define GET_TENSOR(dest, key)         \
    do {                              \
        auto v = tensor(ggml, (key)); \
        if (!v) {                     \
            return false;             \
        }                             \
        stored.emplace_back((key));   \
        dest = v;                     \
    } while (0)

#define GET_TENSOR_N(dest, key, n)           \
    do {                                     \
        std::string name =                   \
            berts::fmt::fmt((key), (n));     \
        auto v = tensor(ggml, name.c_str()); \
        if (!v) {                            \
            return false;                    \
        }                                    \
        stored.push_back(name);              \
        dest = v;                            \
    } while (0)

    GET_TENSOR(this->token_embedding, BERTS_KEY_BERT_EMB_TOKEN);
    GET_TENSOR(this->segment_embedding, BERTS_KEY_BERT_EMB_SEGM);
    GET_TENSOR(this->position_embedding, BERTS_KEY_BERT_EMB_POS);
    GET_TENSOR(this->ln_w, BERTS_KEY_BERT_LN_W);
    GET_TENSOR(this->ln_b, BERTS_KEY_BERT_LN_B);

    hparams hparams;
    get_hparams(ctx, &hparams);

    this->layers.resize(hparams.n_layers);

    for (const auto [n, layer] : this->layers | std::views::enumerate) {
        GET_TENSOR_N(layer.q_w, BERTS_KEY_BERT_ENC_N_Q_W, n);
        GET_TENSOR_N(layer.q_b, BERTS_KEY_BERT_ENC_N_Q_B, n);
        GET_TENSOR_N(layer.k_w, BERTS_KEY_BERT_ENC_N_K_W, n);
        GET_TENSOR_N(layer.k_b, BERTS_KEY_BERT_ENC_N_K_B, n);
        GET_TENSOR_N(layer.v_w, BERTS_KEY_BERT_ENC_N_V_W, n);
        GET_TENSOR_N(layer.v_b, BERTS_KEY_BERT_ENC_N_V_B, n);
        GET_TENSOR_N(layer.ff_w, BERTS_KEY_BERT_ENC_N_FF_W, n);
        GET_TENSOR_N(layer.ff_b, BERTS_KEY_BERT_ENC_N_FF_B, n);
        GET_TENSOR_N(layer.ln_ff_w, BERTS_KEY_BERT_ENC_N_LN_FF_W, n);
        GET_TENSOR_N(layer.ln_ff_b, BERTS_KEY_BERT_ENC_N_LN_FF_B, n);
        GET_TENSOR_N(layer.i_w, BERTS_KEY_BERT_ENC_N_I_W, n);
        GET_TENSOR_N(layer.i_b, BERTS_KEY_BERT_ENC_N_I_B, n);
        GET_TENSOR_N(layer.o_w, BERTS_KEY_BERT_ENC_N_O_W, n);
        GET_TENSOR_N(layer.o_b, BERTS_KEY_BERT_ENC_N_O_B, n);
        GET_TENSOR_N(layer.ln_out_w, BERTS_KEY_BERT_ENC_N_LN_OUT_W, n);
        GET_TENSOR_N(layer.ln_out_b, BERTS_KEY_BERT_ENC_N_LN_OUT_B, n);
    }

    GET_TENSOR(this->pool_w, BERTS_KEY_BERT_POOL_W);
    GET_TENSOR(this->pool_b, BERTS_KEY_BERT_POOL_B);

    // print unused tensors
    log::when(BERTS_LOG_INFO, [&stored, gguf]() {
        const int n_tensors = gguf_get_n_tensors(gguf);
        for (int i = 0; i < n_tensors; ++i) {
            const std::string tensor_name{gguf_get_tensor_name(gguf, i)};
            if (std::find(stored.begin(), stored.end(), tensor_name) == stored.end()) {
                if (tensor_name != BERTS_KEY_ALL_VOCAB_SIZE && tensor_name != BERTS_KEY_ALL_VOCAB_DATA) {
                    log::info("  unused {} {}", i, tensor_name);
                }
            }
        }
    });

    return true;
}

//
// model::tokenize
//

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

static void clean_text_and_split(const ustr &in, std::vector<ustr> &out, const tokenizer_info &cond) {
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
                           const tokenizer_info &cond) {
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

static bool wordpiece_tokenize(const vocab &vocab,
                               const std::vector<ustr> &words,
                               std::vector<bert_token_t> &result) {
    log::debug("start wordpiece_tokenize");

    auto root_node = trie::trie_root(vocab.trie.get());
    auto cont_node = trie::search_node(vocab.trie.get(), ustr{"##", 2});
    if (!cont_node) {
        log::error("corrupted vocab: \"##\" is not found");
        return false;
    }

    auto unk = vocab.special.unk;

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

static bool tokenize(const vocab &vocab,
                     const std::string &text,
                     const std::unordered_set<std::string> &never_split,
                     std::vector<bert_token_t> &result) {
    log::info("start tokenize");

    // usually never_split is small, so
    // i expect this does not cause
    // performance issue :)
    std::unordered_set<ustr> keep{};
    for (const auto &word : never_split) {
        keep.emplace(word);
    }

    std::vector<ustr> split_tokens{};
    basic_tokenize(text, keep, split_tokens, vocab.cond);

    wordpiece_tokenize(vocab, split_tokens, result);

    log::info("end tokenize");

    return true;
}

bool model::tokenize(const berts_context *ctx, const std::string &text, std::vector<bert_token_t> &out) const {
    (void)ctx;

    std::string cls = vocab->cls_token();
    std::string mask = vocab->mask_token();
    std::string pad = vocab->pad_token();
    std::string sep = vocab->sep_token();
    std::string unk = vocab->unk_token();

    std::unordered_set<std::string> never_split{cls, mask, pad, sep, unk};

    return bert::tokenize(*vocab, text, never_split, out);
}

//
// model::eval
//

internal::ggml_size_info
model::get_context_buffer_size(size_t token_count,
                               const internal::hparams &hparams,
                               const berts_eval_info &cond) const {
    internal::ggml_size_info size{};

    const size_t hidden_dim = hparams.hidden_dim;
    const size_t n_layers = hparams.n_layers;
    const size_t n_heads = hparams.attn_heads;
    const size_t intm_dim = hparams.intermediate_dim;

    size.graph += ggml_graph_overhead();

    //
    // embedding
    //

    // token emb: tensor_1d I32 (n,)
    // seg emb  : tensor_1d I32 (n,)
    // pos emb  : tensor_1d I32 (n,)
    size.emb += get_tensor_size(GGML_TYPE_I32, token_count) * 3;

    // apply embs: F32 (n,hidden_dim)
    // ggml_get_rows creates a new tensor with type=F32
    size.emb += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 3;

    // add embs: F32 (n,hidden_dim)
    // ggml_add creates a new tensor with same shape of lhs
    size.emb += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 2;

    // layer norm: F32 (n,hidden_dim)
    // ggml_norm + ggml_add, ggml_mul, ggml_repeat, ggml_repeat
    // ggml_norm creates a new tensor with same shape of arg
    // ggml_mul creates a new tensor with same shape of lhs
    // ggml_repeat creates a new tensor with same shape of rhs
    size.emb += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;

    //
    // self-attention
    //

    // each layer
    if (n_layers != 0) {
        // q, k, v
        // clang-format off
        size.layer += (
            // dense + reshape: F32 (1,n,n_heads,attn_dim) [same size as (n,hidden_dim)]
            // dense = add + mul_mat + repeat
            // reshape is just a view
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // mul_mat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // repeat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // add
            get_tensor_size(GGML_TYPE_F32, 0)                       + // reshape
            // permute + cont: F32 q,k (1,n_heads,n,attn_dim) [same size as (n,hidden_dim)]
            //                 F32 v   (1,n_heads,attn_dim,n) [same size as (n,hidden_dim)]
            // permute is just a view
            // cont creates a new tensor with same shape of arg
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // cont
            get_tensor_size(GGML_TYPE_F32, 0)                         // permute
        ) * 3;
        // clang-format on

        // softmax: F32 (1,n_heads,n,n)
        // mul_mat create a new tensor with the shape (1,n_heads,n,n)
        // sosftmax create a new tensor with same shape of arg
        size.layer += get_tensor_size(GGML_TYPE_F32, token_count, token_count, n_heads) * 2;

        // v * sim: F32 (1,n_heads,n,attn_dim) [same size as (n,hidden_dim)]
        size.layer += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // permute + cont: F32 (1,n,n_heads,attn_dim) [same size as (n,hidden_dim)]
        size.layer += get_tensor_size(GGML_TYPE_F32, 0) + get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // cpy: F32 (n,hidden_dim)
        // cpy creates a view of rhs
        size.layer += get_tensor_size(GGML_TYPE_F32, 0) + get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // dense
        // clang-format off
        size.layer += (
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // mul_mat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // repeat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count)   // add
        );
        // clang-format on

        // add
        size.layer += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // layer norm
        size.layer += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;

        //
        // intermediate
        //

        // dense
        size.layer += (get_tensor_size(GGML_TYPE_F32, intm_dim, token_count) + // mul_mat
                       get_tensor_size(GGML_TYPE_F32, intm_dim, token_count) + // repeat
                       get_tensor_size(GGML_TYPE_F32, intm_dim, token_count)   // add
        );

        // gelu
        // gelu creates a new tensor with same shape of arg
        size.layer += get_tensor_size(GGML_TYPE_F32, intm_dim, token_count);

        // dense
        size.layer += (get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // mul_mat
                       get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // repeat
                       get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count)   // add
        );

        // add
        size.layer += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // layer norm
        size.layer += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;
    }

    //
    // pooler
    //

    switch (cond.pool_type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE:
        return size;
    case BERTS_POOL_CLS:
        // view
        size.pooler += get_tensor_size(GGML_TYPE_F32, 0); // view
        break;
    case BERTS_POOL_AVG:
    case BERTS_POOL_MAX:
        size.pooler += get_tensor_size(GGML_TYPE_F32, hidden_dim); // cont
        break;
    default:
        // must not happen!
        log::error("unknown pooling type: {}", (int)cond.pool_type);
        size.pooler += 0;
        break;
    }

    // dense
    size.pooler += (get_tensor_size(GGML_TYPE_F32, hidden_dim) + // mul_mat
                    get_tensor_size(GGML_TYPE_F32, hidden_dim) + // repeat
                    get_tensor_size(GGML_TYPE_F32, hidden_dim)   // add
    );

    // tanh
    size.pooler += get_tensor_size(GGML_TYPE_F32, hidden_dim);

    return size;
}

bool model::build_graph(ggml_ctx &ggml,
                        const internal::hparams &hparams,
                        const berts_eval_info &cond,
                        const std::vector<bert_token_t> &tokens,
                        const std::vector<bert_segment_t> &segments) const {
#ifdef BERTS_DEBUG
    auto &cc = ggml_context_for_debug::from(ggml.ctx);
#endif

    const auto n = tokens.size();
    auto eps = hparams.eps;
    auto last_layer_index = cond.output_layer;

#ifdef BERTS_DEBUG
    internal::ggml_size_info size = get_context_buffer_size(n, hparams, cond);
#endif

    //
    // embeddings
    //

    auto token_emb = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, n);
    std::memcpy(token_emb->data, tokens.data(), n * ggml_element_size(token_emb));

    auto seg_emb = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, n);
    std::memcpy(seg_emb->data, segments.data(), n * ggml_element_size(seg_emb));

    auto pos_emb = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, n);
    for (size_t i = 0; i < n; ++i) {
        ggml_set_i32_1d(pos_emb, i, i);
    }

    // x = token_emb + pos_emb + seg_emb
    auto x = ggml_get_rows(ggml, weights.token_embedding, token_emb);
    x = ggml_add(ggml, ggml_get_rows(ggml, weights.position_embedding, pos_emb), x);
    x = ggml_add(ggml, ggml_get_rows(ggml, weights.segment_embedding, seg_emb), x);

    // x = layer_norm(x)
    x = bert_layer_norm(ggml, x, weights.ln_w, weights.ln_b, eps);

    // x := (N,hidden_dim)
    GGML_ASSERT(x->n_dims == 2 || (x->ne[2] == 1 && x->ne[3] == 1));
    GGML_ASSERT(x->ne[0] == hparams.hidden_dim);
    GGML_ASSERT((size_t)x->ne[1] == n);

#ifdef BERTS_DEBUG
    cc.check(size.emb, "emb");
#endif

    //
    // encoders
    //

    const auto n_head = hparams.attn_heads;
    const auto attn_dim = hparams.hidden_dim / n_head;
    // hidden_dim := n_head * attn_dim

    // * BertEncoder
    for (const auto [layer_index, layer] : weights.layers | std::views::enumerate) {
        // ** BertLayer

        if (last_layer_index <= layer_index) {
            break;
        }

        // self-attention
        // *** BertAttention
        {
            // **** BertSelfAttention
            auto q = bert_dense(ggml, x, layer.q_w, layer.q_b);
            ggml_format_name(q, "q_%lld", layer_index);
            q = ggml_reshape_4d(ggml, q, attn_dim, n_head, n, 1); // (1,N,head,dim)

            auto k = bert_dense(ggml, x, layer.k_w, layer.k_b);
            ggml_format_name(k, "k_%lld", layer_index);
            k = ggml_reshape_4d(ggml, k, attn_dim, n_head, n, 1); // (1,N,head,dim)

            auto v = bert_dense(ggml, x, layer.v_w, layer.v_b);
            ggml_format_name(v, "v_%lld", layer_index);
            v = ggml_reshape_4d(ggml, v, attn_dim, n_head, n, 1); // (1,N,head,dim)

            // (1,N,head,dim) -> (1,head,N,dim)
            q = ggml_cont(ggml, ggml_permute(ggml, q, 0, 2, 1, 3));
            k = ggml_cont(ggml, ggml_permute(ggml, k, 0, 2, 1, 3));
            // (1,N,head,dim) -> (1,head,dim,N)
            v = ggml_cont(ggml, ggml_permute(ggml, v, 1, 2, 0, 3));

            // sim = softmax(kq / sqrt(attn_dim))
            // (head,N,N)
            const auto scale = 1.0f / std::sqrt((float)attn_dim);
            auto sim = ggml_soft_max_ext(ggml, ggml_mul_mat(ggml, k, q), nullptr, scale);
            ggml_format_name(sim, "sim_%lld", layer_index);

            auto res = ggml_mul_mat(ggml, v, sim);                      // (1,head,N,dim)
            res = ggml_cont(ggml, ggml_permute(ggml, res, 0, 2, 1, 3)); // (1,N,head,dim)

            // (N,hidden_dim)
            res = ggml_cpy(ggml, res, ggml_new_tensor_2d(ggml, GGML_TYPE_F32, hparams.hidden_dim, n)); // (N,hidden_dim)
            ggml_format_name(res, "attn_%lld", layer_index);

            // output
            // **** BertSelfOutput
            res = bert_dense(ggml, res, layer.ff_w, layer.ff_b);
            x = ggml_add(ggml, x, res);
            x = bert_layer_norm(ggml, x, layer.ln_ff_w, layer.ln_ff_b, eps);
            ggml_format_name(x, "ff_%lld", layer_index);
        }

        // intermediate
        {
            // *** BertIntermediate
            auto res = bert_dense(ggml, x, layer.i_w, layer.i_b);
            switch (hparams.hidden_act) {
                using enum hidden_act;
            case BERTS_HIDDEN_ACT_GELU:
                res = ggml_gelu(ggml, res);
                break;
            default:
                log::error("unknown activation function");
                return false;
            }

            // *** BertOutput
            res = bert_dense(ggml, res, layer.o_w, layer.o_b);
            x = ggml_add(ggml, x, res);
            x = bert_layer_norm(ggml, x, layer.ln_out_w, layer.ln_out_b, eps);
            ggml_format_name(x, "intm_%lld", layer_index);
        }
    }

    // x := (1,1,n,hidden_dim)

#ifdef BERTS_DEBUG
    cc.check(size.emb + size.layers(last_layer_index), "layers");
#endif

    //
    // pooler
    //

    switch (cond.pool_type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE:
        // return non-pooled tensor
        ggml_set_name(x, "out");
        goto RUN_COMPUTE;
    case BERTS_POOL_CLS:
        // retrieve first token (hidden_dim,) of (n,hidden_dim)
        x = ggml_view_1d(ggml, x, x->ne[0], 0);
        break;
    case BERTS_POOL_AVG:
        // average pooling
        x = ggml_pool_2d(ggml, x, GGML_OP_POOL_AVG, 1, n, 1, n, 0, 0);
        break;
    case BERTS_POOL_MAX:
        // max pooling
        x = ggml_pool_2d(ggml, x, GGML_OP_POOL_MAX, 1, n, 1, n, 0, 0);
        break;
    default:
        // must not happen!
        log::error("unknown pooling type: {}", (int)cond.pool_type);
        return false;
    }

    GGML_ASSERT(x->ne[0] == hparams.hidden_dim && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1);

    x = bert_dense(ggml, x, weights.pool_w, weights.pool_b);
    x = ggml_tanh(ggml, x);
    ggml_set_name(x, "out");

RUN_COMPUTE:

#ifdef BERTS_DEBUG
    cc.check(size.emb + size.layers(last_layer_index) + size.pooler, "pooler");
#endif

    return true;
}

} // namespace berts::bert
