#include "berts/models/bert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ranges>
#include "berts/models/gguf.hpp"
#include "berts/models/keys.h"
#include "berts/models/utils.hpp"

namespace berts::bert {

using namespace berts::internal;

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

bool model::init_weight(berts_context *ctx) {
    log::info("initializing weights");

    if (!check_model(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);

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
    log::when(BERTS_LOG_INFO, [&stored, ctx]() {
        const auto gguf = get_gguf_context(ctx);
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

static inline bert_token_t get_special_token_id(const berts_context *ctx, const gguf_context *gguf, const char *key, const char *alternate) {
    auto id = gguf::gguf_u32(gguf, key, BERTS_INVALID_TOKEN_ID);
    if (id == BERTS_INVALID_TOKEN_ID) {
        log::warn("{} is not defined; use {} instead", key, alternate);
        id = internal::token_to_id(ctx, alternate);
        if (id == BERTS_INVALID_TOKEN_ID) {
            log::error("{} does not exist in vocab", alternate);
        }
    }
    return id;
}

bool model::load_vocab(berts_context *ctx) {
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
        std::string token{&data[p], token_len};
        p += token_len;

        internal::add_token(ctx, token);
    }

    if (p != vocab_data->ne[0]) {
        log::error("something wrong");
        return false;
    }

    auto cls_id = get_special_token_id(ctx, gguf, BERTS_KEY_TOKENIZER_CLS_ID, "[CLS]");
    auto mask_id = get_special_token_id(ctx, gguf, BERTS_KEY_TOKENIZER_MASK_ID, "[MASK]");
    auto pad_id = get_special_token_id(ctx, gguf, BERTS_KEY_TOKENIZER_PAD_ID, "[PAD]");
    auto sep_id = get_special_token_id(ctx, gguf, BERTS_KEY_TOKENIZER_SEP_ID, "[SEP]");
    auto unk_id = get_special_token_id(ctx, gguf, BERTS_KEY_TOKENIZER_UNK_ID, "[UNK]");

    log::when(BERTS_LOG_INFO, [=]() {
        auto cls = internal::id_to_token(ctx, cls_id);
        auto mask = internal::id_to_token(ctx, mask_id);
        auto pad = internal::id_to_token(ctx, pad_id);
        auto sep = internal::id_to_token(ctx, sep_id);
        auto unk = internal::id_to_token(ctx, unk_id);
        log::info("  cls_id:  {} ({})", cls_id, cls);
        log::info("  mask_id: {} ({})", mask_id, mask);
        log::info("  pad_id:  {} ({})", pad_id, pad);
        log::info("  sep_id:  {} ({})", sep_id, sep);
        log::info("  unk_id:  {} ({})", unk_id, unk);
    });

    if (cls_id == BERTS_INVALID_TOKEN_ID ||
        mask_id == BERTS_INVALID_TOKEN_ID ||
        pad_id == BERTS_INVALID_TOKEN_ID ||
        sep_id == BERTS_INVALID_TOKEN_ID ||
        unk_id == BERTS_INVALID_TOKEN_ID) {
        return false;
    }

    internal::set_cls_id(ctx, cls_id);
    internal::set_mask_id(ctx, mask_id);
    internal::set_pad_id(ctx, pad_id);
    internal::set_sep_id(ctx, sep_id);
    internal::set_unk_id(ctx, unk_id);

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

    berts_tokenizer_info cond{};
    if (do_basic_tokenize) {
        internal::init_tokenizer_info_default(cond);
    } else {
        internal::init_tokenizer_info_no_basic(cond);
    }

    cond.do_lower_case = do_lower_case;
    cond.add_space_around_cjk_char = tokenize_chinese_chars;
    cond.strip_accents = strip_accent;

    log::when(BERTS_LOG_INFO, [&cond, do_basic_tokenize]() {
        log::info("  do_basic_tokenize = {}", do_basic_tokenize);
        log::info(
            "  berts_tokenizer_info {{\n"
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

    internal::set_tokenizer_info(ctx, cond);

    return true;
}

static inline std::string pool_type_str(berts_pool_type type) {
    switch (type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE: return "none";
    case BERTS_POOL_CLS: return "cls";
    case BERTS_POOL_AVG: return "avg";
    case BERTS_POOL_MAX: return "max";
    default: return "";
    }
}

static inline size_t get_data_size(ggml_type type, size_t ne0, size_t ne1 = 1, size_t ne2 = 1, size_t ne3 = 1) {
    size_t data_size = ggml_type_size(type) * (ne0 / ggml_blck_size(type));
    data_size *= ne1;
    data_size *= ne2;
    data_size *= ne3;
    return data_size;
}

static inline size_t get_tensor_size(ggml_type type, size_t ne0, size_t ne1 = 1, size_t ne2 = 1, size_t ne3 = 1) {
    size_t size = get_data_size(type, ne0, ne1, ne2, ne3);
    size += GGML_TENSOR_SIZE;
    size = GGML_PAD(size, GGML_MEM_ALIGN);
    size += GGML_OBJECT_SIZE;
    return size;
}

static inline size_t get_bert_size(size_t token_count,
                                   const hparams &hparams,
                                   const berts_eval_info &cond) {
    size_t size = 0;
    const size_t hidden_dim = hparams.hidden_dim;
    const size_t n_layers = hparams.n_layers;
    const size_t n_heads = hparams.attn_heads;
    const size_t intm_dim = hparams.intermediate_dim;

    //
    // embedding
    //

    // token emb: tensor_1d I32 (n,)
    // seg emb  : tensor_1d I32 (n,)
    // pos emb  : tensor_1d I32 (n,)
    size += get_tensor_size(GGML_TYPE_I32, token_count) * 3;

    // apply embs: F32 (n,hidden_dim)
    // ggml_get_rows creates a new tensor with type=F32
    size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 3;

    // add embs: F32 (n,hidden_dim)
    // ggml_add creates a new tensor with same shape of lhs
    size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 2;

    // layer norm: F32 (n,hidden_dim)
    // ggml_norm + ggml_add, ggml_mul, ggml_repeat, ggml_repeat
    // ggml_norm creates a new tensor with same shape of arg
    // ggml_mul creates a new tensor with same shape of lhs
    // ggml_repeat creates a new tensor with same shape of rhs
    size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;

    //
    // self-attention
    //

    // each layer
    size_t layer_size = 0;

    if (n_layers != 0) {
        // q, k, v
        // clang-format off
        layer_size += (
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
            get_tensor_size(GGML_TYPE_F32, 0)                         // reshape + permute
        ) * 3;
        // clang-format on

        // softmax: F32 (1,n_heads,n,n)
        // mul_mat create a new tensor with the shape (1,n_heads,n,n)
        // sosftmax create a new tensor with same shape of arg
        layer_size += get_tensor_size(GGML_TYPE_F32, token_count, token_count, n_heads) * 2;

        // v * sim: F32 (1,n_heads,n,attn_dim) [same size as (n,hidden_dim)]
        layer_size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // permute + cont: F32 (1,n,n_heads,attn_dim) [same size as (n,hidden_dim)]
        layer_size += get_tensor_size(GGML_TYPE_F32, 0) + get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // cpy: F32 (n,hidden_dim)
        // cpy creates a view of rhs
        layer_size += get_tensor_size(GGML_TYPE_F32, 0) + get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count);

        // dense
        // clang-format off
        layer_size += (
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // mul_mat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // repeat
            get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count)   // add
        );
        // clang-format on

        // add_inplace
        // add_inplace creates a view of lhs
        layer_size += get_tensor_size(GGML_TYPE_F32, 0);

        // layer norm
        layer_size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;

        //
        // intermediate
        //

        // dense
        layer_size += (get_tensor_size(GGML_TYPE_F32, intm_dim, token_count) + // mul_mat
                       get_tensor_size(GGML_TYPE_F32, intm_dim, token_count) + // repeat
                       get_tensor_size(GGML_TYPE_F32, intm_dim, token_count)   // add
        );

        // gelu_inplace
        // gelu_inplace creates a view of arg
        layer_size += get_tensor_size(GGML_TYPE_F32, 0);

        // dense
        layer_size += (get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // mul_mat
                       get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) + // repeat
                       get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count)   // add
        );

        // add_inplace
        layer_size += get_tensor_size(GGML_TYPE_F32, 0);

        // layer norm
        layer_size += get_tensor_size(GGML_TYPE_F32, hidden_dim, token_count) * 5;
    }

    size += layer_size * n_layers;

    //
    // pooler
    //

    size_t pool_size = 0;

    switch (cond.pool_type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE:
        return size;
    case BERTS_POOL_CLS:
        pool_size += get_tensor_size(GGML_TYPE_F32, hidden_dim);
        break;
    case BERTS_POOL_AVG:
        // transpose -> mean -> transpose -> cont
        pool_size += get_tensor_size(GGML_TYPE_F32, 0);          // transpose
        pool_size += get_tensor_size(GGML_TYPE_F32, hidden_dim); // mean
        pool_size += get_tensor_size(GGML_TYPE_F32, 0);          // transpose
        pool_size += get_tensor_size(GGML_TYPE_F32, hidden_dim); // cont
        break;
    case BERTS_POOL_MAX:
        // transpose -> argmax -> cont
        pool_size += get_tensor_size(GGML_TYPE_F32, 0);          // transpose
        pool_size += get_tensor_size(GGML_TYPE_F32, hidden_dim); // argmax
        pool_size += get_tensor_size(GGML_TYPE_F32, hidden_dim); // cont
        break;
    default:
        // must not happen!
        log::error("unknown pooling type: {}", (int)cond.pool_type);
        return false;
    }

    // dense
    pool_size += (get_tensor_size(GGML_TYPE_F32, hidden_dim) + // mul_mat
                  get_tensor_size(GGML_TYPE_F32, hidden_dim) + // repeat
                  get_tensor_size(GGML_TYPE_F32, hidden_dim)   // add
    );

    // gelu_inplace
    pool_size += get_tensor_size(GGML_TYPE_F32, 0);

    size += pool_size;

    return size;
}

bool model::eval(berts_context *ctx,
                 const std::vector<bert_token_t> &tokens,
                 const std::vector<bert_segment_t> &segments,
                 const berts_eval_info &cond,
                 float *out,
                 size_t &out_count) const {
    static_assert(sizeof(bert_token_t) == sizeof(int32_t));
    static_assert(sizeof(bert_segment_t) == sizeof(int32_t));

    log::info("start evaluating BERT");

    if (!check_model(ctx)) {
        return false;
    }

    hparams hparams{};
    get_hparams(ctx, &hparams);
    auto eps = hparams.eps;

    const auto n = tokens.size();

    log::debug("  #tokens = {}", n);

    const size_t input_out_count = out_count;
    size_t needed_out_count;
    switch (cond.pool_type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE: needed_out_count = hparams.hidden_dim * n; break;
    case BERTS_POOL_CLS: needed_out_count = hparams.hidden_dim * 1; break;
    case BERTS_POOL_AVG: needed_out_count = hparams.hidden_dim * 1; break;
    case BERTS_POOL_MAX: needed_out_count = hparams.hidden_dim * 1; break;
    default:
        log::error("unknown pooling type: {}", (int)cond.pool_type);
        return false;
    }

    log::when(BERTS_LOG_DEBUG, [&]() {
        log::debug(
            "  berts_eval_info {{\n"
            "    output_layer = {};\n"
            "    pool_type = {};\n"
            "  }}",
            cond.output_layer,
            pool_type_str(cond.pool_type));
        log::debug("  output size = {}", needed_out_count);
        log::debug("    given     = {}", input_out_count);
    });

    out_count = needed_out_count;

    if (!out) {
        return true;
    }

    const auto n_layers = hparams.n_layers;
    auto last_layer_index = cond.output_layer;
    if (last_layer_index < 0) {
        if (n_layers < -last_layer_index) {
            log::error("invalid output_layer_value: {} (expected: {}..{})", last_layer_index, -n_layers, n_layers);
            return false;
        }
        last_layer_index += n_layers + 1; // -24 -> 1
    } else {
        if (hparams.n_layers < last_layer_index) {
            log::error("invalid output_layer_value: {} (expected: {}..{})", last_layer_index, -n_layers, n_layers);
            return false;
        }
    }

    size_t size = get_bert_size(n, hparams, cond);
    ggml_init_params init{
        /* .mem_size   = */ size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_ctx ggml{init};

    log::debug("  context buffer size = {}", size);

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
    auto x = ggml_get_rows(ggml, this->token_embedding, token_emb);
    x = ggml_add(ggml, ggml_get_rows(ggml, this->position_embedding, pos_emb), x);
    x = ggml_add(ggml, ggml_get_rows(ggml, this->segment_embedding, seg_emb), x);

    // x = layer_norm(x)
    x = bert_layer_norm(ggml, x, this->ln_w, this->ln_b, eps);

    // x := (N,hidden_dim)
    GGML_ASSERT(x->n_dims == 2 || (x->ne[2] == 1 && x->ne[3] == 1));
    GGML_ASSERT(x->ne[0] == hparams.hidden_dim);
    GGML_ASSERT((size_t)x->ne[1] == n);

    //
    // encoders
    //

    const auto n_head = hparams.attn_heads;
    const auto attn_dim = hparams.hidden_dim / n_head;
    // hidden_dim := n_head * attn_dim

    // * BertEncoder
    for (const auto [layer_index, layer] : this->layers | std::views::enumerate) {
        // ** BertLayer

        if (last_layer_index <= layer_index) {
            break;
        }

        // self-attention
        // *** BertAttention
        {
            // **** BertSelfAttention
            auto q = bert_dense(ggml, x, layer.q_w, layer.q_b);
            q = ggml_reshape_4d(ggml, q, attn_dim, n_head, n, 1); // (1,N,head,dim)

            auto k = bert_dense(ggml, x, layer.k_w, layer.k_b);
            k = ggml_reshape_4d(ggml, k, attn_dim, n_head, n, 1); // (1,N,head,dim)

            auto v = bert_dense(ggml, x, layer.v_w, layer.v_b);
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

            auto res = ggml_mul_mat(ggml, v, sim);                      // (1,head,N,dim)
            res = ggml_cont(ggml, ggml_permute(ggml, res, 0, 2, 1, 3)); // (1,N,head,dim)

            // (N,hidden_dim)
            res = ggml_cpy(ggml, res, ggml_new_tensor_2d(ggml, GGML_TYPE_F32, hparams.hidden_dim, n)); // (N,hidden_dim)

            // output
            // **** BertSelfOutput
            res = bert_dense(ggml, res, layer.ff_w, layer.ff_b);
            x = ggml_add_inplace(ggml, x, res);
            x = bert_layer_norm(ggml, x, layer.ln_ff_w, layer.ln_ff_b, eps);
        }

        // intermediate
        {
            // *** BertIntermediate
            auto res = bert_dense(ggml, x, layer.i_w, layer.i_b);
            switch (hparams.hidden_act) {
                using enum hidden_act;
            case BERTS_HIDDEN_ACT_GELU:
                x = ggml_gelu_inplace(ggml, x);
                break;
            default:
                GGML_ASSERT(false && "unknown activation function");
                // unreachable
            }

            // *** BertOutput
            res = bert_dense(ggml, res, layer.o_w, layer.o_b);
            x = ggml_add_inplace(ggml, x, res);
            x = bert_layer_norm(ggml, x, layer.ln_out_w, layer.ln_out_b, eps);
        }
    }

    // x := (1,1,n,hidden_dim)

    //
    // pooler
    //

    switch (cond.pool_type) {
        using enum berts_pool_type;
    case BERTS_POOL_NONE:
        // return non-pooled tensor
        {
            float *data = ggml_get_data_f32(x);
            size_t count = std::min(input_out_count, needed_out_count);
            std::copy(data, data + count, out);
        }
        return true;
    case BERTS_POOL_CLS:
        // retrieve first token (hidden_dim,) of (n,hidden_dim)
        {
            auto xx = ggml_new_tensor_1d(ggml, GGML_TYPE_F32, x->ne[0]);
            float *data = ggml_get_data_f32(x);
            size_t count = std::min(input_out_count, needed_out_count);
            std::memcpy(xx->data, data, count * sizeof(float));
            x = xx;
        }
        break;
    case BERTS_POOL_AVG:
        // average pooling
        {
            x = ggml_transpose(ggml, x); // (1,1,hidden_dim,n)
            x = ggml_mean(ggml, x);      // (1,1,hidden_dim,1)
            x = ggml_transpose(ggml, x); // (1,1,1,hidden_dim)
            x = ggml_cont(ggml, x);
        }
        break;
    case BERTS_POOL_MAX:
        // max pooling
        {
            x = ggml_transpose(ggml, x); // (1,1,hidden_dim,n)
            x = ggml_argmax(ggml, x);    // (1,1,1,hidden_dim)
            x = ggml_cont(ggml, x);
        }
        break;
    default:
        // must not happen!
        log::error("unknown pooling type: {}", (int)cond.pool_type);
        return false;
    }

    GGML_ASSERT(ggml_nelements(x) == hparams.hidden_dim);

    x = bert_dense(ggml, x, this->pool_w, this->pool_b);
    x = ggml_tanh_inplace(ggml, x);

    {
        float *data = ggml_get_data_f32(x);
        size_t count = std::min(input_out_count, needed_out_count);
        std::copy(data, data + count, out);
    }

    log::info("finish evaluating BERT");

    return true;
}

} // namespace berts::bert
