#include "berts/models/bert.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <ranges>
#include "berts/models/utils.hpp"

namespace berts::bert {

using namespace berts::internal;

#define KEY_PREFIX "berts.bert."
#define KEY(s) KEY_PREFIX #s
#define KEY_N(pre, post) KEY_PREFIX #pre ".{}." #post

// vocab keys
const char *BERTS_KEY_BERT_VOCAB_SIZE = KEY(vocab_size);
const char *BERTS_KEY_BERT_VOCAB_DATA = KEY(vocab_data);

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

static inline ggml_tensor *tensor(ggml_context *ctx, const char *key) {
    auto t = ggml_get_tensor(ctx, key);
    if (!t) {
        log::error(berts::fmt("failed to read tensor: {}", key));
    }
    return t;
}

static inline ggml_tensor *tensor_n(ggml_context *ctx, const char *key, int n) {
    std::string msg = berts::fmt(key, n);
    return tensor(ctx, msg.c_str());
}

bool model::init_weight(berts_context *ctx) {
    if (!check_model(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);

#define GET_TENSOR(dest, key)         \
    do {                              \
        auto v = tensor(ggml, (key)); \
        if (!v) {                     \
            return false;             \
        }                             \
        dest = v;                     \
    } while (0)

#define GET_TENSOR_N(dest, key, n)           \
    do {                                     \
        auto v = tensor_n(ggml, (key), (n)); \
        if (!v) {                            \
            return false;                    \
        }                                    \
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

    return true;
}

bool model::load_vocab(berts_context *ctx) {
    if (!check_ctx(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);

    auto vocab_size = ggml_get_tensor(ggml, BERTS_KEY_BERT_VOCAB_SIZE);
    auto vocab_data = ggml_get_tensor(ggml, BERTS_KEY_BERT_VOCAB_DATA);

    if (!vocab_size) {
        log::error(berts::fmt("key {} is not found", BERTS_KEY_BERT_VOCAB_SIZE));
        return false;
    }

    if (!vocab_data) {
        log::error(berts::fmt("key {} is not found", BERTS_KEY_BERT_VOCAB_DATA));
        return false;
    }

    if (vocab_size->n_dims != 1 || vocab_data->n_dims != 1) {
        log::error(berts::fmt("invalid shape: vocab_size={}, vocab_data={}", vocab_size->n_dims, vocab_data->n_dims));
        return false;
    }

    if (vocab_size->type != GGML_TYPE_I32) {
        log::error(berts::fmt("invalid type of vocab_size: {}", (int)vocab_size->type));
        return false;
    }

    if (vocab_data->type != GGML_TYPE_I8) {
        log::error(berts::fmt("invalid type of vocab_data: {}", (int)vocab_data->type));
        return false;
    }
    
    const int64_t vocab_count = vocab_size->ne[0];
    auto token_lengths = static_cast<const int32_t *>(vocab_size->data);
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

    return true;
}

ggml_tensor *model::eval(berts_context *ctx,
                         const std::vector<bert_token_t> &tokens,
                         const std::vector<bert_segment_t> &segments) {
    static_assert(sizeof(bert_token_t) == sizeof(int32_t));
    static_assert(sizeof(bert_segment_t) == sizeof(int32_t));

    if (!check_model(ctx)) {
        return nullptr;
    }

    hparams hparams{};
    get_hparams(ctx, &hparams);
    auto eps = static_cast<float>(get_eps(ctx));

    ggml_ctx ggml{};

    //
    // embeddings
    //

    const auto n = tokens.size();

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
    GGML_ASSERT(x->n_dims == 2);
    GGML_ASSERT(x->ne[0] == hparams.hidden_dim);
    GGML_ASSERT((size_t)x->ne[1] == n);

    //
    // encoders
    //

    const auto n_head = hparams.attn_heads;
    const auto attn_dim = hparams.hidden_dim / n_head;
    // hidden_dim := n_head * attn_dim

    // * BertEncoder
    for (const auto &layer : this->layers) {
        // ** BertLayer

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

            // (1,head,N,dim) -> (head,N,dim)
            q = ggml_cont(ggml, ggml_reshape_3d(ggml, ggml_permute(ggml, q, 0, 2, 1, 3), attn_dim, n, n_head));
            k = ggml_cont(ggml, ggml_reshape_3d(ggml, ggml_permute(ggml, k, 0, 2, 1, 3), attn_dim, n, n_head));
            // (1,head,N,dim) -> (head,dim,N)
            v = ggml_cont(ggml, ggml_reshape_3d(ggml, ggml_permute(ggml, v, 2, 0, 1, 3), n, attn_dim, n_head));

            // sim = softmax(kq / sqrt(attn_dim))
            // (head,N,N)
            const auto scale = 1.0f / std::sqrt((float)attn_dim);
            auto sim = ggml_soft_max_ext(ggml, ggml_mul_mat(ggml, k, q), nullptr, scale);

            auto res = ggml_mul_mat(ggml, v, sim);                      // (head,N,dim)
            res = ggml_reshape_4d(ggml, res, attn_dim, n, n_head, 1);   // (1,head,N,dim)
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

    return x;
}

} // namespace berts::bert
