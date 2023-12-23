#include "berts/models/bert.hpp"

#include <array>
#include <cmath>
#include <format>
#include <ranges>
#include "berts/models/utils.hpp"

namespace berts::bert {

using namespace berts::internal;

#define KEY_PREFIX "berts.bert."
#define KEY(s) KEY_PREFIX #s
#define KEY_N(s) KEY_PREFIX "{}." #s

// vocab keys
const char *BERTS_KEY_BERT_VOCAB_SIZE = KEY(vocab_size);
const char *BERTS_KEY_BERT_VOCAB_DATA = KEY(vocab_data);

// embedding keys
const char *BERTS_KEY_BERT_EMB_TOKEN = KEY(token_embedding);
const char *BERTS_KEY_BERT_EMB_SEGM = KEY(segment_embedding);
const char *BERTS_KEY_BERT_EMB_POS = KEY(position_embedding);
const char *BERTS_KEY_BERT_LN_W = KEY(ln_w);
const char *BERTS_KEY_BERT_LN_B = KEY(ln_b);

// encoder keys
const char *BERTS_KEY_BERT_ENC_N_Q_W = KEY_N(q_w);
const char *BERTS_KEY_BERT_ENC_N_Q_B = KEY_N(q_b);
const char *BERTS_KEY_BERT_ENC_N_K_W = KEY_N(k_w);
const char *BERTS_KEY_BERT_ENC_N_K_B = KEY_N(k_b);
const char *BERTS_KEY_BERT_ENC_N_V_W = KEY_N(v_w);
const char *BERTS_KEY_BERT_ENC_N_V_B = KEY_N(v_b);
const char *BERTS_KEY_BERT_ENC_N_FF_W = KEY_N(ff_w);
const char *BERTS_KEY_BERT_ENC_N_FF_B = KEY_N(ff_b);
const char *BERTS_KEY_BERT_ENC_N_LN_FF_W = KEY_N(ln_ff_w);
const char *BERTS_KEY_BERT_ENC_N_LN_FF_B = KEY_N(ln_ff_b);
const char *BERTS_KEY_BERT_ENC_N_I_W = KEY_N(i_w);
const char *BERTS_KEY_BERT_ENC_N_I_B = KEY_N(i_b);
const char *BERTS_KEY_BERT_ENC_N_O_W = KEY_N(o_w);
const char *BERTS_KEY_BERT_ENC_N_O_B = KEY_N(o_b);
const char *BERTS_KEY_BERT_ENC_N_LN_OUT_W = KEY_N(ln_out_w);
const char *BERTS_KEY_BERT_ENC_N_LN_OUT_B = KEY_N(ln_out_b);

static inline ggml_tensor *tensor(ggml_context *ctx, const char *key) {
    auto t = ggml_get_tensor(ctx, key);
    if (!t) {
        log::error(std::format("fail to read tensor {}", key));
        GGML_ASSERT(false && "fail to read tensor");
    }
    return t;
}

static inline ggml_tensor *tensor_n(ggml_context *ctx, const char *key, int n) {
    auto name = std::vformat(key, std::make_format_args(n));
    return tensor(ctx, name.c_str());
}

bool model::init_weight(berts_context *ctx) {
    if (!check_model(ctx)) {
        return false;
    }

    auto ggml = get_ggml_context(ctx);

    this->token_embedding = tensor(ggml, BERTS_KEY_BERT_EMB_TOKEN);
    this->segment_embedding = tensor(ggml, BERTS_KEY_BERT_EMB_SEGM);
    this->position_embedding = tensor(ggml, BERTS_KEY_BERT_EMB_POS);
    this->ln_w = tensor(ggml, BERTS_KEY_BERT_LN_W);
    this->ln_b = tensor(ggml, BERTS_KEY_BERT_LN_B);

    hparams hparams;
    get_hparams(ctx, &hparams);

    this->layers.resize(hparams.n_layers);

    for (const auto [n, layer] : this->layers | std::views::enumerate) {
        layer.q_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_Q_W, n);
        layer.q_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_Q_B, n);
        layer.k_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_K_W, n);
        layer.k_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_K_B, n);
        layer.v_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_V_W, n);
        layer.v_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_V_B, n);
        layer.ff_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_FF_W, n);
        layer.ff_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_FF_B, n);
        layer.ln_ff_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_LN_FF_W, n);
        layer.ln_ff_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_LN_FF_B, n);
        layer.i_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_I_W, n);
        layer.i_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_I_B, n);
        layer.o_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_O_W, n);
        layer.o_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_O_B, n);
        layer.ln_out_w = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_LN_OUT_W, n);
        layer.ln_out_b = tensor_n(ggml, BERTS_KEY_BERT_ENC_N_LN_OUT_B, n);
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

    GGML_ASSERT(vocab_size && "key vocab_size is not found");
    GGML_ASSERT(vocab_data && "key vocab_data is not found");
    GGML_ASSERT(vocab_size->n_dims == 1);
    GGML_ASSERT(vocab_data->n_dims == 1);
    GGML_ASSERT(vocab_size->ne[0] == vocab_data->ne[0]);
    GGML_ASSERT(vocab_size->type == GGML_TYPE_I32);
    GGML_ASSERT(vocab_data->type == GGML_TYPE_I8);

    const int64_t vocab_count = vocab_size->ne[0];
    auto token_lengths = static_cast<const int32_t *>(vocab_size->data);
    auto data = static_cast<const char *>(vocab_data->data);
    for (int64_t token_id = 0; token_id < vocab_count; ++token_id) {
        size_t token_len = (size_t)token_lengths[token_id];
        std::string token{&data[0], token_len};
        data += token_len;

        internal::add_token(ctx, token);
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
    memcpy(token_emb->data, tokens.data(), n * ggml_element_size(token_emb));

    auto seg_emb = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, n);
    memcpy(seg_emb->data, segments.data(), n * ggml_element_size(seg_emb));

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
