#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "berts/models/ggml.hpp"
#include "berts/models/gguf.hpp"
#include "berts/models/internal.hpp"
#include "berts/models/keys.h"
#include "berts/models/log.hpp"
#include "berts/models/utils.hpp"

namespace berts::internal {

template <typename T>
concept Vocab = requires(T &obj, berts_context *ctx, ggml_context *ggml, gguf_context *gguf, const std::string &str) {
    typename T::self_type;

    new T{};

    { obj.cls_id() } -> std::convertible_to<bert_token_t>;
    { obj.mask_id() } -> std::convertible_to<bert_token_t>;
    { obj.pad_id() } -> std::convertible_to<bert_token_t>;
    { obj.sep_id() } -> std::convertible_to<bert_token_t>;
    { obj.unk_id() } -> std::convertible_to<bert_token_t>;
    { obj.bos_id() } -> std::convertible_to<bert_token_t>;
    { obj.eos_id() } -> std::convertible_to<bert_token_t>;

    { obj.id_to_token(bert_token_t{}) } -> std::convertible_to<std::string>;
    { obj.token_to_id(str) } -> std::convertible_to<bert_token_t>;

    { obj.add_token(str) } -> std::convertible_to<bool>;
    { obj.has_token(str) } -> std::convertible_to<bool>;

    // called from class `base` after tokens have been added
    { obj.init(ctx, ggml, gguf) } -> std::convertible_to<bool>;
    obj.clear();

    // implemented in vocab_base
    { obj.cls_token() } -> std::convertible_to<std::string>;
    { obj.mask_token() } -> std::convertible_to<std::string>;
    { obj.pad_token() } -> std::convertible_to<std::string>;
    { obj.sep_token() } -> std::convertible_to<std::string>;
    { obj.unk_token() } -> std::convertible_to<std::string>;
    { obj.bos_token() } -> std::convertible_to<std::string>;
    { obj.eos_token() } -> std::convertible_to<std::string>;
};

template <typename T>
concept Weights = requires(T &obj, berts_context *ctx, ggml_context *ggml, gguf_context *gguf) {
    typename T::self_type;

    new T{};

    { obj.init(ctx, ggml, gguf) } -> std::convertible_to<bool>;
};

template <typename Self>
struct vocab_base {
    using inherited = vocab_base<Self>;

    bert_token_t get_token_id(const gguf_context *gguf, const char *key, const char *alternate1, const char *alternate2 = nullptr) {
        const char *failed_key = nullptr;

        auto id = gguf::gguf_u32(gguf, key, BERTS_INVALID_TOKEN_ID);
        if (id == BERTS_INVALID_TOKEN_ID) {
            if (!alternate1) {
                failed_key = key;
                goto FAIL;
            }

            log::warn("{} is not defined; use {} instead", key, alternate1);
            id = as_self()->token_to_id(alternate1);

            if (id == BERTS_INVALID_TOKEN_ID) {
                if (!alternate2) {
                    failed_key = alternate1;
                    goto FAIL;
                }

                log::warn("{} is not defined; use {} instead", alternate1, alternate2);
                id = as_self()->token_to_id(alternate2);

                if (id == BERTS_INVALID_TOKEN_ID) {
                    failed_key = alternate2;
                    goto FAIL;
                }
            }
        }

        return id;

    FAIL:
        if (failed_key) {
            log::error("{} does not exist in vocab", failed_key);
        }
        return BERTS_INVALID_TOKEN_ID;
    }

    std::string cls_token() const noexcept { return as_self()->id_to_token(as_self()->cls_id()); };
    std::string mask_token() const noexcept { return as_self()->id_to_token(as_self()->mask_id()); };
    std::string pad_token() const noexcept { return as_self()->id_to_token(as_self()->pad_id()); };
    std::string sep_token() const noexcept { return as_self()->id_to_token(as_self()->sep_id()); };
    std::string unk_token() const noexcept { return as_self()->id_to_token(as_self()->unk_id()); };
    std::string bos_token() const noexcept { return as_self()->id_to_token(as_self()->bos_id()); };
    std::string eos_token() const noexcept { return as_self()->id_to_token(as_self()->eos_id()); };

  protected:
    auto as_self() noexcept { return static_cast<typename Self::self_type *>(this); }
    const auto as_self() const noexcept { return static_cast<const typename Self::self_type *>(this); }
};

template <typename Self>
struct vocab_base2 : public vocab_base<vocab_base2<Self>> {
    using self_type = Self;
    using inherited = vocab_base2<Self>;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, bert_token_t> token_to_id_;

    vocab_base2() = default;

    std::string id_to_token(bert_token_t token_id) const noexcept {
        if (id_to_token_.size() <= token_id) {
            log::error("token id {} is not found (max={})", token_id, id_to_token_.size());
            return "";
        }
        return id_to_token_[token_id];
    }

    bert_token_t token_to_id(const std::string &token) const noexcept {
        const auto p = token_to_id_.find(token);
        if (p == token_to_id_.end()) {
            log::error("token {} is not found", token);
            return BERTS_INVALID_TOKEN_ID;
        }
        return p->second;
    }

    bool add_token(const std::string &token) {
        if (has_token(token)) {
            log::warn("  token {} already exists", token);
            return false;
        }

        const auto next_id = static_cast<bert_token_t>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_[token] = next_id;
        // log::debug("  token {}: {}", next_id, token);
        return true;
    }

    bool has_token(const std::string &token) const noexcept {
        const auto p = token_to_id_.find(token);
        return p != token_to_id_.end();
    }

    void clear() {
        id_to_token_.clear();
        token_to_id_.clear();
    }
};

template <Vocab VocabType, Weights WeightsType>
struct base : public model {
    using vocab_t = VocabType;
    using weights_t = WeightsType;
    using inherited = base<vocab_t, weights_t>;

    // weights
    weights_t weights;

    // tokenizer
    std::unique_ptr<vocab_t> vocab;

    base(ggml_type type)
        : model(type)
        , weights()
        , vocab(new vocab_t{}) {}

    ~base() override = default;

    bool init_vocab(berts_context *ctx) override {
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
            if (token_len == 0) {
                token_len = 256;
            }
            std::string token{&data[p], token_len};
            p += token_len;

            if (!add_token(token)) {
                log::error("failed to add token: {}", token);
                vocab->clear();
                return false;
            }
        }

        if (p != vocab_data->ne[0]) {
            log::error("something wrong");
            vocab->clear();
            return false;
        }

        if (!vocab->init(ctx, ggml, gguf)) {
            log::error("fail to build vocab");
            vocab->clear();
            return false;
        }

        log::info("finish loading vocab");
        return true;
    }

    bool init_weight(berts_context *ctx) override {
        log::info("initializing weights");

        if (!check_ctx(ctx)) {
            return false;
        }

        auto ggml = get_ggml_context(ctx);
        auto gguf = get_gguf_context(ctx);

        if (!weights.init(ctx, ggml, gguf)) {
            return false;
        }

        log::info("finish loading vocab");
        return true;
    }

    // delegate to vocab
    std::string id_to_token(bert_token_t token_id) const noexcept override {
        return vocab->id_to_token(token_id);
    }

    bert_token_t token_to_id(const std::string &token) const noexcept override {
        return vocab->token_to_id(token);
    }

    bool add_token(const std::string &token) override {
        return vocab->add_token(token);
    }

    bool has_token(const std::string &token) const noexcept override {
        return vocab->has_token(token);
    }

    bert_token_t cls_id() const noexcept override {
        return vocab->cls_id();
    }

    bert_token_t mask_id() const noexcept override {
        return vocab->mask_id();
    }

    bert_token_t pad_id() const noexcept override {
        return vocab->pad_id();
    }

    bert_token_t sep_id() const noexcept override {
        return vocab->sep_id();
    }

    bert_token_t unk_id() const noexcept override {
        return vocab->unk_id();
    }

    bert_token_t bos_id() const noexcept override {
        return vocab->bos_id();
    }

    bert_token_t eos_id() const noexcept override {
        return vocab->eos_id();
    }

    virtual std::string model_name() const = 0;

    virtual bool tokenize(const berts_context *ctx,
                          const std::string &text,
                          std::vector<bert_token_t> &out) const override = 0;

    virtual ggml_size_info get_context_buffer_size(
        size_t token_count,
        const hparams &hparams,
        const berts_eval_info &cond) const = 0;

    bool eval(berts_context *ctx,
              const std::vector<bert_token_t> &tokens,
              const std::vector<bert_segment_t> &segments,
              const berts_eval_info &cond,
              float *out,
              size_t &out_count) const override {
        log::info("start evaluating {}", model_name());

        if (!check_model(ctx)) {
            return false;
        }

        //
        // check inputs
        //

        hparams hparams{};
        get_hparams(ctx, &hparams);

        const auto n = tokens.size();

        log::debug("  #tokens = {}", n);

        if ((size_t)hparams.max_tokens < n) {
            log::error("too many tokens ({}) for this model ({})", n, hparams.max_tokens);
            return false;
        }

        if (n != segments.size()) {
            log::error("segment count ({}) is not match for tokens ({})", segments.size(), n);
            return false;
        }

        for (const auto segm : segments) {
            if (hparams.segment_count <= (bert_int)segm) {
                log::error("invalid segment value: {} (allowed = 0..{})", segm, hparams.segment_count - 1);
                return false;
            }
        }

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

        log::when(BERTS_LOG_INFO, [&]() {
            log::info(
                "  berts_eval_info {{\n"
                "    output_layer = {};\n"
                "    pool_type = {};\n"
                "    n_threads = {}\n"
                "  }}",
                cond.output_layer,
                pool_type_str(cond.pool_type),
                cond.n_threads);
            log::debug("  output size = {}", needed_out_count);
            log::debug("    given     = {}", input_out_count);
        });

        out_count = needed_out_count;

        if (!out) {
            log::info("finish evaluating BERT (dry run)");
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
        berts_eval_info new_cond{cond};
        new_cond.output_layer = last_layer_index;

        //
        // build graph and run the computation
        //

        ggml_size_info size = get_context_buffer_size(n, hparams, new_cond);
        ggml_init_params init{
            /* .mem_size   = */ size.calc(last_layer_index),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ggml_ctx ggml{init};

        log::debug("  context buffer size = {}", init.mem_size);

        if (!build_graph(ggml, hparams, new_cond, tokens, segments)) {
            return false;
        }

        // run the computation
        ggml_cgraph *gf = ggml_new_graph(ggml); // allocated in ggml_context
        ggml_tensor *x = ggml_get_tensor(ggml, "out");
        if (!x) {
            log::error("output tensor is not found");
            return false;
        }
        ggml_build_forward_expand(gf, x);
        ggml_cplan cplan = ggml_graph_plan(gf, new_cond.n_threads);

        std::unique_ptr<uint8_t[]> work_data{};
        if (cplan.work_size != 0) {
            work_data.reset(new uint8_t[cplan.work_size]);
            cplan.work_data = work_data.get();
        }

        ggml_graph_compute(gf, &cplan);

// #ifdef BERTS_DEBUG
//     cc.check(size.calc(last_layer_index), "run");
// #endif
//
#ifdef GGML_PERF
        log::when(BERTS_LOG_DEBUG, [=]() {
            ggml_graph_print(gf);
        });
#endif

        //
        // output
        //

        {
            float *data = ggml_get_data_f32(x);
            size_t count = std::min(input_out_count, needed_out_count);
            std::copy(data, data + count, out);
        }

        log::info("finish evaluating {}", model_name());

        return true;
    }

    virtual bool build_graph(ggml_ctx &ctx,
                             const hparams &hparams,
                             const berts_eval_info &cond,
                             const std::vector<bert_token_t> &tokens,
                             const std::vector<bert_segment_t> &segments) const = 0;
};

} // namespace berts::bert
