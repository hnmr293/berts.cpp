#pragma once

#include "berts/models/model_base.hpp"

namespace berts::internal {

template <Vocab VocabType, Weights WeightsType>
struct model_berts : public model_base<VocabType, WeightsType> {
    using inherited = model_base<VocabType, WeightsType>;
    using inherited::inherited;

    ~model_berts() override = default;
    
    // name of model such as "BERT", "RoBERTa" and so on
    virtual std::string model_name() const = 0;

    virtual bool tokenize(const berts_context *ctx,
                          const std::string &text,
                          std::vector<bert_token_t> &out) const override = 0;

    // compute ggml_context allocation memory size
    virtual ggml_size_info get_context_buffer_size(
        size_t token_count,
        const hparams &hparams,
        const berts_eval_info &cond) const = 0;

    // process forward for ggml_new_graph
    // after calling this function,
    // parameter `ctx` must have the tensor named "out"
    virtual bool build_graph(ggml_ctx &ctx,
                             const hparams &hparams,
                             const berts_eval_info &cond,
                             const std::vector<bert_token_t> &tokens,
                             const std::vector<bert_segment_t> &segments) const = 0;

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
};

}
