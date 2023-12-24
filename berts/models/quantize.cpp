#include <array>
#include <fstream>
#include <functional>
#include <numeric>
#include <ranges>
#include <regex>
#include "berts/berts.h"
#include "berts/models/gguf.hpp"
#include "berts/models/internal.hpp"
#include "berts/models/log.hpp"
#include "berts/models/utils.hpp"

using namespace berts;

struct conv_buf {
    std::vector<float> f16_to_f32;
    std::vector<uint8_t> f32_to_q;
    std::vector<int64_t> hist;
    conv_buf() {
        hist.resize(16);
    }
    void reserve(size_t nelem) {
        if (f16_to_f32.size() < nelem) {
            f16_to_f32.resize(nelem);
        }
        if (f32_to_q.size() < nelem * sizeof(float) /* for safety */) {
            f32_to_q.resize(nelem * sizeof(float));
        }
    }
    float *f() { return f16_to_f32.data(); }
    void *q() { return f32_to_q.data(); }
    int64_t *h() { return hist.data(); }
};

static inline void write_zeros(std::ostream &out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out.put(0);
    }
}

static inline bool quantize(const ggml_tensor *t, ggml_type new_type, conv_buf &buffer, size_t *new_size) {
    const size_t n = ggml_nelements(t);
    buffer.reserve(n);

    // cast values to fp32
    float *data;
    switch (t->type) {
        using enum ggml_type;
    case GGML_TYPE_F32:
        data = static_cast<float *>(t->data);
        break;
    case GGML_TYPE_F16:
        data = buffer.f();
        std::transform((ggml_fp16_t *)t->data, (ggml_fp16_t *)t->data + n, data, ggml_fp16_to_fp32);
        break;
    default:
        log::error(berts::fmt("input type must be f16 or f32, but {}", gguf::type_to_str(t->type)));
        return false;
    }

    // cast values to new_type
    size_t new_size_ = ggml_quantize_chunk(new_type, data, buffer.q(), 0, n, buffer.h());
    if (new_size) {
        *new_size = new_size_;
    }

    return true;
}

bool berts_model_quantize(const char *input_path,
                          const char *output_path,
                          ggml_type qtype) {
    log::info("start quantization");

    const auto type_str = gguf::type_to_str(qtype);

    berts_ctx ctx{berts_load_from_file(input_path)};
    if (!ctx) {
        log::error(berts::fmt("fail to load model: {}", input_path));
        return false;
    }

    log::info(berts::fmt("model loaded: {}", input_path));

    auto gguf_src = internal::get_gguf_context(ctx);
    auto ggml_src = internal::get_ggml_context(ctx);

    gguf_ctx gguf_dst{gguf_init_empty()};

    gguf_set_kv(gguf_dst, gguf_src);
    gguf_set_val_u32(gguf_dst, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(gguf_dst, "general.file_type", (uint32_t)qtype);

    std::ofstream out{output_path, std::ios::binary};
    if (!out) {
        log::error(berts::fmt("failed to open {}", output_path));
        return false;
    }

    const int n = gguf_get_n_tensors(gguf_src);
    for (int i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(gguf_src, i);
        auto t = ggml_get_tensor(ggml_src, name);
        gguf_add_tensor(gguf_dst, t);
    }

    const size_t meta_size = gguf_get_meta_size(gguf_dst);
    write_zeros(out, meta_size);

    const std::array<std::regex, 1> quantize_names{{
        std::regex{".*weight"},
    }};

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    conv_buf buffer{};

    //
    // write tensor data
    //
    log::info("converting...");
    for (int i = 0; i < n; ++i) {
        const char *name = gguf_get_tensor_name(gguf_src, i);
        auto t = ggml_get_tensor(ggml_src, name);

        bool q = false;
        for (const auto &exp : quantize_names) {
            if (std::regex_match(name, exp)) {
                q = true;
                break;
            }
        }

        // quantize only 2D tensors
        if (t->n_dims != 2) {
            q = false;
        }

        ggml_type new_type = t->type;
        void *data = t->data;
        size_t size_in_bytes = ggml_nbytes(t);

        const size_t size_org = size_in_bytes;
        total_size_org += size_org;

        if (q) {
            // do quantize
            if (!quantize(t, qtype, buffer, &size_in_bytes)) {
                return false;
            }
            new_type = qtype;
            data = buffer.q();
        }

        const size_t size_new = size_in_bytes;
        total_size_new += size_new;

        gguf_set_tensor_type(gguf_dst, name, new_type);
        gguf_set_tensor_data(gguf_dst, name, data, size_in_bytes);

        size_t pad = GGML_PAD(size_in_bytes, gguf_get_alignment(gguf_dst)) - size_in_bytes;

        out.write((const char *)data, size_in_bytes);
        write_zeros(out, pad);

        log::when(log::log_level::info, [=]() {
            const std::string msg = berts::fmt(
                "{}:\n"
                "  quantized = {}\n"
                "  n_dims = {}\n"
                "  size = {:.1f} KiB -> {:.1f} KiB",
                name,
                q,
                t->n_dims,
                size_org / 1024.0f,
                size_new / 1024.0f);
            log::info(msg);
        });
    }

    //
    // write metadata
    //
    log::info("writing metadata...");

    std::vector<char> meta(meta_size);
    gguf_get_meta_data(gguf_dst, meta.data());
    out.seekp(0, std::ios::beg);
    out.write(meta.data(), meta_size);

    out.close();

    //
    // dump info
    //
    log::when(log::log_level::info, [=, &buffer]() {
        std::string msg = berts::fmt(
            "========================================\n"
            "original size  = {} ({:.1f} MiB)\n"
            "quantized size = {} ({:.1f} MiB)\n",
            total_size_org,
            total_size_org / 1024.0f / 1024.0f,
            total_size_new,
            total_size_new / 1024.0f / 1024.0f);

        const auto hist_sum = (float)std::reduce(buffer.hist.begin(), buffer.hist.end(), (int64_t)0, std::plus<>());
        if (0 < hist_sum) {
            msg += "[histogram]\n";
            for (const auto [i, v] : buffer.hist | std::views::enumerate) {
                float vv = v / hist_sum;
                msg += berts::fmt("  bin #{}: {:.3f}\n", i, vv);
            }
        }
        msg += "========================================";
        log::info(msg);
    });

    log::info("quantization completed");

    return true;
}
