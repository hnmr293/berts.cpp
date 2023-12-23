#include "berts/models/gguf.hpp"

#include <array>
#include <format>
#include <fstream>
#include "berts/berts.h"
#include "berts/models/bert.hpp"
#include "berts/models/internal.hpp"
#include "berts/models/log.hpp"
#include "berts/models/utils.hpp"

namespace internal = berts::internal;

namespace berts::gguf {

//
// getter
//

static inline size_t safe_index(const struct gguf_context *ctx, const std::string &key) {
    auto idx = gguf_find_key(ctx, key.c_str());
    if (idx < 0) {
        auto msg = std::format("key {0} is not found in gguf", key);
        log::error(msg);
        GGML_ASSERT(false && "key is not found in gguf");
    }

    return (size_t)idx;
}

static inline int index(const struct gguf_context *ctx, const std::string &key) {
    return gguf_find_key(ctx, key.c_str());
}

#define DEFINE_GGUF_VALUE_GET(gguf_type, c_type)                                                    \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const std::string &key) { \
        return gguf_get_val_##gguf_type(ctx, safe_index(ctx, key));                                 \
    }

#define DEFINE_GGUF_VALUE_GET_OR(gguf_type, c_type)                                                                    \
    template <typename T>                                                                                              \
    static inline c_type gguf_##gguf_type(const struct gguf_context *ctx, const std::string &key, const T &default_) { \
        auto idx = index(ctx, key);                                                                                    \
        return idx < 0 ? default_ : gguf_get_val_##gguf_type(ctx, idx);                                                \
    }

#define DEFINE_GGUF_VALUE(gguf_type, c_type) \
    DEFINE_GGUF_VALUE_GET(gguf_type, c_type) \
    DEFINE_GGUF_VALUE_GET_OR(gguf_type, c_type)

DEFINE_GGUF_VALUE(u8, uint8_t)
DEFINE_GGUF_VALUE(i8, int8_t)
DEFINE_GGUF_VALUE(u16, uint16_t)
DEFINE_GGUF_VALUE(i16, int16_t)
DEFINE_GGUF_VALUE(u32, uint32_t)
DEFINE_GGUF_VALUE(i32, int32_t)
DEFINE_GGUF_VALUE(f32, float)
DEFINE_GGUF_VALUE(u64, uint64_t)
DEFINE_GGUF_VALUE(i64, int64_t)
DEFINE_GGUF_VALUE(f64, double)
DEFINE_GGUF_VALUE(bool, bool)
DEFINE_GGUF_VALUE(str, const char *)
DEFINE_GGUF_VALUE(data, const void *)

static inline std::string ftype(uint32_t ftype) {
    // gguf.md
    static std::array<const char *, 19> ftypes{{
        "ALL_F32",
        "MOSTLY_F16",
        "MOSTLY_Q4_0",
        "MOSTLY_Q4_1",
        "MOSTLY_Q4_1_SOME_F16",
        "MOSTLY_Q4_2",
        "MOSTLY_Q4_3",
        "MOSTLY_Q8_0",
        "MOSTLY_Q5_0",
        "MOSTLY_Q5_1",
        "MOSTLY_Q2_K",
        "MOSTLY_Q3_K_S",
        "MOSTLY_Q3_K_M",
        "MOSTLY_Q3_K_L",
        "MOSTLY_Q4_K_S",
        "MOSTLY_Q4_K_M",
        "MOSTLY_Q5_K_S",
        "MOSTLY_Q5_K_M",
        "MOSTLY_Q6_K",
    }};

    if (ftypes.size() <= ftype) {
        log::error(std::format("unrecognized file type: {0}", ftype));
        GGML_ASSERT(false && "unrecognized file type");
    }

    return ftypes[ftype];
}

//
// keys
//

const char *BERTS_KEY_HPARAM_BERT_TYPE = "berts.bert_type";
const char *BERTS_KEY_HPARAM_VOCAB_SIZE = "berts.vocab_size";
const char *BERTS_KEY_HPARAM_HIDDEN_DIM = "berts.hidden_dim";
const char *BERTS_KEY_HPARAM_N_LAYERS = "berts.n_layers";
const char *BERTS_KEY_HPARAM_ATTN_HEADS = "berts.attn_heads";
const char *BERTS_KEY_HPARAM_MAX_TOKENS = "berts.max_token";
const char *BERTS_KEY_HPARAM_INTERMEDIATE_DIM = "berts.intermediate_dim";
const char *BERTS_KEY_HPARAM_HIDDEN_ACT = "berts.hidden_act";

//
// gguf loader
//

static gg_ctx init_gg(const std::string &path, size_t *ctx_size) {
    gg_ctx gg{path, true};
    gguf_context *gguf = gg.gguf;
    ggml_context *ggml_meta = gg.ggml;

    if (!gg || !gguf || !ggml_meta) {
        return gg;
    }

    // dump gguf file information
    {
        // general metadata
        const auto arch = gguf_str(gguf, "general.architecture", "");
        const auto quant_version = gguf_u32(gguf, "general.quantization_version", (uint32_t)-1);
        const auto align = gguf_u32(gguf, "general.alignment", (uint32_t)-1);
        const auto name = gguf_str(gguf, "general.name", "");
        const auto author = gguf_str(gguf, "general.author", "");
        const auto url = gguf_str(gguf, "general.url", "");
        const auto desc = gguf_str(gguf, "general.description", "");
        const auto license = gguf_str(gguf, "general.license", "");
        const auto type = ftype(gguf_u32(gguf, "general.file_type"));
        log::info(std::format(
            "model metadata\n"
            "  arch: {0}\n"
            "  quantization_version: {1}\n"
            "  alignment: {2}\n"
            "  name: {3}\n"
            "  author: {4}\n"
            "  url: {5}\n"
            "  description: {6}\n"
            "  license: {7}\n"
            "  type: {8}",
            arch,
            quant_version,
            align,
            name,
            author,
            url,
            desc,
            license,
            type));

        // gguf info
        const auto n_tensors = gguf_get_n_tensors(gguf);
        const auto n_kv = gguf_get_n_kv(gguf);
        log::info(std::format(
            "gguf info\n"
            "  n_tensors: {}\n"
            "  n_kv: {}",
            n_tensors,
            n_kv));

        log::when(log::log_level::debug, [n_kv, gguf]() {
            for (int i = 0; i < n_kv; ++i) {
                auto key = gguf_get_key(gguf, i);
                const auto msg = std::format("  key {0}: {1}{2}", i, key, i == n_kv - 1 ? "" : "\n");
                log::debug(msg);
            }
        });
    }

    size_t ctx_size_ = 0;

    // retrieve model size and dump tensors' information
    {
        const auto n_tensors = gguf_get_n_tensors(gguf);
        for (int i = 0; i < n_tensors; ++i) {
            const char *tensor_name = gguf_get_tensor_name(gguf, i);
            const size_t tensor_offset = gguf_get_tensor_offset(gguf, i);

            struct ggml_tensor *t = ggml_get_tensor(ggml_meta, tensor_name);
            size_t tensor_size = ggml_nbytes(t);
            size_t padded_size = ggml_nbytes_pad(t);
            ctx_size_ += sizeof(struct ggml_tensor) + padded_size + GGML_OBJECT_SIZE;

            log::when(log::log_level::debug, [=]() {
                const auto msg = std::format(
                    "  tensor {}\n"
                    "    name: {} ({})\n"
                    "    n_dims: {}\n"
                    "    size: {}\n"
                    "    padded_size: {}\n"
                    "    offset: {}",
                    i,
                    t->name,
                    tensor_name,
                    t->n_dims,
                    tensor_size,
                    padded_size,
                    tensor_offset);
            });
        }
    }

    log::info(std::format("  model_size: {} ({} MiB)", ctx_size_, ctx_size_ / 1024 / 1024));

    if (ctx_size) {
        *ctx_size = ctx_size_;
    }

    return gg;
}

berts_context *load_from_file(const std::string &path) {
    log::info(std::format("loading model: {}", path));

    size_t ctx_size;
    gg_ctx gg = init_gg(path, &ctx_size);

    gguf_context *gguf = gg.gguf;
    ggml_context *ggml_meta = gg.ggml;

    if (!gg || !gguf || !ggml_meta) {
        log::error(std::format("fail to load gguf file: {}", path));
        return nullptr;
    }

    ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    ggml_ctx ggml{params};
    GGML_ASSERT(ggml && "fail to init ggml");

    std::ifstream in{path, std::ios::binary};
    GGML_ASSERT(in && "fail to open gguf file");

    // load tensors
    {
        const auto n_tensors = gguf_get_n_tensors(gguf);
        for (int i = 0; i < n_tensors; ++i) {
            const auto tensor_name = gguf_get_tensor_name(gguf, i);
            auto t = ggml_get_tensor(ggml_meta, tensor_name);
            auto x = ggml_dup_tensor(ggml, t);
            ggml_set_name(x, tensor_name);

            const auto offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
            in.seekg(offset, std::ios::beg);
            GGML_ASSERT(in && "failed to seek gguf file");
            in.read((char *)x->data, ggml_nbytes(t));
        }
    }

    internal::hparams hparams{};
    hparams.architecture = static_cast<bert_type>(gguf_u32(gguf, BERTS_KEY_HPARAM_BERT_TYPE));
    hparams.vocab_size = gguf_u32(gguf, BERTS_KEY_HPARAM_VOCAB_SIZE);
    hparams.hidden_dim = gguf_u32(gguf, BERTS_KEY_HPARAM_HIDDEN_DIM);
    hparams.n_layers = gguf_u32(gguf, BERTS_KEY_HPARAM_N_LAYERS);
    hparams.attn_heads = gguf_u32(gguf, BERTS_KEY_HPARAM_ATTN_HEADS);
    hparams.max_tokens = gguf_u32(gguf, BERTS_KEY_HPARAM_MAX_TOKENS);
    hparams.intermediate_dim = gguf_u32(gguf, BERTS_KEY_HPARAM_INTERMEDIATE_DIM);
    hparams.hidden_act = static_cast<hidden_act>(gguf_u32(gguf, BERTS_KEY_HPARAM_HIDDEN_ACT));

    const auto type = static_cast<ggml_type>(gguf_u32(gguf, "general.file_type"));

    // check type
    ftype(type);

    // check act
    switch (hparams.hidden_act) {
        using enum hidden_act;
    case BERTS_HIDDEN_ACT_GELU:
        // ok
        break;
    default:
        GGML_ASSERT(false && "unknown hidden_act");
    }

    internal::model *model;

    // create model
    switch (hparams.architecture) {
        using enum bert_type;
    case BERTS_TYPE_BERT:
        // ok
        model = new bert::model(type);
        break;
    default:
        GGML_ASSERT(false && "unknown bert_type");
    }

    auto ctx = internal::new_context(hparams, model, ggml);
    ggml.release();

    return ctx;
}

#if 0
struct memory_stream_buffer : std::streambuf {
    memory_stream_buffer(const void *ptr, size_t size) {
        char *p = (char *)ptr;
        this->setg(p, p, p + size);
    }
};

struct imstream : virtual memory_stream_buffer, std::istream {
    imstream(const void *ptr, size_t size)
        : memory_stream_buffer(ptr, size)
        , std::istream(static_cast<std::streambuf *>(this)) {}
};

berts_context *load_from_memory(const uint8_t *data, size_t data_len) {
    imstream in{data, data_len};
    log::info("loading model from buffer");
    return load_from_stream(in);
}

berts_context *load_from_stream(std::istream &stream) {
    //
    // check magic
    //
    {
        char magic[4] = {0};
        stream.read(magic, 4);
        if (std::strncmp(magic, "ggml", 4) != 0) {
            log::error("invalid format (invalid magic number)");
            return false;
        }
    }

    //
    // hparams
    //

#define READ(buf)                                  \
    do {                                           \
        stream.read((char *)(&buf), sizeof(buf));  \
        if (!stream) {                             \
            log::error("fail to read model file"); \
            return false;                          \
        }                                          \
    } while (0)

    READ(ctx->hparams.architecture);
    READ(ctx->hparams.vocab_size);
    READ(ctx->hparams.hidden_dim);
    READ(ctx->hparams.n_layers);
    READ(ctx->hparams.attn_heads);
    READ(ctx->hparams.max_tokens);
    READ(ctx->hparams.intermediate_dim);
    READ(ctx->hparams.hidden_act);
    READ(ctx->hparams.eps);
    ggml_type type;
    READ(type);

    log::info(std::format(
        "hparams:\n"
        "  vocab_size: {0}\n"
        "  hidden_dim: {1}\n"
        "  n_layers: {2}\n"
        "  attn_heads: {3}\n"
        "  max_tokens: {4}\n"
        "  intermediate_dim: {5}\n"
        "  hidden_act: {6}",
        ctx->hparams.vocab_size,
        ctx->hparams.hidden_dim,
        ctx->hparams.n_layers,
        ctx->hparams.attn_heads,
        ctx->hparams.max_tokens,
        ctx->hparams.intermediate_dim,
        ctx->hparams.hidden_act));

    GGML_ASSERT(!ctx->model);

    {
        std::string type_;
        switch (type) {
        case GGML_TYPE_F32:
            type_ = "GGML_TYPE_F32";
            break;
        case GGML_TYPE_F16:
            type_ = "GGML_TYPE_F16";
            break;
        case GGML_TYPE_Q4_0:
            type_ = "GGML_TYPE_Q4_0";
            break;
        case GGML_TYPE_Q4_1:
            type_ = "GGML_TYPE_Q4_1";
            break;
        case GGML_TYPE_Q5_0:
            type_ = "GGML_TYPE_Q5_0";
            break;
        case GGML_TYPE_Q5_1:
            type_ = "GGML_TYPE_Q5_1";
            break;
        case GGML_TYPE_Q8_0:
            type_ = "GGML_TYPE_Q8_0";
            break;
        case GGML_TYPE_Q8_1:
            type_ = "GGML_TYPE_Q8_1";
            break;
        case GGML_TYPE_Q2_K:
            type_ = "GGML_TYPE_Q2_K";
            break;
        case GGML_TYPE_Q3_K:
            type_ = "GGML_TYPE_Q3_K";
            break;
        case GGML_TYPE_Q4_K:
            type_ = "GGML_TYPE_Q4_K";
            break;
        case GGML_TYPE_Q5_K:
            type_ = "GGML_TYPE_Q5_K";
            break;
        case GGML_TYPE_Q6_K:
            type_ = "GGML_TYPE_Q6_K";
            break;
        case GGML_TYPE_Q8_K:
            type_ = "GGML_TYPE_Q8_K";
            break;
        default:
            log::error(std::format("unknown model type: {0}", type));
            return false;
        }
        log::info(std::format("model type: {0}", ctx->model->type));
    }

    //
    // vocab
    //

    ctx->vocab.id_to_token.clear();
    ctx->vocab.token_to_id.clear();

    int32_t token_count;
    READ(token_count);

    std::vector<uint8_t> token_sizes;
    token_sizes.reserve(token_count);
    ctx->vocab.id_to_token.reserve(token_count);
    ctx->vocab.token_to_id.reserve(token_count);

    for (int32_t i = 0; i < token_count; ++i) {
        uint8_t token_size;
        READ(token_size);
        token_sizes.push_back(token_size);
    }

    char token_buf[256];
    for (int32_t i = 0; i < token_count; ++i) {
        auto len = token_sizes[i];
        stream.read(token_buf, len);
        if (!stream) {
            log::error("fail to read model file");
            return false;
        }
        const std::string token = std::string{token_buf, len};
        ctx->vocab.id_to_token.push_back(token);
        ctx->vocab.token_to_id[token] = i;
    }

    //
    // weights
    //

    size_t model_mem_req = 0;

    {
        const auto vocab_size = ctx->hparams.vocab_size;
        const auto hidden_dim = ctx->hparams.hidden_dim;
        const auto n_layers = ctx->hparams.n_layers;
        const auto max_tokens = ctx->hparams.max_tokens;
        auto elem_size = ggml_type_sizef(type);

        model_mem_req += vocab_size * hidden_dim * elem_size;    // token embedding
        model_mem_req += 2 /* [0,1] */ * hidden_dim * elem_size; // segment embedding
        model_mem_req += max_tokens * hidden_dim * elem_size;    // position embedding

        // layer norm
        model_mem_req += 2 /* weight, bias */ * hidden_dim * elem_size;
    }

    ctx->model.reset(new model{type});

    // embeddings

    return true;
}
#endif

} // namespace berts::gguf