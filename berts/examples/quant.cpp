#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstring>
#include <iostream>
#include <string>
#include "berts/berts.h"

struct quant_option {
    ggml_type ftype;
    std::string fname;
    double bits_per_weight;
    std::string description;
};

std::array<quant_option, 13> quant_types{{
    {GGML_TYPE_F32, "F32", 32, "32-bit float"},
    {GGML_TYPE_F16, "F16", 16, "16-bit float"},
    {GGML_TYPE_Q4_0, "Q4_0", (double)(4 * 32 + 32) / 32, "4-bit x 32 + 32-bit scale"},
    {GGML_TYPE_Q4_1, "Q4_1", (double)(4 * 32 + 32 + 32) / 32, "4-bit x 32 + 32-bit scale + 32-bit bias"},
    {GGML_TYPE_Q5_0, "Q5_0", (double)(5 * 32 + 16) / 32, "5-bit x 32 + 16-bit scale"},
    {GGML_TYPE_Q5_1, "Q5_1", (double)(5 * 32 + 16 + 16) / 32, "5-bit x 32 + 16-bit scale + 16-bit bias"},
    {GGML_TYPE_Q8_0, "Q8_0", (double)(8 * 32 + 32) / 32, "8-bit x 32 + 32-bit scale"},
    {GGML_TYPE_Q8_1, "Q8_1", (double)(8 * 32 + 32 + 32) / 32, "8-bit x 32 + 32-bit scale + 32-bit bias"},
    {GGML_TYPE_Q2_K, "Q2_K", (double)((2 * 16 + 4 + 4) * 16 + 16 + 16) / (16 * 16), "[2-bit x 16 + 4-bit scale + 4-bit bias] x 16 + 16-bit scale + 16-bit bias"},
    {GGML_TYPE_Q3_K, "Q3_K", (double)((3 * 16 + 6) * 16 + 16) / (16 * 16), "[3-bit x 16 + 6-bit scale] x 16 + 16-bit scale"},
    {GGML_TYPE_Q4_K, "Q4_K", (double)((4 * 32 + 6 + 6) * 8 + 16 + 16) / (32 * 8), "[4-bit x 32 + 6-bit scale + 6-bit bias] x 8 + 16-bit scale + 16-bit bias"},
    {GGML_TYPE_Q5_K, "Q5_K", (double)((5 * 32 + 6 + 6) * 8 + 16 + 16) / (32 * 8), "[5-bit x 32 + 6-bit scale + 6-bit bias] x 8 + 16-bit scale + 16-bit bias"},
    {GGML_TYPE_Q6_K, "Q6_K", (double)((6 * 16 + 8) * 16 + 16) / (16 * 16), "[6-bit x 16 + 8-bit scale] x 16 + 16-bit scale"},
}};

static std::string ftype_to_str(ggml_type ftype) {
    for (const auto &opt : quant_types) {
        if (ftype == opt.ftype) {
            return opt.fname;
        }
    }
    return "";
}

static ggml_type parse_ftype(std::string ftype_str) {
    std::transform(ftype_str.begin(),
                   ftype_str.end(),
                   ftype_str.begin(),
                   [](char c) { return (char)std::toupper(c); });

    const char *s0 = ftype_str.data();
    const char *s1 = s0 + ftype_str.size();
    int ftype;
    auto x = std::from_chars(s0, s1, ftype);
    if (x.ptr != s1) {
        ftype = -1;
    }

    for (const auto &opt : quant_types) {
        if (ftype_str == opt.fname) {
            return opt.ftype;
        }
        if (ftype == opt.ftype) {
            return opt.ftype;
        }
    }

    return (ggml_type)-1;
}

static void show_usage(const char *exe) {
    printf("usage: %s input.gguf type output.gguf\n", exe);
    printf("\n");
    printf("Allowed quantization types:\n");
    for (const auto &opt : quant_types) {
        printf("  %d or %s : %s (bpw=%.3f)\n", opt.ftype, opt.fname.c_str(), opt.description.c_str(), (float)opt.bits_per_weight);
    }
}

int main(int argc, char **argv) {
    std::cout << berts_version() << std::endl;

    berts_set_log_level(BERTS_LOG_ALL);

    if (argc != 4) {
        show_usage(argv[0]);
        return 1;
    }

    // ".gguf/bert-base-cased.gguf"
    // ".gguf/bert-base-cased_q8.gguf"
    const char *model_path = argv[1];
    const char *quant_path = argv[3];

    const char *ftype_str = argv[2];
    if (std::strlen(ftype_str) == 0) {
        // must not happen
        show_usage(argv[0]);
        return 1;
    }

    ggml_type ftype = parse_ftype(ftype_str);
    if ((int)ftype < 0) {
        show_usage(argv[0]);
        return 1;
    }

    const std::string ftype_s = ftype_to_str(ftype);

    printf("input:  %s\n", model_path);
    printf("output: %s\n", quant_path);
    printf("type: %s\n", ftype_s.c_str());

    bool ok = berts_model_quantize(model_path, quant_path, ftype);

    if (!ok) {
        std::cout << "fail to quantize model" << std::endl;
        return 0;
    }

    std::cout << "done" << std::endl;

    return 0;
}
