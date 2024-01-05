#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "berts/berts.h"

int main() {
    berts_set_log_level(BERTS_LOG_INFO);
    const char *model_path = ".gguf/roberta-base-f32.gguf";
    auto ctx = berts_load_from_file(model_path);
    assert(ctx);

    const std::string text1 = "Hi, I am <mask> man. How are you?";
    size_t size = text1.size();
    std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};

    bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
    assert(ok);
    assert(size == 13);
    assert(tokens[0] == 0);     // "<s>"
    assert(tokens[1] == 30086); // "Hi"
    assert(tokens[2] == 6);     // ","
    assert(tokens[3] == 38);    // " I"
    assert(tokens[4] == 524);   // " am"
    assert(tokens[5] == 50264); // "<mask>"
    assert(tokens[6] == 313);   // " man"
    assert(tokens[7] == 4);     // "."
    assert(tokens[8] == 1336);  // " How"
    assert(tokens[9] == 32);    // " are"
    assert(tokens[10] == 47);   // " you"
    assert(tokens[11] == 116);  // "?"
    assert(tokens[12] == 2);    // "</s>"

    const auto pool_types = {
        BERTS_POOL_NONE,
        BERTS_POOL_CLS,
        BERTS_POOL_AVG,
        BERTS_POOL_MAX,
    };

    for (const auto pt : pool_types) {
        std::cout << pt << std::endl;

        berts_eval_info cond{};
        berts_init_eval_info(&cond);
        cond.pool_type = pt;

        size_t out_size = 0;
        auto result = berts_eval(ctx, tokens.get(), nullptr, size, &cond, nullptr, &out_size);
        assert(result);
        assert(out_size);

        std::unique_ptr<float[]> out{new float[out_size]};
        result = berts_eval(ctx, tokens.get(), nullptr, size, &cond, out.get(), &out_size);
        assert(result);

        std::stringstream ss{};
        ss << "test_roberta_" << (int)pt << ".bin";
        FILE *fp = fopen(ss.str().c_str(), "wb");
        assert(fp);

        int32_t size_ = (int32_t)out_size;
        fwrite(&size_, sizeof(int32_t), 1, fp);
        fwrite(out.get(), sizeof(float), out_size, fp);
        fclose(fp);
    }

    berts_free(ctx);

    return 0;
}
