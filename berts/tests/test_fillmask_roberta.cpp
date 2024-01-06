#include "berts/berts.h"

#include <array>
#include <memory>

#define BERTS_TEST_SHORTHAND
#include "berts/tests/tests.hpp"

test_def {
    test(bert_fillmask) {
        berts_set_log_level(BERTS_LOG_INFO);
        const char *model_path = ".gguf/roberta-base-f32.gguf";
        auto ctx = berts_load_from_file(model_path);

        testcase(ctx) {
            assert(ctx);
            assert(berts_arch(ctx) == BERTS_TYPE_ROBERTA);
        };

        testcase(tokenize) {
            const std::string text1 = "Hello I'm a <mask> model.";
            size_t size = text1.size();
            std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};

            bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
            assert(ok);
            assert(size == 9);
            assert(tokens[0] == 0);     // <s>
            assert(tokens[1] == 31414); // Hello
            assert(tokens[2] == 38);    //  I
            assert(tokens[3] == 437);   // 'm
            assert(tokens[4] == 10);    //  a
            assert(tokens[5] == 50264); // <mask>
            assert(tokens[6] == 1421);  //  model
            assert(tokens[7] == 4);     // .
            assert(tokens[8] == 2);     // </s>
        };

        testcase(eval) {
            berts_eval_info cond{};
            berts_init_eval_info(&cond);
            cond.pool_type = BERTS_POOL_NONE;

            std::array<bert_token_t, 9> tokens{{0, 31414, 38, 437, 10, 50264, 1421, 4, 2}};

            size_t out_size = 0;
            auto result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, nullptr, &out_size);
            assert(result);
            assert(out_size);

            std::unique_ptr<float[]> out{new float[out_size]};
            result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, out.get(), &out_size);
            assert(result);
        };

        testcase(fillmask) {
            // 1. eval
            berts_eval_info cond{};
            berts_init_eval_info(&cond);
            cond.pool_type = BERTS_POOL_NONE;

            std::array<bert_token_t, 9> tokens{{0, 31414, 38, 437, 10, 50264, 1421, 4, 2}};

            size_t out_size = 0;
            auto result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, nullptr, &out_size);
            assert(result);
            assert(out_size);

            std::unique_ptr<float[]> out{new float[out_size]};
            result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, out.get(), &out_size);
            assert(result);

            // 2. fill mask
            size_t vocab_size = berts_vocab_size(ctx);
            assert(vocab_size != 0);

            berts_eval_lm_info cond2{};
            berts_init_eval_lm_info(&cond2);
            size_t out_size2 = 0;
            result = berts_eval_lm(ctx, out.get(), out_size, &cond2, nullptr, nullptr, &out_size2);
            assert(result);
            assert(out_size2 == tokens.size() * vocab_size);

            std::unique_ptr<bert_token_t[]> out2{new bert_token_t[out_size2]};
            std::unique_ptr<float[]> out2_probs{new float[out_size2]};
            result = berts_eval_lm(ctx, out.get(), out_size, &cond2, out2.get(), out2_probs.get(), &out_size2);
            assert(result);

            // 3. retrieve a token with the most value in mask position (index=6)
            //    the token should be `fashion`
            const size_t mask_index = 5;
            const bert_token_t *masked_values = out2.get() + mask_index * vocab_size;

            size_t token_len = 256;
            std::string detected(token_len, '\0');
            result = berts_id_to_token(ctx, masked_values[0], detected.data(), &token_len);
            assert(result);
            detected.erase(token_len);
            assert(detected == " male");
        };

        testcase(top_k) {
            // 1. eval
            berts_eval_info cond{};
            berts_init_eval_info(&cond);
            cond.pool_type = BERTS_POOL_NONE;

            std::array<bert_token_t, 9> tokens{{0, 31414, 38, 437, 10, 50264, 1421, 4, 2}};

            size_t out_size = 0;
            auto result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, nullptr, &out_size);
            assert(result);
            assert(out_size);

            std::unique_ptr<float[]> out{new float[out_size]};
            result = berts_eval(ctx, tokens.data(), nullptr, tokens.size(), &cond, out.get(), &out_size);
            assert(result);

            // 2. fill mask
            size_t vocab_size = berts_vocab_size(ctx);
            assert(vocab_size != 0);

            const size_t k = 3;

            berts_eval_lm_info cond2{};
            berts_init_eval_lm_info(&cond2);
            cond2.top_k = k;
            size_t out_size2 = 0;
            result = berts_eval_lm(ctx, out.get(), out_size, &cond2, nullptr, nullptr, &out_size2);
            assert(result);
            assert(out_size2 == tokens.size() * k);

            std::unique_ptr<bert_token_t[]> out2{new bert_token_t[out_size2]};
            std::unique_ptr<float[]> out2_probs{new float[out_size2]};
            result = berts_eval_lm(ctx, out.get(), out_size, &cond2, out2.get(), out2_probs.get(), &out_size2);
            assert(result);

            // 3. retrieve a token with the most value in mask position (index=6)
            //    the token should be `fashion`
            const size_t mask_index = 5;
            const bert_token_t *masked_values = out2.get() + mask_index * k;

            std::array<std::string, k> expected{{
                " male",
                " female",
                " professional",
            }};

            std::array<std::string, k> actual{{
                "",
                "",
                "",
            }};

            for (size_t i = 0; i < k; ++i) {
                size_t token_len = 256;
                std::string detected(token_len, '\0');
                result = berts_id_to_token(ctx, masked_values[i], detected.data(), &token_len);
                assert(result);
                detected.erase(token_len);
                actual[i] = detected;
            }
            
            for (size_t i = 0; i < k; ++i) {
                assert(actual[i] == expected[i]);
            }
        };
    };
};

int main() {
    run_tests();
    return 0;
}
