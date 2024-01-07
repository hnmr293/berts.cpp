#include <iostream>
#include <memory>
#include <string>
#include <ranges>
#include <vector>
#include <tuple>

#include <berts/berts.h>

static void show_usage(const char *exe) {
    printf("usage: %s input.gguf k prompt\n", exe);
    printf("\n");
}

int main(int argc, char **argv) {
    std::cerr << "* berts.cpp v" << berts_version() << " fill_mask" << std::endl;

    berts_set_log_level(BERTS_LOG_WARN);

    if (argc != 4) {
        show_usage(argv[0]);
        return 1;
    }

    //
    // parse arguments
    //

    const char *model_path = argv[1];
    const char *k_str = argv[2];
    const char *prompt = argv[3];

    int k = 0;

    try {
        size_t k_rest = 0;
        k = std::stoi(k_str, &k_rest);
    } catch (std::runtime_error &e) {
        std::cerr << "invalid k: " << k_str << std::endl;
        std::cerr << "k must be a positive integer" << std::endl;
        return 1;
    }

    if (k <= 0) {
        std::cerr << "invalid k: " << k_str << std::endl;
        std::cerr << "k must be a positive integer" << std::endl;
        return 1;
    }

    //
    // load model
    //

    berts_context *ctx = berts_load_from_file(model_path);
    if (!ctx) {
        std::cerr << "fail to load model: " << model_path << std::endl;
        return 1;
    }

    size_t vocab_size = berts_vocab_size(ctx);

    if (vocab_size < (size_t)k) {
        std::cerr << "invalid k: " << k_str << std::endl;
        std::cerr << "k must be in range 1.." << vocab_size << std::endl;
        return 1;
    }

    //
    // tokenize
    //

    std::cout << "prompt = " << prompt << std::endl;

    const std::string text{prompt};
    size_t token_count = text.size();
    std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[token_count]};

    if (!berts_tokenize(ctx, text.c_str(), tokens.get(), &token_count)) {
        std::cerr << "fail to tokenize prompt: " << text << std::endl;
        return 1;
    }

    std::cout << "token id = ";
    for (size_t i = 0; i < token_count; ++i) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;

    // check <mask> position
    bert_token_t mask_id = berts_mask_id(ctx);
    size_t mask_pos = 0;
    bool found = false;
    for (size_t i = 0; i < token_count; ++i) {
        if (tokens[i] == mask_id) {
            if (found) {
                // many masks
                std::cerr << "please specify one mask" << std::endl;
                return 1;
            }
            found = true;
            mask_pos = i;
        }
    }

    if (!found) {
        // no masks
        std::cerr << "please specify one mask" << std::endl;
        return 1;
    }

    //
    // retrieve hidden states
    //

    berts_eval_info eval_cond{};
    berts_init_eval_info(&eval_cond);
    eval_cond.pool_type = BERTS_POOL_NONE;
    // dry run
    // to estimate the output length
    size_t hidden_state_size = 0;
    if (!berts_eval(ctx, tokens.get(), nullptr, token_count, &eval_cond, nullptr, &hidden_state_size)) {
        std::cerr << "fail to call `berts_eval`" << std::endl;
        return 1;
    }

    std::unique_ptr<float[]> hidden_states{new float[hidden_state_size]};
    if (!berts_eval(ctx, tokens.get(), nullptr, token_count, &eval_cond, hidden_states.get(), &hidden_state_size)) {
        std::cerr << "fail to call `berts_eval`" << std::endl;
        return 1;
    }

    //
    // fill mask
    //

    berts_eval_lm_info unmask_cond{};
    berts_init_eval_lm_info(&unmask_cond);
    unmask_cond.top_k = k;

    // dry run
    // to estimate the output length
    size_t buffer_size = 0;
    if (!berts_eval_lm(ctx, hidden_states.get(), hidden_state_size, &unmask_cond, nullptr, nullptr, &buffer_size)) {
        std::cerr << "fail to call `berts_eval_lm`" << std::endl;
        return 1;
    }

    // estimate
    std::unique_ptr<bert_token_t[]> estimated_tokens{new bert_token_t[buffer_size]};
    std::unique_ptr<float[]> scores{new float[buffer_size]};
    if (!berts_eval_lm(ctx, hidden_states.get(), hidden_state_size, &unmask_cond, estimated_tokens.get(), scores.get(), &buffer_size)) {
        std::cerr << "fail to call `berts_eval_lm`" << std::endl;
        return 1;
    }

    std::vector<std::tuple<bert_token_t, std::string, float>> results{};
    size_t token_max_len = 0;
    
    for (int i = 0; i < k; ++i) {
        bert_token_t token_id = estimated_tokens[k * mask_pos + i];
        float score = scores[k * mask_pos + i];

        std::string token(256, '\0');
        size_t token_length = token.size();
        berts_id_to_token(ctx, token_id, token.data(), &token_length);
        token.erase(token_length);

        if (token_max_len < token_length) {
            token_max_len = token_length;
        }

        results.emplace_back(token_id, token, score);
    }

    for (const auto &[index, result] : results | std::views::enumerate) {
        auto [id, token, score] = result;
        while (token.size() < token_max_len) {
            token += ' ';
        }
        std::cout << index << ": " << token << " (" << id << "); p = " << score << std::endl;
    }

    berts_free(ctx);

    return 0;
}
