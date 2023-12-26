#include <iostream>
#include <string>
#include <vector>
#include <ranges>
#include "berts/berts.h"

static void show_usage(const char *exe) {
    printf("usage: %s input.gguf text\n", exe);
    printf("\n");
}

int main(int argc, char **argv) {
    std::cout << berts_version() << std::endl;

    berts_set_log_level(BERTS_LOG_ALL);

    if (argc != 3) {
        show_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    std::string text{argv[2]};

    std::cout << "input: " << model_path << std::endl;
    std::cout << "text: " << text << std::endl;

    auto ctx = berts_load_from_file(model_path);

    if (!ctx) {
        std::cout << "fail to load model" << std::endl;
        return 1;
    }
    
    berts_tokenize_info cond{};
    berts_init_tokenize_info_default(&cond);
    
    std::vector<bert_token_t> ids{};
    ids.resize(text.size());
    if (!berts_tokenize(ctx, text.c_str(), &cond, ids.data())) {
        std::cout << "fail to tokenize text" << std::endl;
        return 1;
    }

    for (const auto [i, t] : ids | std::views::enumerate) {
        std::cout << "token " << i << ": " << t << std::endl;
    }
    
    std::cout << "done" << std::endl;

    return 0;
}
