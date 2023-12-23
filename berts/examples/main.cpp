#include "berts/berts.h"
#include <iostream>

int main() {
    std::cout << berts_version() << std::endl;

    berts_set_log_level(BERTS_LOG_ALL);

    const char *model_path = ".gguf/bert-base-cased.gguf";

    auto ctx = berts_load_from_file(model_path);

    if (!ctx) {
        std::cout << "fail to open base model" << std::endl;
        return 0;
    }
    
    berts_free(ctx);

    const char *q8_path = ".gguf/bert-base-cased_q8.gguf";

    bool ok = berts_model_quantize(model_path, q8_path, ggml_type::GGML_TYPE_Q8_0);

    if (!ok) {
        std::cout << "fail to quantize model" << std::endl;
        return 0;
    }
    
    std::cout << "done" << std::endl;
    
    return 0;
}
