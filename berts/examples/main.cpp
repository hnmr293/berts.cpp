#include "berts/berts.h"
#include <iostream>

int main() {
    std::cout << berts_version() << std::endl;

    berts_set_log_level(BERTS_LOG_ALL);

    const char *model_path = ".gguf/bert-base-cased.gguf";

    auto ctx = berts_load_from_file(model_path);
    
    berts_free(ctx);

    std::cout << "done" << std::endl;
    
    return 0;
}
