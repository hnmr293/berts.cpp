#include <cassert>
#include <memory>
#include <string>
#include "berts/berts.h"

int main() {
    berts_set_log_level(BERTS_LOG_ALL);
    const char *model_path = ".gguf/bert-base-cased_q8.gguf";
    auto ctx = berts_load_from_file(model_path);
    assert(ctx);
    
    const std::string text1 = "Hi, I am [MASK] man. How are you?";
    size_t size = text1.size();
    std::unique_ptr<bert_token_t[]> tokens{new bert_token_t[size]};

    bool ok = berts_tokenize(ctx, text1.c_str(), tokens.get(), &size);
    assert(ok);
    assert(size == 13);
    assert(tokens[0] == 101);   // [CLS]
    assert(tokens[1] == 8790);  // Hi
    assert(tokens[2] == 117);   // ,
    assert(tokens[3] == 146);   // I
    assert(tokens[4] == 1821);  // am
    assert(tokens[5] == 103);   // [MASK]
    assert(tokens[6] == 1299);  // man
    assert(tokens[7] == 119);   // .
    assert(tokens[8] == 1731);  // How
    assert(tokens[9] == 1132);  // are
    assert(tokens[10] == 1128); // you
    assert(tokens[11] == 136);  // ?
    assert(tokens[12] == 102);  // [SEP]

    auto result = berts_eval(ctx, tokens.get(), nullptr, size);
    assert(result);

    berts_free(ctx);

    return 0;
}
