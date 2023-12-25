#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include "berts/berts.h"
#include "berts/models/trie.hpp"

namespace berts::tokenizer {

bool tokenize(const std::string &text,
              const vocab::trie *vocab,
              const std::unordered_set<std::string> &never_split,
              std::vector<bert_token_t> &result,
              const berts_tokenize_info &cond);

} // namespace berts::tokenizer
