#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include "berts/berts.h"
#include "berts/models/vocab.hpp"

namespace berts::tokenizer {

struct tokenize_info {
    // ignored, always normalized with NFC
    bool normalize;

    // remove U+FFFD
    bool remove_replacement_char;

    // remove U+0000
    bool remove_null_char;

    // remove control chars (category C*)
    bool remove_control_char;

    // convert all whitespaces to a normal space (U+0020)
    bool normalize_whitespaces;

    // add space around all CJK characters
    bool add_space_around_cjk_char;

    // force input to be lowercase letters
    bool do_lower_case;

    // remove all accent chars
    bool strip_accents;

    // split words at a punctuation
    bool split_on_punc;

    // [UNK] token id
    bert_token_t unknown_token_id;

    static tokenize_info create_default(bert_token_t unknown_token_id) {
        return tokenize_info{
            .normalize = true,
            .remove_replacement_char = true,
            .remove_null_char = true,
            .remove_control_char = true,
            .normalize_whitespaces = true,
            .add_space_around_cjk_char = true,
            .do_lower_case = true,
            .strip_accents = true,
            .split_on_punc = true,
            .unknown_token_id = unknown_token_id,
        };
    }

    static tokenize_info do_not_basic_tokenize(bert_token_t unknown_token_id) {
        return tokenize_info{
            .normalize = true,
            .remove_replacement_char = false,
            .remove_null_char = false,
            .remove_control_char = false,
            .normalize_whitespaces = true,
            .add_space_around_cjk_char = false,
            .do_lower_case = false,
            .strip_accents = false,
            .split_on_punc = false,
            .unknown_token_id = unknown_token_id,
        };
    }
};

bool tokenize(const std::string &text,
              const vocab::trie *vocab,
              const std::unordered_set<std::string> &never_split,
              std::vector<bert_token_t> &result,
              const tokenize_info &cond);

} // namespace berts::tokenizer
