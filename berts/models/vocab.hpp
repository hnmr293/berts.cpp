#pragma once

#include <string>
#include <vector>
#include "berts/berts.h"
#include "berts/models/unicode.hpp"

namespace berts::vocab {

struct trie;
struct trie_node;

trie *build_trie(const std::vector<std::string> &vocab);

void free_trie(trie *t);

const trie_node *trie_root(const trie *t);

/// @return -1 if not found
bert_token_t search_trie(const trie *t,
                         const std::string &s);

bert_token_t search_trie(const trie *t,
                         const unicode::ustr &s);

const trie_node *search_node(const trie *t,
                             const std::string &s);

const trie_node *search_node(const trie *t,
                             const unicode::ustr &s);

const trie_node *search_node(const trie_node *n,
                             const std::string &s);

const trie_node *search_node(const trie_node *n,
                             const unicode::ustr &s);

/// @brief search substr in vocab with greedy longest-match-first algorithm
/// @param found [out] found substring
/// @param rest [out] rest of string
/// @return bert_token_id of `found`; -1 if not found
bert_token_t search_trie_substr(const trie *t,
                                const std::string &s,
                                std::string &found,
                                std::string &rest);

bert_token_t search_trie_substr(const trie *t,
                                const unicode::ustr &s,
                                unicode::ustr &found,
                                unicode::ustr &rest);

bert_token_t search_trie_substr(const trie_node *n,
                                const std::string &s,
                                std::string &found,
                                std::string &rest);

bert_token_t search_trie_substr(const trie_node *n,
                                const unicode::ustr &s,
                                unicode::ustr &found,
                                unicode::ustr &rest);

} // namespace berts::vocab
