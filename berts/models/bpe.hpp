#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "berts/berts.h"
#include "berts/models/unicode.hpp"

namespace std {
template <>
struct hash<std::pair<bert_token_t, bert_token_t>> {
    size_t operator()(const std::pair<bert_token_t, bert_token_t> &pair) const {
        using hasher = std::hash<bert_token_t>;
        return (hasher{}(pair.first)) ^ (hasher{}(pair.second));
    }
};
} // namespace std

namespace berts {

struct bpe {
    using str_t = unicode::ustr;
    using vocab_t = std::unordered_map<str_t, bert_token_t>;
    using vocab_r_t = std::unordered_map<bert_token_t, str_t>;
    using tokenized_t = std::vector<str_t>;
    using cache_t = std::unordered_map<str_t, tokenized_t>;

    using token_id_pair = std::pair<bert_token_t, bert_token_t>;
    using token_pair = std::pair<str_t, str_t>;
    // (id0, id1) -> (rank, new_id)
    using mergemap_t = std::unordered_map<token_id_pair, std::pair<uint32_t, bert_token_t>>;

    const str_t unk;
    double dropout_;
    bool fuse_unk_;

    vocab_t vocab;
    vocab_r_t vocab_r;
    mergemap_t merge;

    str_t continueing_subword_prefix_;
    str_t end_of_word_suffix_;

    bpe(str_t unk, double dropout = 0.0, bool fuse_unk = true);

    double dropout() const noexcept {
        return dropout_;
    }

    void dropout(double v) {
        dropout_ = v;
    }

    bool fuse_unk() const noexcept {
        return fuse_unk_;
    }

    void fuse_unk(bool v) {
        fuse_unk_ = v;
    }

    const str_t &continueing_subword_prefix() const noexcept {
        return continueing_subword_prefix_;
    }

    void continueing_subword_prefix(const str_t &str) {
        continueing_subword_prefix_ = str;
    }

    const str_t &end_of_word_suffix() const noexcept {
        return end_of_word_suffix_;
    }

    void end_of_word_suffix(const str_t &str) {
        end_of_word_suffix_ = str;
    }

    bool id_to_token(bert_token_t id, str_t &token) const;

    bool token_to_id(const str_t &token, bert_token_t &id) const;

    void clear();

    bool load_vocab(const vocab_t &vocab, const std::vector<token_id_pair> &merge);

    bool load_vocab(const vocab_t &vocab, const std::vector<token_pair> &merge);

    bool tokenize(const str_t &text, tokenized_t &result) const;

    bool tokenize(const str_t &text, tokenized_t &result, cache_t &cache) const;
};

} // namespace berts
