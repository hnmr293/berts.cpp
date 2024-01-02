#pragma once

#include <unordered_map>
#include <utility>
#include <vector>
#include "berts/models/unicode.hpp"

namespace berts {

struct bpe {
    using str_t = unicode::ustr;
    using vocab_t = std::unordered_map<str_t, size_t>;
    using merge_t = std::vector<std::pair<str_t, str_t>>;

    const str_t unk;
    double dropout_;
    bool fuse_unk_;

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

    bool load_vocab(const vocab_t &vocab, const merge_t &merge);

    bool tokenize(const str_t &text, std::vector<str_t> &result) const;
};

} // namespace berts
