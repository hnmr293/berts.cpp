#include "berts/models/bpe.hpp"

namespace berts {

bpe::bpe(str_t unk, double dropout, bool fuse_unk)
    : unk(unk)
    , dropout_(dropout)
    , fuse_unk_(fuse_unk) {}

bool bpe::load_vocab(const vocab_t &vocab, const merge_t &merge) {
    return false;
}

bool bpe::tokenize(const str_t &text, std::vector<str_t> &result) const {
    return false;
}

} // namespace berts
