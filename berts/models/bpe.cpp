#include "berts/models/bpe.hpp"

#include <algorithm>
#include <cassert>
#include <compare>
#include <iterator>
#include <queue>
#include <random>
#include <ranges>
#include <sstream>
#include <type_traits>
#include "berts/models/log.hpp"

//
// port of tokenizers/src/models/bpe/model.rs
//   of huggingface/tokenizers
//

namespace berts {

struct symbol_t {
    bert_token_t id;
    int32_t prev;
    int32_t next;
    uint32_t len; // len of code point, not byte length

    void merge_with(const symbol_t &other, bert_token_t new_id) {
        id = new_id;
        len += other.len;
        next = other.next;
    }
};

struct word_t {
    std::vector<symbol_t> symbols;

    void add(bert_token_t id, uint32_t cp_len) {
        if (symbols.empty()) {
            symbols.emplace_back(id, -1, -1, cp_len);
        } else {
            size_t len = symbols.size();
            symbols.back().next = len;
            symbols.emplace_back(id, len - 1, -1, cp_len);
        }
    }

    bool last_is(bert_token_t id) const {
        if (symbols.empty()) {
            return false;
        }
        return symbols.back().id == id;
    }
};

struct token_t {
    bert_token_t id;
    uint32_t begin;
    uint32_t end;
    bpe::str_t value;
};

struct merge_t {
    int32_t index;
    uint32_t rank;
    bert_token_t new_id;

    std::strong_ordering operator<=>(const merge_t &rhs) const noexcept {
        if (rank != rhs.rank) {
            return rank <=> rhs.rank;
        } else {
            return index <=> rhs.index;
        }
    }
};

static bool tokenize_bpe(const bpe &bpe, const bpe::str_t &text, bpe::tokenized_t &result, bpe::cache_t *cache);

static bool merge_all(const bpe &bpe, word_t &word);

static bool merge_word(const bpe &bpe, const bpe::str_t &text, word_t &result);

static bool word_to_tokens(const bpe &bpe, const word_t &word, std::vector<token_t> &result);

template <typename T>
concept Reservable = requires(T &x, size_t n) {
    x.reserve(n);
};

template <Reservable T>
static inline void reserve(T &container, size_t n) {
    container.reserve(container.size() + n);
}

static inline bpe::str_t slice(const bpe::str_t &str, size_t begin = 0, int32_t len = -1) {
    std::vector<unicode::unic32_t> codepoints{};
    size_t index = 0;
    str.each_cp(false, [&](const bpe::str_t::cp &cp) {
        if (index++ < begin) {
            return;
        }
        if (len < 0 || codepoints.size() < (size_t)len) {
            codepoints.push_back(cp.c);
        }
    });
    return {codepoints};
}

bpe::bpe(str_t unk, double dropout, bool fuse_unk)
    : unk(unk)
    , dropout_(dropout)
    , fuse_unk_(fuse_unk)
    , continueing_subword_prefix_("")
    , end_of_word_suffix_("") {}

void bpe::clear() {
    vocab.clear();
    merge.clear();
}

bool bpe::id_to_token(bert_token_t id, str_t &token) const {
    auto it = vocab_r.find(id);
    if (it == vocab_r.end()) {
        return false;
    }
    token = it->second;
    return true;
}

bool bpe::token_to_id(const str_t &token, bert_token_t &id) const {
    auto it = vocab.find(token);
    if (it == vocab.end()) {
        return false;
    }
    id = it->second;
    return true;
}

bool bpe::load_vocab(const vocab_t &vocab, const std::vector<token_id_pair> &merge) {
    log::debug("loading BPE vocab");

    reserve(this->vocab, vocab.size());
    reserve(this->vocab_r, vocab.size());
    reserve(this->merge, merge.size());

    // unordered_map::merge does not receive "const" argument.
    for (const auto &[str, id] : vocab) {
        this->vocab.insert_or_assign(str, id);
        this->vocab_r.insert_or_assign(id, str);
    }
    
    log::when(BERTS_LOG_DEBUG, [this, &vocab]() {
        log::debug("  vocab");
        std::vector<bert_token_t> ids{};
        for (const auto &[str, id] : vocab) {
            ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end());
        for (auto &&id : ids) {
            log::debug("    {:>3}: {}", id, this->vocab_r[id].encode());
        }
    });

    log::debug("  merge");
    const auto prefix_len = continueing_subword_prefix().codepoints();
    const size_t rank_start = this->merge.size();
    for (const auto [index, pair] : merge | std::views::enumerate) {
        const auto rank = rank_start + index;
        const auto [id0, id1] = pair;

        str_t token0{}, token1{}, new_token{};
        if (!id_to_token(id0, token0)) {
            log::error("token id {} is not found in vocab", id0);
            return false;
        }

        if (!id_to_token(id1, token1)) {
            log::error("token id {} is not found in vocab", id1);
            return false;
        }

        if (prefix_len == 0) {
            new_token = token0 + token1;
        } else {
            new_token = token0 + slice(token1, prefix_len);
        }

        bert_token_t new_token_id;
        if (!token_to_id(new_token, new_token_id)) {
            log::error("merged token {} is not found in vocab", new_token.encode());
            return false;
        }

        this->merge.insert_or_assign(pair, std::pair{rank, new_token_id});
        log::debug("    rank={}, [{}({}), {}({})] -> {}({})", rank, token0.encode(), id0, token1.encode(), id1, new_token.encode(), new_token_id);
    }

    log::debug("finish loading BPE vocab");

    return true;
}

bool bpe::load_vocab(const vocab_t &vocab, const std::vector<token_pair> &merge) {
    std::vector<token_id_pair> merge_{};
    for (const auto &[token0, token1] : merge) {
        // this->vocab is not constructed yet,
        // so we cannot call this->token_to_id.
        
        auto it0 = vocab.find(token0);
        if (it0 == vocab.end()) {
            log::error("token {} is not found in vocab", token0.encode());
            return false;
        }
        
        auto it1 = vocab.find(token1);
        if (it1 == vocab.end()) {
            log::error("token {} is not found in vocab", token1.encode());
            return false;
        }
        
        merge_.emplace_back(it0->second, it1->second);
    }
    return load_vocab(vocab, merge_);
}

bool bpe::tokenize(const str_t &text, tokenized_t &result) const {
    return tokenize_bpe(*this, text, result, nullptr);
}

bool bpe::tokenize(const str_t &text, tokenized_t &result, cache_t &cache) const {
    return tokenize_bpe(*this, text, result, &cache);
}

static bool tokenize_bpe(const bpe &bpe, const bpe::str_t &text, bpe::tokenized_t &result, bpe::cache_t *cache) {
    log::when(BERTS_LOG_DEBUG, [&text]() {
        log::debug("start BPE tokenization");
        log::debug("  text = {}", text.encode());
    });

    if (text.empty()) return true;

    if (bpe.dropout() == 0.0 && cache) {
        // search cache
        auto it = cache->find(text);
        if (it != cache->end()) {
            // found
            auto &cached = it->second;
            reserve(result, cached.size());
            std::copy(cached.begin(), cached.end(), std::back_inserter(result));
            return true;
        }
        // not found
    }

    word_t word{};
    if (!merge_word(bpe, text, word)) {
        return false;
    }

    std::vector<token_t> tokens{};
    if (!word_to_tokens(bpe, word, tokens)) {
        return false;
    }

    if (bpe.dropout() == 0.0 && cache) {
        // save cache
        auto p = cache->emplace(text, bpe::tokenized_t{});
        assert(p.second); // it must be created

        auto &cached = p.first->second;

        reserve(result, tokens.size());
        reserve(cached, tokens.size());

        for (const auto &token : tokens) {
            result.push_back(token.value);
            cached.push_back(token.value);
        }
    } else {
        // without cache
        reserve(result, tokens.size());
        for (const auto &token : tokens) {
            result.push_back(token.value);
        }
    }

    log::debug("finish BPE tokenization");
    return true;
}

static bool merge_all(const bpe &bpe, word_t &word) {
    log::when(BERTS_LOG_DEBUG, [&word]() {
        std::stringstream ss{};
        ss << "(";
        for (const auto &sym : word.symbols) {
            ss << sym.id << " ";
        }
        if (!word.symbols.empty()) {
            ss.seekp(-1, ss.cur); // remove last space
        }
        ss << ")";
        log::debug("  id = {}", ss.str());
    });

    //
    // init priority queue
    //
    std::priority_queue<merge_t> q{};
    for (auto const [index, view] : word.symbols | std::views::slide(2) | std::views::enumerate) {
        // const auto &[sym1, sym2] = view;
        const symbol_t &sym1 = view[0];
        const symbol_t &sym2 = view[1];
        auto it = bpe.merge.find({sym1.id, sym2.id});
        if (it != bpe.merge.end()) {
            const auto [rank, new_id] = it->second;
            q.emplace((size_t)index, rank, new_id);
        }
    }

    std::random_device seed{};
    std::default_random_engine e{seed()};
    std::uniform_real_distribution<> rand{0.0, 1.0};

    //
    // main
    //
    const auto d = bpe.dropout();
    std::vector<merge_t> skip;
    while (!q.empty()) {
        const auto top = q.top();
        q.pop();

        if (0.0 < d) {
            if (rand(e) < d) {
                // skip
                skip.push_back(top);
                continue;
            }
        }

        // Re-insert the skipped elements
        for (auto &&skipped : skip) {
            q.push(skipped);
        }
        skip.clear();

        symbol_t &sym = word.symbols[top.index];

        if (sym.len == 0) {
            continue;
        }

        // Do nothing if we are the last symbol
        if (sym.next == -1) {
            continue;
        }

        auto next_pos = sym.next;
        symbol_t &right = word.symbols[next_pos];

        // Make sure we are not processing an expired queue entry
        std::pair target_new_pair{sym.id, right.id};
        auto target_it = bpe.merge.find(target_new_pair);
        if (target_it == bpe.merge.end()) {
            continue;
        }

        static_assert(std::is_same_v<decltype(target_it->second), std::pair<uint32_t, bert_token_t>>);
        if (target_it->second.second /* new_id */ != top.new_id) {
            continue;
        }

        // Otherwise, let's merge
        log::debug("  * merge ({}, {}) -> {}", sym.id, right.id, top.new_id);
        sym.merge_with(right, top.new_id);

        // Tag the right part as removed
        right.len = 0;

        // Update `prev` on the new `next` to the current pos
        if (0 <= right.next && (uint32_t)right.next < word.symbols.size()) {
            word.symbols[right.next].prev = top.index;
        }

        // Insert the new pair formed with the previous symbol
        if (0 <= sym.prev) {
            const symbol_t &prev = word.symbols[sym.prev];
            std::pair new_pair{prev.id, sym.id};
            auto prev_it = bpe.merge.find(new_pair);
            if (prev_it != bpe.merge.end()) {
                q.emplace(
                    sym.prev,
                    prev_it->second.first, // rank
                    prev_it->second.second // new_id
                );
            }
        }

        // Insert the new pair formed with the next symbol
        if ((size_t)sym.next < word.symbols.size()) {
            const symbol_t &next = word.symbols[sym.next];
            std::pair new_pair{sym.id, next.id};
            auto next_it = bpe.merge.find(new_pair);
            if (next_it != bpe.merge.end()) {
                q.emplace(
                    top.index,
                    next_it->second.first, // rank
                    next_it->second.second // new_id
                );
            }
        }
    }

    // Filter out the removed symbols
    auto end_it = std::remove_if(word.symbols.begin(),
                                 word.symbols.end(),
                                 [](const symbol_t &sym) {
                                     return sym.len == 0;
                                 });
    word.symbols.erase(end_it, word.symbols.end());

    return true;
}

static bool merge_word(const bpe &bpe, const bpe::str_t &text, word_t &result) {
    std::vector<std::pair<bpe::str_t, size_t>> chars{};
    text.each_cp(false, [&](const berts::unicode::ustr::cp &cp) {
        chars.push_back({{&cp.c, 1}, 1});
    });

    const bpe::str_t &prefix = bpe.continueing_subword_prefix();
    const bpe::str_t &suffix = bpe.end_of_word_suffix();
    const auto prefix_len = prefix.codepoints();
    const auto suffix_len = suffix.codepoints();

    for (size_t i = 0; i < chars.size(); ++i) {
        if (i != 0) {
            chars[i].first = bpe.continueing_subword_prefix() + chars[i].first;
            chars[i].second += prefix_len;
        }
        if (i == chars.size() - 1) {
            chars[i].first += bpe.end_of_word_suffix();
            chars[i].second += suffix_len;
        }
    }

    reserve(result.symbols, chars.size());

    for (const auto &s : chars) {
        if (bert_token_t id; bpe.token_to_id(s.first, id)) {
            result.add(id, s.second);
        } else {
            // TODO byte_fallback
            // if (bpe.byte_fallback) {
            //
            // }
            if (!bpe.unk.empty()) {
                if (bert_token_t unk_id; bpe.token_to_id(bpe.unk, unk_id)) {
                    // unk found
                    if (!bpe.fuse_unk() || !result.last_is(unk_id)) {
                        result.add(unk_id, s.second + bpe.unk.codepoints());
                        //                 ^~~~~~~~ ???
                    }
                } else {
                    // unk not found
                    // ignore
                }
            } else {
                // unk is not set
                // ignore
            }
        }
    }

    return merge_all(bpe, result);
}

static bool word_to_tokens(const bpe &bpe, const word_t &word, std::vector<token_t> &result) {
    uint32_t pos = 0;
    for (const auto symbol : word.symbols) {
        auto id = symbol.id;
        auto new_pos = pos + symbol.len;

        bpe::str_t token{};
        if (!bpe.id_to_token(id, token)) {
            token = bpe.unk;
        }

        result.emplace_back(id, pos, new_pos, token);

        pos = new_pos;
    }

    return true;
}

} // namespace berts
