#include "berts/models/vocab.hpp"
#include <algorithm>
#include <memory>
#include <unordered_map>

using namespace berts::unicode;

namespace berts::vocab {

struct trie_node;

using node_t = std::unique_ptr<trie_node>;

struct trie_node {
    bert_token_t id; // -1 if not in vacab
    std::unordered_map<unic_t, node_t> children;

    trie_node()
        : id(-1)
        , children() {}

    trie_node(bert_token_t id)
        : id(id)
        , children() {}
};

struct trie {
    node_t root;
    trie()
        : root(new trie_node()) {}
};

static inline void add_str(trie_node *n, const ustr &s, bert_token_t id) {
    if (s.empty()) return;

    auto it = s.begin();
    const auto end = s.end();

    while (it != end) {
        auto c = *it++;
        auto nit = n->children.find(c);
        if (nit == n->children.end()) {
            // not found
            auto nn = new trie_node{};
            n->children.emplace(c, nn);
            nn = n;
        } else {
            n = nit->second.get();
        }
    }

    n->id = id;
}

static inline const trie_node *find_node(const trie_node *n, const ustr &s) {
    if (s.empty()) return nullptr;

    auto it = s.begin();
    const auto end = s.end();

    while (it != end) {
        auto c = *it++;
        auto nit = n->children.find(c);
        if (nit == n->children.end()) {
            // not found
            return nullptr;
        }
        n = nit->second.get();
    }

    return n;
}

static inline const trie_node *find_substr(const trie_node *n, const ustr &s, ustr &found, ustr &rest) {
    if (s.empty()) return nullptr;

    const trie_node *last_found = nullptr;

    auto it = s.begin();
    const auto end = s.end();

    std::vector<unic_t> found_{};
    std::vector<unic_t> rest_{};

    while (it != end) {
        auto c = *it;
        auto nit = n->children.find(c);

        if (nit == n->children.end()) {
            // not found
            std::copy(it, end, std::back_inserter(rest_));
            break;
        }

        found_.push_back(c);
        n = nit->second.get();
        it += 1;

        if (n->id != (bert_token_t)-1) {
            last_found = n;
        }
    }

    found = ustr{found_};
    rest = ustr{rest_};

    return last_found;
}

const trie *build_trie(const std::vector<std::string> &vocab) {
    trie *t = new trie{};
    for (size_t id = 0, n = vocab.size(); id < n; ++id) {
        add_str(t->root.get(), {vocab[id]}, id);
    }
    return t;
}

void free_trie(trie *t) {
    delete t;
}

const trie_node *trie_root(const trie *t) {
    return t->root.get();
}

bert_token_t search_trie(const trie *t, const std::string &s) {
    return search_trie(t, ustr{s});
}

bert_token_t search_trie(const trie *t, const ustr &s) {
    auto n = find_node(t->root.get(), s);
    return n ? n->id : (bert_token_t)-1;
}

const trie_node *search_node(const trie *t, const std::string &s) {
    return search_node(t->root.get(), ustr{s});
}

const trie_node *search_node(const trie *t, const ustr &s) {
    return search_node(t->root.get(), s);
}

const trie_node *search_node(const trie_node *n,
                             const std::string &s) {
    return search_node(n, ustr{s});
}

const trie_node *search_node(const trie_node *n,
                             const ustr &s) {
    return find_node(n, s);
}

bert_token_t search_trie_substr(const trie *t, const std::string &s, std::string &found, std::string &rest) {
    return search_trie_substr(t->root.get(), s, found, rest);
}

bert_token_t search_trie_substr(const trie *t, const ustr &s, ustr &found, ustr &rest) {
    return search_trie_substr(t->root.get(), s, found, rest);
}

bert_token_t search_trie_substr(const trie_node *n, const std::string &s, std::string &found, std::string &rest) {
    ustr s_{s}, found_{}, rest_{};
    const auto id = search_trie_substr(n, s_, found_, rest_);

    if (id != (bert_token_t)-1) {
        found = found_.encode();
        rest = rest_.encode();
    }

    return id;
}

bert_token_t search_trie_substr(const trie_node *n, const ustr &s, ustr &found, ustr &rest) {
    ustr found_{}, rest_{};
    n = find_substr(n, s, found_, rest_);

    const auto id = n ? n->id : (bert_token_t)-1;
    if (id != (bert_token_t)-1) {
        found = found_;
        rest = rest_;
    }

    return id;
}

} // namespace berts::vocab
