#include <cassert>
#include <memory>
#include "berts/models/trie.hpp"

using namespace berts::trie;
using namespace berts::unicode;

static int check_trie_free = 1;

namespace std {
template <>
struct default_delete<trie> {
    void operator()(trie *trie) {
        if (trie) free_trie(trie);
        check_trie_free = 2;
    }
};
} // namespace std

using trie_t = std::unique_ptr<trie>;

int main() {
    std::vector<std::string> vocab{{
        "a",
        "b",
        "c",
        "ab",
        "abc",
        "acb",
        "ca",
        "##d",
    }};

    auto t = build_trie(vocab);
    assert(t);

    auto a = search_trie(t, ustr{"a"});
    assert(a != BERTS_INVALID_TOKEN_ID);

    auto b = search_trie(t, ustr{"b"});
    assert(b != BERTS_INVALID_TOKEN_ID);

    auto c = search_trie(t, ustr{"c"});
    assert(c != BERTS_INVALID_TOKEN_ID);

    auto d = search_trie(t, ustr{"d"});
    assert(d == BERTS_INVALID_TOKEN_ID);

    auto ab = search_trie(t, ustr{"ab"});
    assert(ab != BERTS_INVALID_TOKEN_ID);

    auto abc = search_trie(t, ustr{"abc"});
    assert(abc != BERTS_INVALID_TOKEN_ID);

    auto acb = search_trie(t, ustr{"acb"});
    assert(acb != BERTS_INVALID_TOKEN_ID);

    auto ca = search_trie(t, ustr{"ca"});
    assert(ca != BERTS_INVALID_TOKEN_ID);

    auto ac = search_trie(t, ustr{"ac"});
    assert(ac == BERTS_INVALID_TOKEN_ID);

    auto d_node = search_node(t, ustr{"d"});
    assert(!d_node);

    auto ac_node = search_node(t, ustr{"ac"});
    assert(ac_node);

    auto acb_node = search_node(t, ustr{"acb"});
    assert(acb_node);

    auto a_node = search_node(t, ustr{"a"});
    assert(a_node);

    auto ad_node = search_node(a_node, ustr{"d"});
    assert(!ad_node);

    auto ac_node2 = search_node(a_node, ustr{"c"});
    assert(ac_node2);

    auto acb_node2 = search_node(a_node, ustr{"cb"});
    assert(acb_node2);

    auto acb_node3 = search_node(ac_node, ustr{"b"});
    assert(acb_node3);

    auto acc_node = search_node(ac_node, ustr{"c"});
    assert(!acc_node);

    auto cont_node = search_node(t, ustr{"##"});
    assert(cont_node);

    auto _d_node = search_node(cont_node, ustr{"d"});
    assert(_d_node);

    auto _a_node = search_node(cont_node, ustr{"a"});
    assert(!_a_node);

    // abcd
    // -> abc ##d
    {
        std::string found, rest;
        auto id = search_trie_substr(t, "abcd", found, rest);
        assert(id != BERTS_INVALID_TOKEN_ID);
        assert(found == "abc");
        assert(rest == "d");
        id = search_trie_substr(cont_node, "d", found, rest);
        assert(id != BERTS_INVALID_TOKEN_ID);
        assert(found == "d");
        assert(rest == "");
    }

    assert(check_trie_free == 1);
    {
        trie_t t{build_trie(vocab)};
    }
    assert(check_trie_free == 2);

    return 0;
}
