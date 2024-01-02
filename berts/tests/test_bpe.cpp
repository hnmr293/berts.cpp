#include <cassert>
#include <string>
#include <vector>
#include "berts/models/bpe.hpp"
#include "berts/models/unicode.hpp"

using ustr = berts::unicode::ustr;

static inline std::vector<ustr> tokenize(const berts::bpe &bpe, const ustr &token) {
    std::vector<ustr> result{};
    bool ok = bpe.tokenize(token, result);
    assert(ok);
    return result;
}

int main() {
    using vocab_t = berts::bpe::vocab_t;
    using merge_t = berts::bpe::merge_t;
    
    // unfused <unk>
    {
        berts::bpe bpe{"<unk>", 0.0, false};
        vocab_t vocab{{
            {ustr{"a"}, 0},
            {ustr{"b"}, 1},
        }};
        merge_t merge{};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(ok);

        {
            auto r = tokenize(bpe, "c");
            assert(r.size() == 1);
            assert(r[0] == ustr{"<unk>"});
        }

        {
            auto r = tokenize(bpe, "cc");
            assert(r.size() == 2);
            assert(r[0] == ustr{"<unk>"});
            assert(r[1] == ustr{"<unk>"});
        }

        {
            auto r = tokenize(bpe, "accb");
            assert(r.size() == 4);
            assert(r[0] == ustr{"a"});
            assert(r[1] == ustr{"<unk>"});
            assert(r[2] == ustr{"<unk>"});
            assert(r[3] == ustr{"b"});
        }
    }

    // fused <unk>
    {
        berts::bpe bpe{"<unk>", 0.0, true};
        vocab_t vocab{{
            {ustr{"a"}, 0},
            {ustr{"b"}, 1},
        }};
        merge_t merge{};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(ok);

        {
            auto r = tokenize(bpe, "c");
            assert(r.size() == 1);
            assert(r[0] == ustr{"<unk>"});
        }

        {
            auto r = tokenize(bpe, "cc");
            assert(r.size() == 2);
            assert(r[0] == ustr{"<unk>"});
            assert(r[1] == ustr{"<unk>"});
        }

        {
            auto r = tokenize(bpe, "accb");
            assert(r.size() == 3);
            assert(r[0] == ustr{"a"});
            assert(r[1] == ustr{"<unk>"});
            assert(r[2] == ustr{"b"});
        }
    }

    // merge and dropout
    {
        berts::bpe bpe{"<unk>", 0.0};
        vocab_t vocab{{
            {ustr{"u"}, 0},
            {ustr{"n"}, 1},
            {ustr{"r"}, 2},
            {ustr{"e"}, 3},
            {ustr{"l"}, 4},
            {ustr{"a"}, 5},
            {ustr{"t"}, 6},
            {ustr{"d"}, 7},
            {ustr{"re"}, 8},
            {ustr{"at"}, 9},
            {ustr{"ed"}, 10},
            {ustr{"un"}, 11},
            {ustr{"ated"}, 12},
            {ustr{"rel"}, 13},
            {ustr{"related"}, 14},
            {ustr{"unrelated"}, 15},
        }};
        merge_t merge{{
            {"r", "e"},
            {"a", "t"},
            {"e", "d"},
            {"u", "n"},
            {"at", "ed"},
            {"re", "l"},
            {"rel", "ated"},
            {"un", "related"},
        }};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(ok);

        {
            auto r = tokenize(bpe, "unrelated");
            assert(r.size() == 1);
            assert(r[0] == ustr{"unrelated"});
        }

        {
            bpe.dropout(1.0);
            auto r = tokenize(bpe, "unrelated");
            assert(r.size() == 9);
            assert(r[0] == ustr{"u"});
            assert(r[1] == ustr{"n"});
            assert(r[2] == ustr{"r"});
            assert(r[3] == ustr{"e"});
            assert(r[4] == ustr{"l"});
            assert(r[5] == ustr{"a"});
            assert(r[6] == ustr{"t"});
            assert(r[7] == ustr{"e"});
            assert(r[8] == ustr{"d"});
        }

        {
            bpe.dropout(0.5);
            auto r = tokenize(bpe, "unrelated");
            assert(0 < r.size());
            assert(r.size() < 9);
        }
    }

    // out of vocabulary in merges
    {
        berts::bpe bpe{"<unk>", 0.0, false};
        vocab_t vocab{{
            {ustr{"a"}, 0},
            {ustr{"b"}, 1},
            {ustr{"c"}, 2},
            {ustr{"ab"}, 3},
        }};
        merge_t merge{};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(!ok);
    }

    return 0;
}
