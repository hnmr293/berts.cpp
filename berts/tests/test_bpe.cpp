#include <cassert>
#include <string>
#include <vector>
#include "berts/berts.h"
#include "berts/models/bpe.hpp"

#define TOKENIZE(var, bpe, token)                 \
    berts::bpe::tokenized_t var{};                \
    {                                             \
        bool ok = (bpe).tokenize((token), (var)); \
        assert(ok);                               \
    }

int main() {
    using str_t = berts::bpe::str_t;
    using vocab_t = berts::bpe::vocab_t;
    using merge_t = std::vector<berts::bpe::token_pair>;

    berts_set_log_level(BERTS_LOG_ALL);

    // unfused <unk>
    {
        berts::bpe bpe{"<unk>", 0.0, false};
        vocab_t vocab{{
            {str_t{"<unk>"}, 0},
            {str_t{"a"}, 1},
            {str_t{"b"}, 2},
        }};
        merge_t merge{};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(ok);

        {
            TOKENIZE(r, bpe, "c");
            assert(r.size() == 1);
            assert(r[0] == str_t{"<unk>"});
        }

        {
            TOKENIZE(r, bpe, "cc");
            assert(r.size() == 2);
            assert(r[0] == str_t{"<unk>"});
            assert(r[1] == str_t{"<unk>"});
        }

        {
            TOKENIZE(r, bpe, "accb");
            assert(r.size() == 4);
            assert(r[0] == str_t{"a"});
            assert(r[1] == str_t{"<unk>"});
            assert(r[2] == str_t{"<unk>"});
            assert(r[3] == str_t{"b"});
        }
    }

    // fused <unk>
    {
        berts::bpe bpe{"<unk>", 0.0, true};
        vocab_t vocab{{
            {str_t{"<unk>"}, 0},
            {str_t{"a"}, 1},
            {str_t{"b"}, 2},
        }};
        merge_t merge{};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(ok);

        {
            TOKENIZE(r, bpe, "c");
            assert(r.size() == 1);
            assert(r[0] == str_t{"<unk>"});
        }

        {
            TOKENIZE(r, bpe, "cc");
            assert(r.size() == 1);
            assert(r[0] == str_t{"<unk>"});
        }

        {
            TOKENIZE(r, bpe, "accb");
            assert(r.size() == 3);
            assert(r[0] == str_t{"a"});
            assert(r[1] == str_t{"<unk>"});
            assert(r[2] == str_t{"b"});
        }
    }

    // merge and dropout
    {
        berts::bpe bpe{"<unk>", 0.0};
        vocab_t vocab{{
            {str_t{"u"}, 0},
            {str_t{"n"}, 1},
            {str_t{"r"}, 2},
            {str_t{"e"}, 3},
            {str_t{"l"}, 4},
            {str_t{"a"}, 5},
            {str_t{"t"}, 6},
            {str_t{"d"}, 7},
            {str_t{"re"}, 8},
            {str_t{"at"}, 9},
            {str_t{"ed"}, 10},
            {str_t{"un"}, 11},
            {str_t{"ated"}, 12},
            {str_t{"rel"}, 13},
            {str_t{"related"}, 14},
            {str_t{"unrelated"}, 15},
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
            TOKENIZE(r, bpe, "unrelated");
            assert(r.size() == 1);
            assert(r[0] == str_t{"unrelated"});
        }

        {
            bpe.dropout(1.0);
            TOKENIZE(r, bpe, "unrelated");
            assert(r.size() == 9);
            assert(r[0] == str_t{"u"});
            assert(r[1] == str_t{"n"});
            assert(r[2] == str_t{"r"});
            assert(r[3] == str_t{"e"});
            assert(r[4] == str_t{"l"});
            assert(r[5] == str_t{"a"});
            assert(r[6] == str_t{"t"});
            assert(r[7] == str_t{"e"});
            assert(r[8] == str_t{"d"});
        }

        {
            bpe.dropout(0.5);
            TOKENIZE(r, bpe, "unrelated");
            assert(0 < r.size());
            assert(r.size() < 9);
        }
    }

    // out of vocabulary in merges
    {
        berts::bpe bpe{"<unk>", 0.0, false};
        vocab_t vocab{{
            {str_t{"a"}, 0},
            {str_t{"b"}, 1},
            {str_t{"c"}, 2},
            {str_t{"ab"}, 3},
        }};
        merge_t merge{{
            {str_t{"a"}, str_t{"b"}},
            {str_t{"a"}, str_t{"d"}},
        }};
        bool ok = bpe.load_vocab(vocab, merge);
        assert(!ok);
    }

    return 0;
}
