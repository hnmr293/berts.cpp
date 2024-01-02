#include <cassert>
#include "berts/models/unicode.hpp"

using namespace berts::unicode;

int main() {
    //
    // operator=, operator==
    //

    // alphabetic
    {
        const ustr a1{"a"};
        const ustr a2{"a"};
        const ustr b1{"b"};
        assert(a1 == a2);
        assert(a1 != b1);

        ustr c1{};
        c1 = a1;
        assert(a1 == c1);
        assert(b1 != c1);
    }

    // non-alphabetic
    {
        const ustr a1{"\xe3\x81\x82"}; // あ U+3042
        const ustr a2{"\xe3\x81\x82"}; // あ U+3042
        const ustr b1{"\xe3\x81\x8b"}; // か U+304b
        assert(a1 == a2);
        assert(a1 != b1);

        ustr c1{};
        c1 = a1;
        assert(a1 == c1);
        assert(b1 != c1);
    }

    //
    // operator+, operator+=
    //

    // alphabetic
    {
        const ustr a1{"a"};
        const ustr a2{"b"};
        const ustr a3{"c"};

        const ustr a = a1 + a2;
        const ustr b{"ab"};
        assert(a != a1);
        assert(a != a2);
        assert(a == b);

        ustr c = a;
        c += a3;
        const ustr d{"abc"};
        assert(c != a1);
        assert(c != a2);
        assert(c != a3);
        assert(c != b);
        assert(c == d);

        const ustr e = d + d;
        ustr f{"abcabc"};
        assert(e != a1);
        assert(e != a2);
        assert(e != a3);
        assert(e != b);
        assert(e != c);
        assert(e != d);
        assert(e == f);

        f += f;
        const ustr g{"abcabcabcabc"};
        assert(f == g);
    }

    // non-alphabetic
    {
        const ustr a1{"\xe3\x81\x82"}; // あ U+3042
        const ustr a2{"\xe3\x81\x83"}; // い U+3043
        const ustr a3{"\xe3\x81\x84"}; // う U+3044

        const ustr a = a1 + a2;
        const ustr b{"\xe3\x81\x82"
                     "\xe3\x81\x83"};
        assert(a != a1);
        assert(a != a2);
        assert(a == b);

        ustr c = a;
        c += a3;
        const ustr d{"\xe3\x81\x82"
                     "\xe3\x81\x83"
                     "\xe3\x81\x84"};
        assert(c != a1);
        assert(c != a2);
        assert(c != a3);
        assert(c != b);
        assert(c == d);

        const ustr e = d + d;
        ustr f{"\xe3\x81\x82"
               "\xe3\x81\x83"
               "\xe3\x81\x84"
               "\xe3\x81\x82"
               "\xe3\x81\x83"
               "\xe3\x81\x84"};
        assert(e != a1);
        assert(e != a2);
        assert(e != a3);
        assert(e != b);
        assert(e != c);
        assert(e != d);
        assert(e == f);

        f += f;
        const ustr g{"\xe3\x81\x82"
                     "\xe3\x81\x83"
                     "\xe3\x81\x84"
                     "\xe3\x81\x82"
                     "\xe3\x81\x83"
                     "\xe3\x81\x84"
                     "\xe3\x81\x82"
                     "\xe3\x81\x83"
                     "\xe3\x81\x84"
                     "\xe3\x81\x82"
                     "\xe3\x81\x83"
                     "\xe3\x81\x84"};
        assert(f == g);
    }

    //
    // NFC
    //

    // alphabetic composition
    {
        const ustr a1{"\x61\xcc\x80"}; // a U+0061 + Combining Grave Accent U+0300
        ustr a2;
        const ustr b1{"\xc3\xa0"}; // à U+00e0
        bool ok = normalize_nfc(a1, a2);
        assert(ok);
        assert(a2 == b1);
    }

    // non-alphabetic composition
    {
        const ustr a1{"\xe3\x81\x8b\xe3\x82\x99"}; // か U+304b + Combining Katakana-Hiragana Voiced Sound Mark U+3099
        ustr a2;
        const ustr b1{"\xe3\x81\x8c"}; // が U+304c
        bool ok = normalize_nfc(a1, a2);
        assert(ok);
        assert(a2 == b1);
    }

    //
    // NFD
    //

    // alphabetic decomposition
    {
        const ustr a1{"\xc3\xa0"}; // à U+00e0
        ustr a2;
        const ustr b1{"\x61\xcc\x80"}; // a U+0061 + Combining Grave Accent U+0300
        bool ok = normalize_nfd(a1, a2);
        assert(ok);
        assert(a2 == b1);
    }

    // non-alphabetic decomposition
    {
        const ustr a1{"\xe3\x81\x8c"}; // が U+304c
        ustr a2;
        const ustr b1{"\xe3\x81\x8b\xe3\x82\x99"}; // か U+304b + Combining Katakana-Hiragana Voiced Sound Mark U+3099
        bool ok = normalize_nfd(a1, a2);
        assert(ok);
        assert(a2 == b1);
    }

    return 0;
}
