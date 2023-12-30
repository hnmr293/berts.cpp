#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <ranges>
#include <string>

namespace berts::unicode {

struct ustr;

using unic_t = char16_t;
using unic32_t = int32_t;

bool normalize_nfc(const ustr &in, ustr &out);

bool normalize_nfd(const ustr &in, ustr &out);

// '\t' + Zs
bool is_whitespace(unic32_t c);

// C*
bool is_control(unic32_t c);

// P*
bool is_punct(unic32_t c);

const char *category(unic32_t c);

bool is_category(unic32_t c, const char *cat);

bool to_lower(const ustr &in, ustr &out);

bool to_upper(const ustr &in, ustr &out);

struct ustr_impl;

// UTF-16 string buffer
struct ustr {
    ustr();
    ustr(const char *in_utf8);
    ustr(const char *in_utf8, size_t size);
    ustr(const unic_t *in_utf16, size_t count);
    ustr(const unic32_t *in_utf32, size_t count);
    ustr(const ustr &in);
    ustr(ustr &&in) noexcept;

    template <std::ranges::range View>
    ustr(const View &range)
        : ustr(range.data(), range.size()) {}

    ~ustr();

    ustr &operator=(const ustr &in);
    ustr &operator=(ustr &&in) noexcept;

    bool operator==(const ustr &rhs) const;

    std::strong_ordering operator<=>(const ustr &rhs) const;

    bool ok() const;

    void dispose();

    bool empty() const;

    std::string encode() const;

    size_t bytesize() const;

    size_t packsize() const;

    size_t codepoints() const;

    void pack_to(unic_t *buffer) const;

    unic_t operator[](size_t index) const;

    unic_t *begin() const;

    unic_t *end() const;

    template <typename Fn>
    void each(Fn &&fn) const {
        const size_t n = packsize();
        for (size_t i = 0; i < n; ++i) {
            fn((*this)[i]);
        }
    }

    struct cp {
        unic32_t c;
        unic_t hi, lo;
        cp(unic_t c)
            : c((unic32_t)c)
            , hi(0)
            , lo(0) {}
        cp(unic32_t c, unic_t hi, unic_t lo)
            : c(c)
            , hi(hi)
            , lo(lo) {}
        bool is_pair() const { return hi != 0 || lo != 0; }
    };

    template <typename Fn>
    void each_cp(bool skip_invalid, Fn &&fn) const {
        char16_t surrogate = 0;

        each([&surrogate, skip_invalid, fn = std::move(fn)](unic_t c) {
            if (0xdc00 <= c && c < 0xe000) {
                // low surrogate
                if (surrogate == 0) {
                    // no high surrogate, invalid sequence!
                    // treat as U+FFFD
                    if (!skip_invalid) fn(cp{0xfffd, 0, c});
                    return;
                }
                // now we have complete surrogate pair
                // continue to process
            } else {
                if (surrogate != 0) {
                    // lone high surrogate, invalid sequence!
                    // treat as U+FFFD
                    if (!skip_invalid) fn(cp{0xfffd, surrogate, 0});
                    surrogate = 0;
                }

                if (0xd800 <= c && c < 0xdc00) {
                    // high surrogate
                    surrogate = c;
                    return;
                }
            }

            if (surrogate) {
                unic_t h = surrogate;
                unic_t l = c;
                unic32_t C = ((h - 0xd800) << 10) | (l - 0xdc00);
                fn(cp{C + 0x10000, surrogate, c});
                surrogate = 0;
            } else {
                fn(cp{c});
            }
        });

        if (surrogate != 0) {
            // lone high surrogate, invalid sequence!
            // treat as U+FFFD
            if (!skip_invalid) fn(cp{0xfffd, surrogate, 0});
        }
    }

    ustr_impl *impl;
};

} // namespace berts::unicode

namespace std {

template <>
struct hash<berts::unicode::ustr> {
    size_t operator()(const berts::unicode::ustr &s) const {
        size_t v = 0;
        for (const auto c : s) {
            v ^= std::hash<berts::unicode::unic_t>{}(c);
        }
        return v;
    }
};

} // namespace std
