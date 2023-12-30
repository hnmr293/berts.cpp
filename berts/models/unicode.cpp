#include "berts/models/unicode.hpp"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <icu.h>
#else
#include <unicode/uclean.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include "unicode.hpp"
#endif

namespace berts::unicode {

struct ustr_impl {
    // in ICU lib, int32_t is used as a size type
    using size_type = int32_t;

    UChar *str;
    size_type size;
    UErrorCode e;

    ustr_impl()
        : str(nullptr)
        , e(U_ZERO_ERROR) {}

    ~ustr_impl() {
        free();
    }

    ustr_impl(const ustr_impl &other)
        : ustr_impl() {
        if (other.str) {
            alloc(other.size);
            std::copy(other.str, other.str + other.size, str);
        }
        e = other.e;
    }

    ustr_impl(ustr_impl &&other) noexcept
        : str(other.str)
        , size(other.size)
        , e(other.e) {
        other.str = nullptr;
        other.size = 0;
        other.e = U_ZERO_ERROR;
    }

    ustr_impl &operator=(const ustr_impl &other) {
        if (this != &other) {
            free();
            if (other.str) {
                alloc(other.size);
                std::copy(other.str, other.str + other.size, str);
            }
            e = other.e;
        }
        return *this;
    }

    ustr_impl &operator=(ustr_impl &&other) noexcept {
        if (this != &other) {
            free();
            str = other.str;
            size = other.size;
            e = other.e;
            other.str = nullptr;
            other.size = 0;
            other.e = U_ZERO_ERROR;
        }
        return *this;
    }

    void alloc() {
        alloc(size);
    }

    void alloc(size_type new_size) {
        free();
        if (new_size != 0) {
            str = new UChar[new_size];
        }
        size = new_size;
    }

    void free() {
        if (str) delete[] str;
        size = 0;
    }
};

#if 0
// ref. https://qiita.com/tomolatoon/items/3e14a3172261230ebe83
template <bool IsConst>
struct ustr_iterator {
    template <bool>
    friend struct ustr_iterator;

    using value_type = std::conditional_t<IsConst, const UChar, UChar>;
    using difference_type = ptrdiff_t;
    using iterator_concept = std::contiguous_iterator_tag;

    using I = ustr_iterator;
    using D = difference_type;

    ustr_iterator()
        : ptr(nullptr) {}

    ustr_iterator(UChar *p)
        : ptr(p) {}

    ustr_iterator(const ustr_iterator<!IsConst> &other)
        requires IsConst
        : ptr(other.ptr) {}

    ustr_iterator(ustr_iterator<!IsConst> &&other) noexcept
        requires IsConst
        : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    I &operator=(const ustr_iterator<!IsConst> &other)
        requires IsConst
    {
        if (this != &other) {
            ptr = other.ptr;
        }
        return *this;
    }

    I &operator=(ustr_iterator<!IsConst> &&other) noexcept
        requires IsConst
    {
        if (this != &other) {
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    value_type &operator*() const {
        return *ptr;
    }

    value_type &operator[](D d) const {
        return ptr[d];
    }

    value_type *operator->() const {
        return ptr;
    }

    I &operator++() {
        // ++it
        ptr += 1;
        return *this;
    }

    I operator++(int) {
        // it++
        auto tmp = *this;
        ptr += 1;
        return tmp;
    }

    I &operator--() {
        // --it
        ptr -= 1;
        return *this;
    }

    I operator--(int) {
        // it--
        auto tmp = *this;
        ptr -= 1;
        return tmp;
    }

    friend bool operator==(const I &lhs, const I &rhs) {
        return lhs.ptr == rhs.ptr;
    }

    friend std::strong_ordering operator<=>(const I &lhs, const I &rhs) {
        return lhs.ptr <=> rhs.ptr;
    }

    friend D operator-(const I &lhs, const I &rhs) {
        return lhs.ptr - rhs.ptr;
    }

    I &operator+=(D d) {
        ptr += d;
        return *this;
    }

    friend I operator+(const I &lhs, D d) {
        return {lhs.ptr + d};
    }

    friend I operator+(D d, const I &lhs) {
        return {lhs.ptr + d};
    }

    I &operator-=(D d) {
        ptr -= d;
        return *this;
    }

    friend I operator-(const I &lhs, D d) {
        return {lhs.ptr - d};
    }

    UChar *ptr;
};

static_assert(std::contiguous_iterator<ustr_iterator<true>>);
static_assert(std::contiguous_iterator<ustr_iterator<false>>);
static_assert(requires(
    ustr_iterator<false> a, 
    ustr_iterator<true> b
) {
    // 構築・代入
    ustr_iterator<true>{a};
    ustr_iterator<true>{std::move(a)};
    b = a;
    b = std::move(a);
});
#endif

ustr::ustr()
    : impl(new ustr_impl{}) {}

ustr::ustr(const char *in)
    : ustr(in, std::strlen(in)) {}

ustr::ustr(const char *in, size_t size)
    : ustr() {
    u_strFromUTF8(nullptr, 0, &impl->size, in, size, &impl->e);
    if (!U_SUCCESS(impl->e) && impl->e != U_BUFFER_OVERFLOW_ERROR) return;
    impl->e = U_ZERO_ERROR;
    impl->alloc();
    u_strFromUTF8(impl->str, impl->size, nullptr, in, size, &impl->e);
}

ustr::ustr(const unic_t *in, size_t count)
    : ustr() {
    impl->alloc(count);
    std::copy(in, in + count, impl->str);
}

ustr::ustr(const unic32_t *in, size_t count)
    : ustr() {
    u_strFromUTF32(nullptr, 0, &impl->size, in, count, &impl->e);
    if (!U_SUCCESS(impl->e) && impl->e != U_BUFFER_OVERFLOW_ERROR) return;
    impl->e = U_ZERO_ERROR;
    impl->alloc();
    u_strFromUTF32(impl->str, impl->size, nullptr, in, count, &impl->e);
}

// ustr::ustr(const std::string &in)
//     : ustr(in.c_str(), in.size()) {}

ustr::ustr(const ustr &in)
    : ustr() {
    *impl = *in.impl;
}

ustr::ustr(ustr &&in) noexcept
    : ustr() {
    *impl = std::move(*in.impl);
}

void ustr::dispose() {
    impl->free();
    impl->e = U_ZERO_ERROR;
}

ustr::~ustr() {
    dispose();
}

ustr &ustr::operator=(const ustr &in) {
    *impl = *in.impl;
    return *this;
}

ustr &ustr::operator=(ustr &&in) noexcept {
    *impl = std::move(*in.impl);
    return *this;
}

bool ustr::operator==(const ustr &rhs) const {
    if (impl->size != rhs.impl->size) {
        return false;
    }

    if (impl->size == 0) {
        return true;
    }

    return std::memcmp(impl->str, rhs.impl->str, impl->size * sizeof(UChar)) == 0;
}

std::strong_ordering ustr::operator<=>(const ustr &rhs_) const {
    // a <=> b
    //   a <  b <-> < 0
    //   a == b <-> = 0
    //   a  > b <-> > 0

    using o = std::strong_ordering;

    const auto lhs_size = impl->size;
    const auto rhs_size = rhs_.impl->size;
    const auto lhs = impl->str;
    const auto rhs = rhs_.impl->str;

    if (lhs_size == 0) {
        // "" <=> ""
        // "" <=> "a"
        return rhs_size == 0 ? o::equal : o::less;
    }

    if (rhs_size == 0) {
        // "a" <=> ""
        return o::greater;
    }

    for (ustr_impl::size_type i = 0; i < lhs_size; ++i) {
        if (rhs_size <= i) {
            // "abc" <=> "a"
            return o::greater;
        }
        const auto l = lhs[i];
        const auto r = rhs[i];
        if (l != r) return l <=> r;
    }

    // "abc" <=> "abc"
    // "a" <=> "abc"
    return lhs_size == rhs_size ? o::equal : o::less;
}

bool ustr::ok() const {
    return U_SUCCESS(impl->e);
}

bool ustr::empty() const {
    return !impl->str;
}

std::string ustr::encode() const {
    if (empty()) return "";

    std::string s{};
    int32_t size;
    u_strToUTF8(nullptr, 0, &size, impl->str, impl->size, &impl->e);
    if (!U_SUCCESS(impl->e) && impl->e != U_BUFFER_OVERFLOW_ERROR) return "";
    impl->e = U_ZERO_ERROR;
    s.resize(size);
    u_strToUTF8(s.data(), s.size(), nullptr, impl->str, impl->size, &impl->e);
    return s;
}

size_t ustr::bytesize() const {
    return impl->size * 2;
}

size_t ustr::packsize() const {
    return impl->size;
}

size_t ustr::codepoints() const {
    if (!impl->str || impl->size == 0) return 0;
    return (size_t)u_countChar32(impl->str, impl->size);
}

void ustr::pack_to(unic_t *buffer) const {
    std::copy(impl->str, impl->str + impl->size, buffer);
}

unic_t ustr::operator[](size_t index) const {
    if (index < (size_t)impl->size) {
        return impl->str[index];
    } else {
        return (unic_t)0xfffd;
    }
}

unic_t *ustr::begin() const {
    return impl->str;
}

unic_t *ustr::end() const {
    return impl->str ? (impl->str + impl->size) : nullptr;
}

bool normalize_nfc(const ustr &in, ustr &out) {
    UErrorCode e = U_ZERO_ERROR;
    auto k = unorm2_getNFCInstance(&e);
    if (!U_SUCCESS(e)) return false;

    UErrorCode e1 = U_ZERO_ERROR;
    auto len = unorm2_normalize(k, in.impl->str, in.impl->size, nullptr, 0, &e1);
    out.impl->alloc(len);

    unorm2_normalize(k, in.impl->str, in.impl->size, out.impl->str, out.impl->size, &e);

    return U_SUCCESS(e);
}

bool normalize_nfd(const ustr &in, ustr &out) {
    UErrorCode e = U_ZERO_ERROR;
    auto k = unorm2_getNFDInstance(&e);
    if (!U_SUCCESS(e)) return false;

    UErrorCode e1 = U_ZERO_ERROR;
    auto len = unorm2_normalize(k, in.impl->str, in.impl->size, nullptr, 0, &e1);
    out.impl->alloc(len);

    unorm2_normalize(k, in.impl->str, in.impl->size, out.impl->str, out.impl->size, &e);

    return U_SUCCESS(e);
}

bool is_whitespace(unic32_t c) {
    // doc says:
    //   true for U+0009 (TAB) and characters with general category "Zs" (space separators).
    // we treat '\t' as a whitespace, so everything is ok.
    return u_isblank(c);
}

bool is_control(unic32_t c) {
    // doc says:
    //   True for general categories <em>other</em> than "C" (controls).
    return !u_isprint(c);
}

bool is_punct(unic32_t c) {
    // according to transformers, we treat all
    // non-letter/number ASCII as punctuation
    // such as "^", "$", and "`"

    if ((c >= 33 && c <= 47) ||
        (c >= 58 && c <= 64) ||
        (c >= 91 && c <= 96) ||
        (c >= 123 && c <= 126))
        return true;

    const std::string k = category(c);
    return k.size() != 0 && k[0] == 'P';
}

const char *category(unic32_t c) {
    auto k = u_charType(c);
    switch (k) {
    case U_GENERAL_OTHER_TYPES: return "Cn";
    case U_UPPERCASE_LETTER: return "Lu";
    case U_LOWERCASE_LETTER: return "Ll";
    case U_TITLECASE_LETTER: return "Lt";
    case U_MODIFIER_LETTER: return "Lm";
    case U_OTHER_LETTER: return "Lo";
    case U_NON_SPACING_MARK: return "Mn";
    case U_ENCLOSING_MARK: return "Me";
    case U_COMBINING_SPACING_MARK: return "Mc";
    case U_DECIMAL_DIGIT_NUMBER: return "Nd";
    case U_LETTER_NUMBER: return "Nl";
    case U_OTHER_NUMBER: return "No";
    case U_SPACE_SEPARATOR: return "Zs";
    case U_LINE_SEPARATOR: return "Zl";
    case U_PARAGRAPH_SEPARATOR: return "Zp";
    case U_CONTROL_CHAR: return "Cc";
    case U_FORMAT_CHAR: return "Cf";
    case U_PRIVATE_USE_CHAR: return "Co";
    case U_SURROGATE: return "Cs";
    case U_DASH_PUNCTUATION: return "Pd";
    case U_START_PUNCTUATION: return "Ps";
    case U_END_PUNCTUATION: return "Pe";
    case U_CONNECTOR_PUNCTUATION: return "Pc";
    case U_OTHER_PUNCTUATION: return "Po";
    case U_MATH_SYMBOL: return "Sm";
    case U_CURRENCY_SYMBOL: return "Sc";
    case U_MODIFIER_SYMBOL: return "Sk";
    case U_OTHER_SYMBOL: return "So";
    case U_INITIAL_PUNCTUATION: return "Pi";
    case U_FINAL_PUNCTUATION: return "Pf";
    default: return "";
    }
}

bool is_category(unic32_t c, const char *cat) {
    const std::string k = category(c);
    return k == cat;
}

bool to_lower(const ustr &in, ustr &out) {
    out.dispose();
    auto size = u_strToLower(nullptr, 0, in.impl->str, in.impl->size, "", &in.impl->e);
    if (!U_SUCCESS(in.impl->e)) return false;
    out.impl->alloc(size);
    u_strToLower(out.impl->str, out.impl->size, in.impl->str, in.impl->size, "", &out.impl->e);
    return out.ok();
}

bool to_upper(const ustr &in, ustr &out) {
    out.dispose();
    auto size = u_strToUpper(nullptr, 0, in.impl->str, in.impl->size, "", &in.impl->e);
    if (!U_SUCCESS(in.impl->e)) return false;
    out.impl->alloc(size);
    u_strToUpper(out.impl->str, out.impl->size, in.impl->str, in.impl->size, "", &out.impl->e);
    return out.ok();
}

} // namespace berts::unicode
