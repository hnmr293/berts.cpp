#include <cassert>
#include <cstdio>
#include "berts/berts.h"
#include "berts/models/unicode.hpp"

using namespace berts::unicode;

#define check_(expr, func, lineno)        \
    do {                                  \
        const auto &v = (expr);           \
        if (!v) {                         \
            fprintf(stderr,               \
                    "Assertion failed:\n" \
                    "[%s:%d] %s\n",       \
                    func,                 \
                    lineno,               \
                    #expr);               \
        }                                 \
    } while (0)

#define check(expr) check_(expr, __func__, __LINE__)

int main() {
    berts_set_log_level(BERTS_LOG_ALL);
    
    // regex::test
    {
        regex re1{"a"};
        ustr s1{"abc"};
        check(re1);
        check(re1.test(s1));
        ustr s2{"abc"};
        check(re1.test(s2));
        ustr s3{"bac"};
        check(re1.test(s3));
        ustr s4{"bcd"};
        check(!re1.test(s4));
    }

    {
        regex re1{"^a"};
        ustr s1{"abc"};
        check(re1);
        check(re1.test(s1));
        ustr s2{"abc"};
        check(re1.test(s2));
        ustr s3{"bac"};
        check(!re1.test(s3));
        ustr s4{"bcd"};
        check(!re1.test(s4));
    }

    {
        regex re1{"A"};
        ustr s1{"abc"};
        check(re1);
        check(!re1.test(s1));
        ustr s2{"abc"};
        check(!re1.test(s2));
        ustr s3{"bac"};
        check(!re1.test(s3));
        ustr s4{"bcd"};
        check(!re1.test(s4));
        ustr s5{"Abc"};
        check(re1.test(s5));
        ustr s6{"bAc"};
        check(re1.test(s6));
    }

    // regex::split
    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"abc"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 2);
        check(ss1[0] == ustr{"a"});
        check(ss1[1] == ustr{"bc"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"bac"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 3);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"a"});
        check(ss1[2] == ustr{"c"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"bca"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 2);
        check(ss1[0] == ustr{"bc"});
        check(ss1[1] == ustr{"a"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"bacdaef"};
        //       0122344
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 5);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"a"});
        check(ss1[2] == ustr{"cd"});
        check(ss1[3] == ustr{"a"});
        check(ss1[4] == ustr{"ef"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"bcd"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 1);
        check(ss1[0] == ustr{"bcd"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a"};
        ustr s1{"bAcdAef"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 1);
        check(ss1[0] == ustr{"bAcdAef"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"A"};
        ustr s1{"bAcdAef"};
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 5);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"A"});
        check(ss1[2] == ustr{"cd"});
        check(ss1[3] == ustr{"A"});
        check(ss1[4] == ustr{"ef"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a."};
        ustr s1{"bacdaef"};
        //       0112334
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 5);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"ac"});
        check(ss1[2] == ustr{"d"});
        check(ss1[3] == ustr{"ae"});
        check(ss1[4] == ustr{"f"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"a|c"};
        ustr s1{"bacdaef"};
        //       0123455
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 6);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"a"});
        check(ss1[2] == ustr{"c"});
        check(ss1[3] == ustr{"d"});
        check(ss1[4] == ustr{"a"});
        check(ss1[5] == ustr{"ef"});
    }

    {
        std::vector<ustr> ss1{};
        regex re1{"(a|c)+"};
        ustr s1{"bacdaef"};
        //       0112344
        check(re1);
        check(re1.split(s1, ss1));
        check(ss1.size() == 5);
        check(ss1[0] == ustr{"b"});
        check(ss1[1] == ustr{"ac"});
        check(ss1[2] == ustr{"d"});
        check(ss1[3] == ustr{"a"});
        check(ss1[4] == ustr{"ef"});
    }

    return 0;
}
