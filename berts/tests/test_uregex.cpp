#include <cstdio>
#include "berts/berts.h"
#include "berts/models/unicode.hpp"

#define BERTS_TEST_SHORTHAND
#define BERTS_TEST_SHORTHAND_PREFIX
#include "berts/tests/tests.hpp"
#undef test

using namespace berts::unicode;

test_def {
    berts_test(uregex) {
        berts_set_log_level(BERTS_LOG_ALL);

        // regex::test
        testcase(test_1) {
            regex re1{"a"};
            ustr s1{"abc"};
            assert(re1);
            assert(re1.test(s1));
            ustr s2{"abc"};
            assert(re1.test(s2));
            ustr s3{"bac"};
            assert(re1.test(s3));
            ustr s4{"bcd"};
            assert(!re1.test(s4));
        };

        testcase(test_2) {
            regex re1{"^a"};
            ustr s1{"abc"};
            assert(re1);
            assert(re1.test(s1));
            ustr s2{"abc"};
            assert(re1.test(s2));
            ustr s3{"bac"};
            assert(!re1.test(s3));
            ustr s4{"bcd"};
            assert(!re1.test(s4));
        };

        testcase(test_3) {
            regex re1{"A"};
            ustr s1{"abc"};
            assert(re1);
            assert(!re1.test(s1));
            ustr s2{"abc"};
            assert(!re1.test(s2));
            ustr s3{"bac"};
            assert(!re1.test(s3));
            ustr s4{"bcd"};
            assert(!re1.test(s4));
            ustr s5{"Abc"};
            assert(re1.test(s5));
            ustr s6{"bAc"};
            assert(re1.test(s6));
        };

        // regex::split
        testcase(split_1) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"abc"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 2);
            assert(ss1[0] == ustr{"a"});
            assert(ss1[1] == ustr{"bc"});
        };

        testcase(split_2) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"bac"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 3);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"a"});
            assert(ss1[2] == ustr{"c"});
        };

        testcase(split_3) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"bca"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 2);
            assert(ss1[0] == ustr{"bc"});
            assert(ss1[1] == ustr{"a"});
        };

        testcase(split_4) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"bacdaef"};
            //       0122344
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 5);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"a"});
            assert(ss1[2] == ustr{"cd"});
            assert(ss1[3] == ustr{"a"});
            assert(ss1[4] == ustr{"ef"});
        };

        testcase(split_5) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"bcd"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 1);
            assert(ss1[0] == ustr{"bcd"});
        };

        testcase(split_6) {
            std::vector<ustr> ss1{};
            regex re1{"a"};
            ustr s1{"bAcdAef"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 1);
            assert(ss1[0] == ustr{"bAcdAef"});
        };

        testcase(split_7) {
            std::vector<ustr> ss1{};
            regex re1{"A"};
            ustr s1{"bAcdAef"};
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 5);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"A"});
            assert(ss1[2] == ustr{"cd"});
            assert(ss1[3] == ustr{"A"});
            assert(ss1[4] == ustr{"ef"});
        };

        testcase(split_8) {
            std::vector<ustr> ss1{};
            regex re1{"a."};
            ustr s1{"bacdaef"};
            //       0112334
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 5);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"ac"});
            assert(ss1[2] == ustr{"d"});
            assert(ss1[3] == ustr{"ae"});
            assert(ss1[4] == ustr{"f"});
        };

        testcase(split_9) {
            std::vector<ustr> ss1{};
            regex re1{"a|c"};
            ustr s1{"bacdaef"};
            //       0123455
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 6);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"a"});
            assert(ss1[2] == ustr{"c"});
            assert(ss1[3] == ustr{"d"});
            assert(ss1[4] == ustr{"a"});
            assert(ss1[5] == ustr{"ef"});
        };

        testcase(split_10) {
            std::vector<ustr> ss1{};
            regex re1{"(a|c)+"};
            ustr s1{"bacdaef"};
            //       0112344
            assert(re1);
            assert(re1.split(s1, ss1));
            assert(ss1.size() == 5);
            assert(ss1[0] == ustr{"b"});
            assert(ss1[1] == ustr{"ac"});
            assert(ss1[2] == ustr{"d"});
            assert(ss1[3] == ustr{"a"});
            assert(ss1[4] == ustr{"ef"});
        };
    };
};

int main() {
    run_tests();
    return 0;
}
