#pragma once

/**
 * test utilities
 */

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// #ifdef _WIN32
// #include <cerrno>
// #include <fcntl.h>
// #include <io.h>
// #else
//// do nothing
// #endif

namespace berts::tests {

/* [usage 1]

#include <iostream>
#include "berts/tests/tests.hpp"

BERTS_TEST_DEFN() {
    BERTS_TEST(aaa) {
        BERTS_TESTCASE(a1) {
            std::cout << "x" << std::endl;
        };
        BERTS_TESTCASE(a2) {
            std::cout << "y" << std::endl;
        };
    };

    BERTS_TEST(bbb) {
        BERTS_TESTCASE(a1) {
            std::cout << "z" << std::endl;
        };
        BERTS_TESTCASE(a2) {
            std::cout << "w" << std::endl;
        };
    };
}

int main() {
    run_tests();
    return 0;
}

*/

/* [usage 2]

#define BERTS_TEST_SHORTHAND
#include <iostream>
#include "berts/tests/tests.hpp"

test_def {
    test(aaa) {
        testcase(a1) {
            std::cout << "x" << std::endl;
        };
        testcase(a2) {
            std::cout << "y" << std::endl;
        };
    };

    test(bbb) {
        testcase(a1) {
            std::cout << "z" << std::endl;
        };
        testcase(a2) {
            std::cout << "w" << std::endl;
        };
    };
}

int main() {
    run_tests();
    return 0;
}

*/

using __test_fn_t = std::function<void()>;

struct __test_case;
struct __test_container;
struct __test_driver;

struct __test_case {
    __test_case(__test_container *p, const char *testcasename)
        : __parent(p)
        , __testcasename(testcasename) {}

    template <typename Fn>
    void operator<<(Fn &&fn);

    __test_container *__parent;
    const char *__testcasename;
};

struct __test_container {
    __test_container(__test_driver *p, const char *testname)
        : __parent(p)
        , __testname(testname) {}

    template <typename Fn>
    void operator<<(Fn &&fn);

    template <typename Fn>
    void __add_test(const char *testcasename, Fn &&fn);

    __test_driver *__parent;
    const char *__testname;
};

struct __test_driver {
    __test_driver()
        : __testnames()
        , __testcases() {}

    ~__test_driver();

    template <typename Fn>
    void __add_test(const char *testname, const char *testcasename, Fn &&fn);

    std::vector<const char *> __testnames;
    std::unordered_map<const char *, std::vector<std::pair<const char *, __test_fn_t>>> __testcases;
};

template <typename Fn>
void __test_case::operator<<(Fn &&fn) {
    __parent->__add_test(__testcasename, fn);
}

template <typename Fn>
void __test_container::__add_test(const char *testcasename, Fn &&fn) {
    __parent->__add_test(__testname, testcasename, fn);
}

template <typename Fn>
void __test_driver::__add_test(const char *testname, const char *testcasename, Fn &&fn) {
    if (__testcases.contains(testname)) {
        __testcases[testname].emplace_back(testcasename, fn);
    } else {
        __testnames.push_back(testname);
        std::vector<std::pair<const char *, __test_fn_t>> to_add{{std::pair{testcasename, fn}}};
        __testcases[testname] = to_add;
    }
}

template <typename Fn>
void __test_container::operator<<(Fn &&fn) {
    fn();
}

__test_driver::~__test_driver() {
    // #ifdef _WIN32
    //     const int READ = 0;
    //     const int WRITE = 1;
    //     const int STDERR_FD = 2;
    //
    //     int p[2];
    //     if (_set_errno(0); _pipe(p, 0, _O_BINARY) != 0) {
    //         fprintf(stderr, _strerror(nullptr));
    //         std::exit(1);
    //     }
    //     int saved_stderr = _dup(STDERR_FD);
    //     _dup2(p[WRITE], STDERR_FD);
    // #else
    // #endif

    std::stringstream ss{};

    size_t ok = 0;
    size_t ng = 0;
    for (const auto &testname : __testnames) {
        ss << "\x1b[33m"
              "["
           << testname << "]"
                          "\x1b[0m"
                          "\n";

        std::vector<std::string> errors{};

        if (__testcases.contains(testname)) {
            for (const auto &[testcasename, fn] : __testcases[testname]) {
                try {
                    fn();
                    ss << "\x1b[1m\x1b[32m"
                          "OK"
                          "\x1b[0m";
                    ok += 1;
                } catch (const std::runtime_error &e) {
                    ss << "\x1b[1m\x1b[31m"
                          "NG"
                          "\x1b[0m";
                    // ss << e.what() << "\n";
                    errors.emplace_back(e.what());
                    ng += 1;
                } catch (const std::exception &e) {
                    std::cout << e.what() << std::endl;
                    std::cout << testname << ":" << testcasename << std::endl;
                    std::exit(1);
                } catch (...) {
                    std::cout << "unknown error" << std::endl;
                    std::cout << testname << ":" << testcasename << std::endl;
                    std::exit(1);
                }
                ss << " " << testcasename << "\n";
            }
        }

        for (auto &&e : errors) {
            ss << e << "\n";
        }

        ss << "\n";
    }

    std::cout << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "*** test results ***" << std::endl;
    std::cout << std::endl;
    std::cout << ss.str();
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << (ok + ng) << " tests, \x1b[32m" << ok << " success\x1b[0m, \x1b[31m" << ng << " fail\x1b[0m" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::endl;

    // #ifdef _WIN32
    //     fflush(stderr);
    //     _dup2(saved_stderr, STDERR_FD);
    //     _close(p[READ]);
    //     _close(p[WRITE]);
    // #else
    // #endif
}

namespace __test_assets {

struct __test_fail : public std::runtime_error {
    using inherited = std::runtime_error;

    const char *testname;
    const char *testcasename;
    const char *file;
    const char *func;
    const int lineno;
    std::string msg;

    __test_fail(const std::string &msg,
                const char *testname,
                const char *testcasename,
                const char *file,
                const char *func,
                int lineno)
        : inherited("")
        , testname(testname)
        , testcasename(testcasename)
        , file(file)
        , func(func)
        , lineno(lineno) {
        std::stringstream ss{};
        ss << testname << ":" << testcasename << " [" << file << ":" << lineno << "] " << msg;
        this->msg = ss.str();
    }

    const char *what() const noexcept override {
        return msg.c_str();
    }
};

template <std::convertible_to<bool> T>
inline void berts_assert_(T &&v, const char *expr, const char *testname, const char *testcasename, const char *file, const char *func, int lineno) {
    if (!v) {
        std::stringstream ss{};
        ss << "assertion failed: expression = " << expr;
        throw __test_fail{ss.str(), testname, testcasename, file, func, lineno};
    }
}

} // namespace __test_assets

#define BERTS_TEST_DEFN_FUNCTION __berts_test_main

#define BERTS_TEST_DEFN()                          \
    static berts::tests::__test_driver __driver{}; \
    static void BERTS_TEST_DEFN_FUNCTION()

#define BERTS_TEST(testname)                                                \
    berts::tests::__test_container __test_##testname{&__driver, #testname}; \
    __test_##testname << [__container = &__test_##testname](const char *__testname = #testname)

#define BERTS_TESTCASE(name)                                     \
    berts::tests::__test_case __test_##name{__container, #name}; \
    __test_##name << [ =, __testcasename = #name ]()

#define BERTS_TESTCASE_WITH(name, ...)                           \
    berts::tests::__test_case __test_##name{__container, #name}; \
    __test_##name << [ =, __testcasename = #name, __VA_ARGS__ ]()

#define BERTS_TEST_RUN()            \
    do {                            \
        BERTS_TEST_DEFN_FUNCTION(); \
    } while (0)

#define BERTS_ASSERT(expr)                          \
    do {                                            \
        berts::tests::__test_assets::berts_assert_( \
            (expr),                                 \
            #expr,                                  \
            __testname,                             \
            __testcasename,                         \
            __FILE__,                               \
            __func__,                               \
            __LINE__);                              \
    } while (0)

#ifdef BERTS_TEST_SHORTHAND
#define test_def BERTS_TEST_DEFN()
#define test(testname) BERTS_TEST(testname)
#define testcase(name) BERTS_TESTCASE(name)
#define testcase_with(name, ...) BERTS_TESTCASE_WITH(name, __VA_ARGS__)
#define run_tests() BERTS_TEST_RUN()
#define assert(expr) BERTS_ASSERT(expr)
#endif

#ifdef BERTS_TEST_SHORTHAND_PREFIX
#define berts_test_def BERTS_TEST_DEFN()
#define berts_test(testname) BERTS_TEST(testname)
#define berts_testcase(name) BERTS_TESTCASE(name)
#define berts_testcase_with(name, ...) BERTS_TESTCASE_WITH(name, __VA_ARGS__)
#define berts_run_tests() BERTS_TEST_RUN()
#define berts_assert(expr) BERTS_ASSERT(expr)
#endif

} // namespace berts::tests
