#define BERTS_TEST_SHORTHAND
#include "tests.hpp"
#include <iostream>

test_def {
    test(aaa) {
        testcase(a1) {
            std::cout << "x" << std::endl;
            assert(true);
        };
        testcase(a2) {
            std::cout << "y" << std::endl;
            assert(false);
        };
    };

    test(bbb) {
        testcase(a1) {
            std::cout << "z" << std::endl;
            assert(false);
        };
        testcase(a2) {
            std::cout << "w" << std::endl;
            assert(true);
        };
    };
}

int main() {
    run_tests();
    return 0;
}
