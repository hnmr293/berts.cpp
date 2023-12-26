#pragma once

#include <cstdio>
#include <string>
#include "berts/berts.h"

namespace berts::log {

void set_log_level(berts_log_level level);

berts_log_level get_log_level();

bool is_logging(berts_log_level level);

void set_log_file(FILE *file);

FILE *get_log_file();

template <typename Fn>
bool when(berts_log_level level, Fn &&fn) {
    if (is_logging(level)) {
        fn();
        return true;
    } else {
        return false;
    }
}

void debug(const std::string &msg);

void info(const std::string &msg);

void warn(const std::string &msg);

void error(const std::string &msg);

} // namespace berts::log
