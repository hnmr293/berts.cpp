#pragma once

#include <cstdio>
#include <string>

namespace berts::log {

enum struct log_level {
    all = 0,
    debug = 0,
    info = 1,
    warn = 2,
    normal = 2,
    error = 3,
    quiet = 10,
};

void set_log_level(log_level level);

log_level get_log_level();

bool is_logging(log_level level);

void set_log_file(FILE *file);

FILE *get_log_file();

template <typename Fn>
bool when(log_level level, Fn &&fn) {
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
