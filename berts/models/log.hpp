#pragma once

#include <cstdio>
#include <string>
#include "berts/berts.h"
#include "berts/models/fmt.hpp"

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

template <typename... Args>
void debug(const std::string &fmt, Args... args) {
    if (is_logging(BERTS_LOG_DEBUG)) {
        debug(berts::fmt::fmt(fmt, args...));
    }
}

template <typename... Args>
void info(const std::string &fmt, Args... args) {
    if (is_logging(BERTS_LOG_INFO)) {
        info(berts::fmt::fmt(fmt, args...));
    }
}

template <typename... Args>
void warn(const std::string &fmt, Args... args) {
    if (is_logging(BERTS_LOG_WARN)) {
        warn(berts::fmt::fmt(fmt, args...));
    }
}

template <typename... Args>
void error(const std::string &fmt, Args... args) {
    if (is_logging(BERTS_LOG_INFO)) {
        error(berts::fmt::fmt(fmt, args...));
    }
}

} // namespace berts::log
