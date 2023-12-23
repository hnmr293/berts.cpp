#include "berts/models/log.hpp"

namespace berts::log {

log_level LOG_LEVEL = log_level::normal;

FILE *LOG_FILE = stderr;

void set_log_level(log_level level) {
    LOG_LEVEL = level;
}

log_level get_log_level() {
    return LOG_LEVEL;
}

bool is_logging(log_level level) {
    return static_cast<int>(LOG_LEVEL) <= static_cast<int>(level);
}

void set_log_file(FILE *file) {
    LOG_FILE = file;
}

FILE *get_log_file() {
    return LOG_FILE;
}

static inline void write(const std::string &msg) {
    if (LOG_FILE) {
        fprintf(LOG_FILE, msg.c_str());
        fprintf(LOG_FILE, "\n");
        fflush(LOG_FILE);
    }
}

template <log_level N>
static inline void write_if(const std::string &msg) {
    if ((int)LOG_LEVEL <= (int)N) {
        write(msg);
    }
}

void debug(const std::string &msg) {
    write_if<log_level::debug>(msg);
}

void info(const std::string &msg) {
    write_if<log_level::info>(msg);
}

void warn(const std::string &msg) {
    write_if<log_level::warn>(msg);
}

void error(const std::string &msg) {
    write_if<log_level::error>(msg);
}

} // namespace berts::log
