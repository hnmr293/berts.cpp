#include "berts/common/log.hpp"

namespace berts::log {

berts_log_level LOG_LEVEL = berts_log_level::BERTS_LOG_DEFAULT;

FILE *LOG_FILE = stderr;

void set_log_level(berts_log_level level) {
    LOG_LEVEL = level;
}

berts_log_level get_log_level() {
    return LOG_LEVEL;
}

bool is_logging(berts_log_level level) {
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

template <berts_log_level N>
static inline void write_if(const std::string &msg) {
    if ((int)LOG_LEVEL <= (int)N) {
        write(msg);
    }
}

void debug(const std::string &msg) {
    write_if<berts_log_level::BERTS_LOG_DEBUG>(msg);
}

void info(const std::string &msg) {
    write_if<berts_log_level::BERTS_LOG_INFO>(msg);
}

void warn(const std::string &msg) {
    write_if<berts_log_level::BERTS_LOG_WARN>(msg);
}

void error(const std::string &msg) {
    write_if<berts_log_level::BERTS_LOG_ERROR>(msg);
}

} // namespace berts::log
