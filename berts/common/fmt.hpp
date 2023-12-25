#pragma once

#ifdef BERTS_USE_FMTLIB_FMT
#include <fmt/core.h>
#else
#include <format>
#endif

namespace berts::fmt {

// std::v?format is a bit buggy in some systems such as w64devkit
template <typename... Args>
std::string fmt(const std::string_view fmt, Args &&...args) {
#ifdef BERTS_USE_FMTLIB_FMT
    std::string msg = fmt::vformat(fmt, fmt::make_format_args(args...));
#else
    std::string msg = std::vformat(fmt, std::make_format_args(args...));
#endif
    return msg;
}

}
