#pragma once

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

namespace welding {

inline const char *option_value(int &index, int argc, char **argv,
                                const char *option) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[index];
}

inline int parse_positive_int(const char *text, const char *name) {
    int value = 0;
    const char *end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value <= 0) {
        throw std::invalid_argument(
            std::string("invalid ") + name + ": " + text);
    }
    return value;
}

inline double parse_positive_double(const char *text, const char *name) {
    if (*text == '\0') {
        throw std::invalid_argument(
            std::string("invalid ") + name + ": empty value");
    }
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
            std::string("invalid ") + name + ": " + text);
    }
    return value;
}

inline std::uint64_t parse_uint64(const char *text, const char *name = "seed") {
    std::uint64_t value = 0;
    const char *end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument(
            std::string("invalid ") + name + ": " + text);
    }
    return value;
}

} // namespace welding
