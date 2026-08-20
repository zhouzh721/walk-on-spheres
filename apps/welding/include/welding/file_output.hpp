#pragma once

#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace welding {

inline void ensure_directory(const std::filesystem::path &directory) {
    if (directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error(
            "could not create output directory: " + error.message());
    }
}

inline void ensure_parent_directory(const std::filesystem::path &path) {
    ensure_directory(path.parent_path());
}

} // namespace welding
