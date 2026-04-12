#pragma once
#include <stdexcept>
#include <string>

// Typed runtime errors for structured error code dispatch in SAFE_CALL macros.
// Throw these instead of plain std::runtime_error for cache/filesystem errors.
struct coded_runtime_error : std::runtime_error {
    int code;
    coded_runtime_error(int code, const std::string& msg) : std::runtime_error(msg), code(code) {}
};
struct cache_not_found_error : coded_runtime_error {
    cache_not_found_error(const std::string& msg) : coded_runtime_error(5, msg) {}
};
struct directory_not_found_error : coded_runtime_error {
    directory_not_found_error(const std::string& msg) : coded_runtime_error(6, msg) {}
};
struct default_cache_dir_error : coded_runtime_error {
    default_cache_dir_error(const std::string& msg) : coded_runtime_error(7, msg) {}
};
struct cache_dir_not_set_error : coded_runtime_error {
    cache_dir_not_set_error(const std::string& msg) : coded_runtime_error(8, msg) {}
};
struct corrupted_cache_error : coded_runtime_error {
    corrupted_cache_error(const std::string& msg) : coded_runtime_error(9, msg) {}
};
