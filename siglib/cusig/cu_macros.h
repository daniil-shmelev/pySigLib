/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once

#include <charconv>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include "../shared/errors.h"

// Unified error-handling macro for all cusig exported functions.
// Error codes match pysiglib/error_codes.py:
//   0      = success
//   1      = bad_alloc (memory allocation failure)
//   2      = invalid_argument
//   3      = out_of_range
//   4      = filesystem_error
//   5      = cache not found
//   6      = directory does not exist
//   7      = default cache dir failure
//   8      = cache dir not set
//   9      = corrupted cache file
//   10     = generic runtime_error
//   11     = unknown exception
//   100000+= CUDA errors (100000 + cudaError_t code)

#define CUSIG_SAFE_CALL(function_call)                                  \
    try {                                                               \
        function_call;                                                  \
    }                                                                   \
    catch (const std::bad_alloc&) {                                     \
        std::cerr << "Failed to allocate memory";                       \
        return 1;                                                       \
    }                                                                   \
    catch (const std::invalid_argument& e) {                            \
        std::cerr << e.what();                                          \
        return 2;                                                       \
    }                                                                   \
    catch (const std::out_of_range& e) {                                \
        std::cerr << e.what();                                          \
        return 3;                                                       \
    }                                                                   \
    catch (const std::filesystem::filesystem_error& e) {                \
        std::cerr << e.what();                                          \
        return 4;                                                       \
    }                                                                   \
    catch (const coded_runtime_error& e) {                              \
        std::cerr << e.what();                                          \
        return e.code;                                                  \
    }                                                                   \
    catch (const std::runtime_error& e) {                               \
        std::string msg = e.what();                                     \
        std::cerr << msg;                                               \
        auto cuda_pos = msg.find("CUDA Error (");                       \
        if (cuda_pos != std::string::npos) {                            \
            const auto num_start = cuda_pos + 12;                       \
            const auto num_end = msg.find(')', num_start);              \
            if (num_end != std::string::npos) {                         \
                int code = 0;                                           \
                const char* beg = msg.data() + num_start;               \
                const char* end_ptr = msg.data() + num_end;             \
                if (std::from_chars(beg, end_ptr, code).ec == std::errc{}) \
                    return 100000 + code;                               \
            }                                                           \
        }                                                               \
        return 10;                                                      \
    }                                                                   \
    catch (...) {                                                       \
        std::cerr << "Unknown exception";                               \
        return 11;                                                      \
    }                                                                   \
    return 0;
