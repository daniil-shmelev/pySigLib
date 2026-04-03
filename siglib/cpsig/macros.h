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
#include <iostream>

//#ifndef __APPLE__
//	#define VEC
//#endif


#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
    #define RESTRICT __restrict
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
    #define RESTRICT __restrict__
#endif

// AVX2 intrinsics are available when VEC is defined and we're not on Apple
// (Apple Silicon uses NEON, not AVX)
#if defined(VEC) && !defined(__APPLE__)
    #define HAS_AVX2
#endif

// Error codes match pysiglib/error_codes.py:
//   0  = success
//   1  = bad_alloc (memory allocation failure)
//   2  = invalid_argument
//   3  = out_of_range
//   4  = filesystem_error
//   5  = cache not found
//   6  = directory does not exist
//   7  = default cache dir failure
//   8  = cache dir not set
//   9  = corrupted cache file
//   10 = generic runtime_error
//   11 = unknown exception

#define SAFE_CALL(function_call)                                        \
    try {                                                               \
        function_call;                                                  \
    }                                                                   \
    catch (std::bad_alloc&) {                                           \
        std::cerr << "Failed to allocate memory";                       \
        return 1;                                                       \
    }                                                                   \
    catch (std::invalid_argument& e) {                                  \
        std::cerr << e.what();                                          \
        return 2;                                                       \
    }                                                                   \
    catch (std::out_of_range& e) {                                      \
        std::cerr << e.what();                                          \
        return 3;                                                       \
    }                                                                   \
    catch (std::filesystem::filesystem_error& e) {                      \
        std::cerr << e.what();                                          \
        return 4;                                                       \
    }                                                                   \
    catch (std::runtime_error& e) {                                     \
        std::string msg = e.what();                                     \
        std::cerr << msg;                                               \
        if (msg == "Could not find basis cache")                        \
            return 5;                                                   \
        if (msg.rfind("Directory ", 0) == 0)                            \
            return 6;                                                   \
        if (msg == "Failed to get default cache directory.")             \
            return 7;                                                   \
        if (msg == "Unexpected internal error. Cache directory was not set correctly.") \
            return 8;                                                   \
        if (msg == "Tried to read an invalid cache file. Cache may have been corrupted.") \
            return 9;                                                   \
        return 10;                                                      \
    }                                                                   \
    catch (...) {                                                       \
        std::cerr << "Unknown exception";                               \
        return 11;                                                      \
    }                                                                   \
    return 0;
