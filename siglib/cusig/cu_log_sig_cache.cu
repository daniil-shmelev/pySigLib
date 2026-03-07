/* Copyright 2026 Daniil Shmelev
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

#include "cupch.h"
#include "cusig.h"
#include "cu_log_sig_cache.h"

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#ifndef CU_LOG_SIG_CACHE_SAFE_CALL
#define CU_LOG_SIG_CACHE_SAFE_CALL(function_call)                   \
    try {                                                           \
        function_call;                                              \
    }                                                               \
    catch (std::bad_alloc&) {                                       \
        std::cerr << "Failed to allocate memory";                   \
        return 1;                                                   \
    }                                                               \
    catch (std::invalid_argument& e) {                              \
        std::cerr << e.what();                                      \
        return 2;                                                   \
    }                                                               \
    catch (std::out_of_range& e) {                                  \
        std::cerr << e.what();                                      \
        return 3;                                                   \
    }                                                               \
    catch (std::runtime_error& e) {                                 \
        std::string msg = e.what();                                 \
        std::regex pattern(R"(CUDA Error \((\d+)\):)");             \
        std::smatch match;                                          \
        int ret_code = 10;                                          \
        if (std::regex_search(msg, match, pattern)) {               \
            ret_code = 100000 + std::stoi(match[1]);                \
        }                                                           \
        std::cerr << e.what();                                      \
        return ret_code;                                            \
    }                                                               \
    catch (...) {                                                   \
        std::cerr << "Unknown exception";                           \
        return 11;                                                  \
    }                                                               \
    return 0;
#endif

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int prepare_log_sig_cuda(
		uint64_t dimension, uint64_t degree, int method
	) noexcept {
		CU_LOG_SIG_CACHE_SAFE_CALL(prepare_log_sig_cuda_(dimension, degree, method));
	}

	CUSIG_API int clear_cache_cuda() noexcept {
		CU_LOG_SIG_CACHE_SAFE_CALL(clear_cache_cuda_());
	}
}
