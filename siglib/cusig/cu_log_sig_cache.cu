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
#include "cu_log_sig_combine.h"

// =========================================================================
// SAFE_CALL macro
// =========================================================================

#include "cu_macros.h"

std::unordered_map<
	CuLogSigCacheKey, CUDALogSigCache, CuLogSigCacheKeyHash
>& get_cuda_log_sig_cache_map_() {
	static std::unordered_map<
		CuLogSigCacheKey, CUDALogSigCache, CuLogSigCacheKeyHash
	> cache;
	return cache;
}

std::mutex& get_cuda_log_sig_cache_mu_() {
	static std::mutex mu;
	return mu;
}

// =========================================================================
// Exported C functions
// =========================================================================

extern "C" {

	CUSIG_API int prepare_log_sig_cuda(
		uint64_t dimension, uint64_t degree, int method, bool use_disk
	) noexcept {
		CUSIG_SAFE_CALL(
			if (method != 3)
				prepare_log_sig_cuda_(dimension, degree, method, use_disk);
			if (method == 3)
				prepare_cuda_bch_cache_(dimension, degree)
		);
	}

	CUSIG_API int clear_cache_cuda(bool use_disk) noexcept {
		CUSIG_SAFE_CALL(clear_cache_cuda_(use_disk));
	}

	CUSIG_API int set_cache_dir_cuda(const char* dir) noexcept {
		CUSIG_SAFE_CALL(set_cache_dir_cuda_(dir));
	}
}
