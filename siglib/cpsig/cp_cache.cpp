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

#include "cppch.h"
#include "cpsig.h"
#include "cp_bch.h"
#include "cp_branched_cache.h"
#include "cp_volterra_signature.h"
#include "disk_cache.h"
#include "log_sig_cache.h"
#include "macros.h"

namespace {

void clear_memory_caches() {
	clear_basis_cache();
	clear_bch_cache();
	clear_branched_sig_cache();
	clear_prepared_volterra_sig_cache();
}

void clear_cache_(bool use_disk) {
	auto dir = get_cache_dir();

	clear_memory_caches();

	if (use_disk)
		std::filesystem::remove_all(dir / cache_folder_name);
}

}  // anonymous namespace

extern "C" {

	CPSIG_API int set_cache_dir(const char* dir) noexcept {
		SAFE_CALL(set_cache_dir_(dir));
	}

	CPSIG_API int clear_cache(bool use_disk) noexcept {
		SAFE_CALL(clear_cache_(use_disk));
	}

	CPSIG_API void cpsig_shutdown() noexcept {
		try { clear_memory_caches(); } catch (...) {}
		try { clear_cache_dir_();    } catch (...) {}
	}

}
