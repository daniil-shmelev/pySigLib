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

#include "cp_bch_cache.h"

void prepare_bch_cache(
	uint64_t dimension, uint64_t degree, bool use_disk
) {
	if (degree > BCH_MAX_HARDCODED_DEGREE)
		throw std::invalid_argument(
			"BCH methods support truncation degrees at most 20");
	prepare_basis_cache(dimension, degree, LogSigMethod::LyndonBasis, use_disk);
	LogSigCache& log_cache = get_log_sig_cache_mutable(
		dimension, degree, LogSigMethod::LyndonBasis);
	if (log_cache.has_bch())
		return;

	log_cache.set_bch(make_standard_bch_cache(
		dimension, degree, log_cache.basis(LogSigMethod::LyndonBasis)));
}

const BchCache& get_bch_cache(uint64_t dimension, uint64_t degree) {
	return get_log_sig_cache(dimension, degree, LogSigMethod::DirectBCH).bch();
}

void clear_bch_cache() {
	// BCH data is owned by the LogSigCache registry.
}
