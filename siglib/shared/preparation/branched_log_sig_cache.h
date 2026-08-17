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

#pragma once
#include "bch_cache.h"
#include "branched_log_plan.h"
#include "branched_sig_cache.h"
#include "log_sig_cache.h"

#include <filesystem>
#include <memory>
#include <vector>


struct BranchedBchCache {
	BranchedBchCache(
		const BranchedSigCache& branched_cache,
		const BranchedLogHornerPlan& horner_plan,
		const BasisCache& basis,
		bool use_disk
	);

	// Ordinary BCH data plus the sparse lift of one path increment.
	BchCache bch;
	// Multipliers and flat coordinates for the nonzero segment lift entries.
	std::vector<double> linear_coefficients;
	std::vector<uint64_t> linear_basis_idx;
};


class BranchedLogSigCache {
public:
	BranchedLogSigCache(
		const BranchedSigCache& branched_cache,
		int method,
		const std::filesystem::path& cache_directory = {},
		bool use_disk = false
	);

	bool supports(int method) const noexcept;
	void upgrade(
		const BranchedSigCache& branched_cache,
		int method,
		const std::filesystem::path& cache_directory = {},
		bool use_disk = false);
	const BranchedLogHornerPlan& horner_plan() const noexcept;
	const BasisCache& basis_cache(int method) const;
	const BranchedBchCache& bch_cache() const;

private:
	int method_ = -1;
	BranchedLogHornerPlan horner_plan_;
	std::unique_ptr<BasisCache> basis_cache_;
	std::unique_ptr<BranchedBchCache> bch_cache_;
};
