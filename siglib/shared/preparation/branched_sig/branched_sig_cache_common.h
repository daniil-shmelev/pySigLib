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

#include "branched_sig_cache.h"
#include "../../trees/basis_counts.h"
#include "../../trees/coproduct.h"
#include "../../trees/tree.h"

namespace branched_sig_cache_detail {
void build_chain_indices_(
	BranchedSigCache& cache,
	const TreeTable& trees,
	const std::vector<Forest>* basis_forests = nullptr);
void build_tree_cuts_(
	TreeTable& trees,
	const std::vector<uint64_t>& order_offsets,
	uint64_t max_nodes,
	std::vector<std::vector<TreeCut>>& cuts);
}  // namespace branched_sig_cache_detail
