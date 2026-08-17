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
#include "cp_branched_cache.h"
#include "disk_cache.h"
#include "macros.h"
#include "preparation/branched_sig/branched_sig_cache_io.h"

namespace {
// One cache per dimension, degree, and planar basis kind.
struct BranchedSigCacheRegistry {
	std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BranchedSigCache>,
		PairHash
	> map;
	std::shared_mutex mu;
};
BranchedSigCacheRegistry& branched_sig_cache_registry() {
	static BranchedSigCacheRegistry r;
	return r;
}
}  // anonymous namespace

// ---------------------------------------------------------------------------
// Cache construction
// ---------------------------------------------------------------------------

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool use_disk, bool planar) {
	const auto key = make_branched_sig_cache_key(dimension, max_nodes, planar);
	auto& reg = branched_sig_cache_registry();

	{
		std::shared_lock rlock(reg.mu);
		if (reg.map.find(key) != reg.map.end())
			return;
	}

	// Disk data uses the same flattened BranchedSigCache representation.
	if (use_disk) {
		auto cache = std::make_unique<BranchedSigCache>();
		bool cache_file_exists = read_branched_sig_cache(get_cache_dir() / cache_folder_name, dimension, max_nodes, planar, *cache);
		if (cache_file_exists) {
			cache->planar = planar;
			std::unique_lock wlock(reg.mu);
			reg.map.try_emplace(key, std::move(cache));
			return;
		}
	}

	// Compute from scratch using the shared constructor.
	auto cache = std::make_unique<BranchedSigCache>(dimension, max_nodes, planar);

	if (use_disk) {
		write_branched_sig_cache(get_cache_dir() / cache_folder_name, *cache);
	}

	// try_emplace: if a concurrent caller raced us to the write lock and
	// populated the slot first, keep theirs and drop our rebuild.
	std::unique_lock wlock(reg.mu);
	reg.map.try_emplace(key, std::move(cache));
}

const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool planar) {
	const auto key = make_branched_sig_cache_key(dimension, max_nodes, planar);
	auto& reg = branched_sig_cache_registry();
	std::shared_lock rlock(reg.mu);
	auto it = reg.map.find(key);
	if (it == reg.map.end()) {
		throw cache_not_found_error(
			"Branched signature cache not found - call prepare_branched_sig first");
	}
	return *(it->second);
}

void clear_branched_sig_cache() {
	auto& reg = branched_sig_cache_registry();
	std::unique_lock wlock(reg.mu);
	reg.map.clear();
}
