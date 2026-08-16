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
#include "cp_branched_log_cache.h"

#include "cp_bch.h"
#include "disk_cache.h"

namespace {
struct BranchedLogSigCacheRegistry_ {
	std::unordered_map<
		std::pair<uint64_t, uint64_t>,
		std::unique_ptr<BranchedLogSigCache>,
		PairHash> map;
	std::shared_mutex mu;
};

BranchedLogSigCacheRegistry_& branched_log_sig_cache_registry_() {
	static BranchedLogSigCacheRegistry_ registry;
	return registry;
}
}  // namespace

void prepare_branched_log_sig_cache(
	const BranchedSigCache& cache,
	int method,
	bool use_disk
) {
	const auto key = make_branched_sig_cache_key(
		cache.dimension, cache.max_nodes, cache.planar);
	auto& registry = branched_log_sig_cache_registry_();
	{
		std::shared_lock lock(registry.mu);
		const auto found = registry.map.find(key);
		if (found != registry.map.end() && found->second->supports(method))
			return;
	}
	const std::filesystem::path cache_directory = use_disk
		? get_cache_dir() / cache_folder_name
		: std::filesystem::path{};
	std::unique_lock lock(registry.mu);
	const auto found = registry.map.find(key);
	if (found == registry.map.end()) {
		registry.map.try_emplace(
			key,
			std::make_unique<BranchedLogSigCache>(
				cache, method, cache_directory, use_disk));
	}
	else if (!found->second->supports(method))
		found->second->upgrade(cache, method, cache_directory, use_disk);
}

const BranchedLogSigCache& get_branched_log_sig_cache_(
	const BranchedSigCache& cache,
	int method
) {
	const auto key = make_branched_sig_cache_key(
		cache.dimension, cache.max_nodes, cache.planar);
	auto& registry = branched_log_sig_cache_registry_();
	std::shared_lock lock(registry.mu);
	const auto found = registry.map.find(key);
	if (found == registry.map.end() || !found->second->supports(method))
		throw cache_not_found_error(
			"Branched log signature cache not found - call prepare_branched_log_sig first");
	return *found->second;
}

void clear_branched_log_sig_cache() {
	auto& registry = branched_log_sig_cache_registry_();
	std::unique_lock lock(registry.mu);
	registry.map.clear();
}
