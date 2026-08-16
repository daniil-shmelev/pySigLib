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
#include "cp_branched_sig_coef_cache.h"
#include "disk_cache.h"
#include "errors.h"

namespace {
struct BranchedSigCoefCacheKey {
	uint64_t data_dimension = 0;
	uint64_t dimension = 0;
	uint64_t max_nodes = 0;
	bool planar = false;
	std::vector<uint64_t> tree_data;

	bool operator==(const BranchedSigCoefCacheKey& other) const noexcept {
		return data_dimension == other.data_dimension
			&& dimension == other.dimension
			&& max_nodes == other.max_nodes
			&& planar == other.planar
			&& tree_data == other.tree_data;
	}
};

struct BranchedSigCoefCacheKeyHash {
	size_t operator()(const BranchedSigCoefCacheKey& key) const noexcept {
		size_t h = std::hash<uint64_t>{}(key.data_dimension);
		auto combine = [&h](uint64_t value) {
			h ^= std::hash<uint64_t>{}(value) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
		};
		combine(key.dimension);
		combine(key.max_nodes);
		combine(static_cast<uint64_t>(key.planar));
		for (uint64_t value : key.tree_data)
			combine(value);
		return h;
	}
};

struct BranchedSigCoefCacheRegistry {
	std::unordered_map<
		BranchedSigCoefCacheKey,
		BranchedSigCoefCache,
		BranchedSigCoefCacheKeyHash
	> map;
	std::shared_mutex mu;
};

BranchedSigCoefCacheRegistry& branched_sig_coef_cache_registry() {
	static BranchedSigCoefCacheRegistry registry;
	return registry;
}

BranchedSigCoefCacheKey make_branched_sig_coef_cache_key(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	if (tree_data == nullptr && tree_data_len != 0)
		throw std::invalid_argument("branched_sig_coef received null tree_data");
	BranchedSigCoefCacheKey key;
	key.data_dimension = data_dimension;
	key.dimension = dimension;
	key.max_nodes = max_nodes;
	key.planar = planar;
	if (tree_data_len != 0)
		key.tree_data.assign(tree_data, tree_data + tree_data_len);
	return key;
}
}  // anonymous namespace

void prepare_branched_sig_coef_cache(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar,
	bool use_disk
) {
	if (data_dimension == 0)
		throw std::invalid_argument("prepare_branched_sig_coef received dimension 0");
	auto key = make_branched_sig_coef_cache_key(tree_data, tree_data_len,
		data_dimension, dimension, max_nodes, planar);
	auto& registry = branched_sig_coef_cache_registry();
	{
		std::shared_lock rlock(registry.mu);
		if (registry.map.find(key) != registry.map.end())
			return;
	}

	BranchedSigCoefCache cache;
	std::filesystem::path cache_dir;
	if (use_disk) {
		cache_dir = get_cache_dir() / cache_folder_name;
		std::filesystem::create_directories(cache_dir);
		if (read_branched_sig_coef_cache(
			cache_dir, data_dimension, dimension, max_nodes, planar,
			key.tree_data, cache)) {
			std::unique_lock wlock(registry.mu);
			registry.map.try_emplace(std::move(key), std::move(cache));
			return;
		}
	}

	cache = BranchedSigCoefCache(
		key.tree_data.data(), key.tree_data.size(), data_dimension, dimension,
		max_nodes, planar);
	if (use_disk)
		write_branched_sig_coef_cache(
			cache_dir, data_dimension, dimension, max_nodes, planar,
			key.tree_data, cache);

	std::unique_lock wlock(registry.mu);
	registry.map.try_emplace(std::move(key), std::move(cache));
}

const BranchedSigCoefCache& get_branched_sig_coef_cache(
	const uint64_t* tree_data,
	uint64_t tree_data_len,
	uint64_t data_dimension,
	uint64_t dimension,
	uint64_t max_nodes,
	bool planar
) {
	const auto key = make_branched_sig_coef_cache_key(tree_data, tree_data_len,
		data_dimension, dimension, max_nodes, planar);
	auto& registry = branched_sig_coef_cache_registry();
	std::shared_lock rlock(registry.mu);
	const auto it = registry.map.find(key);
	if (it == registry.map.end())
		throw cache_not_found_error(
			"Branched signature coefficient cache not found - call prepare_branched_sig_coef first");
	return it->second;
}

void clear_branched_sig_coef_cache() {
	auto& registry = branched_sig_coef_cache_registry();
	std::unique_lock wlock(registry.mu);
	registry.map.clear();
}
