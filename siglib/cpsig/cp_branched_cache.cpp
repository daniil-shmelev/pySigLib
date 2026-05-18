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

namespace {
// max_nodes is combinatorially bounded (dozens at most), so bit 63 is free for
// the planar flag. Packing it lets us reuse PairHash instead of maintaining a
// parallel tuple-hash.
inline std::pair<uint64_t, uint64_t> make_key(uint64_t dimension, uint64_t max_nodes, bool planar) {
	return { dimension, max_nodes | (static_cast<uint64_t>(planar) << 63) };
}

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

// Cut enumeration and cache building are in shared/branched_cache.h

// ---------------------------------------------------------------------------
// Disk cache serialization
// ---------------------------------------------------------------------------

static constexpr const char* branched_cache_version = "v3";

static std::filesystem::path branched_cache_file_path(uint64_t dimension, uint64_t max_nodes, bool planar) {
	const char* prefix = planar ? "planar_branched_" : "branched_";
	return get_cache_dir() / cache_folder_name /
		(prefix + std::to_string(dimension) + "_" + std::to_string(max_nodes) + "_" + branched_cache_version + ".bin");
}

static void write_branched_cache(const BranchedSigCache& c) {
	auto dir = get_cache_dir() / cache_folder_name;
	if (!std::filesystem::exists(dir))
		std::filesystem::create_directory(dir);

	std::ofstream out(branched_cache_file_path(c.dimension, c.max_nodes, c.planar), std::ios::binary);
	if (!out) return;

	out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
	out.write(reinterpret_cast<const char*>(&c.dimension), sizeof(c.dimension));
	out.write(reinterpret_cast<const char*>(&c.max_nodes), sizeof(c.max_nodes));
	out.write(reinterpret_cast<const char*>(&c.total_length), sizeof(c.total_length));
	serialize_vector(out, c.order_index);

	uint64_t n = c.inv_tree_factorial.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0) out.write(reinterpret_cast<const char*>(c.inv_tree_factorial.data()), n * sizeof(double));

	n = c.node_labels_data.size();
	out.write(reinterpret_cast<const char*>(&n), sizeof(n));
	if (n > 0) out.write(reinterpret_cast<const char*>(c.node_labels_data.data()), n);

	serialize_vector(out, c.node_labels_offsets);
	serialize_vector(out, c.chain_index_offsets);
	serialize_vector(out, c.chain_indices);
	serialize_vector(out, c.coproduct_data);
	serialize_vector(out, c.coproduct_offsets);
}

static bool read_branched_cache(uint64_t dimension, uint64_t max_nodes, bool planar, BranchedSigCache& c) {
	auto path = branched_cache_file_path(dimension, max_nodes, planar);
	if (!std::filesystem::exists(path)) return false;

	std::ifstream in(path, std::ios::binary);
	if (!in) return false;

	// Read into a fresh local first; only move into `c` on full success.
	BranchedSigCache tmp;

	uint64_t magic;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != cache_magic_number)
		throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");

	in.read(reinterpret_cast<char*>(&tmp.dimension), sizeof(tmp.dimension));
	in.read(reinterpret_cast<char*>(&tmp.max_nodes), sizeof(tmp.max_nodes));
	if (!in || tmp.dimension != dimension || tmp.max_nodes != max_nodes)
		return false;
	in.read(reinterpret_cast<char*>(&tmp.total_length), sizeof(tmp.total_length));
	if (!in || tmp.total_length > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error("Tried to read an invalid cache file: branched total_length exceeds limit");

	deserialize_vector(in, tmp.order_index);

	// inv_tree_factorial: bounded by total_length - 1 (num_trees).
	uint64_t n;
	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > MAX_CACHE_VECTOR_SIZE || n + 1 > tmp.total_length)
		throw std::runtime_error("Tried to read an invalid cache file: branched inv_tree_factorial size invalid");
	check_stream_has_bytes(in, n * sizeof(double), "branched inv_tree_factorial body");
	tmp.inv_tree_factorial.resize(n);
	if (n > 0) in.read(reinterpret_cast<char*>(tmp.inv_tree_factorial.data()), n * sizeof(double));

	// node_labels_data: raw bytes, also bounded.
	in.read(reinterpret_cast<char*>(&n), sizeof(n));
	if (!in || n > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error("Tried to read an invalid cache file: branched node_labels_data size invalid");
	check_stream_has_bytes(in, n, "branched node_labels_data body");
	tmp.node_labels_data.resize(n);
	if (n > 0) in.read(reinterpret_cast<char*>(tmp.node_labels_data.data()), n);

	deserialize_vector(in, tmp.node_labels_offsets);
	deserialize_vector(in, tmp.chain_index_offsets);
	deserialize_vector(in, tmp.chain_indices);
	deserialize_vector(in, tmp.coproduct_data);
	deserialize_vector(in, tmp.coproduct_offsets);

	if (!in.good()) return false;

	c = std::move(tmp);
	return true;
}

// ---------------------------------------------------------------------------
// Cache construction
// ---------------------------------------------------------------------------

void prepare_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool use_disk, bool planar) {
	const auto key = make_key(dimension, max_nodes, planar);
	auto& reg = branched_sig_cache_registry();

	{
		std::shared_lock rlock(reg.mu);
		if (reg.map.find(key) != reg.map.end())
			return;
	}

	// Try loading from disk
	if (use_disk) {
		auto cache = std::make_unique<BranchedSigCache>();
		if (read_branched_cache(dimension, max_nodes, planar, *cache)) {
			cache->planar = planar;
			std::unique_lock wlock(reg.mu);
			reg.map.try_emplace(key, std::move(cache));
			return;
		}
	}

	// Compute from scratch using shared builder (expensive - can take seconds).
	auto cache = std::make_unique<BranchedSigCache>(build_branched_sig_cache(dimension, max_nodes, planar));

	if (use_disk) {
		write_branched_cache(*cache);
	}

	// try_emplace: if a concurrent caller raced us to the write lock and
	// populated the slot first, keep theirs and drop our rebuild.
	std::unique_lock wlock(reg.mu);
	reg.map.try_emplace(key, std::move(cache));
}

const BranchedSigCache& get_branched_sig_cache(uint64_t dimension, uint64_t max_nodes, bool planar) {
	const auto key = make_key(dimension, max_nodes, planar);
	auto& reg = branched_sig_cache_registry();
	std::shared_lock rlock(reg.mu);
	auto it = reg.map.find(key);
	if (it == reg.map.end()) {
		throw std::runtime_error("Branched signature cache not found. Call prepare_branched_sig first.");
	}
	return *(it->second);
}

void clear_branched_sig_cache() {
	auto& reg = branched_sig_cache_registry();
	std::unique_lock wlock(reg.mu);
	reg.map.clear();
}
