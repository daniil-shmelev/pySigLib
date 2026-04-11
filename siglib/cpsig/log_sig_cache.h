/* Copyright 2025 Daniil Shmelev
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
#include "cppch.h"
#include "cp_tensor_poly.h"
#include "words.h"
#include "sparse.h"
#include "disk_cache.h"

struct BasisCache {
	int method;
	std::vector<uint64_t> lyndon_idx;
	SparseIntMatrix inv_proj_mat;
	SparseIntMatrix inv_proj_mat_transpose;

	BasisCache() {}

	BasisCache(
		int method_,
		std::vector<uint64_t>&& lyndon_idx_,
		SparseIntMatrix&& inv_proj_mat_,
		SparseIntMatrix&& inv_proj_mat_transpose_
	) : method{ method_ },
		lyndon_idx{ std::move(lyndon_idx_) },
		inv_proj_mat{ std::move(inv_proj_mat_) },
		inv_proj_mat_transpose{ std::move(inv_proj_mat_transpose_) } {
	}

	void serialize(std::ostream& out) const {
		out.write(reinterpret_cast<const char*>(&method), sizeof(method));
		serialize_vector(out, lyndon_idx);
		inv_proj_mat.serialize(out);
		inv_proj_mat_transpose.serialize(out);
	}

	void deserialize(std::istream& in) {
		in.read(reinterpret_cast<char*>(&method), sizeof(method));
		deserialize_vector(in, lyndon_idx);
		SparseIntMatrix::deserialize(in, inv_proj_mat);
		SparseIntMatrix::deserialize(in, inv_proj_mat_transpose);
	}
};

extern const char* version;
extern std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BasisCache>, PairHash> basis_cache;
extern std::shared_mutex basis_cache_mu;

class CacheFile {
public:

	CacheFile(uint64_t dimension_, uint64_t degree_) {
		if (cache_dir.empty() || !std::filesystem::exists(cache_dir / cache_folder_name))
			throw std::runtime_error("Unexpected internal error. Cache directory was not set correctly.");

		dimension = dimension_;
		degree = degree_;
		file_name = std::to_string(dimension) + "_" + std::to_string(degree) + "_" + version + ".bin";
		file_path = cache_dir / cache_folder_name / file_name;
	}

	bool exists() const {
		return std::filesystem::exists(file_path);
	}

	void write(std::unique_ptr<BasisCache>& obj) const {
		std::ofstream out(file_path, std::ios::binary);
		out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
		obj->serialize(out);
	}

	void read(std::unique_ptr<BasisCache>& obj) {
		std::ifstream in(file_path, std::ios::binary);
		uint64_t magic;
		in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
		if (magic != cache_magic_number)
			throw std::runtime_error("Tried to read an invalid cache file. Cache may have been corrupted.");
		obj->deserialize(in);
	}

private:
	uint64_t dimension;
	uint64_t degree;
	std::string file_name;
	std::filesystem::path file_path;
};

void set_basis_cache(uint64_t dimension, uint64_t degree, int method, bool use_disk = false);
const BasisCache& get_basis_cache(uint64_t dimension, uint64_t degree, int method);
void clear_cache_(bool use_disk);
