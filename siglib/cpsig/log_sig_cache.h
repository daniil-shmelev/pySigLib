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
#include "../shared/errors.h"
#include "words.h"
#include "sparse.h"
#include "disk_cache.h"

struct BasisCache {
	int method = 0;
	std::vector<uint64_t> lyndon_idx;
	SparseIntMatrix inv_proj_mat;
	SparseIntMatrix inv_proj_mat_transpose;

	BasisCache() = default;

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

class CacheFile {
public:

	CacheFile(uint64_t dimension_, uint64_t degree_, std::string prefix = "") {
		auto dir = get_cache_dir();
		if (dir.empty() || !std::filesystem::exists(dir / cache_folder_name))
			throw cache_dir_not_set_error("Unexpected internal error. Cache directory was not set correctly.");

		dimension = dimension_;
		degree = degree_;
		file_name = prefix + std::to_string(dimension) + "_" + std::to_string(degree) + "_" + version + ".bin";
		file_path = dir / cache_folder_name / file_name;
	}

	bool exists() const {
		return std::filesystem::exists(file_path);
	}

	void write(std::unique_ptr<BasisCache>& obj) const {
		std::ofstream out(file_path, std::ios::binary);
		if (!out)
			throw std::filesystem::filesystem_error(
				"Failed to open cache file for writing", file_path,
				std::make_error_code(std::errc::io_error));
		out.write(reinterpret_cast<const char*>(&cache_magic_number), sizeof(cache_magic_number));
		obj->serialize(out);
	}

	void read(std::unique_ptr<BasisCache>& obj) {
		std::ifstream in(file_path, std::ios::binary);
		if (!in)
			throw std::filesystem::filesystem_error(
				"Failed to open cache file for reading", file_path,
				std::make_error_code(std::errc::io_error));
		uint64_t magic;
		in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
		if (magic != cache_magic_number)
			throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");
		obj->deserialize(in);
	}

private:
	uint64_t dimension;
	uint64_t degree;
	std::string file_name;
	std::filesystem::path file_path;
};

void prepare_basis_cache(uint64_t dimension, uint64_t degree, int method, bool use_disk = false);
const BasisCache& get_basis_cache(uint64_t dimension, uint64_t degree, int method);
void clear_cache_(bool use_disk);
