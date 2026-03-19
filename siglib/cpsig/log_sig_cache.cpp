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
#include "log_sig_cache.h"
#include "cp_bch.h"

const char* version = "v1";
std::filesystem::path cache_dir;
const char* cache_folder_name = "pysiglib_cache";
std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BasisCache>, PairHash> basis_cache;

void serialize_vector(std::ostream& out, const std::vector<uint64_t>& v) {

	uint64_t size = v.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(size));

	if (size > 0) {
		out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(uint64_t));
	}
}

void deserialize_vector(std::istream& in, std::vector<uint64_t>& out) {

	uint64_t size;
	in.read(reinterpret_cast<char*>(&size), sizeof(size));

	out.resize(size);
	if (size > 0) {
		in.read(reinterpret_cast<char*>(out.data()), size * sizeof(uint64_t));
	}
}

void set_cache_dir_(const char* dir) {
	std::filesystem::path dir_path = dir;
	if (!std::filesystem::exists(dir_path)) {
		throw std::runtime_error("Directory " + std::string(dir) + " does not exist.");
	}
	std::filesystem::path pysiglib_cache_path = dir_path / cache_folder_name;
	if (!std::filesystem::exists(pysiglib_cache_path)) {
		std::filesystem::create_directories(pysiglib_cache_path);
	}
	cache_dir = dir_path;
}

void set_default_cache_dir() {
#ifdef _WIN32
	char* dir = nullptr;
	size_t len;

	const errno_t err = _dupenv_s(&dir, &len, "LOCALAPPDATA");

	if (err || !dir) {
		throw std::runtime_error("Failed to get default cache directory.");
	}

#elif __APPLE__
	std::string dir_str = std::string(std::getenv("HOME")) + "/Library/Caches";
	const char* dir = dir_str.c_str();
#else
	std::string dir_str = std::string(std::getenv("HOME")) + "/.cache";
	const char* dir = dir_str.c_str();
#endif

	set_cache_dir_(dir);

#ifdef _WIN32
	free(dir);
#endif
}

void set_basis_cache(uint64_t dimension, uint64_t degree, int method, bool use_disk) {
	if (method < 1)
		return;

	if (cache_dir.empty()) {
		set_default_cache_dir();
	}

	if (!std::filesystem::exists(cache_dir / cache_folder_name))
		std::filesystem::create_directory(cache_dir / cache_folder_name);

	std::pair<uint64_t, uint64_t> key(dimension, degree);

	auto it = basis_cache.find(key);
	bool exists_in_memory = it != basis_cache.end() && it->second->method >= method;
	if (!exists_in_memory) {

		CacheFile file(dimension, degree);
		if (use_disk) {
			if (file.exists()) {
				// Pull into memory
				auto basis_obj = std::make_unique<BasisCache>();
				basis_cache.insert_or_assign(key, std::move(basis_obj));
				return;
			}
		}

		std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
		std::vector<uint64_t> lyndon_idx = all_lyndon_idx(dimension, degree);
		SparseIntMatrix p, p_inv, p_inv_t;
		if (method == 2) {
			lyndon_proj_matrix(p, lyndon_words, lyndon_idx, dimension, degree);
			p.inverse(p_inv);
			p_inv.transpose(p_inv_t);
		}


		auto basis_obj = std::make_unique<BasisCache>(
			method,
			std::move(lyndon_idx),
			std::move(p_inv),
			std::move(p_inv_t)
		);

		//Save to disk
		if (use_disk) {
			file.write(basis_obj);
		}

		//Save to memory
		basis_cache.insert_or_assign(key, std::move(basis_obj));
	}
}

const BasisCache& get_basis_cache(uint64_t dimension, uint64_t degree, int method) {

	if (cache_dir.empty()) {
		set_default_cache_dir();
	}

	std::pair<uint64_t, uint64_t> key(dimension, degree);

	auto it = basis_cache.find(key);
	if (it == basis_cache.end() || it->second->method < method) {

		//Check disk
		CacheFile file(dimension, degree);
		if (!file.exists())
			throw std::runtime_error("Could not find basis cache");

		auto basis_obj = std::make_unique<BasisCache>();
		file.read(basis_obj);

		if (basis_obj->method < method)
			throw std::runtime_error("Could not find basis cache");

		auto p = basis_cache.insert_or_assign(key, std::move(basis_obj));
		return *(p.first->second);
	}
	return *(it->second);
}

void clear_cache_(bool use_disk) {
	if (cache_dir.empty()) {
		set_default_cache_dir();
	}

	basis_cache.clear();
	clear_bch_cache();

	if (use_disk)
		std::filesystem::remove_all(cache_dir / cache_folder_name);
}

extern "C" {

	CPSIG_API int set_cache_dir(const char* dir) noexcept {
		SAFE_CALL(set_cache_dir_(dir));
	}

	CPSIG_API int prepare_log_sig(uint64_t dimension, uint64_t degree, int method, bool use_disk) noexcept {
		SAFE_CALL(set_basis_cache(dimension, degree, method, use_disk));
	}

	CPSIG_API int clear_cache(bool use_disk) noexcept {
		SAFE_CALL(clear_cache_(use_disk));
	}

}
