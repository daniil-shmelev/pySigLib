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
#include "cp_branched_cache.h"
#include "cp_branched_log_signature.h"
#include "cp_branched_sig_coef_cache.h"

const char* version = "v1";
const char* cache_folder_name = "pysiglib_cache";

namespace {
struct BasisCacheRegistry {
	std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<BasisCache>, PairHash> map;
	std::shared_mutex mu;
};
BasisCacheRegistry& basis_cache_registry() {
	static BasisCacheRegistry r;
	return r;
}
void clear_basis_cache() {
	auto& reg = basis_cache_registry();
	std::unique_lock lk(reg.mu);
	reg.map.clear();
}
}  // anonymous namespace

static std::filesystem::path cache_dir;
static std::shared_mutex cache_dir_mu;

std::filesystem::path get_cache_dir() {
	{
		std::shared_lock rlk(cache_dir_mu);
		if (!cache_dir.empty())
			return cache_dir;
	}
	set_default_cache_dir();
	std::shared_lock rlk(cache_dir_mu);
	return cache_dir;
}

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
	if (!in)
		throw std::runtime_error("Tried to read an invalid cache file: vector size header");
	if (size > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error("Tried to read an invalid cache file: vector size exceeds limit");

	if (size > 0) {
		check_stream_has_bytes(in, size * sizeof(uint64_t), "vector body");
		out.resize(size);
		in.read(reinterpret_cast<char*>(out.data()), size * sizeof(uint64_t));
		if (!in)
			throw std::runtime_error("Tried to read an invalid cache file: vector body read");
	}
	else {
		out.clear();
	}
}

void set_cache_dir_(const char* dir) {
	std::filesystem::path dir_path = dir;
	if (!std::filesystem::exists(dir_path)) {
		throw directory_not_found_error("Directory " + std::string(dir) + " does not exist.");
	}
	std::filesystem::path pysiglib_cache_path = dir_path / cache_folder_name;
	if (!std::filesystem::exists(pysiglib_cache_path)) {
		std::filesystem::create_directories(pysiglib_cache_path);
	}
	std::unique_lock lk(cache_dir_mu);
	cache_dir = dir_path;
}

void set_default_cache_dir() {
#ifdef _WIN32
	char* dir = nullptr;
	size_t len;

	const errno_t err = _dupenv_s(&dir, &len, "LOCALAPPDATA");

	if (err || !dir) {
		throw default_cache_dir_error("Failed to get default cache directory.");
	}

#elif __APPLE__
	const char* home = std::getenv("HOME");
	if (!home)
		throw default_cache_dir_error("$HOME is not set; cannot determine default cache directory.");
	std::string dir_str = std::string(home) + "/Library/Caches";
	const char* dir = dir_str.c_str();
#else
	const char* home = std::getenv("HOME");
	if (!home)
		throw default_cache_dir_error("$HOME is not set; cannot determine default cache directory.");
	std::string dir_str = std::string(home) + "/.cache";
	const char* dir = dir_str.c_str();
#endif

	std::filesystem::path dir_path = dir;
	std::filesystem::path pysiglib_cache_path = dir_path / cache_folder_name;
	if (!std::filesystem::exists(pysiglib_cache_path)) {
		std::filesystem::create_directories(pysiglib_cache_path);
	}

	// Only install if still unset: don't clobber an explicit set_cache_dir_().
	{
		std::unique_lock lk(cache_dir_mu);
		if (cache_dir.empty())
			cache_dir = dir_path;
	}

#ifdef _WIN32
	free(dir);
#endif
}

void prepare_basis_cache(uint64_t dimension, uint64_t degree, int method, bool use_disk) {
	if (method < 1)
		return;

	std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto& reg = basis_cache_registry();

	{
		std::shared_lock rlock(reg.mu);
		auto it = reg.map.find(key);
		if (it != reg.map.end() && it->second->method >= method) return;
	}

	if (use_disk) {
		auto dir = get_cache_dir();
		if (!std::filesystem::exists(dir / cache_folder_name))
			std::filesystem::create_directory(dir / cache_folder_name);
		CacheFile file(dimension, degree);
		if (file.exists()) {
			auto basis_obj = std::make_unique<BasisCache>();
			file.read(basis_obj);
			if (basis_obj->method >= method) {
				std::unique_lock wlock(reg.mu);
				reg.map.insert_or_assign(key, std::move(basis_obj));
				return;
			}
		}
	}

	std::vector<word> lyndon_words = all_lyndon_words(dimension, degree);
	std::vector<uint64_t> lyndon_idx;
	lyndon_idx.reserve(lyndon_words.size());
	for (const auto& w : lyndon_words)
		lyndon_idx.push_back(word_to_idx(w, dimension));
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

	if (use_disk) {
		CacheFile file(dimension, degree);
		file.write(basis_obj);
	}

	std::unique_lock wlock(reg.mu);
	reg.map.insert_or_assign(key, std::move(basis_obj));
}

const BasisCache& get_basis_cache(uint64_t dimension, uint64_t degree, int method) {

	std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto& reg = basis_cache_registry();

	{
		std::shared_lock rlock(reg.mu);
		auto it = reg.map.find(key);
		if (it != reg.map.end() && it->second->method >= method)
			return *(it->second);
	}

	CacheFile file(dimension, degree);
	if (!file.exists())
		throw cache_not_found_error("Could not find basis cache");

	auto basis_obj = std::make_unique<BasisCache>();
	file.read(basis_obj);

	if (basis_obj->method < method)
		throw cache_not_found_error("Could not find basis cache");

	std::unique_lock wlock(reg.mu);
	auto p = reg.map.insert_or_assign(key, std::move(basis_obj));
	return *(p.first->second);
}

void clear_cache_(bool use_disk) {
	clear_basis_cache();
	clear_bch_cache();
	clear_branched_sig_coef_cache();
	clear_branched_log_sig_cache();
	clear_branched_sig_cache();

	if (use_disk) {
		auto dir = get_cache_dir();
		std::filesystem::remove_all(dir / cache_folder_name);
	}
}

extern "C" {

	CPSIG_API int set_cache_dir(const char* dir) noexcept {
		SAFE_CALL(set_cache_dir_(dir));
	}

	CPSIG_API int prepare_log_sig(uint64_t dimension, uint64_t degree, int method, bool use_disk) noexcept {
		SAFE_CALL(
			if (method != 3)
				prepare_basis_cache(dimension, degree, method, use_disk);
			if (method == 3)
				prepare_bch_cache(dimension, degree, use_disk)
		);
	}

	CPSIG_API int clear_cache(bool use_disk) noexcept {
		SAFE_CALL(clear_cache_(use_disk));
	}

	CPSIG_API void cpsig_shutdown() noexcept {
		try { clear_basis_cache();                                       } catch (...) {}
		try { clear_bch_cache();                                         } catch (...) {}
		try { clear_branched_sig_coef_cache();                            } catch (...) {}
		try { clear_branched_log_sig_cache();                             } catch (...) {}
		try { clear_branched_sig_cache();                                } catch (...) {}
		try { std::unique_lock lk(cache_dir_mu);   cache_dir.clear();    } catch (...) {}
	}

}
