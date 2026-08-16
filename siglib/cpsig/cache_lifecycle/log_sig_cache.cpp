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
#include "disk_cache.h"
#include "cp_bch.h"
#include "cp_branched_cache.h"
#include "cp_branched_log_signature.h"
#include "cp_branched_sig_coef_cache.h"

void clear_sig_poly_table_cache();

const char* version = "v2";
const char* cache_folder_name = "pysiglib_cache";

namespace {
struct LogSigCacheRegistry {
	std::unordered_map<std::pair<uint64_t, uint64_t>, std::unique_ptr<LogSigCache>, PairHash> map;
	std::shared_mutex mu;
};
LogSigCacheRegistry& log_sig_cache_registry() {
	static LogSigCacheRegistry r;
	return r;
}
void clear_log_sig_cache() {
	auto& reg = log_sig_cache_registry();
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
	const std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto& reg = log_sig_cache_registry();
	{
		std::shared_lock rlock(reg.mu);
		auto it = reg.map.find(key);
		if (it != reg.map.end() && it->second->supports(method))
			return;
	}
	const auto cache_directory = get_cache_dir() / cache_folder_name;
	std::unique_lock wlock(reg.mu);
	auto found = reg.map.find(key);
	if (found == reg.map.end()) {
		reg.map.try_emplace(
			key,
			std::make_unique<LogSigCache>(
				dimension, degree, method, cache_directory, use_disk));
	}
	else if (!found->second->supports(method))
		found->second->upgrade(method, cache_directory, use_disk);
}

const LogSigCache& get_log_sig_cache(
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	const std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto& reg = log_sig_cache_registry();
	{
		std::shared_lock rlock(reg.mu);
		auto it = reg.map.find(key);
		if (it != reg.map.end() && it->second->supports(method))
			return *(it->second);
	}

	BasisCache disk_basis;
	const auto cache_directory = get_cache_dir() / cache_folder_name;
	if (!read_log_sig_basis_cache(
		cache_directory, dimension, degree, disk_basis)
		|| !disk_basis.supports((std::min)(method, 2)))
		throw cache_not_found_error("Could not find basis cache");
	auto cache = std::make_unique<LogSigCache>(
		dimension, degree, disk_basis.method, cache_directory, true);
	std::unique_lock wlock(reg.mu);
	auto p = reg.map.insert_or_assign(key, std::move(cache));
	return *(p.first->second);
}

LogSigCache& get_log_sig_cache_mutable(
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	const std::pair<uint64_t, uint64_t> key(dimension, degree);
	auto& reg = log_sig_cache_registry();
	std::shared_lock lock(reg.mu);
	const auto found = reg.map.find(key);
	if (found == reg.map.end()
		|| !found->second->basis((std::min)(method, 2)).supports(
			(std::min)(method, 2)))
		throw cache_not_found_error("Could not find basis cache");
	return *found->second;
}

const BasisCache& get_basis_cache(
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	return get_log_sig_cache(dimension, degree, method).basis(method);
}

void clear_cache_(bool use_disk) {
	clear_log_sig_cache();
	clear_bch_cache();
	clear_branched_sig_coef_cache();
	clear_branched_log_sig_cache();
	clear_branched_sig_cache();
	clear_sig_poly_table_cache();

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
		try { clear_log_sig_cache();                                     } catch (...) {}
		try { clear_bch_cache();                                         } catch (...) {}
		try { clear_branched_sig_coef_cache();                            } catch (...) {}
		try { clear_branched_log_sig_cache();                             } catch (...) {}
		try { clear_branched_sig_cache();                                } catch (...) {}
		try { clear_sig_poly_table_cache();                              } catch (...) {}
		try { std::unique_lock lk(cache_dir_mu);   cache_dir.clear();    } catch (...) {}
	}

}
