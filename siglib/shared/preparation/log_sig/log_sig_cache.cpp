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

#include "log_sig_cache.h"

#include "../../errors.h"
#include "bch_cache.h"
#include "bch_data.h"
#include "../cache_io.h"
#include "lyndon_words.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace {
inline constexpr const char* basis_cache_version_ = "v2";

void build_basis_projection_(
	BasisCache& basis,
	const std::vector<word>& lyndon_words,
	uint64_t dimension,
	uint64_t degree
) {
	SparseIntMatrix projection;
	lyndon_proj_matrix(
		projection, lyndon_words, basis.lyndon_idx,
		dimension, degree);
	projection.inverse(basis.inv_proj_mat);
	basis.inv_proj_mat.transpose(basis.inv_proj_mat_transpose);
	basis.method = 2;
}

BasisCache build_basis_cache_(
	uint64_t dimension,
	uint64_t degree,
	int method
) {
	BasisCache result;
	result.method = method;
	const std::vector<word> lyndon_words = all_lyndon_words(
		dimension, degree);
	result.lyndon_idx.reserve(lyndon_words.size());
	for (const word& value : lyndon_words)
		result.lyndon_idx.push_back(word_to_idx(value, dimension));
	if (method >= 2)
		build_basis_projection_(result, lyndon_words, dimension, degree);
	return result;
}
}  // namespace

void BasisCache::serialize(std::ostream& out) const {
	out.write(reinterpret_cast<const char*>(&method), sizeof(method));
	serialize_cache_vector(out, lyndon_idx);
	inv_proj_mat.serialize(out);
	inv_proj_mat_transpose.serialize(out);
}

void BasisCache::deserialize(std::istream& in) {
	in.read(reinterpret_cast<char*>(&method), sizeof(method));
	if (!in || method < 0 || method > 2)
		throw std::runtime_error(
			"Tried to read an invalid cache file: basis method");
	deserialize_cache_vector(in, lyndon_idx, "basis Lyndon indices");
	SparseIntMatrix::deserialize(in, inv_proj_mat);
	SparseIntMatrix::deserialize(in, inv_proj_mat_transpose);
}

std::filesystem::path log_sig_basis_cache_file_path(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	const std::string& prefix
) {
	return cache_directory / (
		prefix + std::to_string(dimension) + "_"
		+ std::to_string(degree) + "_" + basis_cache_version_ + ".bin");
}

bool read_log_sig_basis_cache(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	BasisCache& cache,
	const std::string& prefix
) {
	const auto path = log_sig_basis_cache_file_path(
		cache_directory, dimension, degree, prefix);
	if (!std::filesystem::exists(path))
		return false;
	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;
	uint64_t magic = 0;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	if (!in || magic != cache_magic_number)
		throw corrupted_cache_error(
			"Tried to read an invalid cache file. Cache may have been corrupted.");
	BasisCache loaded;
	loaded.deserialize(in);
	cache = std::move(loaded);
	return true;
}

void write_log_sig_basis_cache(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	const BasisCache& cache,
	const std::string& prefix
) {
	std::filesystem::create_directories(cache_directory);
	const auto path = log_sig_basis_cache_file_path(
		cache_directory, dimension, degree, prefix);
	std::ofstream out(path, std::ios::binary);
	if (!out) {
		throw std::filesystem::filesystem_error(
			"Failed to open cache file for writing", path,
			std::make_error_code(std::errc::io_error));
	}
	out.write(
		reinterpret_cast<const char*>(&cache_magic_number),
		sizeof(cache_magic_number));
	cache.serialize(out);
}

LogSigCache::LogSigCache(
	uint64_t dimension,
	uint64_t degree,
	int method,
	const std::filesystem::path& cache_directory,
	bool use_disk,
	const std::string& file_prefix
) : dimension_{ dimension }, degree_{ degree } {
	upgrade(method, cache_directory, use_disk, file_prefix);
}

LogSigCache::~LogSigCache() = default;
LogSigCache::LogSigCache(LogSigCache&&) noexcept = default;
LogSigCache& LogSigCache::operator=(LogSigCache&&) noexcept = default;

bool LogSigCache::supports(int method) const noexcept {
	return basis_.supports((std::min)(method, 2))
		&& (method < 3 || bch_ != nullptr);
}

int LogSigCache::method() const noexcept {
	return basis_.method;
}

const BasisCache& LogSigCache::basis(int method) const {
	if (!supports(method))
		throw cache_not_found_error("Could not find basis cache");
	return basis_;
}

BasisCache& LogSigCache::basis(int method) {
	if (!supports(method))
		throw cache_not_found_error("Could not find basis cache");
	return basis_;
}

bool LogSigCache::has_bch() const noexcept {
	return bch_ != nullptr;
}

const BchCache& LogSigCache::bch() const {
	if (!bch_)
		throw cache_not_found_error("Could not find BCH cache");
	return *bch_;
}

BchCache& LogSigCache::bch() {
	if (!bch_)
		throw cache_not_found_error("Could not find BCH cache");
	return *bch_;
}

void LogSigCache::set_bch(std::unique_ptr<BchCache> cache) {
	bch_ = std::move(cache);
}

void LogSigCache::upgrade(
	int method,
	const std::filesystem::path& cache_directory,
	bool use_disk,
	const std::string& file_prefix
) {
	if (method < 1 || method > 3)
		throw std::invalid_argument("log signature method must be 1, 2, or 3");
	const int basis_method = (std::min)(method, 2);
	if (basis_.supports(basis_method)) {
		if (method == 3 && !bch_)
			bch_ = make_standard_bch_cache(
				dimension_, degree_, basis_);
		return;
	}
	if (use_disk && !cache_directory.empty()) {
		BasisCache loaded;
		if (read_log_sig_basis_cache(
			cache_directory, dimension_, degree_, loaded, file_prefix)
			&& loaded.supports(basis_method)) {
			basis_ = std::move(loaded);
			if (method == 3)
				bch_ = make_standard_bch_cache(
					dimension_, degree_, basis_);
			return;
		}
	}
	if (basis_.method == 1 && basis_method == 2) {
		const std::vector<word> lyndon_words = all_lyndon_words(
			dimension_, degree_);
		build_basis_projection_(
			basis_, lyndon_words, dimension_, degree_);
		if (use_disk && !cache_directory.empty()) {
			write_log_sig_basis_cache(
				cache_directory, dimension_, degree_, basis_, file_prefix);
		}
		if (method == 3)
			bch_ = make_standard_bch_cache(dimension_, degree_, basis_);
		return;
	}
	basis_ = build_basis_cache_(dimension_, degree_, basis_method);
	if (use_disk && !cache_directory.empty()) {
		write_log_sig_basis_cache(
			cache_directory, dimension_, degree_, basis_, file_prefix);
	}
	if (method == 3)
		bch_ = make_standard_bch_cache(dimension_, degree_, basis_);
}
