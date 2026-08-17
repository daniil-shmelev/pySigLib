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

#pragma once

#include "sparse_int_matrix.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct BchCache;

struct BasisCache {
	int method = 0;
	std::vector<uint64_t> lyndon_idx;
	SparseIntMatrix inv_proj_mat;
	SparseIntMatrix inv_proj_mat_transpose;

	bool supports(int requested_method) const noexcept {
		return method >= requested_method;
	}

	void serialize(std::ostream& out) const;
	void deserialize(std::istream& in);
};

class LogSigCache {
public:
	LogSigCache(
		uint64_t dimension,
		uint64_t degree,
		int method,
		const std::filesystem::path& cache_directory = {},
		bool use_disk = false,
		const std::string& file_prefix = "");
	~LogSigCache();
	LogSigCache(LogSigCache&&) noexcept;
	LogSigCache& operator=(LogSigCache&&) noexcept;
	LogSigCache(const LogSigCache&) = delete;
	LogSigCache& operator=(const LogSigCache&) = delete;

	bool supports(int method) const noexcept;
	int method() const noexcept;
	const BasisCache& basis(int method) const;
	BasisCache& basis(int method);
	bool has_bch() const noexcept;
	const BchCache& bch() const;
	BchCache& bch();
	void set_bch(std::unique_ptr<BchCache> cache);

	void upgrade(
		int method,
		const std::filesystem::path& cache_directory = {},
		bool use_disk = false,
		const std::string& file_prefix = "");

private:
	uint64_t dimension_ = 0;
	uint64_t degree_ = 0;
	BasisCache basis_;
	std::unique_ptr<BchCache> bch_;
};

std::filesystem::path log_sig_basis_cache_file_path(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	const std::string& prefix = "");

bool read_log_sig_basis_cache(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	BasisCache& cache,
	const std::string& prefix = "");

void write_log_sig_basis_cache(
	const std::filesystem::path& cache_directory,
	uint64_t dimension,
	uint64_t degree,
	const BasisCache& cache,
	const std::string& prefix = "");
