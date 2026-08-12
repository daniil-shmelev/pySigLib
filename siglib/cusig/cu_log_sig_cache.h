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
#include "cupch.h"
#include "cu_disk_cache.h"
#include "cu_utils.h"
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>
#include <utility>

// =========================================================================
// Lyndon word utilities (host-side, ported from cpsig/words.h + words.cpp)
// =========================================================================

typedef std::vector<uint64_t> cu_word;

// Golden-ratio (Fibonacci) hash spreader - same constant used by boost::hash_combine.
static constexpr std::size_t kFibHashConst = 0x9e3779b97f4a7c15ULL;

struct CuWordHash {
	std::size_t operator()(const cu_word& w) const noexcept {
		std::size_t h = 0;
		for (uint64_t x : w) {
			h ^= std::hash<uint64_t>{}(x)
				+ kFibHashConst
				+ (h << 6)
				+ (h >> 2);
		}
		return h;
	}
};

struct CuPairHash {
	std::size_t operator()(const std::pair<uint64_t, uint64_t>& p) const noexcept {
		std::size_t h1 = std::hash<uint64_t>{}(p.first);
		std::size_t h2 = std::hash<uint64_t>{}(p.second);
		return h1 ^ (h2 + kFibHashConst + (h1 << 6) + (h1 >> 2));
	}
};

inline bool cu_is_lyndon(cu_word w) {
	const uint64_t n = w.size();
	if (n == 0) return false;
	if (n == 1) return true;
	for (uint64_t i = 1; i < n; ++i) {
		if (!std::lexicographical_compare(
			w.begin(), w.end(),
			w.begin() + i, w.end()
		))
			return false;
	}
	return true;
}

inline void cu_all_lyndon_words_of_length_n(std::vector<cu_word>& res, uint64_t n, uint64_t dimension) {
	cu_word w;
	w.push_back(0);

	while (!w.empty()) {
		uint64_t m = w.size();
		if (m == n)
			res.push_back(w);

		while (w.size() < n)
			w.push_back(w[w.size() - m]);

		while (!w.empty() && w.back() == dimension - 1)
			w.pop_back();

		if (!w.empty())
			++w.back();
	}
}

inline std::vector<cu_word> cu_all_lyndon_words(uint64_t dimension, uint64_t degree) {
	std::vector<cu_word> res;
	for (uint64_t n = 1; n <= degree; ++n)
		cu_all_lyndon_words_of_length_n(res, n, dimension);
	return res;
}

inline uint64_t cu_word_to_idx(const cu_word& w, uint64_t dimension) {
	if (!w.size()) return 0;
	uint64_t idx = 0;
	for (uint64_t i : w) {
		if (dimension != 0 && idx > UINT64_MAX / dimension)
			throw std::overflow_error("cu_word_to_idx: index overflow");
		const uint64_t mul = idx * dimension;
		const uint64_t add = i + 1;
		if (mul > UINT64_MAX - add)
			throw std::overflow_error("cu_word_to_idx: index overflow");
		idx = mul + add;
	}
	return idx;
}

inline std::vector<uint64_t> cu_all_lyndon_idx(uint64_t dimension, uint64_t degree) {
	std::vector<cu_word> words = cu_all_lyndon_words(dimension, degree);
	std::vector<uint64_t> res;
	res.reserve(words.size());
	for (const cu_word& w : words) {
		res.push_back(cu_word_to_idx(w, dimension));
	}
	return res;
}

inline cu_word cu_longest_lyndon_suffix_(cu_word w, const std::unordered_set<cu_word, CuWordHash>& lyndon_set) {
	uint64_t n = w.size();
	for (uint64_t i = 1; i < n; ++i) {
		cu_word suffix(w.begin() + i, w.end());
		if (lyndon_set.find(suffix) != lyndon_set.end()) {
			return suffix;
		}
	}
	throw std::runtime_error("Error looking for lyndon suffix");
}

inline uint64_t cu_concatenate_idx(uint64_t i, uint64_t j, uint64_t len_j, uint64_t dimension) {
	uint64_t idx = i;
	idx *= host_power(dimension, len_j);
	idx += j;
	return idx;
}

// =========================================================================
// Minimal SparseIntMatrix (ported from cpsig/sparse.h)
// =========================================================================

struct CuEntry {
	uint64_t col;
	int val;
	CuEntry() : col(0), val(0) {}
	CuEntry(uint64_t c, int v) : col(c), val(v) {}
};

class CuSparseIntMatrix {
public:
	uint64_t n;
	uint64_t m;
	std::vector<std::vector<CuEntry>> rows;

	CuSparseIntMatrix() : n(0), m(0), rows(0) {}
	CuSparseIntMatrix(uint64_t n_) : n(n_), m(n_), rows(n_) {}
	CuSparseIntMatrix(uint64_t n_, uint64_t m_) : n(n_), m(m_), rows(n_) {}

	CuSparseIntMatrix(const CuSparseIntMatrix&) = default;
	CuSparseIntMatrix& operator=(const CuSparseIntMatrix&) = default;

	CuSparseIntMatrix(CuSparseIntMatrix&& other) noexcept
		: n{ std::exchange(other.n, 0) },
		  m{ std::exchange(other.m, 0) },
		  rows{ std::move(other.rows) } {}

	CuSparseIntMatrix& operator=(CuSparseIntMatrix&& other) noexcept {
		if (this != &other) {
			n = std::exchange(other.n, 0);
			m = std::exchange(other.m, 0);
			rows = std::move(other.rows);
		}
		return *this;
	}

	void resize(uint64_t n_, uint64_t m_) {
		n = n_; m = m_;
		rows.resize(n);
	}

	void insert_entry(uint64_t i, uint64_t j, int v) {
		rows[i].emplace_back(j, v);
	}

	void add_to_entry(uint64_t i, uint64_t j, int v) {
		for (auto& e : rows[i]) {
			if (e.col == j) { e.val += v; return; }
		}
		rows[i].emplace_back(j, v);
	}

	void drop_diagonal() {
		for (uint64_t i = 0; i < n; ++i) {
			auto& row = rows[i];
			for (auto it = row.begin(); it != row.end(); ++it) {
				if (it->col == i) { row.erase(it); break; }
			}
		}
	}

	void transpose(CuSparseIntMatrix& out) const {
		out.resize(m, n);
		for (uint64_t i = 0; i < n; ++i) {
			for (const auto& e : rows[i]) {
				out.rows[e.col].emplace_back(i, e.val);
			}
		}
	}

	void inverse(CuSparseIntMatrix& out) const {
		// Assumes lower triangular with ones on the diagonal.
		// Output omits the diagonal.
		out.resize(n, n);

		for (uint64_t i = 0; i < n; ++i) {
			out.insert_entry(i, i, 1);
		}

		for (uint64_t i = 0; i < n; ++i) {
			std::unordered_map<uint64_t, int> row_i;

			for (const auto& e : rows[i]) {
				uint64_t k = e.col;
				int Lik = e.val;
				if (k >= i) continue;

				for (const auto& ek : out.rows[k]) {
					uint64_t j = ek.col;
					row_i[j] -= Lik * ek.val;
				}

				row_i[k] -= Lik;
			}

			out.rows[i].clear();
			for (const auto& kv : row_i) {
				if (kv.second != 0) {
					out.rows[i].emplace_back(kv.first, kv.second);
				}
			}
		}
	}

	void serialize(std::ostream& out) const {
		out.write(reinterpret_cast<const char*>(&n), sizeof(n));
		out.write(reinterpret_cast<const char*>(&m), sizeof(m));
		for (uint64_t i = 0; i < n; ++i) {
			uint64_t nnz = rows[i].size();
			out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
			for (const auto& e : rows[i]) {
				out.write(reinterpret_cast<const char*>(&e.col), sizeof(e.col));
				out.write(reinterpret_cast<const char*>(&e.val), sizeof(e.val));
			}
		}
	}

	static void deserialize(std::istream& in, CuSparseIntMatrix& out) {
		in.read(reinterpret_cast<char*>(&out.n), sizeof(out.n));
		in.read(reinterpret_cast<char*>(&out.m), sizeof(out.m));
		if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix header");
		if (out.n > kCuMaxCacheVectorSize || out.m > kCuMaxCacheVectorSize)
			throw std::runtime_error("Tried to read an invalid cache file: sparse matrix dimension exceeds limit");

		cu_check_stream_has_bytes_(in, out.n * sizeof(uint64_t), "sparse matrix row headers");

		out.rows.clear();
		out.rows.resize(out.n);
		for (uint64_t i = 0; i < out.n; ++i) {
			uint64_t nnz;
			in.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
			if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix nnz");
			if (nnz > out.m)
				throw std::runtime_error("Tried to read an invalid cache file: sparse matrix row nnz exceeds column count");

			if (nnz > 0)
				cu_check_stream_has_bytes_(in, nnz * (sizeof(uint64_t) + sizeof(int)), "sparse matrix row body");

			out.rows[i].resize(nnz);
			for (uint64_t k = 0; k < nnz; ++k) {
				in.read(reinterpret_cast<char*>(&out.rows[i][k].col), sizeof(uint64_t));
				in.read(reinterpret_cast<char*>(&out.rows[i][k].val), sizeof(int));
			}
			if (!in) throw std::runtime_error("Tried to read an invalid cache file: sparse matrix entry read");
		}
	}
};

// =========================================================================
// lyndon_proj_matrix (ported from cpsig/words.cpp)
// =========================================================================

inline void cu_lyndon_proj_matrix(
	CuSparseIntMatrix& out,
	const std::vector<cu_word>& lyndon_words,
	std::vector<uint64_t> lyndon_idx,
	uint64_t dimension,
	uint64_t degree
) {
	std::unordered_set<cu_word, CuWordHash> lyndon_set(lyndon_words.begin(), lyndon_words.end());
	uint64_t n = host_sig_length(dimension, degree);
	uint64_t m_ = lyndon_words.size();

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	host_populate_level_index(level_index, dimension, degree + 2);

	CuSparseIntMatrix full_mat_transpose(m_, n);

	std::unordered_map<cu_word, uint64_t, CuWordHash> col_idx;
	for (uint64_t i = 0; i < m_; ++i) {
		col_idx[lyndon_words[i]] = i;
	}

	for (uint64_t i = 0; i < m_; ++i) {
		cu_word w = lyndon_words[i];

		if (w.size() == 1) {
			full_mat_transpose.insert_entry(i, w[0] + 1, 1);
		}
		else {
			cu_word v = cu_longest_lyndon_suffix_(w, lyndon_set);
			cu_word u(w.begin(), w.end() - v.size());

			uint64_t jw = col_idx[w];
			uint64_t jv = col_idx[v];
			uint64_t ju = col_idx[u];

			for (const auto& eu : full_mat_transpose.rows[ju]) {
				if (eu.val) {
					for (const auto& ev : full_mat_transpose.rows[jv]) {
						if (ev.val) {
							uint64_t ic = cu_concatenate_idx(eu.col, ev.col, v.size(), dimension);
							int val = eu.val * ev.val;
							full_mat_transpose.add_to_entry(jw, ic, val);
							ic = cu_concatenate_idx(ev.col, eu.col, u.size(), dimension);
							full_mat_transpose.add_to_entry(jw, ic, -val);
						}
					}
				}
			}
		}
	}

	CuSparseIntMatrix full_mat;
	full_mat_transpose.transpose(full_mat);
	out.resize(full_mat.m, full_mat.m);

	for (uint64_t i = 0; i < m_; ++i) {
		out.rows[i] = full_mat.rows[lyndon_idx[i]];
	}

	out.drop_diagonal();
}

// =========================================================================
// CUDALogSigCache - holds GPU-resident arrays for methods 1 and 2
// =========================================================================

struct CUDALogSigCache {
	uint64_t* d_lyndon_idx = nullptr;
	uint64_t log_sig_len = 0;

	// Cached level index (avoids per-call allocation)
	uint64_t* d_level_index = nullptr;
	uint64_t sig_len = 0;
	uint64_t buff1_len = 0;
	unsigned int threads_per_block = 32;

	// CSR sparse matrix for method 2
	int* d_sparse_vals = nullptr;
	uint64_t* d_sparse_cols = nullptr;
	uint64_t* d_sparse_row_ptr = nullptr;

	// Transpose CSR sparse matrix for method 2 backprop
	int* d_sparse_vals_t = nullptr;
	uint64_t* d_sparse_cols_t = nullptr;
	uint64_t* d_sparse_row_ptr_t = nullptr;

	~CUDALogSigCache() {
		if (d_lyndon_idx) cudaFree(d_lyndon_idx);
		if (d_level_index) cudaFree(d_level_index);
		if (d_sparse_vals) cudaFree(d_sparse_vals);
		if (d_sparse_cols) cudaFree(d_sparse_cols);
		if (d_sparse_row_ptr) cudaFree(d_sparse_row_ptr);
		if (d_sparse_vals_t) cudaFree(d_sparse_vals_t);
		if (d_sparse_cols_t) cudaFree(d_sparse_cols_t);
		if (d_sparse_row_ptr_t) cudaFree(d_sparse_row_ptr_t);
	}

	// No copy
	CUDALogSigCache(const CUDALogSigCache&) = delete;
	CUDALogSigCache& operator=(const CUDALogSigCache&) = delete;
	CUDALogSigCache() = default;
	CUDALogSigCache(CUDALogSigCache&& o) noexcept
		: d_lyndon_idx(o.d_lyndon_idx), log_sig_len(o.log_sig_len),
		  d_level_index(o.d_level_index), sig_len(o.sig_len),
		  buff1_len(o.buff1_len), threads_per_block(o.threads_per_block),
		  d_sparse_vals(o.d_sparse_vals), d_sparse_cols(o.d_sparse_cols),
		  d_sparse_row_ptr(o.d_sparse_row_ptr),
		  d_sparse_vals_t(o.d_sparse_vals_t), d_sparse_cols_t(o.d_sparse_cols_t),
		  d_sparse_row_ptr_t(o.d_sparse_row_ptr_t)
	{
		o.d_lyndon_idx = nullptr;
		o.d_level_index = nullptr;
		o.d_sparse_vals = nullptr;
		o.d_sparse_cols = nullptr;
		o.d_sparse_row_ptr = nullptr;
		o.d_sparse_vals_t = nullptr;
		o.d_sparse_cols_t = nullptr;
		o.d_sparse_row_ptr_t = nullptr;
	}
};

struct CuLogSigCacheKey {
	int device = 0;
	uint64_t dimension = 0;
	uint64_t degree = 0;

	bool operator==(const CuLogSigCacheKey& other) const noexcept {
		return device == other.device
			&& dimension == other.dimension
			&& degree == other.degree;
	}
};

struct CuLogSigCacheKeyHash {
	size_t operator()(const CuLogSigCacheKey& key) const noexcept {
		size_t h = std::hash<int>{}(key.device);
		h ^= std::hash<uint64_t>{}(key.dimension) + kFibHashConst
			+ (h << 6) + (h >> 2);
		h ^= std::hash<uint64_t>{}(key.degree) + kFibHashConst
			+ (h << 6) + (h >> 2);
		return h;
	}
};

inline CuLogSigCacheKey make_cuda_log_sig_cache_key_(
	uint64_t dimension, uint64_t degree
) {
	CuLogSigCacheKey key;
	CUDA_CHECK(cudaGetDevice(&key.device));
	key.dimension = dimension;
	key.degree = degree;
	return key;
}

// =========================================================================
// Cache management (host-side)
// =========================================================================

std::unordered_map<
	CuLogSigCacheKey, CUDALogSigCache, CuLogSigCacheKeyHash
>& get_cuda_log_sig_cache_map_();
std::mutex& get_cuda_log_sig_cache_mu_();

inline void upload_csr_to_gpu_(
	const CuSparseIntMatrix& mat,
	int*& d_vals, uint64_t*& d_cols, uint64_t*& d_row_ptr
) {
	uint64_t nnz = 0;
	for (uint64_t i = 0; i < mat.n; ++i)
		nnz += mat.rows[i].size();

	std::vector<int> h_vals(nnz);
	std::vector<uint64_t> h_cols(nnz);
	std::vector<uint64_t> h_row_ptr(mat.n + 1);

	uint64_t idx = 0;
	for (uint64_t i = 0; i < mat.n; ++i) {
		h_row_ptr[i] = idx;
		for (const auto& e : mat.rows[i]) {
			h_vals[idx] = e.val;
			h_cols[idx] = e.col;
			++idx;
		}
	}
	h_row_ptr[mat.n] = idx;

	CudaBuf<int> buf_vals;
	CudaBuf<uint64_t> buf_cols;
	CudaBuf<uint64_t> buf_row_ptr;

	if (nnz > 0) {
		buf_vals = CudaBuf<int>(nnz * sizeof(int));
		CUDA_CHECK(cudaMemcpy(buf_vals.get(), h_vals.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

		buf_cols = CudaBuf<uint64_t>(nnz * sizeof(uint64_t));
		CUDA_CHECK(cudaMemcpy(buf_cols.get(), h_cols.data(), nnz * sizeof(uint64_t), cudaMemcpyHostToDevice));
	}

	buf_row_ptr = CudaBuf<uint64_t>((mat.n + 1) * sizeof(uint64_t));
	CUDA_CHECK(cudaMemcpy(buf_row_ptr.get(), h_row_ptr.data(), (mat.n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

	d_vals = buf_vals.release();
	d_cols = buf_cols.release();
	d_row_ptr = buf_row_ptr.release();
}

inline void upload_sparse_matrix_(CUDALogSigCache& entry, uint64_t dimension, uint64_t degree) {
	std::vector<cu_word> lyndon_words = cu_all_lyndon_words(dimension, degree);
	std::vector<uint64_t> lyndon_idx = cu_all_lyndon_idx(dimension, degree);

	CuSparseIntMatrix proj_mat;
	cu_lyndon_proj_matrix(proj_mat, lyndon_words, lyndon_idx, dimension, degree);

	// Compute the inverse (lower triangular, diagonal dropped)
	CuSparseIntMatrix inv_proj_mat;
	proj_mat.inverse(inv_proj_mat);

	// Upload inverse to GPU (used by forward pass)
	upload_csr_to_gpu_(inv_proj_mat, entry.d_sparse_vals, entry.d_sparse_cols, entry.d_sparse_row_ptr);

	// Upload transpose of inverse to GPU (used by backprop)
	CuSparseIntMatrix inv_proj_mat_t;
	inv_proj_mat.transpose(inv_proj_mat_t);
	upload_csr_to_gpu_(inv_proj_mat_t, entry.d_sparse_vals_t, entry.d_sparse_cols_t, entry.d_sparse_row_ptr_t);
}

// =========================================================================
// Disk cache infrastructure (shared format with cpsig)
// =========================================================================

constexpr const char* cu_cache_version = "v1";

class CuCacheFile {
public:
	CuCacheFile(uint64_t dimension, uint64_t degree) {
		auto& dir = get_cuda_cache_dir_();
		if (dir.empty() || !std::filesystem::exists(dir / cu_cache_folder_name))
			throw cache_dir_not_set_error("Unexpected internal error. Cache directory was not set correctly.");
		file_name_ = std::to_string(dimension) + "_" + std::to_string(degree) + "_" + cu_cache_version + ".bin";
		file_path_ = dir / cu_cache_folder_name / file_name_;
	}

	bool exists() const {
		return std::filesystem::exists(file_path_);
	}

	void write(int method, const std::vector<uint64_t>& lyndon_idx,
	           const CuSparseIntMatrix& inv_proj_mat,
	           const CuSparseIntMatrix& inv_proj_mat_t) const {
		std::ofstream out(file_path_, std::ios::binary);
		if (!out)
			throw std::filesystem::filesystem_error(
				"Failed to open CUDA cache file for writing", file_path_,
				std::make_error_code(std::errc::io_error));
		out.write(reinterpret_cast<const char*>(&cu_cache_magic_number), sizeof(cu_cache_magic_number));
		out.write(reinterpret_cast<const char*>(&method), sizeof(method));
		cu_serialize_vector_(out, lyndon_idx);
		inv_proj_mat.serialize(out);
		inv_proj_mat_t.serialize(out);
	}

	void read(int& method, std::vector<uint64_t>& lyndon_idx,
	          CuSparseIntMatrix& inv_proj_mat,
	          CuSparseIntMatrix& inv_proj_mat_t) const {
		std::ifstream in(file_path_, std::ios::binary);
		if (!in)
			throw std::filesystem::filesystem_error(
				"Failed to open CUDA cache file for reading", file_path_,
				std::make_error_code(std::errc::io_error));
		uint64_t magic;
		in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
		if (magic != cu_cache_magic_number)
			throw corrupted_cache_error("Tried to read an invalid cache file. Cache may have been corrupted.");
		in.read(reinterpret_cast<char*>(&method), sizeof(method));
		cu_deserialize_vector_(in, lyndon_idx);
		CuSparseIntMatrix::deserialize(in, inv_proj_mat);
		CuSparseIntMatrix::deserialize(in, inv_proj_mat_t);
	}

private:
	std::string file_name_;
	std::filesystem::path file_path_;
};

// =========================================================================
// prepare / clear (host-side)
// =========================================================================

inline void populate_cuda_cache_entry_(
	CUDALogSigCache& entry,
	const std::vector<uint64_t>& lyndon_idx,
	uint64_t dimension, uint64_t degree
) {
	uint64_t log_sig_len = lyndon_idx.size();
	entry.log_sig_len = log_sig_len;

	CUDA_CHECK(cudaMalloc(&entry.d_lyndon_idx, log_sig_len * sizeof(uint64_t)));
	CUDA_CHECK(cudaMemcpy(entry.d_lyndon_idx, lyndon_idx.data(), log_sig_len * sizeof(uint64_t), cudaMemcpyHostToDevice));

	auto level_index_host = std::make_unique<uint64_t[]>(degree + 2);
	host_populate_level_index(level_index_host.get(), dimension, degree + 2);
	const size_t level_index_bytes = (degree + 2) * sizeof(uint64_t);
	CUDA_CHECK(cudaMalloc(&entry.d_level_index, level_index_bytes));
	CUDA_CHECK(cudaMemcpy(entry.d_level_index, level_index_host.get(), level_index_bytes, cudaMemcpyHostToDevice));

	entry.sig_len = host_sig_length(dimension, degree);
	entry.buff1_len = degree >= 2 ? host_sig_length(dimension, degree - 1) : 1;

	uint64_t max_level_size = level_index_host[degree + 1] - level_index_host[degree];
	entry.threads_per_block = host_choose_threads_per_block(max_level_size);
}

inline void prepare_log_sig_cuda_(uint64_t dimension, uint64_t degree, int method, bool use_disk) {
	auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	auto& cache_map = get_cuda_log_sig_cache_map_();
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());

	auto it = cache_map.find(key);
	if (it != cache_map.end()) {
		// Entry exists - upgrade to method 2 if needed
		if (method == 2 && it->second.d_sparse_row_ptr == nullptr) {
			// Try loading sparse matrices from disk
			if (use_disk) {
				ensure_cuda_cache_dir_();
				CuCacheFile file(dimension, degree);
				if (file.exists()) {
					int disk_method;
					std::vector<uint64_t> disk_lyndon_idx;
					CuSparseIntMatrix disk_inv, disk_inv_t;
					file.read(disk_method, disk_lyndon_idx, disk_inv, disk_inv_t);
					if (disk_method >= 2) {
						upload_csr_to_gpu_(disk_inv, it->second.d_sparse_vals, it->second.d_sparse_cols, it->second.d_sparse_row_ptr);
						upload_csr_to_gpu_(disk_inv_t, it->second.d_sparse_vals_t, it->second.d_sparse_cols_t, it->second.d_sparse_row_ptr_t);
						return;
					}
				}
			}
			upload_sparse_matrix_(it->second, dimension, degree);
		}
		return;
	}

	// Full cache miss - try loading from disk
	if (use_disk) {
		ensure_cuda_cache_dir_();
		CuCacheFile file(dimension, degree);
		if (file.exists()) {
			int disk_method;
			std::vector<uint64_t> lyndon_idx;
			CuSparseIntMatrix inv_proj_mat, inv_proj_mat_t;
			file.read(disk_method, lyndon_idx, inv_proj_mat, inv_proj_mat_t);

			if (disk_method >= method) {
				CUDALogSigCache entry;
				populate_cuda_cache_entry_(entry, lyndon_idx, dimension, degree);

				if (method == 2 && disk_method >= 2) {
					upload_csr_to_gpu_(inv_proj_mat, entry.d_sparse_vals, entry.d_sparse_cols, entry.d_sparse_row_ptr);
					upload_csr_to_gpu_(inv_proj_mat_t, entry.d_sparse_vals_t, entry.d_sparse_cols_t, entry.d_sparse_row_ptr_t);
				}

				cache_map.emplace(key, std::move(entry));
				return;
			}
		}
	}

	// Compute from scratch
	std::vector<cu_word> lyndon_words;
	std::vector<uint64_t> lyndon_idx;
	if (method == 2) {
		lyndon_words = cu_all_lyndon_words(dimension, degree);
		lyndon_idx.reserve(lyndon_words.size());
		for (const auto& w : lyndon_words)
			lyndon_idx.push_back(cu_word_to_idx(w, dimension));
	} else {
		lyndon_idx = cu_all_lyndon_idx(dimension, degree);
	}

	CUDALogSigCache entry;
	populate_cuda_cache_entry_(entry, lyndon_idx, dimension, degree);

	// For method 2, compute and upload the sparse matrix
	CuSparseIntMatrix inv_proj_mat, inv_proj_mat_t;
	if (method == 2) {
		CuSparseIntMatrix proj_mat;
		cu_lyndon_proj_matrix(proj_mat, lyndon_words, lyndon_idx, dimension, degree);
		proj_mat.inverse(inv_proj_mat);
		inv_proj_mat.transpose(inv_proj_mat_t);
		upload_csr_to_gpu_(inv_proj_mat, entry.d_sparse_vals, entry.d_sparse_cols, entry.d_sparse_row_ptr);
		upload_csr_to_gpu_(inv_proj_mat_t, entry.d_sparse_vals_t, entry.d_sparse_cols_t, entry.d_sparse_row_ptr_t);
	}

	// Save to disk
	if (use_disk) {
		ensure_cuda_cache_dir_();
		CuCacheFile file(dimension, degree);
		file.write(method, lyndon_idx, inv_proj_mat, inv_proj_mat_t);
	}

	cache_map.emplace(key, std::move(entry));
}

inline const CUDALogSigCache& get_cuda_log_sig_cache(uint64_t dimension, uint64_t degree, int method = 1) {
	auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	auto& cache_map = get_cuda_log_sig_cache_map_();
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
	auto it = cache_map.find(key);

	// Auto-reload from disk if memory cache is empty
	if (it == cache_map.end()) {
		auto& dir = get_cuda_cache_dir_();
		if (!dir.empty()) {
			try {
				CuCacheFile file(dimension, degree);
				if (file.exists()) {
					int disk_method;
					std::vector<uint64_t> lyndon_idx;
					CuSparseIntMatrix inv_proj_mat, inv_proj_mat_t;
					file.read(disk_method, lyndon_idx, inv_proj_mat, inv_proj_mat_t);

					CUDALogSigCache entry;
					populate_cuda_cache_entry_(entry, lyndon_idx, dimension, degree);

					if (disk_method >= 2) {
						upload_csr_to_gpu_(inv_proj_mat, entry.d_sparse_vals, entry.d_sparse_cols, entry.d_sparse_row_ptr);
						upload_csr_to_gpu_(inv_proj_mat_t, entry.d_sparse_vals_t, entry.d_sparse_cols_t, entry.d_sparse_row_ptr_t);
					}

					cache_map.emplace(key, std::move(entry));
					it = cache_map.find(key);
				}
			} catch (...) {
				// Disk load failed - fall through to error
			}
		}
	}

	if (it == cache_map.end()) {
		throw cache_not_found_error(
			"CUDA log sig cache not found - call prepare_log_sig with device='cuda' first");
	}
	if (method == 2 && it->second.d_sparse_row_ptr == nullptr) {
		throw cache_not_found_error(
			"CUDA log sig cache not found for method 2 - call prepare_log_sig with method=2 and device='cuda' first");
	}
	return it->second;
}

// Forward declarations - defined in cu_log_signature.cu
void free_cuda_log_sig_workspace_();
void free_cuda_log_sig_backprop_workspace_();

// Forward declaration - defined in cu_log_sig_combine.h
void clear_cuda_bch_cache_();

// Forward declaration - defined in cu_branched_signature.cu
void clear_cuda_branched_sig_gpu_cache_();
void clear_cuda_branched_log_sig_gpu_cache_();
void release_sig_kernel_poly_state();

inline void clear_cache_cuda_(bool use_disk) {
	{
		std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
		get_cuda_log_sig_cache_map_().clear();
	}
	clear_cuda_bch_cache_();
	clear_cuda_branched_sig_gpu_cache_();
	clear_cuda_branched_log_sig_gpu_cache_();
	release_sig_kernel_poly_state();
	free_cuda_log_sig_workspace_();
	free_cuda_log_sig_backprop_workspace_();

	if (use_disk) {
		auto& dir = get_cuda_cache_dir_();
		if (dir.empty()) {
			set_default_cuda_cache_dir_();
		}
		auto cache_path = dir / cu_cache_folder_name;
		if (std::filesystem::exists(cache_path)) {
			std::filesystem::remove_all(cache_path);
		}
	}
}
