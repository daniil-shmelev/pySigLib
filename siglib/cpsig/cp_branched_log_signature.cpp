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
#include "cp_branched_log_signature.h"
#include "cp_branched_cache.h"
#include "multithreading.h"
#include "macros.h"

template<std::floating_point T>
void butcher_product_general_(
	const T* X,
	const T* Y,
	T* out,
	const BranchedSigCache& cache
) {
	out[0] = X[0] * Y[0];
	uint64_t num_trees = cache.total_length - 1;

	for (uint64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
		uint64_t flat_idx = tree_idx + 1;
		T val = X[flat_idx] * Y[0] + X[0] * Y[flat_idx];

		uint64_t pos = cache.coproduct_offsets[tree_idx];
		uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

		while (pos < pos_end) {
			uint64_t num_forest = cache.coproduct_data[pos++];
			uint64_t trunk_flat = cache.coproduct_data[pos++];
			T term = Y[trunk_flat];

			for (uint64_t j = 0; j < num_forest; ++j) {
				term *= X[cache.coproduct_data[pos++]];
			}

			val += term;
		}

		out[flat_idx] = val;
	}
}


template<std::floating_point T>
void butcher_product_general_deriv_(
	const T* X,
	const T* Y,
	const T* d_out,
	T* d_X,
	T* d_Y,
	const BranchedSigCache& cache
) {
	d_X[0] += d_out[0] * Y[0];
	d_Y[0] += d_out[0] * X[0];

	uint64_t num_trees = cache.total_length - 1;
	for (uint64_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
		uint64_t flat_idx = tree_idx + 1;
		T d = d_out[flat_idx];
		if (d == static_cast<T>(0)) continue;

		d_X[flat_idx] += d * Y[0];
		d_Y[0] += d * X[flat_idx];
		d_X[0] += d * Y[flat_idx];
		d_Y[flat_idx] += d * X[0];

		uint64_t pos = cache.coproduct_offsets[tree_idx];
		uint64_t pos_end = cache.coproduct_offsets[tree_idx + 1];

		while (pos < pos_end) {
			uint64_t num_forest = cache.coproduct_data[pos++];
			uint64_t trunk_flat = cache.coproduct_data[pos++];
			uint64_t forest_start = pos;
			T forest_product = static_cast<T>(1);

			for (uint64_t j = 0; j < num_forest; ++j) {
				forest_product *= X[cache.coproduct_data[pos++]];
			}

			d_Y[trunk_flat] += d * forest_product;
			for (uint64_t k = 0; k < num_forest; ++k) {
				uint64_t fk_flat = cache.coproduct_data[forest_start + k];
				T partial = d * Y[trunk_flat];
				for (uint64_t j = 0; j < num_forest; ++j) {
					if (j != k)
						partial *= X[cache.coproduct_data[forest_start + j]];
				}
				d_X[fk_flat] += partial;
			}
		}
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_degree2_dim2_one_(
	const T* bsig,
	T* out,
	bool scalar_term
) {
	const T* tail = scalar_term ? bsig + 1 : bsig;
	T* out_tail = scalar_term ? out + 1 : out;
	const T x0 = tail[0];
	const T x1 = tail[1];
	const T half = static_cast<T>(0.5);

	if (scalar_term)
		out[0] = static_cast<T>(0);
	out_tail[0] = x0;
	out_tail[1] = x1;
	out_tail[2] = tail[2] - half * x0 * x0;
	out_tail[3] = tail[3] - half * x0 * x1;
	out_tail[4] = tail[4] - half * x1 * x0;
	out_tail[5] = tail[5] - half * x1 * x1;
}


template<std::floating_point T>
void branched_sig_to_log_sig_degree2_dim2_backprop_one_(
	const T* bsig,
	const T* derivs,
	T* out,
	bool scalar_term
) {
	const T* tail = scalar_term ? bsig + 1 : bsig;
	const T* deriv_tail = scalar_term ? derivs + 1 : derivs;
	T* out_tail = scalar_term ? out + 1 : out;
	const T x0 = tail[0];
	const T x1 = tail[1];
	const T d00 = deriv_tail[2];
	const T d01 = deriv_tail[3];
	const T d10 = deriv_tail[4];
	const T d11 = deriv_tail[5];
	const T neg_half = static_cast<T>(-0.5);

	if (scalar_term)
		out[0] = static_cast<T>(0);
	out_tail[0] = deriv_tail[0] - d00 * x0 + neg_half * (d01 + d10) * x1;
	out_tail[1] = deriv_tail[1] + neg_half * (d01 + d10) * x0 - d11 * x1;
	out_tail[2] = d00;
	out_tail[3] = d01;
	out_tail[4] = d10;
	out_tail[5] = d11;
}


template<std::floating_point T>
void branched_sig_to_log_sig_degree2_one_(
	const T* bsig,
	T* out,
	uint64_t dimension,
	bool scalar_term
) {
	if (dimension == 2) {
		branched_sig_to_log_sig_degree2_dim2_one_(bsig, out, scalar_term);
		return;
	}
	const T* tail = scalar_term ? bsig + 1 : bsig;
	T* out_tail = scalar_term ? out + 1 : out;
	if (scalar_term)
		out[0] = static_cast<T>(0);

	for (uint64_t i = 0; i < dimension; ++i)
		out_tail[i] = tail[i];

	const T* level1 = tail;
	const T* level2 = tail + dimension;
	T* out_level2 = out_tail + dimension;
	const T half = static_cast<T>(0.5);

	for (uint64_t root = 0; root < dimension; ++root) {
		for (uint64_t child = 0; child < dimension; ++child) {
			uint64_t idx = root * dimension + child;
			out_level2[idx] = level2[idx] - half * level1[root] * level1[child];
		}
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_degree2_backprop_one_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t dimension,
	bool scalar_term
) {
	if (dimension == 2) {
		branched_sig_to_log_sig_degree2_dim2_backprop_one_(bsig, derivs, out, scalar_term);
		return;
	}
	const T* tail = scalar_term ? bsig + 1 : bsig;
	const T* deriv_tail = scalar_term ? derivs + 1 : derivs;
	T* out_tail = scalar_term ? out + 1 : out;
	if (scalar_term)
		out[0] = static_cast<T>(0);

	for (uint64_t i = 0; i < dimension; ++i)
		out_tail[i] = deriv_tail[i];

	const T* level1 = tail;
	const T* deriv_level2 = deriv_tail + dimension;
	T* out_level2 = out_tail + dimension;
	const T neg_half = static_cast<T>(-0.5);

	for (uint64_t root = 0; root < dimension; ++root) {
		for (uint64_t child = 0; child < dimension; ++child) {
			uint64_t idx = root * dimension + child;
			T d = deriv_level2[idx];
			out_level2[idx] = d;
			out_tail[root] += neg_half * d * level1[child];
			out_tail[child] += neg_half * d * level1[root];
		}
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_one_(
	const T* bsig,
	T* out,
	const BranchedSigCache& cache,
	bool scalar_term
) {
	uint64_t total_len = cache.total_length;
	if (cache.max_nodes == 0) {
		if (scalar_term)
			out[0] = static_cast<T>(0);
		return;
	}
	if (cache.max_nodes == 2) {
		branched_sig_to_log_sig_degree2_one_(bsig, out, cache.dimension, scalar_term);
		return;
	}
	std::vector<T> h(total_len, static_cast<T>(0));
	std::vector<T> power(total_len, static_cast<T>(0));
	std::vector<T> next_power(total_len, static_cast<T>(0));

	const T* tail = scalar_term ? bsig + 1 : bsig;
	std::memcpy(h.data() + 1, tail, (total_len - 1) * sizeof(T));
	std::memcpy(power.data(), h.data(), total_len * sizeof(T));

	std::vector<T> full_out(total_len, static_cast<T>(0));
	std::memcpy(full_out.data(), power.data(), total_len * sizeof(T));

	for (uint64_t k = 2; k <= cache.max_nodes; ++k) {
		butcher_product_general_(power.data(), h.data(), next_power.data(), cache);
		T coeff = (k % 2 == 0)
			? static_cast<T>(-1.) / static_cast<T>(k)
			: static_cast<T>(1.) / static_cast<T>(k);
		for (uint64_t i = 1; i < total_len; ++i)
			full_out[i] += coeff * next_power[i];
		std::swap(power, next_power);
	}

	full_out[0] = static_cast<T>(0);
	if (scalar_term) {
		std::memcpy(out, full_out.data(), total_len * sizeof(T));
	} else {
		std::memcpy(out, full_out.data() + 1, (total_len - 1) * sizeof(T));
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_backprop_one_(
	const T* bsig,
	const T* derivs,
	T* out,
	const BranchedSigCache& cache,
	bool scalar_term
) {
	uint64_t total_len = cache.total_length;
	if (cache.max_nodes == 0) {
		if (scalar_term)
			out[0] = static_cast<T>(0);
		return;
	}
	if (cache.max_nodes == 2) {
		branched_sig_to_log_sig_degree2_backprop_one_(bsig, derivs, out, cache.dimension, scalar_term);
		return;
	}
	std::vector<std::vector<T>> powers(cache.max_nodes + 1, std::vector<T>(total_len, static_cast<T>(0)));
	std::vector<std::vector<T>> d_powers(cache.max_nodes + 1, std::vector<T>(total_len, static_cast<T>(0)));
	std::vector<T> h(total_len, static_cast<T>(0));
	std::vector<T> d_h(total_len, static_cast<T>(0));
	std::vector<T> full_derivs(total_len, static_cast<T>(0));

	const T* tail = scalar_term ? bsig + 1 : bsig;
	std::memcpy(h.data() + 1, tail, (total_len - 1) * sizeof(T));
	std::memcpy(powers[1].data(), h.data(), total_len * sizeof(T));

	for (uint64_t k = 2; k <= cache.max_nodes; ++k)
		butcher_product_general_(powers[k - 1].data(), h.data(), powers[k].data(), cache);

	if (scalar_term) {
		std::memcpy(full_derivs.data(), derivs, total_len * sizeof(T));
	} else {
		std::memcpy(full_derivs.data() + 1, derivs, (total_len - 1) * sizeof(T));
	}
	full_derivs[0] = static_cast<T>(0);

	for (uint64_t k = 1; k <= cache.max_nodes; ++k) {
		T coeff;
		if (k == 1) {
			coeff = static_cast<T>(1);
		} else {
			coeff = (k % 2 == 0)
				? static_cast<T>(-1.) / static_cast<T>(k)
				: static_cast<T>(1.) / static_cast<T>(k);
		}
		for (uint64_t i = 1; i < total_len; ++i)
			d_powers[k][i] += coeff * full_derivs[i];
	}

	for (uint64_t k = cache.max_nodes; k >= 2; --k) {
		butcher_product_general_deriv_(
			powers[k - 1].data(), h.data(), d_powers[k].data(),
			d_powers[k - 1].data(), d_h.data(), cache);
	}
	for (uint64_t i = 1; i < total_len; ++i)
		d_h[i] += d_powers[1][i];
	d_h[0] = static_cast<T>(0);

	if (scalar_term) {
		std::memcpy(out, d_h.data(), total_len * sizeof(T));
	} else {
		std::memcpy(out, d_h.data() + 1, (total_len - 1) * sizeof(T));
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_(
	const T* bsig,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = scalar_term ? total_len : total_len - 1;

	if (max_nodes == 2) {
		auto work_range = [&](uint64_t start, uint64_t end) {
			for (uint64_t b = start; b < end; ++b)
				branched_sig_to_log_sig_degree2_one_(bsig + b * stride, out + b * stride, dimension, scalar_term);
		};

		uint64_t work_items = batch_size * dimension * dimension;
		if (n_jobs == 1 || work_items <= 1000000) {
			work_range(0, batch_size);
			return;
		}
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");
		if (n_jobs < 0 && dimension <= 2)
			max_threads = std::min<unsigned int>(max_threads, 4);
		else if (n_jobs < 0 && dimension <= 16 && sizeof(T) == sizeof(float))
			max_threads = std::min<unsigned int>(max_threads, 16);

		uint64_t num_threads = std::min<uint64_t>(max_threads, batch_size);
		uint64_t chunk = (batch_size + num_threads - 1) / num_threads;
		std::vector<std::thread> workers;
		for (uint64_t t = 0; t < num_threads; ++t) {
			uint64_t start = t * chunk;
			uint64_t end = std::min<uint64_t>(batch_size, start + chunk);
			if (start >= end) break;
			workers.emplace_back([&, start, end]() { work_range(start, end); });
		}
		for (auto& w : workers) w.join();
		return;
	}

	auto work = [&](uint64_t b) {
		branched_sig_to_log_sig_one_(bsig + b * stride, out + b * stride, cache, scalar_term);
	};

	if (n_jobs == 1 || batch_size == 1) {
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b);
	}
	else {
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		const unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");

		std::vector<std::thread> workers;
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				for (uint64_t b = t; b < batch_size; b += max_threads)
					work(b);
			});
		}
		for (auto& w : workers) w.join();
	}
}


template<std::floating_point T>
void branched_sig_to_log_sig_backprop_(
	const T* bsig,
	const T* derivs,
	T* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t max_nodes,
	int n_jobs,
	bool planar,
	bool scalar_term
) {
	const auto& cache = get_branched_sig_cache(dimension, max_nodes, planar);
	uint64_t total_len = cache.total_length;
	uint64_t stride = scalar_term ? total_len : total_len - 1;

	if (max_nodes == 2) {
		auto work_range = [&](uint64_t start, uint64_t end) {
			for (uint64_t b = start; b < end; ++b) {
				branched_sig_to_log_sig_degree2_backprop_one_(
					bsig + b * stride, derivs + b * stride, out + b * stride, dimension, scalar_term);
			}
		};

		uint64_t work_items = batch_size * dimension * dimension;
		if (n_jobs == 1 || work_items <= 1000000) {
			work_range(0, batch_size);
			return;
		}
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");
		if (n_jobs < 0 && dimension <= 2)
			max_threads = std::min<unsigned int>(max_threads, 4);
		else if (n_jobs < 0 && dimension <= 16 && sizeof(T) == sizeof(float))
			max_threads = std::min<unsigned int>(max_threads, 16);

		uint64_t num_threads = std::min<uint64_t>(max_threads, batch_size);
		uint64_t chunk = (batch_size + num_threads - 1) / num_threads;
		std::vector<std::thread> workers;
		for (uint64_t t = 0; t < num_threads; ++t) {
			uint64_t start = t * chunk;
			uint64_t end = std::min<uint64_t>(batch_size, start + chunk);
			if (start >= end) break;
			workers.emplace_back([&, start, end]() { work_range(start, end); });
		}
		for (auto& w : workers) w.join();
		return;
	}

	auto work = [&](uint64_t b) {
		branched_sig_to_log_sig_backprop_one_(
			bsig + b * stride, derivs + b * stride, out + b * stride, cache, scalar_term);
	};

	if (n_jobs == 1 || batch_size == 1) {
		for (uint64_t b = 0; b < batch_size; ++b)
			work(b);
	}
	else {
		if (n_jobs == 0)
			throw std::invalid_argument("n_jobs cannot be 0");
		const unsigned int max_threads = n_jobs > 0
			? static_cast<unsigned int>(n_jobs)
			: get_max_threads() + 1 + n_jobs;
		if (max_threads < 1)
			throw std::invalid_argument("n_jobs too low");

		std::vector<std::thread> workers;
		for (unsigned int t = 0; t < max_threads && t < batch_size; ++t) {
			workers.emplace_back([&, t, max_threads]() {
				for (uint64_t b = t; b < batch_size; b += max_threads)
					work(b);
			});
		}
		for (auto& w : workers) w.join();
	}
}



extern "C" {

	CPSIG_API int branched_sig_to_log_sig_f(const float* bsig, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<float>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_d(const double* bsig, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_<double>(bsig, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_f(const float* bsig, const float* derivs, float* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<float>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

	CPSIG_API int branched_sig_to_log_sig_backprop_d(const double* bsig, const double* derivs, double* out, uint64_t batch_size, uint64_t dimension, uint64_t max_nodes, int n_jobs, bool planar, bool scalar_term) noexcept {
		SAFE_CALL(branched_sig_to_log_sig_backprop_<double>(bsig, derivs, out, batch_size, dimension, max_nodes, n_jobs, planar, scalar_term));
	}

}
