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
#include <exception>


inline unsigned int get_max_threads() {
	static const unsigned int max_threads = std::thread::hardware_concurrency();
	return max_threads;
}

// Partitions [0, batch_size) into contiguous chunks and spawns up to `n_jobs`
// threads to run `worker(start, end)` on each chunk.
template<typename Worker>
inline void spawn_batch_threads(uint64_t batch_size, int n_jobs, Worker worker) {
	if (n_jobs == 0)
		throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : get_max_threads() + 1 + n_jobs;
	if (max_threads < 1)
		throw std::invalid_argument("received negative n_jobs which is less than max_threads + 1; n_jobs too low");

	const uint64_t num_threads = std::min(static_cast<uint64_t>(max_threads), batch_size);
	std::vector<std::thread> workers;
	workers.reserve(num_threads);

	std::mutex err_mu;
	std::exception_ptr first_err;

	auto wrapped_worker = [&](uint64_t s, uint64_t e) {
		try {
			worker(s, e);
		}
		catch (...) {
			std::lock_guard<std::mutex> lk(err_mu);
			if (!first_err) first_err = std::current_exception();
		}
	};

	const uint64_t chunk = batch_size / num_threads;
	const uint64_t remainder = batch_size % num_threads;
	uint64_t start = 0;
	try {
		for (uint64_t t = 0; t < num_threads; ++t) {
			uint64_t end = start + chunk + (t < remainder ? 1 : 0);
			workers.emplace_back(wrapped_worker, start, end);
			start = end;
		}
	}
	catch (...) {
		for (auto& w : workers) if (w.joinable()) w.join();
		throw;
	}

	for (auto& w : workers)
		w.join();

	if (first_err)
		std::rethrow_exception(first_err);
}

// Contiguous chunk assignment: each thread processes a contiguous block
// of batch items for better cache locality. The fast path lives inline so
// single-path calls (batch_size == 1 || n_jobs == 1) keep their original
// codegen and never touch std::thread.
template<typename T, typename U, typename FN>
void multi_threaded_batch(FN& thread_func, T* path, U* out, uint64_t batch_size, uint64_t flat_path_length, uint64_t result_length, int n_jobs) {
	if (n_jobs == 1 || batch_size == 1) {
		T* path_ptr = path;
		U* out_ptr = out;
		for (uint64_t i = 0; i < batch_size; ++i, path_ptr += flat_path_length, out_ptr += result_length) {
			thread_func(path_ptr, out_ptr);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		T* path_ptr = path + start * flat_path_length;
		U* out_ptr = out + start * result_length;
		for (uint64_t i = start; i < end; ++i, path_ptr += flat_path_length, out_ptr += result_length)
			thread_func(path_ptr, out_ptr);
	});
}


template<typename S, typename T, typename U, typename FN>
void multi_threaded_batch_2(FN& thread_func, S* path1, T* path2, U* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t result_length, int n_jobs) {
	if (n_jobs == 1 || batch_size == 1) {
		S* p1 = path1;
		T* p2 = path2;
		U* o = out;
		for (uint64_t i = 0; i < batch_size; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, o += result_length) {
			thread_func(p1, p2, o);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		S* p1 = path1 + start * flat_path_length_1;
		T* p2 = path2 + start * flat_path_length_2;
		U* o = out + start * result_length;
		for (uint64_t i = start; i < end; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, o += result_length)
			thread_func(p1, p2, o);
	});
}

template<typename R, typename S, typename T, typename U, typename FN>
void multi_threaded_batch_3(FN& thread_func, R* path1, S* path2, T* path3, U* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t flat_path_length_3, uint64_t result_length, int n_jobs) {
	if (n_jobs == 1 || batch_size == 1) {
		R* p1 = path1;
		S* p2 = path2;
		T* p3 = path3;
		U* o = out;
		for (uint64_t i = 0; i < batch_size; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, p3 += flat_path_length_3, o += result_length) {
			thread_func(p1, p2, p3, o);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		R* p1 = path1 + start * flat_path_length_1;
		S* p2 = path2 + start * flat_path_length_2;
		T* p3 = path3 + start * flat_path_length_3;
		U* o = out + start * result_length;
		for (uint64_t i = start; i < end; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, p3 += flat_path_length_3, o += result_length)
			thread_func(p1, p2, p3, o);
	});
}

template<typename T, typename U, typename FN>
void multi_threaded_batch_4(FN& thread_func, const T* path1, T* path2, T* path3, const T* path4, const U* out, uint64_t batch_size, uint64_t flat_path_length_1, uint64_t flat_path_length_2, uint64_t flat_path_length_3, uint64_t flat_path_length_4, uint64_t result_length, int n_jobs) {
	if (n_jobs == 1 || batch_size == 1) {
		const T* p1 = path1;
		T* p2 = path2;
		T* p3 = path3;
		const T* p4 = path4;
		const U* o = out;
		for (uint64_t i = 0; i < batch_size; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, p3 += flat_path_length_3, p4 += flat_path_length_4, o += result_length) {
			thread_func(p1, p2, p3, p4, o);
		}
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, [&](uint64_t start, uint64_t end) {
		const T* p1 = path1 + start * flat_path_length_1;
		T* p2 = path2 + start * flat_path_length_2;
		T* p3 = path3 + start * flat_path_length_3;
		const T* p4 = path4 + start * flat_path_length_4;
		const U* o = out + start * result_length;
		for (uint64_t i = start; i < end; ++i, p1 += flat_path_length_1, p2 += flat_path_length_2, p3 += flat_path_length_3, p4 += flat_path_length_4, o += result_length)
			thread_func(p1, p2, p3, p4, o);
	});
}
