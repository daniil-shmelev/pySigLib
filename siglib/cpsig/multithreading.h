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

template<typename T>
struct batch_data {
	T* ptr;
	uint64_t stride;
};

template<typename T>
inline batch_data<T> make_batch(T* ptr, uint64_t stride) {
	return { ptr, stride };
}

template<typename T>
inline auto batch_ptr(batch_data<T> arg, uint64_t i) {
	return arg.ptr + i * arg.stride;
}

template<typename FN, typename... Args>
void multi_threaded_batch(FN& thread_func, uint64_t batch_size, int n_jobs, Args... args) {
	auto work_range = [&](uint64_t start, uint64_t end) {
		for (uint64_t i = start; i < end; ++i)
			thread_func(batch_ptr(args, i)...);
	};

	if (batch_size == 0 || n_jobs == 1 || batch_size == 1) {
		work_range(0, batch_size);
		return;
	}

	spawn_batch_threads(batch_size, n_jobs, work_range);
}
