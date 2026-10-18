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

#include "multithreading.h"
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>


unsigned int get_max_threads() {
	static const unsigned int max_threads = std::thread::hardware_concurrency();
	return max_threads;
}

struct tbb_arena_cache {
	std::unique_ptr<oneapi::tbb::task_arena> arena;
	int threads = 0;
};

oneapi::tbb::task_arena& get_tbb_arena(int threads) {
	thread_local tbb_arena_cache cache;
	if (cache.threads != threads || !cache.arena) {
		cache.arena = std::make_unique<oneapi::tbb::task_arena>(threads);
		cache.threads = threads;
	}
	return *cache.arena;
}

// Partitions [0, batch_size) into contiguous chunks and spawns up to `n_jobs`
// threads to run `worker(start, end)` on each chunk.
void run_batch_threads(uint64_t batch_size, int n_jobs, void* context, void (*worker)(void*, uint64_t, uint64_t)) {
	if (n_jobs == 0)
		throw std::invalid_argument("n_jobs cannot be 0");
	const int max_threads = n_jobs > 0 ? n_jobs : get_max_threads() + 1 + n_jobs;
	if (max_threads < 1)
		throw std::invalid_argument("received negative n_jobs which is less than max_threads + 1; n_jobs too low");
	if (batch_size == 0)
		return;

	const uint64_t num_threads = std::min(static_cast<uint64_t>(max_threads), batch_size);
	std::mutex err_mu;
	std::exception_ptr first_err;

	auto wrapped_worker = [&](uint64_t s, uint64_t e) {
		try {
			worker(context, s, e);
		}
		catch (...) {
			std::lock_guard<std::mutex> lk(err_mu);
			if (!first_err) first_err = std::current_exception();
		}
	};

	const uint64_t chunk = batch_size / num_threads;
	const uint64_t remainder = batch_size % num_threads;
	get_tbb_arena(static_cast<int>(num_threads)).execute([&] {
		oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint64_t>(0, num_threads, 1),
			[&](const oneapi::tbb::blocked_range<uint64_t>& range) {
				for (uint64_t t = range.begin(); t != range.end(); ++t) {
					const uint64_t start = t * chunk + std::min(t, remainder);
					const uint64_t end = start + chunk + (t < remainder ? 1 : 0);
					wrapped_worker(start, end);
				}
			});
	});

	if (first_err)
		std::rethrow_exception(first_err);
}
