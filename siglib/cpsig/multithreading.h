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
#include <memory>

unsigned int get_max_threads();
void run_batch_threads(uint64_t batch_size, int n_jobs, void* context, void (*worker)(void*, uint64_t, uint64_t));

template<typename Worker>
inline void spawn_batch_threads(uint64_t batch_size, int n_jobs, Worker worker) {
    run_batch_threads(batch_size, n_jobs, &worker, [](void* context, uint64_t begin, uint64_t end) {
        (*static_cast<Worker*>(context))(begin, end);
    });
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
