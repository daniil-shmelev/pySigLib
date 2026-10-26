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

#include "cp_simd_config.h"
#include "hwy/highway.h"
#if defined(CPSIG_CP_BCH_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef CPSIG_CP_BCH_H_
#undef CPSIG_CP_BCH_H_
#else
#define CPSIG_CP_BCH_H_
#endif
#include "cppch.h"
#include "cp_sig_combine.h"
#include "cp_tensor_log.h"
#include "preparation/log_sig/bch_data.h"
#include "../shared/preparation/log_sig/lyndon_words.h"
#include "../shared/preparation/log_sig/bch_cache.h"
#include "macros.h"
#include "cp_simd-inl.h"
#include "cache_lifecycle/cp_bch_cache.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {

// One prepared arithmetic schedule for all dimensions and degrees.
inline constexpr uint64_t bch_tile = 16;
template<std::floating_point T, bool scan = false>
void bch_combine_batch_(
	const T* ls1, const T* ls2, T* out, const BchSchedule& cache, T* memo, uint64_t count, uint64_t input_stride = 0, uint64_t right_width = 0
) {
	const hn::CappedTag<T, 64 / sizeof(T)> d;
	constexpr uint64_t lanes = hn::MaxLanes(decltype(d)());
	const uint64_t m = cache.m, stride = cache.bch_memo_width;
	if (!right_width) right_width = m;
	const uint64_t active_lanes = (count + lanes - 1) / lanes * lanes;
	for (uint64_t k = 0; k < stride; ++k)
		for (uint64_t lane = 0; lane < count; ++lane) {
			if constexpr (!scan) memo[k * bch_tile + lane] = ls1[lane * m + k];
			memo[(stride + k) * bch_tile + lane] = k < right_width
				? ls2[scan ? k * input_stride + lane : lane * right_width + k] : T(0);
		}
	for (const auto& row : cache.sums) {
		if (!out && row.destination == UINT64_MAX) continue;
		alignas(64) T sum[bch_tile];
		T* destination = row.destination != UINT64_MAX
			? memo + row.destination * bch_tile : sum;
		decltype(hn::Zero(d)) totals[bch_tile / lanes];
		for (auto& total : totals) total = hn::Zero(d);
		for (uint64_t q = row.begin; q < row.end; ++q) {
			const auto& term = cache.products[q];
			const auto coefficient = hn::Set(d, static_cast<T>(term.coefficient));
			const T* a = memo + term.left * bch_tile;
			const T* b = memo + term.right * bch_tile;
			HWY_UNROLL(16)
			for (uint64_t block = 0; block < bch_tile / lanes; ++block)
				if (block * lanes < count)
					totals[block] = hn::MulAdd(hn::Mul(hn::LoadU(d, a + block * lanes),
						hn::LoadU(d, b + block * lanes)), coefficient, totals[block]);
		}
		HWY_UNROLL(16)
		for (uint64_t block = 0; block < bch_tile / lanes; ++block)
			if (block * lanes < count) hn::StoreU(totals[block], d, destination + block * lanes);
		if (row.destination != UINT64_MAX) continue;
		if constexpr (scan) {
			if (row.coordinate >= stride) {
				if (row.coordinate < right_width)
					for (uint64_t lane = 0; lane < count; ++lane) sum[lane] += ls2[row.coordinate * input_stride + lane];
				std::fill(sum + count, sum + active_lanes, T(0));
				auto total = hn::Zero(d);
				for (uint64_t lane = 0; lane < active_lanes; lane += lanes) total = hn::Add(total, hn::Load(d, sum + lane));
				out[row.coordinate] += hn::ReduceSum(d, total);
				continue;
			}
			T prefix = out[row.coordinate];
			for (uint64_t lane = 0; lane < count; ++lane) {
				if (row.coordinate < stride) memo[row.coordinate * bch_tile + lane] = prefix;
				prefix += (row.coordinate < right_width ? ls2[row.coordinate * input_stride + lane] : T(0)) + sum[lane];
			}
			out[row.coordinate] = prefix;
		}
		else
			for (uint64_t lane = 0; lane < count; ++lane) {
				const uint64_t i = lane * m + row.coordinate;
				out[i] = ls1[i] + (row.coordinate < right_width ? ls2[lane * right_width + row.coordinate] : T(0)) + sum[lane];
			}
	}
}

template<std::floating_point T>
void batch_bch_combine_(
	const T* left, const T* right, T* out, uint64_t batch,
	const BchSchedule& cache, uint64_t right_width, int n_jobs
) {
	auto work = [&](uint64_t begin, uint64_t end) {
		thread_local std::vector<T> scratch;
		scratch.resize(cache.bch_memo_size * bch_tile);
		for (; begin < end; begin += bch_tile)
			bch_combine_batch_(left + begin * cache.m, right + begin * right_width,
				out + begin * cache.m, cache, scratch.data(), std::min(bch_tile, end - begin), 0, right_width);
	};
	if (n_jobs == 1 || batch < 2) work(0, batch);
	else spawn_batch_threads(batch, n_jobs, work);
}

inline uint64_t bch_backward_workspace_size(const BchSchedule& cache) {
	return (cache.bch_memo_size + cache.bch_gradient_size + cache.m) * bch_tile + cache.m;
}

// Reverse the same arithmetic program. Contributions with a common primal
// factor are summed before multiplying, avoiding repeated gradient scatters.
template<std::floating_point T, bool scan = false>
void bch_combine_backprop_batch_(
	const T* d_out, T* d_left, T* d_right, const T* left, const T* right,
	const BchSchedule& cache, T* workspace, uint64_t count,
	uint64_t input_stride = 0, uint64_t right_width = 0
) {
	const hn::CappedTag<T, 64 / sizeof(T)> d;
	constexpr uint64_t lanes = hn::MaxLanes(decltype(d)());
	const uint64_t m = cache.m, stride = cache.bch_memo_width;
	if (!right_width) right_width = m;
	T* memo = workspace;
	T* gradient = memo + cache.bch_memo_size * bch_tile;
	T* output_gradient = gradient + cache.bch_gradient_size * bch_tile;
	T* result = output_gradient + m * bch_tile;
	if constexpr (scan) std::copy_n(left, m, result);
	bch_combine_batch_<T, scan>(left, right, scan ? result : nullptr,
		cache, memo, count, input_stride, right_width);
	for (const auto& row : cache.reverse_sums) {
		if (row.destination == UINT64_MAX) {
			const uint64_t k = row.coordinate;
			T prefix = scan ? d_out[k] : T(0);
			for (uint64_t lane = count; lane-- > 0;) {
				const T value = scan ? prefix : d_out[lane * m + k];
				output_gradient[k * bch_tile + lane] = value;
				const T dl = k < stride ? gradient[lane] : T(0);
				const T dr = k < stride ? gradient[bch_tile + lane] : T(0);
				if constexpr (scan) prefix += dl;
				else d_left[lane * m + k] = value + dl;
				if (k < right_width)
					d_right[scan ? k * input_stride + lane : lane * right_width + k] = value + dr;
			}
			if constexpr (scan) d_left[k] = prefix;
			continue;
		}
		decltype(hn::Zero(d)) totals[bch_tile / lanes];
		for (auto& total : totals) total = hn::Zero(d);
		for (uint64_t q = row.begin; q < row.end;) {
			const uint64_t factor = cache.reverse_products[q].right;
			decltype(hn::Zero(d)) sums[bch_tile / lanes];
			for (auto& sum : sums) sum = hn::Zero(d);
			do {
				const auto& term = cache.reverse_products[q++];
				const auto coefficient = hn::Set(d, static_cast<T>(term.coefficient));
				HWY_UNROLL(16)
				for (uint64_t block = 0; block < bch_tile / lanes; ++block)
					if (block * lanes < count)
						sums[block] = hn::MulAdd(coefficient,
							hn::LoadU(d, gradient + term.left * bch_tile + block * lanes), sums[block]);
			} while (q < row.end && cache.reverse_products[q].right == factor);
			HWY_UNROLL(16)
			for (uint64_t block = 0; block < bch_tile / lanes; ++block)
				if (block * lanes < count)
					totals[block] = hn::MulAdd(sums[block],
						hn::LoadU(d, memo + factor * bch_tile + block * lanes), totals[block]);
		}
		HWY_UNROLL(16)
		for (uint64_t block = 0; block < bch_tile / lanes; ++block)
			if (block * lanes < count)
				hn::StoreU(totals[block], d, gradient + row.destination * bch_tile + block * lanes);
	}
}

template<std::floating_point T>
void batch_bch_backprop_(
	const T* d_out, T* d_left, T* d_right, const T* left, const T* right,
	uint64_t batch, const BchSchedule& cache, uint64_t right_width, int n_jobs
) {
	auto work = [&](uint64_t begin, uint64_t end) {
		thread_local std::vector<T> scratch;
		scratch.resize(bch_backward_workspace_size(cache));
		for (; begin < end; begin += bch_tile)
			bch_combine_backprop_batch_(d_out + begin * cache.m, d_left + begin * cache.m,
				d_right + begin * right_width, left + begin * cache.m, right + begin * right_width,
				cache, scratch.data(), std::min(bch_tile, end - begin), 0, right_width);
	};
	if (n_jobs == 1 || batch < 2) work(0, batch);
	else spawn_batch_threads(batch, n_jobs, work);
}

// Recover each tile's starting state by BCH inversion, then differentiate its
// fused scan. Workspace is independent of path length; lifts supply geometry.
template<std::floating_point T, typename Lift, typename Pullback>
void bch_path_backprop_(
	const T* d_out, T* d_path, const T* path, uint64_t length, uint64_t dimension,
	const BchSchedule& cache, uint64_t right_width, Lift lift, Pullback pullback
) {
	const uint64_t m = cache.m, ws = bch_backward_workspace_size(cache);
	thread_local std::vector<T> scratch;
	scratch.resize(ws + 2 * m + 2 * right_width * bch_tile + 2 * dimension);
	T* current = scratch.data() + ws;
	T* cotangent = current + m;
	T* segments = cotangent + m;
	T* segment_gradient = segments + right_width * bch_tile;
	T* increment = segment_gradient + right_width * bch_tile;
	T* increment_gradient = increment + dimension;
	std::fill_n(current, m, T(0));
	std::fill_n(d_path, length * dimension, T(0));
	std::copy_n(d_out, m, cotangent);
	auto pack = [&](uint64_t first, uint64_t count, bool reverse) {
		for (uint64_t lane = 0; lane < count; ++lane) {
			const uint64_t segment = first + (reverse ? count - 1 - lane : lane);
			for (uint64_t k = 0; k < dimension; ++k)
				increment[k] = path[(segment + 1) * dimension + k] - path[segment * dimension + k];
			lift(increment, segments + lane, bch_tile);
			if (reverse)
				for (uint64_t k = 0; k < right_width; ++k) segments[k * bch_tile + lane] *= T(-1);
		}
	};
	for (uint64_t first = 0; first < length - 1; first += bch_tile) {
		const uint64_t count = std::min(bch_tile, length - 1 - first);
		pack(first, count, false);
		bch_combine_batch_<T, true>(nullptr, segments, current, cache,
			scratch.data(), count, bch_tile, right_width);
	}
	for (uint64_t end = length - 1; end;) {
		const uint64_t first = (end - 1) / bch_tile * bch_tile, count = end - first;
		pack(first, count, true);
		bch_combine_batch_<T, true>(nullptr, segments, current, cache,
			scratch.data(), count, bch_tile, right_width);
		pack(first, count, false);
		bch_combine_backprop_batch_<T, true>(cotangent, cotangent, segment_gradient,
			current, segments, cache, scratch.data(), count, bch_tile, right_width);
		for (uint64_t lane = 0; lane < count; ++lane) {
			for (uint64_t k = 0; k < dimension; ++k)
				increment[k] = path[(first + lane + 1) * dimension + k] - path[(first + lane) * dimension + k];
			pullback(segment_gradient + lane, increment, increment_gradient, bch_tile);
			for (uint64_t k = 0; k < dimension; ++k) {
				d_path[(first + lane + 1) * dimension + k] += increment_gradient[k];
				d_path[(first + lane) * dimension + k] -= increment_gradient[k];
			}
		}
		end = first;
	}
}

}  // namespace HWY_NAMESPACE
}  // namespace cpsig
HWY_AFTER_NAMESPACE();
#endif
