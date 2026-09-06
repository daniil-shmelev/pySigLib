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

#include "cu_runtime_utils.h"
#include "cache_lifecycle/cu_log_sig_cache.h"
#include "../shared/preparation/log_sig/bch_cache.h"
#include "../shared/preparation/log_sig/bch_data.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

template<typename T>
inline void upload_bch_vector_(T*& device, const std::vector<T>& host) {
	if (host.empty())
		return;
	CudaBuf<T> buffer(host.size() * sizeof(T));
	CUDA_CHECK(cudaMemcpy(
		buffer.get(), host.data(), host.size() * sizeof(T),
		cudaMemcpyHostToDevice));
	device = buffer.release();
}

inline void upload_balanced_commutator_plan_(
	CUDABchCache& device, const BchCache& host
) {
	std::vector<uint32_t> order(host.m);
	std::iota(order.begin(), order.end(), 0u);
	const auto row_work = [&host](uint32_t row) {
		return host.comm_k_ptr[row + 1] - host.comm_k_ptr[row];
	};
	std::stable_sort(order.begin(), order.end(),
		[&](uint32_t left, uint32_t right) {
			if (host.coordinate_weights[left] != host.coordinate_weights[right])
				return host.coordinate_weights[left]
					< host.coordinate_weights[right];
			return row_work(left) < row_work(right);
		});

	struct DivergentWork {
		uint64_t canonical = 0;
		uint64_t balanced = 0;
	};
	const auto divergent_range_work = [&order, &row_work](
		uint64_t begin, uint64_t end
	) {
		DivergentWork work;
		for (uint64_t base = begin; base < end; base += 32) {
			DivergentWork longest;
			for (uint64_t logical = base;
				logical < (std::min)(base + 32, end); ++logical) {
				longest.canonical = (std::max)(longest.canonical,
					static_cast<uint64_t>(row_work(
						static_cast<uint32_t>(logical))));
				longest.balanced = (std::max)(longest.balanced,
					static_cast<uint64_t>(row_work(order[logical])));
			}
			work.canonical += longest.canonical;
			work.balanced += longest.balanced;
		}
		return work;
	};
	const auto linear_divergent_work = [&] {
		DivergentWork work;
		for (uint32_t node : host.live_bch_nodes) {
			const auto [begin, end] = host.linear_range[node];
			const auto range_work = divergent_range_work(begin, end);
			work.canonical += range_work.canonical;
			work.balanced += range_work.balanced;
		}
		return work;
	};
	const auto dense_work = host.live_bch_nodes.empty()
		? DivergentWork{}
		: divergent_range_work(0, host.m);
	const auto linear_work = linear_divergent_work();
	const auto worthwhile = [](const auto& work) {
		return work.canonical != 0
			&& static_cast<long double>(work.balanced)
				<= 0.8L * static_cast<long double>(work.canonical);
	};
	const bool use_for_dense = worthwhile(dense_work);
	const bool use_for_linear = worthwhile(linear_work);
	if (!use_for_dense && !use_for_linear)
		return;

	std::vector<uint32_t> ptr(host.m + 1), i, j;
	std::vector<int> value;
	i.reserve(host.comm_k_i.size());
	j.reserve(host.comm_k_j.size());
	value.reserve(host.comm_k_val.size());
	for (uint64_t logical = 0; logical < host.m; ++logical) {
		const uint32_t k = order[logical];
		const uint32_t begin = host.comm_k_ptr[k];
		const uint32_t end = host.comm_k_ptr[k + 1];
		i.insert(i.end(),
			host.comm_k_i.begin() + begin, host.comm_k_i.begin() + end);
		j.insert(j.end(),
			host.comm_k_j.begin() + begin, host.comm_k_j.begin() + end);
		value.insert(value.end(),
			host.comm_k_val.begin() + begin, host.comm_k_val.begin() + end);
		ptr[logical + 1] = static_cast<uint32_t>(i.size());
	}

	static_assert(sizeof(int) == sizeof(uint32_t));
	constexpr size_t alignment = 128 / sizeof(uint32_t);
	const auto padded = [](size_t size) {
		return (size + alignment - 1) / alignment * alignment;
	};
	const size_t ptr_offset = padded(order.size());
	const size_t i_offset = ptr_offset + padded(ptr.size());
	const size_t j_offset = i_offset + padded(i.size());
	const size_t value_offset = j_offset + padded(j.size());
	const size_t elements = value_offset + value.size();
	CudaBuf<uint32_t> storage(checked_cuda_size_mul(
		elements, sizeof(uint32_t), "CUDA balanced commutator plan"));
	auto* cursor = storage.get();
	device.commutator_plan = CUDACommutatorPlan{
		std::move(storage),
		{
			cursor,
			cursor + ptr_offset,
			cursor + i_offset,
			cursor + j_offset,
			reinterpret_cast<int*>(cursor + value_offset)
		},
		use_for_dense,
		use_for_linear
	};
	CUDA_CHECK(cudaMemcpy(cursor, order.data(),
		order.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cursor + ptr_offset, ptr.data(),
		ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cursor + i_offset, i.data(),
		i.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cursor + j_offset, j.data(),
		j.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<int*>(cursor + value_offset), value.data(),
		value.size() * sizeof(int), cudaMemcpyHostToDevice));
}

inline void upload_linear_forward_plan_(CUDABchCache& device, const BchCache& host) {
	if (host.linear_forward.empty())
		return;
	const size_t rows = checked_cuda_size_mul(
		static_cast<size_t>(device.m2), static_cast<size_t>(device.m), "CUDA BCH forward plan");
	std::vector<uint32_t> ptr(checked_cuda_size_add(rows, size_t(1), "CUDA BCH forward plan"), 0);
	std::vector<uint32_t> i, j;
	std::vector<int> value;
	std::vector<uint8_t> live(device.m2, 0);
	for (uint32_t node : host.live_bch_nodes)
		live[node] = 1;
	for (uint64_t node = 0; node < device.m2; ++node) {
		const auto& plan = host.linear_forward[node];
		const auto [begin, end] = host.linear_range[node];
		for (uint64_t k = 0; k < device.m; ++k) {
			ptr[node * device.m + k] = static_cast<uint32_t>(i.size());
			if (!live[node] || k < begin || k >= end)
				continue;
			for (uint32_t q = plan.row_ptr[k - begin]; q < plan.row_ptr[k - begin + 1]; ++q) {
				const auto& entry = plan.entries[q];
				if (entry.orientation & 1) {
					i.push_back(entry.i);
					j.push_back(entry.j);
					value.push_back(entry.coefficient);
				}
				if (entry.orientation & 2) {
					i.push_back(entry.j);
					j.push_back(entry.i);
					value.push_back(-entry.coefficient);
				}
			}
			if (i.size() > std::numeric_limits<uint32_t>::max())
				throw std::overflow_error("CUDA BCH forward plan exceeds uint32 offsets");
		}
	}
	ptr.back() = static_cast<uint32_t>(i.size());
	const size_t i_offset = ptr.size();
	const size_t j_offset = checked_cuda_size_add(i_offset, i.size(), "CUDA BCH forward plan");
	const size_t value_offset = checked_cuda_size_add(j_offset, j.size(), "CUDA BCH forward plan");
	CudaBuf<uint32_t> storage(checked_cuda_size_mul(
		checked_cuda_size_add(value_offset, value.size(), "CUDA BCH forward plan"),
		sizeof(uint32_t), "CUDA BCH forward plan"));
	auto* cursor = storage.get();
	CUDA_CHECK(cudaMemcpy(cursor, ptr.data(), ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	if (!i.empty()) {
		CUDA_CHECK(cudaMemcpy(cursor + i_offset, i.data(), i.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cursor + j_offset, j.data(), j.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cursor + value_offset, value.data(), value.size() * sizeof(int), cudaMemcpyHostToDevice));
	}
	device.linear_forward = { nullptr, cursor, cursor + i_offset, cursor + j_offset,
		reinterpret_cast<int*>(cursor + value_offset) };
	device.linear_forward_storage = std::move(storage);
}

inline CUDABchCache upload_bch_cache_to_device_(const BchCache& host) {
	CUDABchCache device;
	device.m = host.m;
	device.m2 = host.bch_coefficients.size();
	device.bch_plan.size = host.live_bch_nodes.size();
	upload_bch_vector_(device.d_bch_coefficients, host.bch_coefficients);
	upload_bch_vector_(device.d_bch_left_factor, host.bch_left_factor);
	upload_bch_vector_(device.d_bch_right_factor, host.bch_right_factor);
	if (!host.all_bch_nodes_live)
		upload_bch_vector_(device.bch_plan.nodes, host.live_bch_nodes);
	upload_bch_vector_(device.d_comm_k_ptr, host.comm_k_ptr);
	upload_bch_vector_(device.d_comm_k_i, host.comm_k_i);
	upload_bch_vector_(device.d_comm_k_j, host.comm_k_j);
	upload_bch_vector_(device.d_comm_k_val, host.comm_k_val);
	upload_balanced_commutator_plan_(device, host);
	upload_linear_forward_plan_(device, host);
	upload_bch_vector_(device.d_comm_a_ptr, host.comm_a_ptr);
	upload_bch_vector_(device.d_comm_a_k, host.comm_a_k);
	upload_bch_vector_(device.d_comm_a_partner, host.comm_a_partner);
	upload_bch_vector_(device.d_comm_a_signed_c, host.comm_a_signed_c);

	std::vector<uint64_t> ranges(2 * device.m2, 0);
	for (uint64_t node = 0; node < device.m2; ++node) {
		ranges[2 * node] = host.linear_range[node].first;
		ranges[2 * node + 1] = host.linear_range[node].second;
	}
	upload_bch_vector_(device.d_linear_range, ranges);

	if (host.comm_a_k.size()
		> (std::numeric_limits<uint32_t>::max() >> 1))
		throw std::overflow_error(
			"CUDA BCH linear reverse plan exceeds uint32 packing");
	const auto supported = [&host](uint64_t node, uint64_t input) {
		const auto [begin, end] = host.linear_range[node];
		if (input < begin || input >= end)
			return false;
		if (node < 2 || host.linear_forward.empty())
			return true;
		const auto& ptr = host.linear_forward[node].row_ptr;
		return ptr[input - begin] != ptr[input - begin + 1];
	};
	std::vector<uint32_t> linear_a_ptr(device.m2 * device.m + 1, 0);
	std::vector<uint32_t> linear_a_idx;
	for (uint64_t node = 0; node < device.m2; ++node) {
		for (uint64_t input = 0; input < device.m; ++input) {
			const uint64_t row = node * device.m + input;
			linear_a_ptr[row] = static_cast<uint32_t>(linear_a_idx.size());
			if (node < 2)
				continue;
			const uint64_t left = host.bch_left_factor[node];
			const uint64_t right = host.bch_right_factor[node];
			const bool active_left = supported(left, input);
			const bool active_right = supported(right, input);
			for (uint32_t index = host.comm_a_ptr[input];
				index < host.comm_a_ptr[input + 1]; ++index) {
				const uint32_t partner = host.comm_a_partner[index];
				if (active_left && supported(right, partner))
					linear_a_idx.push_back(index << 1);
				if (active_right && supported(left, partner))
					linear_a_idx.push_back((index << 1) | 1);
			}
			if (linear_a_idx.size() > std::numeric_limits<uint32_t>::max())
				throw std::overflow_error("CUDA BCH reverse plan exceeds uint32 offsets");
		}
	}
	linear_a_ptr.back() = static_cast<uint32_t>(linear_a_idx.size());
	upload_bch_vector_(device.d_linear_a_ptr32, linear_a_ptr);
	upload_bch_vector_(device.d_linear_a_idx, linear_a_idx);

	const uint64_t nodes = device.bch_plan.size;
	device.linear_dense_forward_work = nodes
		* (device.m + host.comm_k_i.size());
	for (uint32_t node : host.live_bch_nodes) {
		const auto [begin, end] = host.linear_range[node];
		device.linear_active_forward_work += end - begin;
		device.linear_zero_work += begin + device.m - end;
		for (uint64_t output = begin; output < end; ++output)
			device.linear_active_forward_work +=
				host.comm_k_ptr[output + 1] - host.comm_k_ptr[output];
	}
	return device;
}

inline void prepare_cuda_log_sig_method3_(
	uint64_t dimension,
	uint64_t degree,
	bool use_disk = false
) {
	if (degree > BCH_MAX_HARDCODED_DEGREE)
		throw std::runtime_error(
			"log_sig_combine_cuda: degree > 20 not supported");
	const auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	{
		std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
		const auto found = get_cuda_log_sig_cache_map_().find(key);
		if (found != get_cuda_log_sig_cache_map_().end()
			&& found->second.method >= 2
			&& found->second.bch)
			return;
	}
	if (use_disk)
		ensure_cuda_cache_dir_();
	const auto cache_directory = get_cuda_cache_dir_() / cu_cache_folder_name;
	LogSigCache host_cache(
		dimension, degree, 3, cache_directory, use_disk);
	auto uploaded = std::make_unique<CUDABchCache>(
		upload_bch_cache_to_device_(host_cache.bch()));
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
	auto [found, inserted] = get_cuda_log_sig_cache_map_().try_emplace(key);
	if (found->second.method < 2)
		upload_log_sig_basis_(
			found->second, host_cache.basis(2), dimension, degree);
	if (!found->second.bch)
		found->second.bch = std::move(uploaded);
}

inline const CUDABchCache& get_cuda_bch_cache_(
	uint64_t dimension,
	uint64_t degree
) {
	const auto key = make_cuda_log_sig_cache_key_(dimension, degree);
	std::lock_guard<std::mutex> lock(get_cuda_log_sig_cache_mu_());
	const auto found = get_cuda_log_sig_cache_map_().find(key);
	if (found == get_cuda_log_sig_cache_map_().end() || !found->second.bch)
		throw cache_not_found_error(
			"CUDA BCH cache not found - call prepare_log_sig with method=3 and device='cuda' first");
	return *found->second.bch;
}
