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
	std::vector<uint64_t> linear_a_ptr(device.m2 * device.m + 1, 0);
	std::vector<uint32_t> linear_a_idx;
	for (uint64_t node = 0; node < device.m2; ++node) {
		for (uint64_t input = 0; input < device.m; ++input) {
			const uint64_t row = node * device.m + input;
			linear_a_ptr[row] = linear_a_idx.size();
			if (node < 2)
				continue;
			const uint64_t left = host.bch_left_factor[node];
			const uint64_t right = host.bch_right_factor[node];
			const auto [left_begin, left_end] = host.linear_range[left];
			const auto [right_begin, right_end] = host.linear_range[right];
			const bool active_left = input >= left_begin && input < left_end;
			const bool active_right = input >= right_begin && input < right_end;
			for (uint32_t index = host.comm_a_ptr[input];
				index < host.comm_a_ptr[input + 1]; ++index) {
				const uint32_t partner = host.comm_a_partner[index];
				if (active_left && partner >= right_begin && partner < right_end)
					linear_a_idx.push_back(index << 1);
				if (active_right && partner >= left_begin && partner < left_end)
					linear_a_idx.push_back((index << 1) | 1);
			}
		}
	}
	linear_a_ptr.back() = linear_a_idx.size();
	upload_bch_vector_(device.d_linear_a_ptr, linear_a_ptr);
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
			"log_sig_combine_cuda: degree > 12 not supported");
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
