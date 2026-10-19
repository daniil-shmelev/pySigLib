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

#include "cusig.h"
#include "cu_runtime_utils.h"
#include "cu_sig_coef.h"
#include "cu_atomic.h"
#include "cu_macros.h"
#include "pascal_coefficient.h"
#include <algorithm>
#include <mutex>
#include <type_traits>
#include <vector>

__constant__ double c_one_over_fact_d[SIG_COEF_CUDA_MAX_DEGREE + 1];
__constant__ float c_one_over_fact_f[SIG_COEF_CUDA_MAX_DEGREE + 1];
static std::mutex c_one_over_fact_mu;

template<typename T, uint64_t DEGREE, bool Augmented>
__global__ void sig_coef_kernel_fixed_degree(
	const T* __restrict__ path,
	T* __restrict__ out,
	const uint64_t* __restrict__ multi_idx,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ out_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t result_length,
	bool prefixes,
	uint64_t work_offset,
	uint64_t work_size,
	const uint64_t* __restrict__ word_order,
	bool time_aug, bool lead_lag, T end_time
) {
	const uint64_t local_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (local_idx >= work_size) return;
	const uint64_t global_idx = work_offset + local_idx;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = word_order[global_idx % num_multi_idx];

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t out_offset = out_offsets[word_idx];

	const uint64_t* my_multi_idx_ptr = multi_idx + idx_offset;
	T* my_out = out + batch_idx * result_length + out_offset;
	const T* my_path = path + batch_idx * path_length * dimension;
	const uint64_t segments = (path_length - 1) * (Augmented && lead_lag ? 2 : 1);
	auto point = [&](uint64_t vertex, uint64_t channel) {
		if constexpr (!Augmented) return my_path[vertex * dimension + channel];
		else return pascal_coefficient::point(my_path, vertex, channel, dimension, segments, time_aug, lead_lag, end_time);
	};

	// Cache channel indices and 1/k! in registers
	uint64_t channels[DEGREE];
	T ovf[DEGREE + 1];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		channels[i] = my_multi_idx_ptr[i];
		ovf[i + 1] = one_over_fact[i + 1];
	}

	T coefs[DEGREE + 1];

	// First segment
	coefs[0] = T(1);
	T init_prod = T(1);
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		init_prod *= (point(1, channels[i]) - point(0, channels[i]));
		coefs[i + 1] = init_prod * ovf[i + 1];
	}

	// Main loop: Horner-based Chen's relation
	for (uint64_t t = 2; t <= segments; ++t) {
		T dx[DEGREE];
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i) {
			dx[i] = point(t, channels[i]) - point(t - 1, channels[i]);
		}

		// Horner scheme, high-to-low for in-place coefs update
		#pragma unroll
		for (uint64_t i = DEGREE; i >= 1; --i) {
			T temp = ovf[i];
			#pragma unroll
			for (uint64_t k = 0; k + 1 < i; ++k) {
				temp = temp * dx[k] + coefs[k + 1] * ovf[i - k - 1];
			}
			coefs[i] += temp * dx[i - 1];
		}
	}

	if (!prefixes) {
		*my_out = coefs[DEGREE];
	}
	else {
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i) {
			my_out[i] = coefs[i + 1];
		}
	}
}

template<typename T, uint64_t DEGREE, bool Augmented>
__global__ void sig_coef_backprop_kernel_fixed_degree(
	const T* __restrict__ path,
	T* __restrict__ out,
	const T* __restrict__ coefs_in,
	const T* __restrict__ derivs_in,
	const uint64_t* __restrict__ multi_idx,
	const T* __restrict__ one_over_fact,
	const uint64_t* __restrict__ idx_offsets,
	const uint64_t* __restrict__ coef_offsets,
	uint64_t num_multi_idx,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t path_length,
	uint64_t coefs_length,
	uint64_t work_offset,
	uint64_t work_size,
	const uint64_t* __restrict__ word_order,
	bool time_aug, bool lead_lag, T end_time
) {
	const uint64_t local_idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
	if (local_idx >= work_size) return;
	const uint64_t global_idx = work_offset + local_idx;

	const uint64_t batch_idx = global_idx / num_multi_idx;
	const uint64_t word_idx = word_order[global_idx % num_multi_idx];

	const uint64_t idx_offset = idx_offsets[word_idx];
	const uint64_t coef_offset = coef_offsets[word_idx];

	const uint64_t* my_idx = multi_idx + idx_offset;
	const T* my_coefs = coefs_in + batch_idx * coefs_length + coef_offset;
	const T* my_derivs = derivs_in + batch_idx * coefs_length + coef_offset;
	const T* my_path = path + batch_idx * path_length * dimension;
	const uint64_t segments = (path_length - 1) * (Augmented && lead_lag ? 2 : 1);
	auto point = [&](uint64_t vertex, uint64_t channel) {
		if constexpr (!Augmented) return my_path[vertex * dimension + channel];
		else return pascal_coefficient::point(my_path, vertex, channel, dimension, segments, time_aug, lead_lag, end_time);
	};
	T* my_out = out + batch_idx * path_length * dimension;

	uint32_t ch[DEGREE];
	T ovf[DEGREE + 1];

	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		ch[i] = static_cast<uint32_t>(my_idx[i]);
		ovf[i + 1] = one_over_fact[i + 1];
	}
	ovf[0] = T(1);

	T c[DEGREE + 1];
	c[0] = T(1);
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		c[i + 1] = my_coefs[i];

	T d[DEGREE];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		d[i] = my_derivs[i];

	T buff[DEGREE];
	T prev_upd[DEGREE];
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i)
		prev_upd[i] = T(0);

	for (uint64_t t = segments; t >= 1; --t) {
		T dx[DEGREE];
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i)
			dx[i] = point(t, ch[i]) - point(t - 1, ch[i]);

		// uncombine_coefs_ (sovf computed inline: (-1)^k * ovf[k])
		{
			T last_c = c[0];
			#pragma unroll
			for (uint64_t i = 1; i <= DEGREE; ++i) {
				const T inc = dx[i - 1];
				#pragma unroll
				for (uint64_t j = 0; j + 1 < i; ++j)
					buff[j] *= inc;
				buff[i - 1] = last_c * inc;
				T acc = T(0);
				#pragma unroll
				for (uint64_t j = 0; j < i; ++j)
					acc += buff[j] * (((i - j) & 1) ? -ovf[i - j] : ovf[i - j]);
				last_c = c[i];
				c[i] += acc;
			}
		}

		// update_path_derivs_ -> collect into cur_upd
		T cur_upd[DEGREE];
		{
			T upd = d[0];
			T rp = T(1);
			#pragma unroll
			for (uint64_t m = 1; m < DEGREE; ++m) {
				rp *= dx[m];
				upd += d[m] * rp * ovf[m + 1];
			}
			cur_upd[0] = upd;

			#pragma unroll
			for (uint64_t i = 1; i < DEGREE; ++i) {
				const T inc = dx[i - 1];
				#pragma unroll
				for (uint64_t k = 0; k + 1 < i; ++k)
					buff[k] *= inc;
				buff[i - 1] = c[i - 1] * inc;

				T s = c[i];
				#pragma unroll
				for (uint64_t k = 0; k < i; ++k)
					s += buff[k] * ovf[i - k + 1];
				upd = d[i] * s;

				rp = T(1);
				#pragma unroll
				for (uint64_t m = i + 1; m < DEGREE; ++m) {
					rp *= dx[m];
					s = c[i] * ovf[m - i + 1];
					#pragma unroll
					for (uint64_t k = 0; k < i; ++k)
						s += buff[k] * ovf[m - k + 1];
					s *= rp;
					upd += d[m] * s;
				}

				cur_upd[i] = upd;
			}
		}

		// Merged atomicAdds: net = cur_upd - prev_upd
		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i) {
			if constexpr (Augmented) if (time_aug && ch[i] == dimension * (lead_lag ? 2 : 1)) continue;
			const uint64_t row = Augmented && lead_lag ? t / 2 + ((t % 2) && ch[i] >= dimension) : t;
			const uint64_t column = Augmented && lead_lag && ch[i] >= dimension ? ch[i] - dimension : ch[i];
			myAtomicAdd(&my_out[row * dimension + column], cur_upd[i] - prev_upd[i]);
		}

		#pragma unroll
		for (uint64_t i = 0; i < DEGREE; ++i)
			prev_upd[i] = cur_upd[i];

		// update_coef_derivs_
		{
			#pragma unroll
			for (uint64_t i = 0; i + 1 < DEGREE; ++i) {
				T acc = T(0);
				T rp = T(1);
				#pragma unroll
				for (uint64_t k = i + 1; k < DEGREE; ++k) {
					rp *= dx[k];
					acc += d[k] * rp * ovf[k - i];
				}
				d[i] += acc;
			}
		}
	}

	// Final: write -prev_upd to position 0
	#pragma unroll
	for (uint64_t i = 0; i < DEGREE; ++i) {
		if constexpr (Augmented) if (time_aug && ch[i] == dimension * (lead_lag ? 2 : 1)) continue;
		const uint64_t column = Augmented && lead_lag && ch[i] >= dimension ? ch[i] - dimension : ch[i];
		myAtomicAdd(&my_out[column], -prev_upd[i]);
	}
}

template<typename T>
__global__ void sig_coef_identity_kernel(T* out, const uint64_t* offsets, const uint64_t* order,
    uint64_t words, uint64_t width, uint64_t degree, bool prefixes, uint64_t offset, uint64_t work) {
    const uint64_t local = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= work) return;
    const uint64_t job = offset + local, sample = job / words, word = order[job % words];
    T* result = out + sample * width + offsets[word];
    const uint64_t count = prefixes && degree ? degree : 1;
    for (uint64_t j = 0; j < count; ++j) result[j] = degree ? T(0) : T(1);
}

template<class T, bool Backward>
void sig_coef_horner_cuda(const T* path, T* out, const T* coefs, const T* deriv,
    const uint64_t* indices, uint64_t count, const uint64_t* degrees, uint64_t batch,
    uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time,
    bool prefixes, bool device_metadata, void* stream_ptr) {
    namespace pc = pascal_coefficient;
    if (!dimension) throw std::invalid_argument("Invalid coefficient dimension");
    if (!batch || !count) return;
    if (!degrees) throw std::invalid_argument("Missing coefficient degrees");
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    std::vector<uint64_t> host_degrees(count);
    if (device_metadata) {
        CUDA_CHECK(cudaMemcpyAsync(host_degrees.data(), degrees, pc::mul(count, sizeof(uint64_t)), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else std::copy_n(degrees, count, host_degrees.data());
    uint64_t max_degree = 0, index_count = 0, width = 0;
    std::vector<uint64_t> metadata(pc::mul(3, count));
    for (uint64_t w = 0; w < count; ++w) {
        const uint64_t degree = host_degrees[w];
        max_degree = std::max(max_degree, degree);
        metadata[w] = index_count;
        metadata[count + w] = width;
        metadata[2 * count + w] = w;
        index_count = pc::add(index_count, degree);
        width = pc::add(width, (Backward || prefixes) ? std::max<uint64_t>(degree, 1) : 1);
    }
    if (max_degree > SIG_COEF_CUDA_MAX_DEGREE) throw std::invalid_argument("CUDA coefficients support word lengths through 20");
    if (index_count && !indices) throw std::invalid_argument("Missing coefficient word indices");
    std::stable_sort(metadata.begin() + 2 * count, metadata.end(), [&](uint64_t a, uint64_t b) { return host_degrees[a] < host_degrees[b]; });
    if (!device_metadata) {
        const uint64_t augmented_dimension = pc::add(pc::mul(dimension, lead_lag ? 2 : 1), time_aug ? 1 : 0);
        for (uint64_t j = 0; j < index_count; ++j)
            if (indices[j] >= augmented_dimension) throw std::invalid_argument("Coefficient word index out of range");
        if (index_count) metadata.insert(metadata.end(), indices, indices + index_count);
    }
    const uint64_t output_elements = Backward ? pc::mul(batch, pc::mul(length, dimension)) : pc::mul(batch, width);
    const uint64_t output_bytes = pc::mul(output_elements, sizeof(T));
    CudaBuf<uint64_t> device_words(pc::mul(metadata.size(), sizeof(uint64_t)));
    std::lock_guard<std::mutex> lock(c_one_over_fact_mu);
    try {
        CUDA_CHECK(cudaMemcpyAsync(device_words.get(), metadata.data(), metadata.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        const uint64_t* device_indices = device_metadata ? indices : device_words.get() + 3 * count;
        T factors[SIG_COEF_CUDA_MAX_DEGREE + 1];
        factors[0] = T(1);
        for (uint64_t j = 1; j <= max_degree; ++j) factors[j] = factors[j - 1] / T(j);
        void* symbol = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(c_one_over_fact_f, factors, (max_degree + 1) * sizeof(T), 0, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGetSymbolAddress(&symbol, c_one_over_fact_f));
        } else {
            CUDA_CHECK(cudaMemcpyToSymbolAsync(c_one_over_fact_d, factors, (max_degree + 1) * sizeof(T), 0, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGetSymbolAddress(&symbol, c_one_over_fact_d));
        }
        const T* device_factors = static_cast<const T*>(symbol);
        if constexpr (Backward) if (output_bytes) CUDA_CHECK(cudaMemsetAsync(out, 0, output_bytes, stream));
        for (uint64_t begin = 0; begin < count;) {
            uint64_t end = begin + 1;
            const uint64_t degree = host_degrees[metadata[2 * count + begin]];
            while (end < count && host_degrees[metadata[2 * count + end]] == degree) ++end;
            const uint64_t group_count = end - begin;
            const uint64_t* order = device_words.get() + 2 * count + begin;
            const uint64_t jobs = Backward && (!degree || length <= 1) ? 0 : pc::mul(batch, group_count);
            for (uint64_t offset = 0; offset < jobs;) {
                const uint64_t work = std::min<uint64_t>(jobs - offset, CUDA_GRID_X_LIMIT * 256);
                const unsigned int blocks = static_cast<unsigned int>((work + 255) / 256);
                if (!degree || length <= 1) {
                    sig_coef_identity_kernel<T><<<blocks, 256, 0, stream>>>(out, device_words.get() + count,
                        order, group_count, width, degree, prefixes, offset, work);
                } else {
                    auto run = [&]<uint64_t Degree>() {
                        auto launch = [&]<bool Augmented>() {
                            if constexpr (Backward)
                                sig_coef_backprop_kernel_fixed_degree<T, Degree, Augmented><<<blocks, 256, 0, stream>>>(path, out, coefs, deriv,
                                    device_indices, device_factors, device_words.get(), device_words.get() + count,
                                    group_count, batch, dimension, length, width, offset, work, order, time_aug, lead_lag, end_time);
                            else
                                sig_coef_kernel_fixed_degree<T, Degree, Augmented><<<blocks, 256, 0, stream>>>(path, out,
                                    device_indices, device_factors, device_words.get(), device_words.get() + count,
                                    group_count, batch, dimension, length, width, prefixes, offset, work, order, time_aug, lead_lag, end_time);
                        };
                        if (time_aug || lead_lag) launch.template operator()<true>();
                        else launch.template operator()<false>();
                    };
#define HORNER_CASE(M) case M: run.template operator()<M>(); break
                    switch (degree) {
                        HORNER_CASE(1); HORNER_CASE(2); HORNER_CASE(3); HORNER_CASE(4); HORNER_CASE(5);
                        HORNER_CASE(6); HORNER_CASE(7); HORNER_CASE(8); HORNER_CASE(9); HORNER_CASE(10);
                        HORNER_CASE(11); HORNER_CASE(12); HORNER_CASE(13); HORNER_CASE(14); HORNER_CASE(15);
                        HORNER_CASE(16); HORNER_CASE(17); HORNER_CASE(18); HORNER_CASE(19); HORNER_CASE(20);
                    }
#undef HORNER_CASE
                }
                CUDA_CHECK(cudaGetLastError());
                offset += work;
            }
            begin = end;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

#define INSTANTIATE_HORNER(T, B) \
template void sig_coef_horner_cuda<T, B>(const T*, T*, const T*, const T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, bool, void*)
INSTANTIATE_HORNER(float, false);
INSTANTIATE_HORNER(float, true);
INSTANTIATE_HORNER(double, false);
INSTANTIATE_HORNER(double, true);
#undef INSTANTIATE_HORNER

extern "C" {
	CUSIG_API int sig_coef_cuda_f(const float* path, float* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes, bool time_aug, bool lead_lag, float end_time, int method, void* stream) noexcept {
		if (method == 0) {
			CUDA_SAFE_CALL((sig_coef_horner_cuda<float, false>(path, out, nullptr, nullptr, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, !time_aug && !lead_lag, stream)));
		} else {
			CUDA_SAFE_CALL((sig_coef_dispatch_cuda<float, false>(path, out, nullptr, nullptr, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, method, stream)));
		}
	}

	CUSIG_API int sig_coef_backprop_cuda_f(const float* path, float* out, const float* coefs, const float* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, float end_time, int method, void* stream) noexcept {
		if (method == 0) {
			CUDA_SAFE_CALL((sig_coef_horner_cuda<float, true>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, true, !time_aug && !lead_lag, stream)));
		} else {
			CUDA_SAFE_CALL((sig_coef_dispatch_cuda<float, true>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, true, method, stream)));
		}
	}

	CUSIG_API int sig_coef_cuda_d(const double* path, double* out, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool prefixes, bool time_aug, bool lead_lag, double end_time, int method, void* stream) noexcept {
		if (method == 0) {
			CUDA_SAFE_CALL((sig_coef_horner_cuda<double, false>(path, out, nullptr, nullptr, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, !time_aug && !lead_lag, stream)));
		} else {
			CUDA_SAFE_CALL((sig_coef_dispatch_cuda<double, false>(path, out, nullptr, nullptr, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, prefixes, method, stream)));
		}
	}

	CUSIG_API int sig_coef_backprop_cuda_d(const double* path, double* out, const double* coefs, const double* derivs, const uint64_t* multi_idx, uint64_t num_multi_idx, const uint64_t* degrees, uint64_t batch_size, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, double end_time, int method, void* stream) noexcept {
		if (method == 0) {
			CUDA_SAFE_CALL((sig_coef_horner_cuda<double, true>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, true, !time_aug && !lead_lag, stream)));
		} else {
			CUDA_SAFE_CALL((sig_coef_dispatch_cuda<double, true>(path, out, coefs, derivs, multi_idx, num_multi_idx, degrees, batch_size, dimension, length, time_aug, lead_lag, end_time, true, method, stream)));
		}
	}
}
