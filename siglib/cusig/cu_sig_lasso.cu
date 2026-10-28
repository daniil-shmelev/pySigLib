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
#include "cu_macros.h"
#include "sig_lasso_core.h"
#include "sig_lasso_math.h"
#include "cu_sig_lasso_search.cuh"
#include <cublas_v2.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include "cu_sig_lasso_active.cuh"

namespace {
using siglasso::Array;
using siglasso::Column;

void blas_check(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure in sig_lasso: " + std::to_string(static_cast<int>(status)));
}

class DeviceScope {
    int previous_;
public:
    explicit DeviceScope(int device) { CUDA_CHECK(cudaGetDevice(&previous_)); CUDA_CHECK(cudaSetDevice(device)); }
    ~DeviceScope() { cudaSetDevice(previous_); }
};

template<typename T>
__global__ void convert_input(const T* source, double* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) out[i] = source[i];
}
__global__ void normalize_paths(double* paths, int* invalid, size_t n, size_t length, size_t dimension) {
    using Reduction = cub::BlockReduce<double, 128>;
    __shared__ typename Reduction::TempStorage storage;
    __shared__ double total;
    for (size_t i = blockIdx.x; i < n; i += gridDim.x) {
        double* sample = paths + i * length * dimension;
        const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        double partial = 0;
        for (size_t a = threadIdx.x; a < dimension; a += blockDim.x)
            if (!isfinite(sample[a])) atomicExch(invalid, 1);
        for (size_t j = 1 + warp; j < length; j += 4) {
            double step = 0;
            for (size_t a = lane; a < dimension; a += 32) {
                const double difference = sample[j * dimension + a] - sample[(j - 1) * dimension + a];
                if (!isfinite(difference)) atomicExch(invalid, 1);
                step = fmax(step, fabs(difference));
            }
            for (int offset = 16; offset; offset /= 2) step = fmax(step, __shfl_down_sync(0xffffffff, step, offset));
            if (lane == 0) partial += step;
        }
        const double variation = Reduction(storage).Sum(partial);
        if (threadIdx.x == 0) total = variation;
        __syncthreads();
        if (!isfinite(total)) {
            if (threadIdx.x == 0) atomicExch(invalid, 1);
        } else {
            for (size_t a = threadIdx.x; a < dimension; a += blockDim.x) {
                const double origin = sample[a];
                for (size_t j = length; j-- > 0;)
                    sample[j * dimension + a] = total ? (sample[j * dimension + a] - origin) / total : 0;
            }
        }
        __syncthreads();
    }
}
__global__ void vector_op(const double* x, double* y, size_t n, double a, double c, int operation) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (operation == 0) y[i] = a;
        else if (operation == 1) y[i] = a * y[i] + c;
        else y[i] += a * x[i];
    }
}
__global__ void extract_response(const double* y, double* out, size_t n, size_t targets, size_t target) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) out[i] = y[i * targets + target];
}
__global__ void prepare_paths(const double* paths, double* increments, double* variation, double* after,
    size_t n, size_t l, size_t d) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
            increments[(j * d + a) * n + i] = paths[(i * (l + 1) + j + 1) * d + a] - paths[(i * (l + 1) + j) * d + a];
        siglasso::prepare_sample(increments, variation, after, i, n, l, d);
    }
}
__global__ void make_factors(const double* variation, const double* after, const double* constants,
    double* out, size_t n, size_t l, size_t k, size_t levels) {
    const size_t size = n * l * levels * (k + 1);
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % n, j = (index / n) % l, a = (index / (n * l)) % (k + 1), q = index / (n * l * (k + 1)) + 1;
        out[index] = siglasso::factor_value(variation, after, constants, k, q, a, j, i, n) / n;
    }
}
__global__ void make_moments(const double* coefficients, const double* time_values, const double* factors, const double* variation, const double* after,
    double* moments, double* integrals, double* linear_integrals, double* scratch, double weight,
    size_t n, size_t l, size_t k, size_t levels, size_t scratch_size, bool exact, bool envelope) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < n * l; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % n, j = index / n;
        if (exact) {
            double integral = 0, linear = 0;
            for (size_t a = 0; a <= k; ++a) {
                const double c = coefficients[(j * (k + 1) + a) * n + i]; integral += c;
                if (linear_integrals) linear += (a + 1) * c;
            }
            integrals[index] = integral / (k + 1);
            if (linear_integrals) linear_integrals[index] = linear / ((k + 1) * (k + 2));
            siglasso::absolute_moments(coefficients, factors, variation[index], after[index], k, levels, j, i, n, l,
                scratch + (blockIdx.x * blockDim.x + threadIdx.x) * scratch_size, moments + index, envelope);
        } else {
            siglasso::trapezoidal_moments(time_values[index], time_values[index + n], variation[index], after[index],
                weight, k, levels, n * l, moments + index);
            if (!k && integrals) integrals[index] = 1;
            if (!k && linear_integrals) linear_integrals[index] = .5;
        }
    }
}

struct DeviceChild { const double* coefficients; const double* integrals; double* child; double* values; uint64_t letter, k; };
struct DeviceNodeChild {
    const double* const* ancestors;
    const uint64_t* word;
    double *time_values, *integrals, *linear_integrals, *values;
    size_t degree;
};
__global__ void node_child_batch(const DeviceNodeChild* tasks, const double* increments, size_t count, size_t n, size_t l, size_t d) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count * n; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % n;
        const auto task = tasks[index / n];
        double value = 0;
        if (task.time_values) task.time_values[i] = 0;
        for (size_t j = 0; j < l; ++j) {
            double product = 1, integral = value, linear = value / 2;
            for (size_t h = 1; h <= task.degree; ++h) {
                product *= increments[(j * d + task.word[task.degree - h]) * n + i] / h;
                const double term = task.ancestors[task.degree - h][j * n + i] * product;
                value += term;
                if (task.integrals) integral += term / (h + 1);
                if (task.linear_integrals) linear += term / (h + 2);
            }
            if (task.time_values) task.time_values[(j + 1) * n + i] = value;
            if (task.integrals) task.integrals[j * n + i] = integral;
            if (task.linear_integrals) task.linear_integrals[j * n + i] = linear;
        }
        task.values[i] = value;
    }
}

template<typename T>
__global__ void export_signatures(const double* const* columns, const double* means, T* output, size_t n, size_t p) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < n * p; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t j = index % p;
        output[index] = static_cast<T>(columns[j][index / p] + means[j]);
    }
}
struct DeviceInterval { uint64_t low, high; };

__global__ void make_alphabet(const DeviceInterval* intervals, const double* increments, const double* variation,
    double* ratios, size_t blocks, size_t n, size_t l, size_t d) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < n * l; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % n, j = index / n;
        for (size_t block = blocks; block-- > 0;) {
            const auto interval = intervals[block];
            const size_t right = block + 2 * ((interval.low + interval.high) / 2 - interval.low);
            ratios[block * n * l + index] = interval.high - interval.low == 1
                ? (variation[index] ? fabs(increments[(j * d + interval.low) * n + i]) / variation[index] : 0)
                : fmax(ratios[(block + 1) * n * l + index], ratios[right * n * l + index]);
        }
    }
}
__global__ void make_energy_gram(const double* increments, double* gram, size_t low, size_t high, size_t n, size_t l, size_t d) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < n * l * l; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index % n, h = index / n % l, j = index / (n * l);
        if (h > j) continue;
        double value = 0;
        for (size_t a = low; a < high; ++a) value += increments[(j * d + a) * n + i] * increments[(h * d + a) * n + i];
        gram[index] = value;
    }
}

__device__ double block_sum(double value, double* scratch) {
    scratch[threadIdx.x] = value; __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const double total = scratch[0]; __syncthreads(); return total;
}
__global__ void child_batch(const DeviceChild* tasks, const double* increments, size_t count, size_t n, size_t l, size_t d) {
    using Scan = cub::BlockScan<double, 32>;
    __shared__ Scan::TempStorage temporary;
    __shared__ double carry;
    for (size_t index = blockIdx.x; index < count * n; index += gridDim.x) {
        const auto task = tasks[index / n]; const size_t i = index % n;
        if (!threadIdx.x) carry = 0;
        __syncthreads();
        for (size_t begin = 0; begin < l; begin += 32) {
            const size_t j = begin + threadIdx.x;
            double total = j < l ? increments[(j * d + task.letter) * n + i] * task.integrals[j * n + i] : 0;
            double offset, aggregate;
            Scan(temporary).ExclusiveSum(total, offset, aggregate);
            offset += carry;
            if (task.child && j < l) {
                task.child[j * (task.k + 2) * n + i] = offset;
                const double step = increments[(j * d + task.letter) * n + i] / (task.k + 1);
                for (size_t a = 0; a <= task.k; ++a) {
                    offset += step * task.coefficients[(j * (task.k + 1) + a) * n + i];
                    task.child[(j * (task.k + 2) + a + 1) * n + i] = offset;
                }
            }
            __syncthreads();
            if (!threadIdx.x) carry += aggregate;
            __syncthreads();
        }
        if (!threadIdx.x) tasks[index / n].values[i] = carry;
        __syncthreads();
    }
}

__global__ void column_correlations(const double* const* columns, const double* vector, double* out, size_t p, size_t n) {
    __shared__ double scratch[128];
    for (size_t j = blockIdx.x; j < p; j += gridDim.x) {
        double value = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) value += columns[j][i] * vector[i];
        const double sum = block_sum(value, scratch);
        if (!threadIdx.x) out[j] = sum / n;
        __syncthreads();
    }
}
__global__ void certified_correlations(const double* const* columns, const double* vector, double* out, size_t p, size_t n) {
    __shared__ double sums[128], errors[128], magnitudes[128];
    for (size_t j = blockIdx.x; j < p; j += gridDim.x) {
        double value = 0, error = 0, magnitude = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            const double product = columns[j][i] * vector[i], next = value + product;
            error += fma(columns[j][i], vector[i], -product);
            error += fabs(value) >= fabs(product) ? (value - next) + product : (product - next) + value;
            value = next; magnitude += fabs(product);
        }
        sums[threadIdx.x] = value; errors[threadIdx.x] = error; magnitudes[threadIdx.x] = magnitude; __syncthreads();
        for (unsigned int step = blockDim.x / 2; step; step /= 2) {
            if (threadIdx.x < step) {
                const double a = sums[threadIdx.x], b = sums[threadIdx.x + step], next = a + b;
                errors[threadIdx.x] += errors[threadIdx.x + step] + (fabs(a) >= fabs(b) ? (a - next) + b : (b - next) + a);
                sums[threadIdx.x] = next; magnitudes[threadIdx.x] += magnitudes[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (!threadIdx.x) {
            const double total = fabs(sums[0] + errors[0]);
            out[j] = nextafter((total + 8 * DBL_EPSILON * total + 8 * n * DBL_EPSILON * DBL_EPSILON * magnitudes[0]) / n, CUDART_INF);
        }
        __syncthreads();
    }
}
struct EligibleColumn {
    const double* scores;
    const uint64_t* excluded;
    size_t excluded_count;
    double alpha;
    __device__ bool operator()(uint64_t slot) const {
        if (!(fabs(scores[slot]) > alpha)) return false;
        size_t low = 0, high = excluded_count;
        while (low < high) { const size_t middle = (low + high) / 2; if (excluded[middle] < slot) low = middle + 1; else high = middle; }
        return low == excluded_count || excluded[low] != slot;
    }
};
__global__ void select_dense_candidates(double* scores, const uint64_t* eligible, size_t count, uint64_t* selected, size_t capacity) {
    __shared__ double values[128];
    __shared__ uint64_t indices[128];
    for (size_t k = 0; k < capacity; ++k) {
        double value = -1; uint64_t best = UINT64_MAX;
        for (size_t j = threadIdx.x; j < count; j += blockDim.x) {
            const uint64_t slot = eligible[j]; const double score = fabs(scores[slot]);
            if (score > value || (score == value && slot < best)) { value = score; best = slot; }
        }
        values[threadIdx.x] = value; indices[threadIdx.x] = best; __syncthreads();
        for (unsigned int step = blockDim.x / 2; step; step /= 2) {
            if (threadIdx.x < step && (values[threadIdx.x + step] > values[threadIdx.x]
                || (values[threadIdx.x + step] == values[threadIdx.x] && indices[threadIdx.x + step] < indices[threadIdx.x]))) {
                values[threadIdx.x] = values[threadIdx.x + step]; indices[threadIdx.x] = indices[threadIdx.x + step];
            }
            __syncthreads();
        }
        if (!threadIdx.x) { selected[k] = indices[0]; scores[indices[0]] = 0; }
        __syncthreads();
    }
}
__global__ void insert_full(const double* source, double* out, const double* weights, const uint64_t* degrees,
    size_t rows, size_t p, size_t n, size_t begin) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rows * p; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = index / p, j = index % p;
        out[j * n + begin + i] = source[index] * weights[degrees[j]];
    }
}
__global__ void centre_columns(double* matrix, double* means, double* norms, size_t n, size_t p, bool centre) {
    __shared__ double scratch[128];
    for (size_t j = blockIdx.x; j < p; j += gridDim.x) {
        double sum = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) sum += matrix[j * n + i];
        double mean = block_sum(sum, scratch) / n; if (!centre) mean = 0;
        double norm = 0;
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) { const double value = matrix[j * n + i] - mean; matrix[j * n + i] = value; norm += value * value; }
        const double norm2 = block_sum(norm, scratch);
        if (!threadIdx.x) { means[j] = mean; norms[j] = norm2; }
        __syncthreads();
    }
}

class LassoCUDA final : public siglasso::Backend {
    cudaStream_t stream_;
    int device_;
    cublasHandle_t blas_ = nullptr;
    std::shared_ptr<void> blas_owner_, blas_account_;
    Array paths_, responses_, device_weights_, ones_, pointers_, correlations_, blas_workspace_;
    Array stored_pointers_, stored_scores_, selection_scratch_, eligible_, excluded_, selected_, selected_count_, maximum_;
    std::vector<const double*> stored_addresses_;
    std::vector<Array> search_factors_, search_grams_;
    Array factor_addresses_, gram_addresses_, search_intervals_, search_frontier_;
    size_t search_blocks_ = 0, root_groups_ = 0, traversal_lanes_ = 0;
    Array exclusion_hashes_, exclusion_lengths_, exclusion_words_, exclusion_count_;
    lasso_device::Seen exclusions_{};
    size_t excluded_count_ = 0;
    bool search_running_ = false;
    Array seen_hashes_, seen_lengths_, seen_words_, seen_count_;
    lasso_device::Seen seen_{};
    std::array<Array, 27> traversal_buffers_;
    Array prefix_cache_, prefix_words_, prefix_lengths_, prefix_ticks_, prefix_clocks_;
    Array terminal_cache_storage_;
    double terminal_generation_ = 0;
    size_t prefix_slots_ = 0;
    uint64_t prefix_hits_ = 0, prefix_writes_ = 0;
    size_t prefix_queries_ = 0;
    bool prefix_disabled_ = false;

    unsigned int grid(size_t n) { return make_cuda_1d_grid(n, 128); }
    void check() { CUDA_CHECK(cudaGetLastError()); }
    void set_blas_stream(cudaStream_t stream) {
        stream_ = stream; blas_check(cublasSetStream(blas_, stream_));
        blas_check(cublasSetWorkspace(blas_, blas_workspace_.data, blas_workspace_.bytes()));
    }
    template<typename T> Array descriptors(const std::vector<T>& values) {
        auto buffer = alloc((values.size() * sizeof(T) + 7) / 8);
        if (!values.empty()) CUDA_CHECK(cudaMemcpyAsync(buffer.data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream_));
        return buffer;
    }
    void grow_seen(size_t capacity) {
        auto hashes = alloc(capacity), lengths = alloc(capacity), words = alloc(siglasso::multiply(capacity, options.degree)), count = alloc(1);
        fill(hashes, 0); fill(lengths, 0); fill(count, 0);
        lasso_device::Seen next{reinterpret_cast<uint64_t*>(hashes.data), reinterpret_cast<uint64_t*>(lengths.data), reinterpret_cast<uint64_t*>(words.data),
            reinterpret_cast<unsigned long long*>(count.data), capacity, options.degree};
        if (seen_.capacity) lasso_device::rehash<<<grid(seen_.capacity), 128, 0, stream_>>>(seen_, next);
        seen_hashes_ = std::move(hashes); seen_lengths_ = std::move(lengths); seen_words_ = std::move(words); seen_count_ = std::move(count); seen_ = next; check();
    }
    Array traversal_buffer(size_t slot, size_t size) {
        if (traversal_buffers_[slot].size < size) traversal_buffers_[slot] = alloc(size);
        const auto& buffer = traversal_buffers_[slot];
        return {buffer.data, size, buffer.owner};
    }
    void stored_columns(const std::vector<Column>& columns) {
        if (stored_addresses_.size() != columns.size()) {
            stored_addresses_.clear(); stored_addresses_.reserve(columns.size());
            for (const auto& column : columns) stored_addresses_.push_back(column.values.data);
            stored_pointers_ = descriptors(stored_addresses_); stored_scores_ = alloc(columns.size());
        }
    }
public:
    template<typename T>
    LassoCUDA(const SigLassoOptions& o, const T* path, const T* y, cudaStream_t stream, int device)
        : Backend(o), stream_(stream), device_(device) {
        if (o.samples > INT_MAX) throw std::out_of_range("sig_lasso CUDA sample count exceeds cuBLAS limits");
        int major = 0; CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        int multiprocessors = 0; CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        search_blocks_ = std::min<size_t>(128, 4 * multiprocessors);
        blas_account_ = account((major >= 9 ? 32 : 4) * 1024 * 1024);
        blas_check(cublasCreate(&blas_));
        blas_owner_ = std::shared_ptr<void>(blas_, [](void* handle) { cublasDestroy(static_cast<cublasHandle_t>(handle)); });
        blas_workspace_ = alloc(std::min<size_t>(512 * 1024, std::max<size_t>(2048, budget->available() / 64)));
        set_blas_stream(stream_);
        auto constants = weights;
        constants.insert(constants.end(), reciprocals.begin(), reciprocals.end());
        device_weights_ = from_host(constants);
        paths_ = alloc(siglasso::multiply(siglasso::multiply(o.samples, o.length), o.dimension));
        responses_ = alloc(siglasso::multiply(o.samples, o.targets));
        convert_input<<<grid(paths_.size), 128, 0, stream_>>>(path, paths_.data, paths_.size);
        convert_input<<<grid(responses_.size), 128, 0, stream_>>>(y, responses_.data, responses_.size);
        auto invalid = alloc(1);
        CUDA_CHECK(cudaMemsetAsync(invalid.data, 0, sizeof(int), stream_));
        normalize_paths<<<std::min<size_t>(65535, o.samples), 128, 0, stream_>>>(paths_.data, reinterpret_cast<int*>(invalid.data), o.samples, o.length, o.dimension);
        check();
        int invalid_host = 0;
        CUDA_CHECK(cudaMemcpyAsync(&invalid_host, invalid.data, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (invalid_host) throw std::runtime_error("Nonfinite path variation in sig_lasso");
        ones_ = alloc(o.samples); fill(ones_, 1); check();
    }
    Array alloc(size_t n) override {
        if (!n) return {};
        const size_t bytes = checked_cuda_size_mul(n, sizeof(double), "sig_lasso");
        if (bytes > budget->available()) trim_caches();
        budget->acquire(bytes);
        double* data = nullptr;
        const auto status = cudaMallocAsync(reinterpret_cast<void**>(&data), bytes, stream_);
        if (status != cudaSuccess) { budget->release(bytes); if (status == cudaErrorMemoryAllocation) throw std::bad_alloc(); CUDA_CHECK(status); }
        return {data, n, std::shared_ptr<void>(data, [account = budget, bytes, stream = stream_, device = device_](void* p) {
            int previous = device; cudaGetDevice(&previous); cudaSetDevice(device); cudaFreeAsync(p, stream); cudaSetDevice(previous); account->release(bytes);
        })};
    }
    void fill(const Array& a, double value) override { if (a.size) { vector_op<<<grid(a.size), 128, 0, stream_>>>(nullptr, a.data, a.size, value, 0, 0); check(); } }
    void copy(const double* from, double* to, size_t n) override { if (n) CUDA_CHECK(cudaMemcpyAsync(to, from, n * sizeof(double), cudaMemcpyDeviceToDevice, stream_)); }
    void upload(const double* from, double* to, size_t n) override { if (n) CUDA_CHECK(cudaMemcpyAsync(to, from, n * sizeof(double), cudaMemcpyHostToDevice, stream_)); }
    void download(const double* from, double* to, size_t n) override {
        if (n) CUDA_CHECK(cudaMemcpyAsync(to, from, n * sizeof(double), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    double dot(const double* a, const double* b, size_t n) override {
        if (!n) return 0;
        double result = 0;
        blas_check(cublasDdot(blas_, static_cast<int>(n), a, 1, b, 1, &result));
        return siglasso::finite(result);
    }
    void axpy(double alpha, const double* x, double* y, size_t n) override {
        if (n) { vector_op<<<grid(n), 128, 0, stream_>>>(x, y, n, alpha, 0, 2); check(); }
    }
    double absolute_sum(const Array& values) override {
        double result = 0;
        if (values.size > INT_MAX) throw std::out_of_range("CUDA Lasso coefficient count exceeds cuBLAS limits");
        if (values.size) blas_check(cublasDasum(blas_, static_cast<int>(values.size), values.data, 1, &result));
        return siglasso::finite(result);
    }
    void scale(double alpha, double* y, size_t n) override {
        if (n) { vector_op<<<grid(n), 128, 0, stream_>>>(nullptr, y, n, alpha, 0, 1); check(); }
    }
    double center(double* data, size_t n, bool enabled) override {
        if (!enabled || !n) return 0;
        const double mean = dot(data, ones_.data, n) / n;
        vector_op<<<grid(n), 128, 0, stream_>>>(nullptr, data, n, 1, -mean, 1); check();
        return mean;
    }
    void correlations(const std::vector<Column>& columns, const Array& vector, std::vector<double>& out, const std::vector<size_t>* selected = nullptr) override {
        out.resize(selected ? selected->size() : columns.size()); if (out.empty()) return;
        std::vector<const double*> pointers; pointers.reserve(out.size());
        for (size_t j = 0; j < out.size(); ++j) pointers.push_back(columns[selected ? (*selected)[j] : j].values.data);
        if (pointers_.size < out.size()) pointers_ = alloc(out.size());
        if (correlations_.size < out.size()) correlations_ = alloc(out.size());
        CUDA_CHECK(cudaMemcpyAsync(pointers_.data, pointers.data(), pointers.size() * sizeof(double*), cudaMemcpyHostToDevice, stream_));
        column_correlations<<<std::min<size_t>(65535, out.size()), 128, 0, stream_>>>(reinterpret_cast<const double* const*>(pointers_.data), vector.data, correlations_.data, out.size(), vector.size);
        check(); download(correlations_.data, out.data(), out.size());
    }
    void trim_caches() override {
        Backend::trim_caches();
        prefix_cache_ = {}; prefix_words_ = {}; prefix_lengths_ = {}; prefix_ticks_ = {}; prefix_clocks_ = {}; prefix_slots_ = 0;
        prefix_hits_ = prefix_writes_ = 0; prefix_queries_ = 0;
        if (!search_running_) {
            terminal_cache_storage_ = {};
            search_factors_.clear(); search_grams_.clear(); gram_addresses_ = {};
            for (auto& buffer : traversal_buffers_) buffer = {};
            traversal_lanes_ = 0;
        }
    }
    std::unique_ptr<siglasso::ActiveSet> active_set(const Array& y, double alpha) override {
        return siglasso::make_cuda_active(*this, y, alpha, stream_, blas_, device_);
    }
    double correlation_max(const std::vector<Column>& columns, const Array& vector) override {
        if (columns.empty()) return 0;
        stored_columns(columns);
        if (!maximum_.owner) maximum_ = alloc(1);
        certified_correlations<<<static_cast<unsigned>(std::min<size_t>(65535, columns.size())), 128, 0, stream_>>>(
            reinterpret_cast<const double* const*>(stored_pointers_.data), vector.data, stored_scores_.data, columns.size(), vector.size);
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceReduce::Max(nullptr, bytes, stored_scores_.data, maximum_.data, columns.size(), stream_));
        if (selection_scratch_.bytes() < bytes) selection_scratch_ = alloc((bytes + 7) / 8);
        CUDA_CHECK(cub::DeviceReduce::Max(selection_scratch_.data, bytes, stored_scores_.data, maximum_.data, columns.size(), stream_));
        double value = 0; download(maximum_.data, &value, 1); return siglasso::finite(value);
    }
    std::vector<size_t> dense_candidates(const std::vector<Column>& columns, const Array& vector,
        const std::unordered_set<size_t>& offered, double alpha, size_t capacity) override {
        stored_columns(columns);
        if (columns.empty()) return {};
        column_correlations<<<static_cast<unsigned>(std::min<size_t>(65535, columns.size())), 128, 0, stream_>>>(
            reinterpret_cast<const double* const*>(stored_pointers_.data), vector.data, stored_scores_.data, columns.size(), vector.size);
        std::vector<uint64_t> excluded(offered.begin(), offered.end()); std::sort(excluded.begin(), excluded.end());
        excluded_ = descriptors(excluded);
        if (eligible_.size < columns.size()) eligible_ = alloc(columns.size());
        if (selected_.size < capacity) selected_ = alloc(capacity);
        if (!selected_count_.owner) selected_count_ = alloc(1);
        EligibleColumn predicate{stored_scores_.data, reinterpret_cast<const uint64_t*>(excluded_.data), excluded.size(), alpha};
        cub::CountingInputIterator<uint64_t> indices(0);
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceSelect::If(nullptr, bytes, indices, reinterpret_cast<uint64_t*>(eligible_.data),
            reinterpret_cast<uint64_t*>(selected_count_.data), columns.size(), predicate, stream_));
        if (selection_scratch_.bytes() < bytes) selection_scratch_ = alloc((bytes + 7) / 8);
        CUDA_CHECK(cub::DeviceSelect::If(selection_scratch_.data, bytes, indices, reinterpret_cast<uint64_t*>(eligible_.data),
            reinterpret_cast<uint64_t*>(selected_count_.data), columns.size(), predicate, stream_));
        uint64_t count = 0; download(selected_count_.data, reinterpret_cast<double*>(&count), 1);
        capacity = std::min<size_t>(capacity, count);
        std::vector<size_t> result(capacity);
        if (capacity) {
            select_dense_candidates<<<1, 128, 0, stream_>>>(stored_scores_.data, reinterpret_cast<const uint64_t*>(eligible_.data),
                count, reinterpret_cast<uint64_t*>(selected_.data), capacity);
            download(selected_.data, reinterpret_cast<double*>(result.data()), capacity);
        }
        check(); return result;
    }
    void prepare() override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        increments = alloc(siglasso::multiply(n * l, d)); variation = alloc(n * l); after = alloc(n * l);
        prepare_paths<<<grid(n), 128, 0, stream_>>>(paths_.data, increments.data, variation.data, after.data, n, l, d); check();
        prepare_alphabet();
        std::vector<DeviceInterval> intervals;
        for (auto [low, high] : alphabet) intervals.push_back({low, high});
        auto device_intervals = descriptors(intervals);
        if (n * l) make_alphabet<<<grid(n * l), 128, 0, stream_>>>(reinterpret_cast<const DeviceInterval*>(device_intervals.data), increments.data, variation.data,
            alphabet_ratios.data, alphabet.size(), n, l, d);
        paths_ = {}; check();
    }
    Array factors(size_t k) override {
        const size_t n = options.samples, l = options.length - 1, levels = options.degree - k;
        if (!l) return {};
        const size_t values = siglasso::multiply(n * l, levels * (k + 1));
        const size_t constant_count = siglasso::multiply(siglasso::multiply(levels, levels + 1) / 2, k + 1);
        auto result = alloc(values + constant_count);
        auto host_constants = account(siglasso::multiply(constant_count, sizeof(double)));
        auto constants = from_host(siglasso::integration_coefficients(k, levels, weights));
        copy(constants.data, result.data + values, constants.size);
        if (values) make_factors<<<grid(values), 128, 0, stream_>>>(variation.data, after.data, constants.data, result.data, n, l, k, levels);
        check(); return result;
    }
    void moments(siglasso::Node& node, const Array& factors) override {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), levels = options.degree - k;
        if (!l) return;
        node.moments = alloc(n * l * levels);
        if ((options.bound_method || (options.search_method & 1) || ((options.search_method & 2) && levels <= 2)) && !node.integrals.owner) node.integrals = alloc(n * l);
        if ((options.search_method & 2) && levels <= 2 && !node.linear_integrals.owner) node.linear_integrals = alloc(n * l);
        const size_t blocks = std::min<size_t>(32, grid(n * l)), scratch_size = siglasso::integral_scratch_size(options.degree);
        auto scratch = options.bound_method ? alloc(blocks * 128 * scratch_size) : Array{};
        if (n * l) make_moments<<<blocks, 128, 0, stream_>>>(node.coefficients.data, node.time_values.data, factors.data, variation.data, after.data,
            node.moments.data, node.integrals.data, node.linear_integrals.data, scratch.data, weights[k + 1] / n, n, l, k, levels, scratch_size,
            options.bound_method != 0, (options.search_method & 2) && levels <= 2);
        check();
    }
    Array energy_gram(size_t low, size_t high) override {
        const size_t n = options.samples, l = options.length - 1;
        auto result = alloc(siglasso::multiply(n * l, l));
        if (result.size) make_energy_gram<<<grid(result.size), 128, 0, stream_>>>(increments.data, result.data, low, high, n, l, options.dimension);
        check(); return result;
    }
    void children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        const size_t n = options.samples, l = options.length - 1;
        if (!options.bound_method) {
            std::vector<DeviceNodeChild> records;
            std::vector<Array> buffers;
            out.clear();
            for (const auto& task : tasks) {
                auto node = std::make_shared<siglasso::Node>(); node->word = task.node->word; node->word.push_back(task.low);
                node->metadata = account(512 + node->word.size() * 64); node->values = alloc(n);
                if (node->word.size() < options.degree) {
                    node->time_values = alloc(n * (l + 1));
                    if ((options.search_method & 1) || ((options.search_method & 2) && options.degree - node->word.size() <= 2)) node->integrals = alloc(n * l);
                    if ((options.search_method & 2) && options.degree - node->word.size() <= 2) node->linear_integrals = alloc(n * l);
                }
                std::vector<const double*> ancestors;
                for (const auto& parent : task.ancestors->nodes) ancestors.push_back(parent->time_values.data);
                auto addresses = descriptors(ancestors), word = descriptors(node->word);
                records.push_back({reinterpret_cast<const double* const*>(addresses.data), reinterpret_cast<const uint64_t*>(word.data),
                    node->time_values.data, node->integrals.data, node->linear_integrals.data, node->values.data, node->word.size()});
                buffers.push_back(std::move(addresses)); buffers.push_back(std::move(word)); out.push_back(std::move(node));
            }
            auto input = descriptors(records);
            if (!tasks.empty()) node_child_batch<<<grid(tasks.size() * n), 128, 0, stream_>>>(reinterpret_cast<const DeviceNodeChild*>(input.data), increments.data, tasks.size(), n, l, options.dimension);
            check(); return;
        }
        std::vector<DeviceChild> descriptors_host;
        out.clear();
        for (const auto& task : tasks) {
            auto node = std::make_shared<siglasso::Node>(); node->word = task.node->word; node->word.push_back(task.low);
            node->metadata = account(512 + node->word.size() * 64);
            node->values = alloc(n);
            if (node->word.size() < options.degree) node->coefficients = alloc(n * l * (node->word.size() + 1));
            descriptors_host.push_back({task.node->coefficients.data, task.node->integrals.data, node->coefficients.data, node->values.data, task.low, task.node->word.size()});
            out.push_back(std::move(node));
        }
        auto input = descriptors(descriptors_host);
        child_batch<<<std::min<size_t>(65535, tasks.size() * n), 32, 0, stream_>>>(reinterpret_cast<const DeviceChild*>(input.data), increments.data, tasks.size(), n, l, options.dimension); check();
    }
    void full_columns(std::vector<Column>& columns) override {
        const size_t n = options.samples, d = options.dimension, m = options.degree;
        size_t count = 0, level = 1;
        for (size_t k = 1; k <= m; ++k) {
            level = siglasso::multiply(level, d);
            if (level > std::numeric_limits<size_t>::max() - count) throw std::out_of_range("Full signature size overflow");
            count += level;
        }
        auto column_metadata = account(siglasso::multiply(count, 1024 + m * 64 + options.targets * 32));
        auto matrix = alloc(siglasso::multiply(count, n));
        std::vector<uint64_t> degrees; degrees.reserve(count); columns.reserve(count);
        for (size_t k = 1; k <= m; ++k) {
            siglasso::Word word(k, 0);
            do {
                columns.push_back({word, Array{matrix.data + columns.size() * n, n, matrix.owner}, 0, 0, column_metadata}); degrees.push_back(k);
                size_t j = k;
                while (j && ++word[j - 1] == d) { word[j - 1] = 0; --j; }
                if (!j) break;
            } while (true);
        }
        auto device_degrees = descriptors(degrees);
        const size_t tile = std::min<size_t>(n, std::max<size_t>(1, (8 * 1024 * 1024) / std::max<size_t>(1, count * sizeof(double))));
        auto temporary = alloc(tile * count);
        auto kernel_workspace = account(siglasso::multiply(8 * (count + 1) + 8 * d * m, tile * sizeof(double)));
        for (size_t begin = 0; begin < n; begin += tile) {
            const size_t rows = std::min(tile, n - begin);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            ++stats.full_transform_calls;
            const int status = signature_cuda_d(paths_.data + begin * options.length * d, temporary.data, rows, d, options.length, m, false, false, 1.0, true, false);
            if (status == 1) throw std::bad_alloc();
            if (status) throw std::runtime_error("Full CUDA signature kernel failed in sig_lasso");
            insert_full<<<grid(rows * count), 128, 0, stream_>>>(temporary.data, matrix.data, device_weights_.data, reinterpret_cast<const uint64_t*>(device_degrees.data), rows, count, n, begin);
        }
        auto means = alloc(count), norms = alloc(count);
        centre_columns<<<std::min<size_t>(65535, count), 128, 0, stream_>>>(matrix.data, means.data, norms.data, n, count, options.intercept != 0);
        auto host_means = host(means), host_norms = host(norms);
        for (size_t j = 0; j < count; ++j) { columns[j].mean = siglasso::finite(host_means[j]); columns[j].norm2 = siglasso::finite(host_norms[j]); }
        stats.evaluated = stats.constructed = count; check();
    }
    Array response(size_t target) override {
        auto result = alloc(options.samples);
        extract_response<<<grid(options.samples), 128, 0, stream_>>>(responses_.data, result.data, options.samples, options.targets, target); check(); return result;
    }
    template<typename T> void export_typed(const siglasso::Result& result, const SigLassoOutput& output) {
        const std::vector<double>* inputs[] = {&result.coefficients, &result.intercepts, &result.objectives, &result.gaps};
        void* outputs[] = {output.coefficients, output.intercepts, output.objectives, output.gaps};
        for (size_t j = 0; j < 4; ++j) {
            auto reservation = account(inputs[j]->size() * sizeof(T));
            std::vector<T> converted(inputs[j]->size());
            std::transform(inputs[j]->begin(), inputs[j]->end(), converted.begin(), [](double value) { return static_cast<T>(siglasso::finite(static_cast<T>(value))); });
            if (!converted.empty()) {
                CUDA_CHECK(cudaMemcpyAsync(outputs[j], converted.data(), converted.size() * sizeof(T), cudaMemcpyHostToDevice, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            }
        }
        auto host_storage = account(siglasso::multiply(result.selected_columns.size(), sizeof(double*) + sizeof(double)));
        std::vector<const double*> pointers; std::vector<double> means;
        pointers.reserve(result.selected_columns.size()); means.reserve(result.selected_columns.size());
        for (const auto& column : result.selected_columns) { pointers.push_back(column.values.data); means.push_back(column.mean); }
        auto addresses = descriptors(pointers), device_means = from_host(means);
        const size_t count = options.samples * result.words.size();
        if (count) export_signatures<<<grid(count), 128, 0, stream_>>>(reinterpret_cast<const double* const*>(addresses.data),
            device_means.data, static_cast<T*>(output.signatures), options.samples, result.words.size());
        check();
    }
    void export_result(const siglasso::Result& result, const SigLassoOutput& output) override {
        if (options.single_precision) export_typed<float>(result, output);
        else export_typed<double>(result, output);
    }
    bool has_native_search() const override { return true; }
    bool reuse_search_bound() const override { return !options.single_precision; }
    bool native_search(const Array& residual, double alpha, const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active,
        size_t capacity, siglasso::SearchResult& result) override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree, nl = n * l;
        if (!l) return true;
        struct SearchPin { bool& flag; ~SearchPin() { flag = false; } } pin{search_running_};
        search_running_ = true;
        search_factors_.resize(m);
        if (options.bound_method && !factor_addresses_.owner) factor_addresses_ = alloc(m);
        if (options.bound_method && !search_factors_[0].owner) search_factors_[0] = factors(0);
        std::vector<const double*> factors_host(m);
        for (size_t k = 0; k < m; ++k) factors_host[k] = search_factors_[k].data;
        if (options.bound_method) CUDA_CHECK(cudaMemcpyAsync(factor_addresses_.data, factors_host.data(), m * sizeof(double*), cudaMemcpyHostToDevice, stream_));
        if (search_grams_.empty()) {
            search_grams_.resize(alphabet.size()); size_t bytes = 0;
            for (size_t block = 0; block < alphabet.size(); ++block) {
                const auto [low, high] = alphabet[block];
                if (high - low > l && nl && (bytes + nl * l * sizeof(double) <= options.cache_bytes / 8)) {
                    search_grams_[block] = energy_gram(low, high); bytes += search_grams_[block].bytes();
                }
            }
        }
        if (!gram_addresses_.owner) {
            std::vector<const double*> grams_host;
            for (const auto& value : search_grams_) grams_host.push_back(value.data);
            gram_addresses_ = descriptors(grams_host);
        }
        if (!search_intervals_.owner) {
            std::vector<lasso_device::Interval> intervals;
            for (size_t block = 0; block < alphabet.size(); ++block) {
                const auto [low, high] = alphabet[block]; intervals.push_back({low, high, block + 2 * (high - low) - 1});
            }
            search_intervals_ = descriptors(intervals);
        }
        if (excluded_count_ != active.size()) {
            if (exclusions_.capacity < 2 * active.size()) {
                size_t capacity = 256; while (capacity < 2 * active.size()) capacity *= 2;
                auto hashes = alloc(capacity), lengths = alloc(capacity), words = alloc(siglasso::multiply(capacity, m)), count = alloc(1);
                fill(hashes, 0); fill(lengths, 0); fill(count, 0);
                lasso_device::Seen next{reinterpret_cast<uint64_t*>(hashes.data), reinterpret_cast<uint64_t*>(lengths.data), reinterpret_cast<uint64_t*>(words.data),
                    reinterpret_cast<unsigned long long*>(count.data), capacity, m};
                if (exclusions_.capacity) lasso_device::rehash<<<grid(exclusions_.capacity), 128, 0, stream_>>>(exclusions_, next);
                exclusion_hashes_ = std::move(hashes); exclusion_lengths_ = std::move(lengths); exclusion_words_ = std::move(words);
                exclusion_count_ = std::move(count); exclusions_ = next;
            }
            std::vector<lasso_device::Active> records; std::vector<uint64_t> letters;
            for (const auto& [word, index] : active) if (index >= excluded_count_) {
                records.push_back({0, word.size(), letters.size()}); letters.insert(letters.end(), word.begin(), word.end());
            }
            auto device_records = descriptors(records), device_letters = descriptors(letters);
            lasso_device::seed_seen<<<grid(records.size()), 128, 0, stream_>>>(exclusions_, reinterpret_cast<const lasso_device::Active*>(device_records.data),
                reinterpret_cast<const uint64_t*>(device_letters.data), records.size());
            excluded_count_ = active.size();
        }
        if (!seen_.capacity) {
            size_t size = 256; while (size < 2 * constructed.size()) size *= 2;
            grow_seen(size);
            std::vector<lasso_device::Active> records; std::vector<uint64_t> letters;
            for (const auto& word : constructed) { records.push_back({0, word.size(), letters.size()}); letters.insert(letters.end(), word.begin(), word.end()); }
            auto device_records = descriptors(records), seed_letters = descriptors(letters);
            if (!records.empty()) lasso_device::seed_seen<<<grid(records.size()), 128, 0, stream_>>>(seen_, reinterpret_cast<const lasso_device::Active*>(device_records.data),
                reinterpret_cast<const uint64_t*>(seed_letters.data), records.size());
        }
        const size_t triangular = siglasso::multiply(m, m + 1) / 2;
        const bool exact = options.bound_method != 0, signed_blocks = (options.search_method & 2) != 0;
        const size_t prefix_size = exact ? triangular * nl : m * n * (l + 1);
        const size_t integral_size = exact || options.search_method ? m * nl : 0;
        const size_t linear_size = signed_blocks ? m * nl : 0;
        const size_t scratch_size = exact ? siglasso::integral_scratch_size(m) : 0;
        const size_t product_size = signed_blocks && m > 1 ? nl : 0;
        const size_t per_lane = prefix_size + triangular * nl + integral_size + linear_size + product_size + 128 * scratch_size
            + (capacity + 1) * n + 2 * m + capacity * (m + 2) + sizeof(lasso_device::Lane) / sizeof(double) + 1;
        const size_t groups = m >= 3 && (options.search_method & 1) && nl * d >= 16384 ? std::min(d, (search_blocks_ + d - 1) / d) : 1;
        size_t lanes = std::min<size_t>(128, d * groups);
        const size_t workspace_limit = options.memory_limit ? options.memory_limit - budget->used() : 64 * 1024 * 1024;
        const size_t merge_bytes = capacity ? 2 * (capacity * (n + m + 2) + 1) * sizeof(double) : 0;
        const size_t lane_budget = workspace_limit > merge_bytes ? workspace_limit - merge_bytes : 0;
        lanes = std::min(lanes, std::max<size_t>(1, lane_budget / (per_lane * sizeof(double))));
        if (traversal_lanes_) lanes = std::min(lanes, traversal_lanes_);
        const size_t root_groups = std::max<size_t>(1, lanes / d);
        if (root_groups_ != root_groups) {
            std::vector<uint64_t> frontier{0};
            while (frontier.size() < root_groups) {
                auto largest = std::max_element(frontier.begin(), frontier.end(), [&](size_t a, size_t b) {
                    return alphabet[a].second - alphabet[a].first < alphabet[b].second - alphabet[b].first;
                });
                const auto [low, high] = alphabet[*largest];
                const size_t block = *largest;
                *largest = block + 1; frontier.push_back(block + 2 * ((low + high) / 2 - low));
            }
            std::sort(frontier.begin(), frontier.end());
            search_frontier_ = descriptors(frontier); root_groups_ = root_groups;
        }
        const size_t roots = d * root_groups;
        lanes = std::min(lanes, roots);
        traversal_lanes_ = lanes;
        auto coefficients = exact ? traversal_buffer(0, lanes * prefix_size) : Array{};
        auto time_values = exact ? Array{} : traversal_buffer(22, lanes * prefix_size);
        auto linear_integrals = signed_blocks ? traversal_buffer(23, lanes * linear_size) : Array{};
        auto terminal_products = product_size ? traversal_buffer(26, lanes * product_size) : Array{};
        auto integral_scratch = exact ? traversal_buffer(24, lanes * 128 * scratch_size) : Array{};
        auto moments_buffer = traversal_buffer(1, lanes * triangular * nl);
        auto integrals_buffer = integral_size ? traversal_buffer(2, lanes * integral_size) : Array{};
        auto values = traversal_buffer(3, lanes * n);
        auto candidates = traversal_buffer(4, lanes * capacity * n), scores = traversal_buffer(5, lanes * capacity);
        auto words = traversal_buffer(6, lanes * m), next = traversal_buffer(7, lanes * m);
        auto candidate_words = traversal_buffer(8, lanes * capacity * m), lengths = traversal_buffer(9, lanes * capacity);
        auto states = traversal_buffer(10, lanes * sizeof(lasso_device::Lane) / sizeof(double));
        auto lane_ids = traversal_buffer(11, lanes);
        std::array<lasso_device::CandidateBatch, 2> merged;
        for (size_t j = 0; j < 2 && capacity; ++j) {
            const size_t start = 12 + j * 5;
            auto v = traversal_buffer(start, capacity * n), s = traversal_buffer(start + 1, capacity);
            auto w = traversal_buffer(start + 2, capacity * m), l = traversal_buffer(start + 3, capacity), c = traversal_buffer(start + 4, 1);
            fill(c, 0); merged[j] = {v.data, s.data, reinterpret_cast<uint64_t*>(w.data), reinterpret_cast<uint64_t*>(l.data), reinterpret_cast<uint64_t*>(c.data)};
        }
        size_t merge_index = 0;
        auto host_workspace = account(capacity * (m + 2) * sizeof(double) + states.bytes() + 4096);
        const size_t cache_stride = exact ? (m + 2 + signed_blocks) * nl + n
            : (m + (options.search_method != 0) + signed_blocks) * nl + 2 * n;
        if (!prefix_slots_ && !prefix_disabled_ && m > 1) {
            const size_t available = std::min<size_t>(options.cache_bytes / 8 * 3, budget->available() / 2);
            const size_t slots = std::min(lanes * 64, available / ((cache_stride + m + 3) * sizeof(double)));
            if (slots >= lanes) {
                auto cache = alloc(slots * cache_stride), keys = alloc(slots * m), lengths = alloc(slots), ticks = alloc(slots), clocks = alloc(128);
                fill(lengths, 0); fill(ticks, 0); fill(clocks, 0);
                prefix_cache_ = std::move(cache); prefix_words_ = std::move(keys); prefix_lengths_ = std::move(lengths);
                prefix_ticks_ = std::move(ticks); prefix_clocks_ = std::move(clocks); prefix_slots_ = slots;
            }
        }
        lasso_device::SearchData data{n, l, d, m, lanes, capacity, alphabet.size(), increments.data, alphabet_ratios.data,
            device_weights_.data, residual.data, reinterpret_cast<const double* const*>(factor_addresses_.data), reinterpret_cast<const double* const*>(gram_addresses_.data),
            reinterpret_cast<const lasso_device::Interval*>(search_intervals_.data), exclusions_, coefficients.data, moments_buffer.data, integrals_buffer.data, values.data, candidates.data, scores.data,
            reinterpret_cast<uint64_t*>(words.data), reinterpret_cast<uint64_t*>(next.data), reinterpret_cast<uint64_t*>(candidate_words.data),
            reinterpret_cast<uint64_t*>(lengths.data), reinterpret_cast<lasso_device::Lane*>(states.data), seen_};
        data.lane_ids = reinterpret_cast<uint64_t*>(lane_ids.data);
        data.segment_scan = n <= 16 && l >= 64;
        data.search_method = options.search_method; data.intercept = options.intercept != 0;
        data.variation = variation.data; data.after = after.data;
        data.exact = exact; data.time_values = time_values.data; data.linear_integrals = linear_integrals.data;
        data.integral_scratch = integral_scratch.data; data.scratch_size = scratch_size; data.cache_stride = cache_stride;
        data.frontier = reinterpret_cast<const uint64_t*>(search_frontier_.data); data.root_groups = root_groups;
        data.terminal_products = terminal_products.data;
        data.reciprocals = device_weights_.data + weights.size();
        if (signed_blocks) {
            auto terminal_residual = traversal_buffer(25, 3);
            data.terminal_residual = terminal_residual.data;
            lasso_device::prepare_terminal_residual<<<1, 128, 0, stream_>>>(data); check();
            if (!terminal_cache_storage_.owner && stats.terminal_queries > stats.terminal_pruned) {
                const size_t available = std::min(options.cache_bytes / 8, budget->available() / 4);
                size_t possible = 1;
                for (size_t k = 1; k < m && possible < 2048; ++k)
                    possible = d >= 2047 / possible ? 2048 : 1 + d * possible;
                size_t slots = 1;
                while (slots < 4 * possible && 2 * slots * lanes * (m + 3) * sizeof(double) <= available) slots *= 2;
                if (slots >= 64) {
                    terminal_cache_storage_ = alloc(slots * lanes * (m + 3));
                    fill(terminal_cache_storage_, 0);
                }
            }
            if (++terminal_generation_ >= 9007199254740992.0) { fill(terminal_cache_storage_, 0); terminal_generation_ = 1; }
            size_t slots = 1;
            while (2 * slots * lanes * (m + 3) <= terminal_cache_storage_.size) slots *= 2;
            if (terminal_cache_storage_.owner && slots >= 64) data.terminal_cache = {terminal_cache_storage_.data, slots, m, terminal_generation_};
        }
        std::vector<lasso_device::Lane> host_states(lanes);
        uint64_t cache_hits = 0, cache_writes = 0;
        for (size_t first = 0; first < roots; first += lanes) {
            const size_t current = std::min(lanes, roots - first);
            lasso_device::initialise<<<static_cast<unsigned int>(current), 128, 0, stream_>>>(data, first);
            size_t running = current;
            do {
                data.prefix_cache = prefix_cache_.data; data.cache_words = reinterpret_cast<uint64_t*>(prefix_words_.data);
                data.cache_lengths = reinterpret_cast<uint64_t*>(prefix_lengths_.data); data.cache_ticks = reinterpret_cast<uint64_t*>(prefix_ticks_.data);
                data.cache_clocks = reinterpret_cast<uint64_t*>(prefix_clocks_.data); data.cache_slots = prefix_slots_ / lanes;
                if (exact) lasso_device::traverse<true><<<static_cast<unsigned int>(running), 128, 0, stream_>>>(data, alpha, 512);
                else lasso_device::traverse<false><<<static_cast<unsigned int>(running), 128, 0, stream_>>>(data, alpha, 512);
                check();
                download(states.data, reinterpret_cast<double*>(host_states.data()), current * sizeof(lasso_device::Lane) / sizeof(double));
                uint64_t seen_count = 0; download(seen_count_.data, reinterpret_cast<double*>(&seen_count), 1);
                stats.constructed = seen_count;
                const size_t previous = running;
                running = 0;
                bool added = false;
                for (size_t lane = 0; lane < current; ++lane) {
                    running += host_states[lane].depth != 0;
                    const size_t k = host_states[lane].requested;
                    if (k && !search_factors_[k].owner) { search_factors_[k] = factors(k); factors_host[k] = search_factors_[k].data; added = true; }
                }
                if (added) CUDA_CHECK(cudaMemcpyAsync(factor_addresses_.data, factors_host.data(), m * sizeof(double*), cudaMemcpyHostToDevice, stream_));
                if (running && running < previous) { lasso_device::compact_lanes<<<1, 128, 0, stream_>>>(data, current); check(); }
                if (running && seen_count > seen_.capacity / 2) { grow_seen(seen_.capacity * 2); data.seen = seen_; }
            } while (running);
            if (capacity) {
                lasso_device::merge_candidates<<<1, 128, 0, stream_>>>(data, current, merged[merge_index], merged[1 - merge_index]); check();
                merge_index = 1 - merge_index;
            }
            for (size_t lane = 0; lane < current; ++lane) {
                result.omitted = std::max(result.omitted, siglasso::finite(host_states[lane].omitted));
                stats.evaluated += host_states[lane].evaluated; stats.cache_misses += host_states[lane].evaluated;
                stats.cache_hits += host_states[lane].cache_hits;
                stats.pruned += host_states[lane].pruned; stats.block_queries += host_states[lane].queries;
                stats.terminal_queries += host_states[lane].terminal_queries; stats.terminal_pruned += host_states[lane].terminal_pruned;
                stats.terminal_correlations += host_states[lane].terminal_correlations;
                cache_hits = std::min(UINT64_MAX - host_states[lane].cache_hits, cache_hits) + host_states[lane].cache_hits;
                cache_writes = std::min(UINT64_MAX - host_states[lane].cache_writes, cache_writes) + host_states[lane].cache_writes;
            }
        }
        if (capacity) {
            const auto& batch = merged[merge_index]; uint64_t count = 0;
            download(reinterpret_cast<const double*>(batch.count), reinterpret_cast<double*>(&count), 1);
            std::vector<double> host_scores(count); std::vector<uint64_t> host_words(count * m), host_lengths(count);
            download(batch.scores, host_scores.data(), count);
            download(reinterpret_cast<const double*>(batch.words), reinterpret_cast<double*>(host_words.data()), host_words.size());
            download(reinterpret_cast<const double*>(batch.lengths), reinterpret_cast<double*>(host_lengths.data()), count);
            for (size_t j = 0; j < count; ++j) result.candidates.push_back({
                siglasso::Word(host_words.begin() + j * m, host_words.begin() + j * m + host_lengths[j]),
                {batch.values + j * n, n, traversal_buffers_[12 + merge_index * 5].owner}, siglasso::finite(host_scores[j])});
        }
        if (prefix_slots_ && (cache_hits || cache_writes)) {
            constexpr size_t warm_queries = 5; // Includes the first, cold query.
            constexpr uint64_t min_opportunities = 1024, misses_per_hit = 199;
            if (prefix_queries_ < warm_queries) ++prefix_queries_;
            if (prefix_queries_ > 1) {
                prefix_hits_ = std::min(UINT64_MAX - cache_hits, prefix_hits_) + cache_hits;
                prefix_writes_ = std::min(UINT64_MAX - cache_writes, prefix_writes_) + cache_writes;
            }
            const bool enough = prefix_hits_ >= min_opportunities || prefix_writes_ >= min_opportunities - prefix_hits_;
            const bool low_hit_rate = prefix_hits_ < prefix_writes_ / misses_per_hit ||
                (prefix_hits_ == prefix_writes_ / misses_per_hit && prefix_writes_ % misses_per_hit);
            if (prefix_queries_ >= warm_queries && enough && low_hit_rate) {
                prefix_cache_ = {}; prefix_words_ = {}; prefix_lengths_ = {}; prefix_ticks_ = {}; prefix_clocks_ = {};
                prefix_slots_ = 0; prefix_disabled_ = true;
            }
        }
        size_t factor_bytes = 0; for (const auto& value : search_factors_) factor_bytes += value.bytes();
        if (factor_bytes > options.cache_bytes / 4) search_factors_.clear();
        return true;
    }
    void finish() override { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
};

template<typename T>
void run_lasso(const T* path, const T* y, const double* alphas, const uint64_t* letters, const uint64_t* offsets,
    uint64_t words, const SigLassoOptions* options, void* stream_ptr, int device, SigLassoAllocator allocate, void* context) {
    DeviceScope scope(device);
    LassoCUDA backend(*options, path, y, static_cast<cudaStream_t>(stream_ptr), device);
    std::vector<siglasso::Word> initial;
    for (size_t j = 0; j < words; ++j) initial.emplace_back(letters + offsets[j], letters + offsets[j + 1]);
    auto result = siglasso::fit(backend, std::vector<double>(alphas, alphas + options->targets), initial);
    siglasso::export_result(backend, result, allocate, context);
}
} // namespace

extern "C" {
CUSIG_API int sig_lasso_cuda_f(const float* path, const float* y, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, void* stream, int device, SigLassoAllocator allocate, void* context) noexcept {
    CUDA_SAFE_CALL(run_lasso(path, y, alphas, letters, offsets, words, options, stream, device, allocate, context));
}
CUSIG_API int sig_lasso_cuda_d(const double* path, const double* y, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, void* stream, int device, SigLassoAllocator allocate, void* context) noexcept {
    CUDA_SAFE_CALL(run_lasso(path, y, alphas, letters, offsets, words, options, stream, device, allocate, context));
}
}
