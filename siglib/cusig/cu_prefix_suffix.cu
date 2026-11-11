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

#include "cu_signature.h"
#include "cu_macros.h"
#include "cu_path_reduction.h"
#include "prefix_suffix.h"
#include "prefix_suffix_endpoint.h"
#include "prefix_suffix_selection.h"
#include <cmath>
#include "cu_prefix_suffix_backprop.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <climits>
#include <type_traits>

namespace {
using namespace prefix_suffix;

int device_parallelism(int device) {
    static thread_local int cached_device = -1;
    static thread_local int processors = 1;
    if (device != cached_device) {
        CUDA_CHECK(cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device));
        cached_device = device;
    }
    return 4 * processors;
}

void blas_check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("prefix-suffix cuBLAS error " + std::to_string(static_cast<int>(code)));
}

#include "cu_fp64_emulation.cuh"

struct Resources {
    int device;
    cublasHandle_t handle = nullptr;
    std::unique_ptr<CudaBatchWorkspace<std::byte>> storage;
    uint64_t requested_bytes = 0;
    CudaBuf<std::byte> blas_workspace;
    std::unique_ptr<Fp64Emulation> emulation;
    explicit Resources(int device_) : device(device_), blas_workspace(4ULL << 20) {
        blas_check(cublasCreate(&handle));
    }
    // The caller holds the registry lock until the stream has finished.
    static Resources& get(int device) {
        auto& state = cuda_path_workspaces().devices[device].prefix_suffix_state;
        if (!state) state = std::make_shared<Resources>(device);
        return *std::static_pointer_cast<Resources>(state);
    }

    uint64_t reserved() const {
        return storage ? mul(storage->row_elements(), storage->capacity()) : 0;
    }

    uint64_t reserve(uint64_t paths, uint64_t row_bytes, uint64_t max_cached, const char* name) {
        const uint64_t requested = mul(paths, row_bytes);
        const uint64_t allocated = reserved();
        if (allocated < row_bytes || allocated > max_cached || (requested > requested_bytes && allocated < requested)) {
            storage.reset();
            storage = std::make_unique<CudaBatchWorkspace<std::byte>>(paths, row_bytes, name);
            requested_bytes = requested;
        }
        return std::min(paths, reserved() / row_bytes);
    }

    ~Resources() {
        int previous = device;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        if (handle) (void)cublasDestroy(handle);
        storage.reset(); blas_workspace.reset(); emulation.reset();
        (void)cudaSetDevice(previous);
    }
};

template<typename T, typename Run>
void with_factor_workspace(const Layout& l, int device, uint64_t batch, uint64_t max_paths,
    uint64_t stride, uint64_t metadata_bytes, uint64_t blas_charge, bool trim_cache, const char* name,
    cudaStream_t stream, bool fp64_emulation, Run run) {
    const uint64_t row_bytes = add(mul(stride, sizeof(T)), metadata_bytes);
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX);
    const uint64_t emulation_limit = fp64_emulation && std::is_same_v<T, double>
        ? std::min(option("PYSIGLIB_FP64_EMULATION_WORKSPACE_BYTES", 512ULL << 20),
            budget > add(row_bytes, blas_charge) ? budget - row_bytes - blas_charge : 0) : 0;
    blas_charge = add(blas_charge, emulation_limit);
    const uint64_t required = add(row_bytes, blas_charge);
    if (required > budget) throw std::runtime_error(std::string("CUDA ") + name + " requires "
        + std::to_string(required) + " workspace bytes for one path; configured limit is " + std::to_string(budget));
    const uint64_t available = budget - blas_charge;
    const uint64_t paths = std::min({batch, max_paths, available / row_bytes});
    std::lock_guard lock(cuda_path_workspaces().mutex);
    Resources& resources = Resources::get(device);
    if (emulation_limit) {
        if (!resources.emulation || resources.emulation->limit != emulation_limit)
            resources.emulation = std::make_unique<Fp64Emulation>(device, emulation_limit);
        resources.emulation->stream = stream;
    } else resources.emulation.reset();
    const uint64_t capacity = resources.reserve(paths, row_bytes, trim_cache || emulation_limit ? available : UINT64_MAX, name);
    T* memory = reinterpret_cast<T*>(resources.storage->get());
    // Store metadata after the last sample; all numeric sample strides stay equal.
    auto* power = reinterpret_cast<uint64_t*>(memory + capacity * stride);
    auto* offset = power + l.power.size();
    try {
        CUDA_CHECK(cudaMemcpyAsync(power, l.power.data(), l.power.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(offset, l.offset.data(), l.offset.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        blas_check(cublasSetStream(resources.handle, stream));
        blas_check(cublasSetWorkspace(resources.handle, resources.blas_workspace.get(), 4ULL << 20));
        blas_check(cublasSetMathMode(resources.handle, CUBLAS_PEDANTIC_MATH));
        run(resources.handle, resources.emulation.get(), capacity, memory, power, offset);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

template<typename T>
__device__ T point_value(const T* path, uint64_t vertex, uint64_t channel, uint64_t dimension,
    uint64_t points, bool time_aug, bool lead_lag, T end_time) {
    if (time_aug && channel == (lead_lag ? 2 : 1) * dimension)
        return points > 1 ? T(vertex) * (end_time / T(points - 1)) : T(0);
    const uint64_t index = lead_lag ? vertex / 2 + ((vertex % 2) && channel >= dimension) : vertex;
    return path[index * dimension + channel % dimension];
}

template<typename T>
__global__ void increments_kernel(const T* path, T* increments, uint64_t batch, uint64_t dimension,
    uint64_t length, uint64_t d, uint64_t n, uint64_t stride, bool ta, bool ll, T end_time) {
    const uint64_t total = batch * n * d;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / (n * d), i = x / d % n, c = x % d;
        const T* p = path + sample * length * dimension;
        increments[sample * stride + i * d + c] = point_value(p, i + 1, c, dimension, n + 1, ta, ll, end_time) - point_value(p, i, c, dimension, n + 1, ta, ll, end_time);
    }
}

template<typename T>
__global__ void linear_factors_kernel(const T* path, T* increments, T* prefixes, T* suffixes, T* output,
    uint64_t batch, uint64_t dimension, uint64_t length, uint64_t d, uint64_t n, uint64_t plen, uint64_t qlen,
    uint64_t stride, uint64_t output_stride, bool scalar, bool ta, bool ll, T end_time) {
    const uint64_t total = batch * (n + 1) * d;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / ((n + 1) * d), i = x / d % (n + 1), c = x % d;
        const T* p = path + sample * length * dimension;
        const T current = point_value(p, i, c, dimension, n + 1, ta, ll, end_time);
        const T first = point_value(p, 0, c, dimension, n + 1, ta, ll, end_time);
        const T last = point_value(p, n, c, dimension, n + 1, ta, ll, end_time);
        if (plen > 1) prefixes[sample * stride + i * plen + 1 + c] = current - first;
        suffixes[sample * stride + i * qlen + 1 + c] = last - current;
        if (!c) prefixes[sample * stride + i * plen] = suffixes[sample * stride + i * qlen] = 1;
        if (i < n) increments[sample * stride + i * d + c] = point_value(p, i + 1, c, dimension, n + 1, ta, ll, end_time) - current;
        if (!i) {
            output[sample * output_stride + (scalar ? 1 : 0) + c] = last - first;
            if (scalar && !c) output[sample * output_stride] = 1;
        }
    }
}

template<typename T, int D, int A>
__device__ T small_left(const T* prefix, const T* v, int word, int j) {
    int tail = 1, po = 0, power = 1;
    for (int k = 0; k < j + 1; ++k) tail *= D;
    for (int k = 0; k < A - 1 - j; ++k) { po += power; power *= D; }
    T value = prefix[po + word / tail];
    for (int k = 0; k < j + 1; ++k) { value *= v[word % D]; word /= D; }
    return value;
}

template<typename T, int D, int B>
__device__ T small_right(const T* suffix, const T* v, int word, int j) {
    T value = 0, head = 1, reciprocal = 1;
    for (int k = 2; k <= j + 1; ++k) reciprocal /= T(k);
    int tail = 1, offset = 0;
    for (int k = 0; k < B; ++k) { offset += tail; tail *= D; }
    #pragma unroll
    for (int k = 0; k <= B; ++k) {
        value += head * suffix[offset + word % tail] * reciprocal;
        if (k < B) {
            tail /= D; offset -= tail;
            head *= v[word / tail % D];
            reciprocal /= T(j + k + 2);
        }
    }
    return value;
}

template<typename T, int D, int K>
__global__ void small_factors_kernel(const T* prefixes, const T* suffixes, const T* increments,
    T* left, T* right, uint64_t batch, uint64_t plen, uint64_t qlen, uint64_t base, uint64_t rows, uint64_t stride, uint64_t pitch) {
    constexpr int A = (K + 1) / 2, B = K / 2, P = small_power(D, A), Q = small_power(D, B);
    const uint64_t total = batch * rows * (P + Q);
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / (rows * (P + Q)), row = x / (P + Q) % rows, i = (base + row) / A;
        const int word = x % (P + Q), j = (base + row) % A;
        const T* v = increments + sample * stride + i * D;
        if (word < P) left[sample * stride + row * P + word] = small_left<T, D, A>(prefixes + sample * stride + i * plen, v, word, j);
        else right[sample * stride + row * pitch + word - P] = small_right<T, D, B>(suffixes + sample * stride + (i + 1) * qlen, v, word - P, j);
    }
}

template<typename T, int D, int K>
__global__ void small_contract_kernel(const T* prefixes, const T* suffixes, const T* increments,
    T* partial, uint64_t batch, uint64_t n, uint64_t plen, uint64_t qlen, uint64_t stride,
    uint64_t output_stride, uint64_t chunks) {
    constexpr int A = (K + 1) / 2, B = K / 2, P = small_power(D, A), Q = small_power(D, B);
    constexpr int tiles = (P * Q + 127) / 128;
    for (uint64_t task = blockIdx.x; task < batch * chunks * tiles; task += gridDim.x) {
        const uint64_t sample = task / (chunks * tiles), chunk = task / tiles % chunks;
        const int word = (task % tiles) * 128 + threadIdx.x;
        if (word >= P * Q) continue;
        T value = 0;
        const uint64_t end = min(n, (chunk + 1) * 256);
        for (uint64_t i = chunk * 256; i < end; ++i) {
            const T* v = increments + sample * stride + i * D;
            #pragma unroll
            for (int j = 0; j < A; ++j)
                value += small_left<T, D, A>(prefixes + sample * stride + i * plen, v, word / Q, j)
                    * small_right<T, D, B>(suffixes + sample * stride + (i + 1) * qlen, v, word % Q, j);
        }
        partial[sample * stride + chunk * output_stride + word] = value;
    }
}

template<typename T>
__global__ void sum_partials_kernel(const T* partial, T* output, uint64_t batch, uint64_t chunks,
    uint64_t stride, uint64_t output_stride, uint64_t first) {
    const uint64_t width = output_stride - first;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * width; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / width, word = first + x % width;
        T value = 0;
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) value += partial[sample * stride + chunk * output_stride + word];
        output[sample * output_stride + word] = value;
    }
}

template<typename T>
__global__ void trivial_kernel(const T* path, T* out, uint64_t batch, uint64_t dimension, uint64_t length,
    uint64_t d, uint64_t points, uint64_t degree, uint64_t stride, bool scalar, bool ta, bool ll, T end_time) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * stride; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / stride;
        uint64_t word = x % stride + (scalar ? 0 : 1), size = 1, level = 0;
        while (word >= size && level < degree) { word -= size; size *= d; ++level; }
        T value = level && points < 2 ? T(0) : T(1);
        const T* p = path + sample * length * dimension;
        if (points >= 2) for (uint64_t j = 1; j <= level; ++j) {
            const uint64_t c = word % d; word /= d;
            value *= (point_value(p, points - 1, c, dimension, points, ta, ll, end_time) - point_value(p, 0, c, dimension, points, ta, ll, end_time)) / T(j);
        }
        out[x] = value;
    }
}

template<typename T>
__device__ void fused_step(T* state, const T* v, T* first, T* second, uint64_t d,
    uint64_t h, const uint64_t* power, const uint64_t* offset, bool prepend) {
    for (uint64_t k = h; k > 1; --k) {
        for (uint64_t x = threadIdx.x; x < d; x += blockDim.x) first[x] = v[x] / T(k);
        __syncthreads();
        T* a = first; T* b = second;
        for (uint64_t level = 1; level < k; ++level) {
            const uint64_t previous = power[level];
            for (uint64_t x = threadIdx.x; x < power[level + 1]; x += blockDim.x) {
                const uint64_t parent = prepend ? x % previous : x / d;
                const uint64_t c = prepend ? x / previous : x % d;
                const T value = (state[offset[level] + parent] + a[parent]) * v[c] / T(k - level);
                if (level + 1 == k) state[offset[k] + x] += value;
                else b[x] = value;
            }
            __syncthreads();
            T* temp = a; a = b; b = temp;
        }
    }
    for (uint64_t x = threadIdx.x; x < d; x += blockDim.x) state[1 + x] += v[x];
    __syncthreads();
}

template<typename T>
__global__ void scan_kernel(const T* increments, T* prefixes, T* suffixes, T* summaries, const T* boundary,
    T* scratch, T* output, uint64_t batch, uint64_t d, uint64_t n, uint64_t h, uint64_t plen, uint64_t qlen,
    uint64_t chunks, uint64_t chunk_length, uint64_t stride, uint64_t output_stride, bool scalar,
    bool reverse, bool summary_only, const uint64_t* power, const uint64_t* offset, bool local_state) {
    extern __shared__ unsigned char scan_raw[];
    const uint64_t scratch_chunk = qlen + 2 * power[h];
    for (uint64_t task = blockIdx.x; task < batch * chunks; task += gridDim.x) {
        const uint64_t sample = task / chunks, chunk = task % chunks;
        T* state = local_state ? reinterpret_cast<T*>(scan_raw) : scratch + sample * stride + chunk * scratch_chunk;
        T* work = state + qlen;
        const T* seed = nullptr;
        if (!summary_only && ((!reverse && chunk) || (reverse && chunk + 1 < chunks)))
            seed = boundary + sample * stride + (reverse ? chunk + 1 : chunk - 1) * qlen;
        for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x) state[x] = seed ? seed[x] : T(x == 0);
        __syncthreads();
        const uint64_t first = chunk * chunk_length, end = min(n, first + chunk_length);
        for (uint64_t t = first; t < end; ++t) {
            const uint64_t i = reverse ? end - 1 - (t - first) : t;
            if (!summary_only) {
                T* destination = reverse ? suffixes + sample * stride + (i + 1) * qlen : prefixes + sample * stride + i * plen;
                for (uint64_t x = threadIdx.x; x < (reverse ? qlen : plen); x += blockDim.x) destination[x] = state[x];
            }
            __syncthreads();
            fused_step(state, increments + sample * stride + i * d, work, work + power[h], d, h, power, offset, reverse);
        }
        if (summary_only) {
            for (uint64_t x = threadIdx.x; x < qlen; x += blockDim.x) summaries[sample * stride + chunk * qlen + x] = state[x];
        } else if (!reverse && chunk + 1 == chunks) {
            for (uint64_t x = threadIdx.x + (scalar ? 0 : 1); x < qlen; x += blockDim.x)
                output[sample * output_stride + x - (scalar ? 0 : 1)] = state[x];
        }
        __syncthreads();
    }
}

template<typename T>
__global__ void summary_scan_kernel(const T* in, T* out, uint64_t batch, uint64_t chunks, uint64_t qlen,
    uint64_t stride, uint64_t shift, uint64_t h, bool reverse, const uint64_t* power, const uint64_t* offset) {
    const uint64_t total = batch * chunks * qlen;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < total; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / (chunks * qlen), chunk = x / qlen % chunks, word = x % qlen;
        const T* current = in + sample * stride + chunk * qlen;
        T value = current[word];
        if ((!reverse && chunk >= shift) || (reverse && chunk + shift < chunks)) {
            const T* other = in + sample * stride + (reverse ? chunk + shift : chunk - shift) * qlen;
            const T* left = reverse ? current : other; const T* right = reverse ? other : current;
            uint64_t level = 0;
            while (level < h && word >= offset[level + 1]) ++level;
            const uint64_t index = word - offset[level];
            value = 0;
            for (uint64_t j = 0; j <= level; ++j)
                value += left[offset[j] + index / power[level - j]] * right[offset[level - j] + index % power[level - j]];
        }
        out[sample * stride + chunk * qlen + word] = value;
    }
}

// The reverse pass also needs the signature of each chunk for its adjoint scan.
template<typename T>
void prepare_scans(const T* path, T* increments, T* prefixes, T* suffixes, T* output,
    T* summary, T* pscan, T* qscan, T* swap, T* work, uint64_t count, uint64_t dimension,
    uint64_t length, const Layout& l, uint64_t n, uint64_t plen, uint64_t qlen, uint64_t chunks,
    uint64_t chunk_length, uint64_t stride, uint64_t output_stride, bool scalar, bool ta, bool ll,
    T end_time, const uint64_t* power, const uint64_t* offset, bool backward, bool local_scan,
    unsigned threads, uint64_t shared, cudaStream_t stream) {
    const uint64_t d = l.d;
    const bool linear = l.scan_degree == 1;
    if (linear) {
        linear_factors_kernel<<<make_cuda_1d_grid(mul(count, mul(n + 1, d)), 256), 256, 0, stream>>>(
            path, increments, prefixes, suffixes, output, count, dimension, length, d, n,
            plen, qlen, stride, output_stride, scalar, ta, ll, end_time);
        CUDA_CHECK(cudaGetLastError());
        if (!backward || chunks == 1) return;
    } else {
        increments_kernel<<<make_cuda_1d_grid(mul(count, mul(n, d)), 256), 256, 0, stream>>>(
            path, increments, count, dimension, length, d, n, stride, ta, ll, end_time);
        CUDA_CHECK(cudaGetLastError());
    }
    const unsigned grid = make_cuda_1d_grid(mul(count, chunks), 1);
    auto scan = [&](T* boundary, bool reverse, bool summary_only) {
        scan_kernel<<<grid, threads, shared, stream>>>(increments, prefixes, suffixes, summary,
            boundary, work, output, count, d, n, l.scan_degree, plen, qlen, chunks, chunk_length,
            stride, output_stride, scalar, reverse, summary_only, power, offset, local_scan);
        CUDA_CHECK(cudaGetLastError());
    };
    if (chunks > 1) scan(summary, false, true);
    if (linear) return;
    for (bool reverse : {false, true}) {
        T* current = summary; T* next = reverse ? qscan : pscan; T* alternate = swap;
        for (uint64_t shift = 1; shift < chunks; shift *= 2) {
            summary_scan_kernel<<<make_cuda_1d_grid(mul(count, mul(chunks, qlen)), 256), 256, 0, stream>>>(
                current, next, count, chunks, qlen, stride, shift, l.scan_degree, reverse, power, offset);
            CUDA_CHECK(cudaGetLastError());
            current = next; next = alternate; alternate = current;
        }
        scan(current, reverse, false);
    }
}

template<typename T> struct FactorCoefficients {
    T inverse[64], factorial[64];
    FactorCoefficients() {
        inverse[0] = 0; factorial[0] = 1;
        for (int i = 1; i < 64; ++i) {
            inverse[i] = T(1) / T(i);
            factorial[i] = factorial[i - 1] / T(i);
        }
    }
};

template<typename T, int D>
__device__ T coefficient_right(const T* suffix, const T* v, unsigned word, unsigned dimension,
    unsigned j, unsigned b, const FactorCoefficients<T>& coefficients) {
    const unsigned d = D ? D : dimension;
    unsigned index = 0, previous = 1;
    uint64_t offset = 1;
    T value = 1;
    for (unsigned level = 1; level <= b; ++level) {
        const unsigned channel = word % d; word /= d;
        index += channel * previous;
        value = suffix[offset + index] + v[channel] * value * coefficients.inverse[j + b - level + 2];
        previous *= d; offset += previous;
    }
    return value * coefficients.factorial[j + 1];
}

template<typename T, int D>
__global__ void coefficient_factors_kernel(const T* prefixes, const T* suffixes, const T* increments,
    T* left, T* right, uint64_t batch, unsigned dimension, uint64_t plen, uint64_t qlen,
    unsigned a, unsigned b, unsigned p, unsigned q, uint64_t base, uint64_t rows,
    uint64_t stride, uint64_t pitch, const uint64_t* offset, FactorCoefficients<T> coefficients) {
    const unsigned d = D ? D : dimension, lane = threadIdx.x % 32;
    for (uint64_t task = uint64_t(blockIdx.x) * 8 + threadIdx.x / 32; task < batch * rows; task += uint64_t(gridDim.x) * 8) {
        const uint64_t sample = task / rows, row = task % rows, i = (base + row) / a;
        const unsigned j = (base + row) % a;
        const T* v = increments + sample * stride + i * d;
        const T* prefix = prefixes + sample * stride + i * plen + offset[a - 1 - j];
        const T* suffix = suffixes + sample * stride + (i + 1) * qlen;
        for (unsigned word = lane; word < p + q; word += 32) {
            if (word < p) {
                unsigned index = word;
                T value = 1;
                for (unsigned level = 0; level <= j; ++level) { value *= v[index % d]; index /= d; }
                left[sample * stride + row * p + word] = prefix[index] * value;
            } else right[sample * stride + row * pitch + word - p] = coefficient_right<T, D>(suffix, v, word - p, d, j, b, coefficients);
        }
    }
}

template<typename T, int D>
__global__ void extend_factors_kernel(T* right, const T* suffixes, const T* increments, T* work,
    uint64_t slots, uint64_t maxq, uint64_t batch, unsigned dimension, uint64_t qlen, unsigned a,
    unsigned b, unsigned q, uint64_t base, uint64_t rows, uint64_t stride, uint64_t pitch,
    const uint64_t* offset, FactorCoefficients<T> coefficients, bool shared) {
    extern __shared__ unsigned char raw[];
    const unsigned d = D ? D : dimension;
    const uint64_t first = base / a, segments = (base + rows - 1) / a - first + 1;
    T* extra = shared ? reinterpret_cast<T*>(raw) : work + uint64_t(blockIdx.x) / slots * stride + blockIdx.x % slots * 2 * maxq;
    for (uint64_t task = blockIdx.x; task < batch * segments; task += gridDim.x) {
        const uint64_t sample = task / segments, i = first + task % segments;
        const uint64_t begin = max(base, i * a), end = min(base + rows, (i + 1) * a);
        const T* v = increments + sample * stride + i * d;
        const T* suffix = suffixes + sample * stride + (i + 1) * qlen;
        for (unsigned word = threadIdx.x; word < q; word += blockDim.x)
            extra[word] = coefficient_right<T, D>(suffix, v, word, d, unsigned(end - i * a), b, coefficients);
        __syncthreads();
        for (uint64_t t = begin; t < end; ++t) {
            const unsigned j = t - i * a;
            const uint64_t row = t - base;
            const T* previous = t + 1 == end ? extra : right + sample * stride + (row + 1) * pitch;
            for (uint64_t word = threadIdx.x; word < pitch; word += blockDim.x)
                right[sample * stride + row * pitch + word] = suffix[offset[b + 1] + word] * coefficients.factorial[j + 1]
                    + v[word / q] * previous[word % q];
            __syncthreads();
        }
    }
}

template<typename T>
__global__ void factors_kernel(const T* prefixes, const T* suffixes, const T* increments, T* left, T* right,
    T* work, uint64_t slots, uint64_t maxq, uint64_t batch, uint64_t d, uint64_t plen, uint64_t qlen,
    uint64_t a, uint64_t b, uint64_t p, uint64_t q, uint64_t base, uint64_t rows, uint64_t stride,
    const uint64_t* power, const uint64_t* offset, bool shared, uint64_t pitch, FactorCoefficients<T> coefficients) {
    extern __shared__ unsigned char raw[];
    T* first = shared ? reinterpret_cast<T*>(raw) : work + uint64_t(blockIdx.x) / slots * stride + (blockIdx.x % slots) * 2 * maxq;
    T* second = first + max(q, power[a - 1]);
    for (uint64_t task = blockIdx.x; task < batch * rows; task += gridDim.x) {
        const uint64_t sample = task / rows, row = task % rows, i = (base + row) / a, j = (base + row) % a;
        const T* v = increments + sample * stride + i * d;
        const T* suffix = suffixes + sample * stride + (i + 1) * qlen;
        if (!threadIdx.x) first[0] = 1;
        __syncthreads();
        T* current = first; T* next = second;
        for (uint64_t level = 1; level <= j; ++level) {
            for (uint64_t word = threadIdx.x; word < power[level]; word += blockDim.x)
                next[word] = current[word / d] * v[word % d];
            __syncthreads(); T* temp = current; current = next; next = temp;
        }
        const uint64_t tail = power[j + 1];
        const T* prefix = prefixes + sample * stride + i * plen + offset[a - 1 - j];
        for (uint64_t word = threadIdx.x; word < p; word += blockDim.x)
            left[sample * stride + row * p + word] = prefix[word / tail] * current[(word % tail) / d] * v[word % d];
        __syncthreads();
        if (!threadIdx.x) first[0] = 1;
        __syncthreads(); current = first; next = second;
        for (uint64_t level = 1; level <= b; ++level) {
            const uint64_t previous = power[level - 1];
            for (uint64_t word = threadIdx.x; word < power[level]; word += blockDim.x)
                next[word] = suffix[offset[level] + word] + v[word / previous] * current[word % previous] * coefficients.inverse[j + b - level + 2];
            __syncthreads(); T* temp = current; current = next; next = temp;
        }
        const T scale = coefficients.factorial[j + 1];
        for (uint64_t word = threadIdx.x; word < q; word += blockDim.x) right[sample * stride + row * pitch + word] = current[word] * scale;
        __syncthreads();
    }
}

template<typename T>
__global__ void zero_adjoints_kernel(T* data, uint64_t batch, uint64_t width, uint64_t stride) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * width;
        x += uint64_t(gridDim.x) * blockDim.x)
        data[(x / width) * stride + x % width] = 0;
}

template<typename T> struct CudaMatrices {
    cublasHandle_t handle;
    uint64_t batch;
    Fp64Emulation* emulation = nullptr;
    void gemm(Matrix<T> a, Matrix<T> b, Matrix<T> c, bool first) {
        uint64_t remaining = batch;
        if constexpr (std::is_same_v<T, double>) {
            while (emulation && remaining) {
                const uint64_t count = emulation->gemm(a, b, c, remaining, first);
                if (!count) break;
                remaining -= count;
                if (remaining) { a.data += count * a.bs; b.data += count * b.bs; c.data += count * c.bs; }
            }
            if (!remaining) return;
        }
        const T beta = first ? T(0) : T(1);
        const auto op_a = a.rs == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
        const auto op_b = b.rs == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
        constexpr auto dtype = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
        const auto compute = std::is_same_v<T, float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_64F_PEDANTIC;
        const T alpha = 1;
        if (remaining == 1) blas_check(cublasGemmEx_64(handle, op_b, op_a, b.cols, a.rows, a.cols, &alpha,
            b.data, dtype, b.leading_dimension(), a.data, dtype, a.leading_dimension(), &beta, c.data, dtype, c.rs, compute, CUBLAS_GEMM_DEFAULT));
        else blas_check(cublasGemmStridedBatchedEx_64(handle, op_b, op_a, b.cols, a.rows, a.cols, &alpha,
            b.data, dtype, b.leading_dimension(), b.bs, a.data, dtype, a.leading_dimension(), a.bs, &beta, c.data, dtype, c.rs, c.bs,
            remaining, compute, CUBLAS_GEMM_DEFAULT));
    }
};

#include "cu_prefix_suffix_endpoint.cuh"
#include "cu_prefix_suffix_correction.cuh"
} // namespace

template<bool Backward, typename T>
void signature_prefix_suffix_correction_cuda(const T* path, T* out, const T* deriv, uint64_t batch,
    uint64_t dimension, uint64_t length, uint64_t degree, bool ta, T end_time, bool scalar, cudaStream_t stream,
    uint64_t split, const T* correction, uint64_t correction_len, uint64_t batch_stride, uint64_t segment_stride, bool fp64_emulation) {
    gpu_correction::compute<Backward>(path, out, deriv, batch, dimension, length, degree, ta, end_time, scalar,
        stream, split, correction, correction_len, batch_stride, segment_stride, fp64_emulation);
}

template void signature_prefix_suffix_correction_cuda<false, float>(const float*, float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, float, bool, cudaStream_t, uint64_t, const float*, uint64_t, uint64_t, uint64_t, bool);
template void signature_prefix_suffix_correction_cuda<true, float>(const float*, float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, float, bool, cudaStream_t, uint64_t, const float*, uint64_t, uint64_t, uint64_t, bool);
template void signature_prefix_suffix_correction_cuda<false, double>(const double*, double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, double, bool, cudaStream_t, uint64_t, const double*, uint64_t, uint64_t, uint64_t, bool);
template void signature_prefix_suffix_correction_cuda<true, double>(const double*, double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, double, bool, cudaStream_t, uint64_t, const double*, uint64_t, uint64_t, uint64_t, bool);

template<typename T>
void signature_prefix_suffix_backprop_cuda(const T* path, T* out, const T* deriv, uint64_t batch,
    uint64_t dimension, uint64_t length, uint64_t degree, bool ta, bool ll, T end_time, bool scalar, cudaStream_t stream, uint64_t ps_split, bool fp64_emulation) {
    if (!dimension) throw std::invalid_argument("path has zero channels");
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    const uint64_t d = add(mul(dimension, ll ? 2 : 1), ta ? 1 : 0), points = ll && length ? add(length, length - 1) : length;
    Layout l(d, degree, ps_split);
    const uint64_t sig_stride = l.offset[degree + 1] - (scalar ? 0 : 1), path_stride = mul(length, dimension);
    mul(mul(batch, sig_stride), sizeof(T)); mul(mul(batch, path_stride), sizeof(T));
    if (!batch || !path_stride) return;
    if (points < 2 || degree < 2 || d == 1) {
        cuda_detail::trivial_backprop_kernel<<<make_cuda_1d_grid(mul(batch, path_stride), 256), 256, 0, stream>>>(
            path, deriv, out, batch, dimension, length, d, degree, sig_stride, scalar, ll);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    if (!ps_split) {
        const auto choice = prefix_suffix_selection::select_cuda(true, batch, points - 1, l, 1, ta || ll);
        if (choice.a) l.set_split(choice.a);
    }
    const uint64_t n = points - 1, qlen = l.offset[l.scan_degree + 1], scan_power = l.power[l.scan_degree];
    const uint64_t chunk_length = option("PYSIGLIB_PS_SCAN_CHUNK", l.scan_degree == 1 ? n : 16), chunks = (n - 1) / chunk_length + 1;
    const uint64_t maxp = l.power[l.left_degree(degree)], maxq = l.power[l.right_degree(degree)];
    const uint64_t factor_bytes = option("PYSIGLIB_PS_FACTOR_BYTES", 128ULL << 20);
    const uint64_t rows = l.scan_degree < l.m ? std::max<uint64_t>(1, std::min(mul(n, l.left_degree(degree)),
        factor_bytes / mul(mul(2, add(maxp, maxq)), sizeof(T)))) : 0;
    const uint64_t temporary = l.offset[std::max(l.scan_degree, l.left_degree(degree)) + 1];
    const uint64_t slots = std::max<uint64_t>(1, std::min({n, uint64_t(256), factor_bytes / mul(mul(4, temporary), sizeof(T))}));
    const uint64_t inc_count = mul(n, d), scan_count = mul(add(n, 1), qlen), summary_count = mul(chunks, qlen);
    const uint64_t scan_scratch = mul(chunks, add(qlen, mul(2, scan_power)));
    const uint64_t work_count = std::max({scan_scratch, mul(4, mul(slots, temporary)), mul(3, mul(chunks, qlen))});
    uint64_t stride = add(mul(2, inc_count), mul(4, scan_count));
    stride = add(stride, add(mul(5, summary_count), add(work_count, qlen)));
    stride = add(stride, mul(2, mul(rows, add(maxp, maxq))));
    stride = mul(add(stride, 63) / 64, 64);
    const uint64_t metadata_bytes = mul(add(l.power.size(), l.offset.size()), sizeof(uint64_t));
    const FactorCoefficients<T> coefficients;
    const unsigned scan_threads = scan_power <= 64 ? 32 : 128;
    const bool local_scan = mul(add(qlen, mul(2, scan_power)), sizeof(T)) <= 32768;
    const uint64_t scan_shared = local_scan ? mul(add(qlen, mul(2, scan_power)), sizeof(T)) : 0;
    const bool local_backprop = mul(3, mul(qlen, sizeof(T))) <= 32768;
    const uint64_t backprop_shared = local_backprop ? mul(3, mul(qlen, sizeof(T))) : 0;
    with_factor_workspace<T>(l, device, batch, 32, stride, mul(add(metadata_bytes, 255) / 256, 256),
        4ULL << 20, false, "prefix-suffix backprop", stream, fp64_emulation,
        [&](cublasHandle_t handle, Fp64Emulation* emulation, uint64_t capacity, T* memory, uint64_t* power, uint64_t* offset) {
        for (uint64_t base_batch = 0; base_batch < batch; base_batch += capacity) {
            const uint64_t count = std::min(capacity, batch - base_batch);
            T* inc = memory; T* prefixes = inc + inc_count; T* suffixes = prefixes + scan_count;
            T* dprefix = suffixes + scan_count; T* dsuffix = dprefix + scan_count;
            T* dinc = dsuffix + scan_count; T* summary = dinc + inc_count;
            T* pscan = summary + summary_count; T* qscan = pscan + summary_count;
            T* swap = qscan + summary_count; T* boundary = swap + summary_count;
            T* work = boundary + summary_count;
            T* low_output = work + work_count; T* left = low_output + qlen;
            T* right = left + rows * maxp; T* dleft = right + rows * maxq;
            T* dright = dleft + rows * maxp;
            const T* input = path + base_batch * path_stride;
            const T* gradient = deriv + base_batch * sig_stride;
            CudaMatrices<T> matrices{handle, count, emulation};
            const uint64_t zero_count = add(mul(2, scan_count), inc_count);
            zero_adjoints_kernel<<<make_cuda_1d_grid(mul(count, zero_count), 256), 256, 0, stream>>>(
                dprefix, count, zero_count, stride);
            CUDA_CHECK(cudaGetLastError());
            prepare_scans(input, inc, prefixes, suffixes, low_output, summary, pscan, qscan, swap, work,
                count, dimension, length, l, n, qlen, qlen, chunks, chunk_length, stride, stride, true, ta, ll,
                end_time, power, offset, true, local_scan, scan_threads, scan_shared, stream);
            for (uint64_t k = l.scan_degree + 1; k <= degree; ++k) {
                const uint64_t a = l.left_degree(k), b = l.right_degree(k), p = l.power[a], q = l.power[b], total = mul(n, a);
                for (uint64_t base = 0; base < total; base += rows) {
                    const uint64_t nr = std::min(rows, total - base);
                    if (add(p, q) <= INT_MAX) {
                        coefficient_factors_kernel<T, 0><<<make_cuda_1d_grid(mul(count, nr), 8), 256, 0, stream>>>(
                            prefixes, suffixes, inc, left, right, count, unsigned(d), qlen, qlen,
                            unsigned(a), unsigned(b), unsigned(p), unsigned(q), base, nr, stride, q, offset, coefficients);
                    } else {
                        const uint64_t width = std::max(q, l.power[a - 1]);
                        const bool shared = mul(2, mul(width, sizeof(T))) <= 32768;
                        const uint64_t factor_slots = std::min(slots, nr);
                        factors_kernel<<<make_cuda_1d_grid(mul(count, factor_slots), 1), 128, shared ? 2 * width * sizeof(T) : 0, stream>>>(
                            prefixes, suffixes, inc, left, right, work, factor_slots, temporary, count, d, qlen, qlen,
                            a, b, p, q, base, nr, stride, power, offset, shared, q, coefficients);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    T* g = const_cast<T*>(gradient + l.offset[k] - !scalar);
                    matrices.gemm(Matrix<T>{right, nr, q, q, 1, stride}, Matrix<T>{g, q, p, 1, q, sig_stride},
                        Matrix<T>{dleft, nr, p, p, 1, stride}, true);
                    matrices.gemm(Matrix<T>{left, nr, p, p, 1, stride}, Matrix<T>{g, p, q, q, 1, sig_stride},
                        Matrix<T>{dright, nr, q, q, 1, stride}, true);
                    const uint64_t segments = (base + nr - 1) / a - base / a + 1;
                    const uint64_t active_slots = std::min(slots, segments);
                    cuda_detail::factors_backprop_kernel<<<make_cuda_1d_grid(mul(count, active_slots), 1), 128, 0, stream>>>(
                        prefixes, suffixes, inc, dleft, dright, dprefix, dsuffix, dinc, work,
                        count, d, qlen, a, b, base, nr, active_slots, temporary, stride, power, offset);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            auto reverse_scan = [&]<bool Prepend>() {
                const T* sources = Prepend ? dsuffix : dprefix;
                T* current = swap; T* next = boundary;
                if (chunks > 1) {
                    cuda_detail::scan_adjoint_seeds_kernel<Prepend><<<make_cuda_1d_grid(mul(count, chunks), 1), scan_threads, backprop_shared, stream>>>(
                        inc, sources, gradient, current, work, count, d, n, l.scan_degree, qlen, chunks, chunk_length,
                        stride, sig_stride, scalar, power, offset, local_backprop);
                    CUDA_CHECK(cudaGetLastError());
                    T* sig_current = summary; T* sig_next = pscan; T* sig_alternate = qscan;
                    for (uint64_t shift = 1; shift < chunks; shift *= 2) {
                        cuda_detail::summary_adjoint_kernel<Prepend><<<make_cuda_1d_grid(mul(count, summary_count), 256), 256, 0, stream>>>(
                            sig_current, current, next, count, chunks, qlen, stride, shift, l.scan_degree, power, offset);
                        CUDA_CHECK(cudaGetLastError());
                        summary_scan_kernel<<<make_cuda_1d_grid(mul(count, summary_count), 256), 256, 0, stream>>>(
                            sig_current, sig_next, count, chunks, qlen, stride, shift, l.scan_degree, !Prepend, power, offset);
                        CUDA_CHECK(cudaGetLastError());
                        std::swap(current, next);
                        sig_current = sig_next; sig_next = sig_alternate; sig_alternate = sig_current;
                    }
                }
                cuda_detail::scans_backprop_kernel<Prepend><<<make_cuda_1d_grid(mul(count, chunks), 1), scan_threads, backprop_shared, stream>>>(
                    Prepend ? suffixes : prefixes, inc, sources, current, gradient, dinc, work, count, d, n,
                    l.scan_degree, qlen, chunks, chunk_length, stride, sig_stride, scalar, power, offset, local_backprop);
                CUDA_CHECK(cudaGetLastError());
            };
            reverse_scan.template operator()<false>();
            reverse_scan.template operator()<true>();
            cuda_detail::path_backprop_kernel<<<make_cuda_1d_grid(mul(count, path_stride), 256), 256, 0, stream>>>(
                dinc, out + base_batch * path_stride, count, dimension, length, d, n, stride, ll);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

template<typename T>
void signature_prefix_suffix_cuda(const T* path, T* out, uint64_t batch, uint64_t dimension, uint64_t length,
    uint64_t degree, bool ta, bool ll, T end_time, bool scalar, cudaStream_t stream, uint64_t ps_split, bool fp64_emulation) {
    if (!dimension) throw std::invalid_argument("path has zero channels");
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    const uint64_t d = add(mul(dimension, ll ? 2 : 1), ta ? 1 : 0), points = ll && length ? add(length, length - 1) : length;
    Layout l(d, degree, ps_split);
    const uint64_t output_stride = l.offset[degree + 1] - (scalar ? 0 : 1), path_stride = mul(length, dimension);
    mul(mul(batch, output_stride), sizeof(T)); mul(mul(batch, path_stride), sizeof(T));
    if (!batch || !output_stride) return;
    if (points < 2 || degree < 2 || points == 2 || d == 1) {
        trivial_kernel<<<make_cuda_1d_grid(mul(batch, output_stride), 256), 256, 0, stream>>>(path, out, batch, dimension, length, d, points, degree, output_stride, scalar, ta, ll, end_time);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    const auto choice = !ps_split || ps_split == degree
        ? prefix_suffix_selection::select_cuda(false, batch, points - 1, l, device_parallelism(device), ta || ll)
        : prefix_suffix_selection::Choice{};
    const uint64_t endpoint_block = ps_split ? choice.block : option("PYSIGLIB_PS_ENDPOINT_BLOCK", choice.block, true);
    if (endpoint_block) {
        try {
            gpu_endpoint::compute(path, out, prefix_suffix_endpoint::Shape(batch, d, points, degree, endpoint_block),
                stream, dimension, length, ta, ll, end_time, scalar);
            return;
        } catch (const std::bad_alloc&) {
            if (!choice.block) throw;
        }
    }
    const uint64_t n = points - 1, plen = l.offset[l.prefix_degree + 1], qlen = l.offset[l.scan_degree + 1];
    const uint64_t scan_power = l.power[l.scan_degree];
    const uint64_t chunk_length = option("PYSIGLIB_PS_SCAN_CHUNK", 64), chunks = (n - 1) / chunk_length + 1;
    const uint64_t maxp = l.power[l.left_degree(degree)], maxq = l.power[l.right_degree(degree)];
    uint64_t rows = std::max<uint64_t>(1, std::min(mul(n, l.left_degree(degree)), option("PYSIGLIB_PS_FACTOR_BYTES", 128ULL << 20) / mul(add(maxp, maxq), sizeof(T))));
    const bool low = degree <= 6;
    const bool specialized = !l.ps_split && low && (d == 2 || d == 3 || d == 4 || d == 8);
    const bool fused = specialized && l.power[degree] <= 1024;
    const bool pair_levels = !l.ps_split;
    const bool coefficient_factors = option("PYSIGLIB_PS_COEFFICIENT_FACTORS", 1, true) != 0;
    const FactorCoefficients<T> coefficients;
    const uint64_t contract_chunks = (n - 1) / 256 + 1;
    const uint64_t partial_count = fused ? mul(contract_chunks, output_stride) : 0;
    if (fused) rows = 0;
    const uint64_t inc_count = mul(n, d), p_count = mul(add(n, 1), plen), q_count = mul(add(n, 1), qlen);
    const uint64_t factor_width = std::max(maxq, l.power[l.left_degree(degree) - 1]);
    const uint64_t factor_work = mul(2, mul(factor_width, std::min<uint64_t>(rows, 256)));
    const bool local_scan = low && mul(add(qlen, mul(2, scan_power)), sizeof(T)) <= 32768;
    const uint64_t scan_shared = local_scan ? mul(add(qlen, mul(2, scan_power)), sizeof(T)) : 0;
    const unsigned scan_threads = low && scan_power <= 64 ? 32 : 128;
    const uint64_t summary_count = low && l.scan_degree == 1 ? 0 : mul(chunks, qlen);
    const uint64_t scan_work = std::max(local_scan ? 0 : mul(chunks, add(qlen, mul(2, scan_power))), mul(2 * factor_width, sizeof(T)) > 32768 ? factor_work : 0);
    uint64_t stride = add(add(inc_count, add(p_count, q_count)), add(mul(4, summary_count), add(scan_work, add(mul(rows, add(maxp, maxq)), partial_count))));
    const uint64_t metadata_bytes = mul(add(l.power.size(), l.offset.size()), sizeof(uint64_t));
    // Align each sample and the metadata for cuBLAS and uint64_t access.
    stride = add(stride, 63) / 64 * 64;
    with_factor_workspace<T>(l, device, batch, low ? 256 : 32, stride, mul(add(metadata_bytes, 255) / 256, 256),
        0, false, "prefix-suffix", stream, fp64_emulation,
        [&](cublasHandle_t handle, Fp64Emulation* emulation, uint64_t capacity, T* memory, uint64_t* power, uint64_t* offset) {
        for (uint64_t base_batch = 0; base_batch < batch; base_batch += capacity) {
            const uint64_t count = std::min(capacity, batch - base_batch);
            T* inc = memory; T* prefixes = inc + inc_count; T* suffixes = prefixes + p_count;
            T* summary = suffixes + q_count; T* pscan = summary + summary_count;
            T* qscan = pscan + summary_count; T* swap = qscan + summary_count;
            T* work = swap + summary_count; T* left = work + scan_work;
            T* right = left + rows * maxp;
            T* partial = right + rows * maxq;
            T* output = out + base_batch * output_stride;
            prepare_scans(path + base_batch * path_stride, inc, prefixes, suffixes, output, summary, pscan,
                qscan, swap, work, count, dimension, length, l, n, plen, qlen, chunks, chunk_length, stride,
                output_stride, scalar, ta, ll, end_time, power, offset, false, local_scan, scan_threads, scan_shared, stream);
            CudaMatrices<T> matrices{handle, count, emulation};
            for (uint64_t k = l.scan_degree + 1; k <= degree; ++k) {
                const uint64_t a = l.left_degree(k), b = l.right_degree(k), p = l.power[a], q = l.power[b], total = mul(n, a);
                const bool paired = pair_levels && !fused && k % 2 && k < degree && add(p, q) <= INT_MAX;
                const uint64_t pitch = paired ? l.power[b + 1] : q;
                auto dispatch = [&](bool contract, uint64_t base, uint64_t nr) {
                    auto dimension = [&]<int D>() {
                        auto level = [&]<int K>() {
                            if (contract) {
                                const uint64_t tiles = (l.power[k] + 127) / 128;
                                small_contract_kernel<T, D, K><<<make_cuda_1d_grid(mul(count, mul(contract_chunks, tiles)), 1), 128, 0, stream>>>(
                                    prefixes, suffixes, inc, partial + l.offset[k] - (scalar ? 0 : 1), count, n, plen, qlen, stride, output_stride, contract_chunks);
                            } else {
                                small_factors_kernel<T, D, K><<<make_cuda_1d_grid(mul(count, mul(nr, p + q)), 256), 256, 0, stream>>>(
                                    prefixes, suffixes, inc, left, right, count, plen, qlen, base, nr, stride, pitch);
                            }
                        };
                        switch (k) {
                        case 2: level.template operator()<2>(); break;
                        case 3: level.template operator()<3>(); break;
                        case 4: level.template operator()<4>(); break;
                        case 5: level.template operator()<5>(); break;
                        case 6: level.template operator()<6>(); break;
                        }
                    };
                    switch (d) {
                    case 2: dimension.template operator()<2>(); break;
                    case 3: dimension.template operator()<3>(); break;
                    case 4: dimension.template operator()<4>(); break;
                    case 8: dimension.template operator()<8>(); break;
                    }
                    CUDA_CHECK(cudaGetLastError());
                };
                if (fused) { dispatch(true, 0, total); continue; }
                for (uint64_t base = 0; base < total; base += rows) {
                    const uint64_t nr = std::min(rows, total - base);
                    auto factor_dimension = [&]<int D>(bool extend) {
                        if (extend) {
                            const uint64_t segments = (base + nr - 1) / a - base / a + 1;
                            const uint64_t slots = std::min<uint64_t>(segments, 256);
                            const bool shared = mul(q, sizeof(T)) <= 32768;
                            extend_factors_kernel<T, D><<<make_cuda_1d_grid(mul(count, slots), 1), 128,
                                shared ? q * sizeof(T) : 0, stream>>>(right, suffixes, inc, work, slots, maxq,
                                count, unsigned(d), qlen, unsigned(a), unsigned(b), unsigned(q), base, nr, stride,
                                pitch, offset, coefficients, shared);
                        } else coefficient_factors_kernel<T, D><<<make_cuda_1d_grid(mul(count, nr), 8), 256, 0, stream>>>(
                            prefixes, suffixes, inc, left, right, count, unsigned(d), plen, qlen, unsigned(a), unsigned(b),
                            unsigned(p), unsigned(q), base, nr, stride, pitch, offset, coefficients);
                    };
                    auto factor_dispatch = [&](bool extend) {
                        switch (d) {
                        case 2: factor_dimension.template operator()<2>(extend); break;
                        case 3: factor_dimension.template operator()<3>(extend); break;
                        case 4: factor_dimension.template operator()<4>(extend); break;
                        case 5: factor_dimension.template operator()<5>(extend); break;
                        case 7: factor_dimension.template operator()<7>(extend); break;
                        case 8: factor_dimension.template operator()<8>(extend); break;
                        case 16: factor_dimension.template operator()<16>(extend); break;
                        case 32: factor_dimension.template operator()<32>(extend); break;
                        default: factor_dimension.template operator()<0>(extend); break;
                        }
                    };
                    if (specialized) dispatch(false, base, nr);
                    else if (coefficient_factors && add(p, q) <= INT_MAX) factor_dispatch(false);
                    else {
                    const uint64_t width = std::max(q, l.power[a - 1]);
                    const bool shared = mul(2 * width, sizeof(T)) <= 32768;
                    const uint64_t slots = std::min<uint64_t>(nr, 256);
                    factors_kernel<<<make_cuda_1d_grid(mul(count, slots), 1), 128, shared ? 2 * width * sizeof(T) : 0, stream>>>(prefixes, suffixes, inc, left, right, work, slots, factor_width, count, d, plen, qlen, a, b, p, q, base, nr, stride, power, offset, shared, pitch, coefficients);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    matrices.gemm(Matrix<T>{left, p, nr, 1, p, stride}, Matrix<T>{right, nr, q, pitch, 1, stride},
                        Matrix<T>{output + l.offset[k] - (scalar ? 0 : 1), p, q, q, 1, output_stride}, base == 0);
                    if (paired) {
                        factor_dispatch(true);
                        CUDA_CHECK(cudaGetLastError());
                        matrices.gemm(Matrix<T>{left, p, nr, 1, p, stride}, Matrix<T>{right, nr, pitch, pitch, 1, stride},
                            Matrix<T>{output + l.offset[k + 1] - (scalar ? 0 : 1), p, pitch, pitch, 1, output_stride},
                            base == 0);
                    }
                }
                if (paired) ++k;
            }
            if (fused && l.scan_degree < degree) {
                const uint64_t first = l.offset[l.scan_degree + 1] - (scalar ? 0 : 1);
                sum_partials_kernel<<<make_cuda_1d_grid(mul(count, output_stride - first), 256), 256, 0, stream>>>(
                    partial, output, count, contract_chunks, stride, output_stride, first);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    });
}
template void signature_prefix_suffix_cuda<float>(const float*, float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, float, bool, cudaStream_t, uint64_t, bool);
template void signature_prefix_suffix_backprop_cuda<float>(const float*, float*, const float*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, float, bool, cudaStream_t, uint64_t, bool);
template void signature_prefix_suffix_cuda<double>(const double*, double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, cudaStream_t, uint64_t, bool);
template void signature_prefix_suffix_backprop_cuda<double>(const double*, double*, const double*, uint64_t, uint64_t, uint64_t, uint64_t, bool, bool, double, bool, cudaStream_t, uint64_t, bool);
