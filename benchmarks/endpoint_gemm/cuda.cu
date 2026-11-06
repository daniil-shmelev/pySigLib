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

#include "common.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>

namespace endpoint {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(int(code)));
}
struct DeviceShape {
    uint64_t d, m, n, block, chunks, width, lower, total;
    uint64_t power[33], offset[33];
    explicit DeviceShape(const Shape& s) : d(s.d), m(s.m), n(s.n), block(s.block), chunks(s.chunks),
        width(s.width), lower(s.lower), total(s.total) {
        std::copy(s.power.begin(), s.power.end(), power);
        std::copy(s.offset.begin(), s.offset.end(), offset);
    }
};
struct Resources {
    int device;
    cublasHandle_t handle{};
    void* buffer = nullptr;
    uint64_t bytes = 0;
    explicit Resources(int d) : device(d) { check(cublasCreate(&handle)); }
    ~Resources() { cudaFree(buffer); cublasDestroy(handle); }
    void reserve(uint64_t size) {
        if (size <= bytes) return;
        check(cudaFree(buffer)); buffer = nullptr; bytes = 0;
        check(cudaMalloc(&buffer, size)); bytes = size;
    }
};

template<typename T>
__global__ void prepare(const T* path, T* blocks, T* w, T* v, T* scratch, DeviceShape s,
    uint64_t first_task, uint64_t count, bool shared) {
    extern __shared__ unsigned char raw[];
    const uint64_t temporary = s.power[s.m - 2];
    const uint64_t work_size = s.lower + 2 * temporary + s.d;
    for (uint64_t slot = blockIdx.x; slot < count; slot += gridDim.x) {
        const uint64_t task = first_task + slot, sample = task / s.chunks;
        const uint64_t start = task % s.chunks * s.block;
        T* state = shared ? reinterpret_cast<T*>(raw) : scratch + slot * work_size;
        T* one = state + s.lower; T* two = one + temporary; T* inc = two + temporary;
        T* wi = w + slot * s.block * s.width; T* vi = v + slot * s.block * s.d;
        for (uint64_t x = threadIdx.x; x < s.lower; x += blockDim.x) state[x] = T(x == 0);
        __syncthreads();
        for (uint64_t i = 0; i < s.block; ++i) {
            if (start + i >= s.n) {
                for (uint64_t x = threadIdx.x; x < s.width; x += blockDim.x) wi[i * s.width + x] = 0;
                for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x) vi[i * s.d + c] = 0;
                continue;
            }
            const T* point = path + (sample * (s.n + 1) + start + i) * s.d;
            for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x)
                vi[i * s.d + c] = inc[c] = point[s.d + c] - point[c];
            __syncthreads();
            for (uint64_t k = s.m; k > 1; --k) {
                if (k == 2 && k == s.m) {
                    for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x)
                        wi[i * s.width + c] = state[1 + c] + inc[c] / T(2);
                    __syncthreads();
                    continue;
                }
                for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x) one[c] = inc[c] / T(k);
                __syncthreads();
                T* a = one; T* b = two;
                const uint64_t stop = k == s.m ? k - 2 : k - 1;
                for (uint64_t level = 1; level < stop; ++level) {
                    const T reciprocal = T(1) / T(k - level);
                    for (uint64_t x = threadIdx.x; x < s.power[level + 1]; x += blockDim.x)
                        b[x] = (state[s.offset[level] + x / s.d] + a[x / s.d]) * inc[x % s.d] * reciprocal;
                    __syncthreads();
                    T* swap = a; a = b; b = swap;
                }
                if (k == s.m) {
                    for (uint64_t x = threadIdx.x; x < s.width; x += blockDim.x)
                        wi[i * s.width + x] = state[s.offset[k - 1] + x]
                            + (state[s.offset[k - 2] + x / s.d] + a[x / s.d]) * inc[x % s.d] / T(2);
                } else {
                    for (uint64_t x = threadIdx.x; x < s.power[k]; x += blockDim.x)
                        state[s.offset[k] + x] += (state[s.offset[k - 1] + x / s.d] + a[x / s.d]) * inc[x % s.d];
                }
                __syncthreads();
            }
            for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x) state[1 + c] += inc[c];
            __syncthreads();
        }
        for (uint64_t x = threadIdx.x; x < s.lower; x += blockDim.x) blocks[task * s.total + x] = state[x];
        __syncthreads();
    }
}

template<typename T>
__global__ void merge(const T* input, T* output, DeviceShape s, uint64_t batch, uint64_t count) {
    const uint64_t next = (count + 1) / 2, size = batch * next * s.total;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < size; x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t task = x / s.total, word = x % s.total, sample = task / next, pair = task % next;
        const T* left = input + (sample * count + 2 * pair) * s.total;
        if (2 * pair + 1 == count || !word) { output[x] = left[word]; continue; }
        const T* right = left + s.total;
        uint64_t k = 1;
        while (word >= s.offset[k + 1]) ++k;
        const uint64_t index = word - s.offset[k];
        T value = left[word] + right[word];
        for (uint64_t j = 1; j < k; ++j)
            value += left[s.offset[j] + index / s.power[k - j]] * right[s.offset[k - j] + index % s.power[k - j]];
        output[x] = value;
    }
}

template<typename T>
void compute(const T* path, T* out, const Shape& s, cudaStream_t stream) {
    const uint64_t tasks = mul(s.batch, s.chunks);
    if (!tasks) return;
    int device; check(cudaGetDevice(&device));
    thread_local std::unique_ptr<Resources> cache;
    if (!cache || cache->device != device) cache = std::make_unique<Resources>(device);
    auto& r = *cache;
    const uint64_t work_size = add(s.lower, add(mul(2, s.power[s.m - 2]), s.d));
    const bool shared = work_size * sizeof(T) <= 48 * 1024;
    const uint64_t worker = add(mul(s.block, add(s.width, s.d)), shared ? 0 : work_size);
    const uint64_t stored = mul(2, mul(tasks, s.total));
    size_t free, total; check(cudaMemGetInfo(&free, &total));
    const uint64_t budget = std::min<uint64_t>(512ULL << 20, (free + r.bytes) / 2) / sizeof(T);
    if (stored >= budget || worker > budget - stored) throw std::bad_alloc();
    const uint64_t slots = std::min(tasks, (budget - stored) / worker);
    r.reserve(mul(add(stored, mul(slots, worker)), sizeof(T)));
    T* first = static_cast<T*>(r.buffer); T* second = first + tasks * s.total;
    T* w = second + tasks * s.total; T* v = w + slots * s.block * s.width;
    T* scratch = v + slots * s.block * s.d;
    check(cublasSetStream(r.handle, stream));
    constexpr auto dtype = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
    constexpr auto compute_type = std::is_same_v<T, float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_64F_PEDANTIC;
    const T alpha = 1, beta = 0;
    const DeviceShape shape(s);
    for (uint64_t begin = 0; begin < tasks; begin += slots) {
        const uint64_t count = std::min(slots, tasks - begin);
        prepare<<<unsigned(std::min<uint64_t>(count, 65535)), 128, shared ? work_size * sizeof(T) : 0, stream>>>(
            path, first, w, v, scratch, shape, begin, count, shared);
        check(cudaGetLastError());
        check(cublasGemmStridedBatchedEx(r.handle, CUBLAS_OP_N, CUBLAS_OP_T, int(s.d), int(s.width), int(s.block),
            &alpha, v, dtype, int(s.d), s.block * s.d, w, dtype, int(s.width), s.block * s.width,
            &beta, first + begin * s.total + s.lower, dtype, int(s.d), s.total, int(count), compute_type, CUBLAS_GEMM_DEFAULT));
    }
    for (uint64_t count = s.chunks; count > 1; count = (count + 1) / 2) {
        const uint64_t size = s.batch * ((count + 1) / 2) * s.total;
        merge<<<unsigned(std::min<uint64_t>((size + 255) / 256, 65535)), 256, 0, stream>>>(first, second, shape, s.batch, count);
        check(cudaGetLastError()); std::swap(first, second);
    }
    check(cudaMemcpyAsync(out, first, s.batch * s.total * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    check(cudaStreamSynchronize(stream));
}

template<typename T>
int call(const T* path, T* out, uint64_t batch, uint64_t d, uint64_t points, uint64_t m, uint64_t block, void* stream) noexcept {
    try { compute(path, out, Shape(batch, d, points, m, block), static_cast<cudaStream_t>(stream)); return 0; }
    catch (const std::exception& e) { error = e.what(); return 1; }
}
}

ENDPOINT_API const char* endpoint_error() noexcept { return endpoint::error.c_str(); }
ENDPOINT_API int endpoint_f(const float* p, float* o, uint64_t b, uint64_t d, uint64_t n, uint64_t m, uint64_t block, void* stream) noexcept {
    return endpoint::call(p, o, b, d, n, m, block, stream);
}
ENDPOINT_API int endpoint_d(const double* p, double* o, uint64_t b, uint64_t d, uint64_t n, uint64_t m, uint64_t block, void* stream) noexcept {
    return endpoint::call(p, o, b, d, n, m, block, stream);
}
