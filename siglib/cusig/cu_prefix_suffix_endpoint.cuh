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

namespace gpu_endpoint {
using prefix_suffix_endpoint::Shape;
struct DeviceShape {
    uint64_t d, m, n, block, chunks, width, lower, total;
    uint64_t power[33], offset[33];
    explicit DeviceShape(const Shape& s) : d(s.d), m(s.m), n(s.n), block(s.block), chunks(s.chunks),
        width(s.width), lower(s.lower), total(s.total) {
        std::copy(s.power.begin(), s.power.end(), power);
        std::copy(s.offset.begin(), s.offset.end(), offset);
    }
};

template<typename T>
__global__ void prepare(const T* path, T* blocks, T* w, T* v, T* scratch, DeviceShape s,
    uint64_t first_task, uint64_t count, bool shared, uint64_t dimension, uint64_t length, bool ta, bool ll, T end_time) {
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
            const T* point = path + sample * length * dimension;
            for (uint64_t c = threadIdx.x; c < s.d; c += blockDim.x)
                vi[i * s.d + c] = inc[c] = point_value(point, start + i + 1, c, dimension, s.n + 1, ta, ll, end_time)
                    - point_value(point, start + i, c, dimension, s.n + 1, ta, ll, end_time);
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
void compute(const T* path, T* out, const Shape& s, cudaStream_t stream, uint64_t dimension, uint64_t length, bool ta, bool ll, T end_time, bool scalar) {
    const uint64_t tasks = mul(s.batch, s.chunks);
    if (!tasks) return;
    int device; CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard lock(cuda_path_workspaces().mutex);
    Resources& r = Resources::get(device);
    const uint64_t reserved = r.reserved();
    const uint64_t work_size = add(s.lower, add(mul(2, s.power[s.m - 2]), s.d));
    const bool shared = work_size * sizeof(T) <= 48 * 1024;
    const uint64_t worker = add(mul(s.block, add(s.width, s.d)), shared ? 0 : work_size);
    const uint64_t stored = mul(2, mul(tasks, s.total));
    size_t free, total; CUDA_CHECK(cudaMemGetInfo(&free, &total));
    const uint64_t budget = std::min<uint64_t>(option("PYSIGLIB_PS_WORKSPACE_BYTES", 512ULL << 20), (free + reserved) / 2) / sizeof(T);
    if (stored >= budget || worker > budget - stored) throw std::bad_alloc();
    const uint64_t slots = std::min(tasks, (budget - stored) / worker);
    const uint64_t requested = mul(add(stored, mul(slots, worker)), sizeof(T));
    r.reserve(1, requested, UINT64_MAX, "endpoint signature factors");
    T* first = reinterpret_cast<T*>(r.storage->get()); T* second = first + tasks * s.total;
    T* w = second + tasks * s.total; T* v = w + slots * s.block * s.width;
    T* scratch = v + slots * s.block * s.d;
    blas_check(cublasSetStream(r.handle, stream));
    blas_check(cublasSetWorkspace(r.handle, r.blas_workspace.get(), 4ULL << 20));
    constexpr auto dtype = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
    constexpr auto compute_type = std::is_same_v<T, float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_64F_PEDANTIC;
    const T alpha = 1, beta = 0;
    const DeviceShape shape(s);
    try {
        for (uint64_t begin = 0; begin < tasks; begin += slots) {
            const uint64_t count = std::min(slots, tasks - begin);
            prepare<<<unsigned(std::min<uint64_t>(count, 65535)), 128, shared ? work_size * sizeof(T) : 0, stream>>>(
                path, first, w, v, scratch, shape, begin, count, shared, dimension, length, ta, ll, end_time);
            CUDA_CHECK(cudaGetLastError());
            blas_check(cublasGemmStridedBatchedEx_64(r.handle, CUBLAS_OP_N, CUBLAS_OP_T, s.d, s.width, s.block,
                &alpha, v, dtype, s.d, s.block * s.d, w, dtype, s.width, s.block * s.width,
                &beta, first + begin * s.total + s.lower, dtype, s.d, s.total, count, compute_type, CUBLAS_GEMM_DEFAULT));
        }
        for (uint64_t count = s.chunks; count > 1; count = (count + 1) / 2) {
            const uint64_t size = s.batch * ((count + 1) / 2) * s.total;
            merge<<<unsigned(std::min<uint64_t>((size + 255) / 256, 65535)), 256, 0, stream>>>(first, second, shape, s.batch, count);
            CUDA_CHECK(cudaGetLastError()); std::swap(first, second);
        }
        const uint64_t skip = scalar ? 0 : 1;
        CUDA_CHECK(cudaMemcpy2DAsync(out, (s.total - skip) * sizeof(T), first + skip, s.total * sizeof(T),
            (s.total - skip) * sizeof(T), s.batch, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } catch (...) { (void)cudaStreamSynchronize(stream); throw; }
}

} // namespace gpu_endpoint
