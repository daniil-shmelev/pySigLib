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

namespace gpu_symmetric {

template<typename T>
__global__ void check_seeds(const T* seeds, uint64_t count, T limit, int* unsafe) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count;
        x += uint64_t(gridDim.x) * blockDim.x) {
        if (!(abs(seeds[x]) <= limit)) { atomicExch(unsafe, 1); return; }
    }
}

struct Maps {
    SymmetricPlan plan;
    std::vector<unsigned> indices;
    std::vector<uint64_t> starts;
    Maps(const Layout& l, uint64_t n, uint64_t rows, uint64_t itemsize) : plan(l, n, rows, itemsize) {
        for (const auto& map : plan.powers) {
            starts.push_back(indices.size());
            indices.insert(indices.end(), map.representative.begin(), map.representative.end());
            indices.insert(indices.end(), map.group.begin(), map.group.end());
            indices.insert(indices.end(), map.offsets.begin(), map.offsets.end());
            indices.insert(indices.end(), map.members.begin(), map.members.end());
        }
    }
};

template<bool Scatter, typename T>
__global__ void pack(T* left, T* compressed, const unsigned* rep, const unsigned* group, uint64_t batch,
    uint64_t n, uint64_t p, uint64_t tail, uint64_t unique, uint64_t heads, uint64_t stride) {
    const uint64_t width = heads * (Scatter ? tail : unique);
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * n * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / (n * width), row = x / width % n, word = x % width;
        T* full = left + sample * stride + row * p;
        T* packed = compressed + sample * stride + row * heads * unique;
        if constexpr (Scatter) {
            const unsigned g = group[word % tail];
            full[word] = rep[g] == word % tail ? packed[word / tail * unique + g] : T(0);
        } else packed[word] = full[word / unique * tail + rep[word % unique]];
    }
}

template<bool Fold, typename T>
__global__ void map_level(T* full, T* compressed, const unsigned* group, const unsigned* offsets,
    const unsigned* members, uint64_t batch, uint64_t heads, uint64_t q, uint64_t tail,
    uint64_t unique, uint64_t full_stride, uint64_t stride) {
    const uint64_t width = heads * (Fold ? unique : tail) * q;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / width, idx = x % width, row = idx / q, col = idx % q;
        T* ordered = full + sample * full_stride;
        T* packed = compressed + sample * stride;
        if constexpr (Fold) {
            const unsigned g = unsigned(row % unique);
            const T* src = ordered + row / unique * tail * q + col;
            T sum = 0;
            for (unsigned t = offsets[g]; t < offsets[g + 1]; ++t) sum += src[uint64_t(members[t]) * q];
            packed[idx] = sum;
        } else ordered[idx] += packed[(row / tail * unique + group[row % tail]) * q + col];
    }
}

enum class Product { forward, left_adjoint, right_adjoint };

template<Product product, typename T>
void gemm(const Maps& maps, const unsigned* indices, CudaMatrices<T>& matrices, cudaStream_t stream, T* scratch,
    T* left, T* right, T* level, uint64_t a, uint64_t p, uint64_t q, uint64_t pitch, uint64_t base,
    uint64_t rows, uint64_t stride, uint64_t sig_stride) {
    constexpr bool forward = product == Product::forward;
    constexpr bool left_adjoint = product == Product::left_adjoint;
    const uint64_t count = matrices.batch;
    for (uint64_t j = 0; j < a; ++j) {
        const uint64_t start = (j + a - base % a) % a;
        if (start >= rows) continue;
        const uint64_t n = (rows - 1 - start) / a + 1;
        T* lrow = left + start * p;
        T* rrow = right + start * pitch;
        if (!j) {
            if constexpr (forward)
                matrices.gemm({lrow, p, n, 1, a * p, stride}, {rrow, n, q, a * pitch, 1, stride},
                    {level, p, q, q, 1, sig_stride}, !base);
            else if constexpr (left_adjoint)
                matrices.gemm({rrow, n, q, a * pitch, 1, stride}, {level, q, p, 1, q, sig_stride},
                    {lrow, n, p, a * p, 1, stride}, true);
            else matrices.gemm({lrow, n, p, a * p, 1, stride}, {level, p, q, q, 1, sig_stride},
                    {rrow, n, q, a * pitch, 1, stride}, true);
            continue;
        }
        const auto& map = maps.plan.powers[j - 1];
        const uint64_t unique = map.representative.size(), tail = map.size, heads = p / tail;
        const unsigned* rep = indices + maps.starts[j - 1];
        const unsigned* group = rep + unique;
        const unsigned* offsets = group + tail;
        const unsigned* members = offsets + unique + 1;
        const uint64_t tile = std::max<uint64_t>(1, maps.plan.scratch / ((n + q) * unique));
        for (uint64_t head = 0; head < heads; head += tile) {
            const uint64_t h = std::min(tile, heads - head), width = h * unique;
            T* packed = scratch;
            T* reduced = packed + n * width;
            T* full = level + head * tail * q;
            T* ordered = lrow + head * tail;
            auto map_left = [&] {
                const uint64_t size = n * h * (left_adjoint ? tail : unique);
                pack<left_adjoint><<<make_cuda_1d_grid(mul(count, size), 256), 256, 0, stream>>>(
                    ordered, packed, rep, group, count, n, a * p, tail, unique, h, stride);
                CUDA_CHECK(cudaGetLastError());
            };
            auto map_output = [&] {
                const uint64_t size = h * (forward ? tail : unique) * q;
                map_level<!forward><<<make_cuda_1d_grid(mul(count, size), 256), 256, 0, stream>>>(
                    full, reduced, group, offsets, members, count, h, q, tail, unique, sig_stride, stride);
                CUDA_CHECK(cudaGetLastError());
            };
            if constexpr (!left_adjoint) map_left();
            if constexpr (!forward) map_output();
            if constexpr (forward) {
                matrices.gemm({packed, width, n, 1, width, stride}, {rrow, n, q, a * pitch, 1, stride},
                    {reduced, width, q, q, 1, stride}, true);
                map_output();
            } else if constexpr (left_adjoint) {
                matrices.gemm({rrow, n, q, a * pitch, 1, stride}, {reduced, q, width, 1, q, stride},
                    {packed, n, width, width, 1, stride}, true);
                map_left();
            } else matrices.gemm({packed, n, width, width, 1, stride}, {reduced, width, q, q, 1, stride},
                {rrow, n, q, a * pitch, 1, stride}, head == 0);
        }
    }
}
} // namespace gpu_symmetric
