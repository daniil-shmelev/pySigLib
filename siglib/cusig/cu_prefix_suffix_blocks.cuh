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

// Included in gpu_legendre after the single-block factorisation.
template<typename T>
__global__ void pack_block_paths(const T* input, T* packed, uint64_t batch, uint64_t n,
    uint64_t dimension, uint64_t d, unsigned blocks, unsigned tile, bool ta, bool ll, T end_time) {
    const uint64_t length = (ll ? n / 2 : n) + 1, width = (tile + 1ULL) * d;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * blocks * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t leaf = x / width, b = leaf % blocks, c = x % d;
        const uint64_t start = b * (n / blocks) + min(b, n % blocks);
        const uint64_t count = n / blocks + (b < n % blocks);
        const uint64_t i = start + min(x % width / d, count);
        packed[x] = point_value(input + leaf / blocks * length * dimension, i, c, dimension, n + 1, ta, ll, end_time);
    }
}

template<typename T>
__global__ void join_blocks(const T* children, T* parents, uint64_t pairs, unsigned width,
    Degrees l, bool scalar) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < pairs * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const unsigned index = unsigned(x % width), pos = index + !scalar, h = level(pos, l);
        const T* left = children + x / width * 2 * width;
        const T* right = left + width;
        T sum = left[index] + right[index];
        const unsigned w = pos - l.offset[h];
        for (unsigned k = 1; k < h; ++k) {
            const unsigned tail = l.power[h - k];
            sum += left[l.offset[k] - !scalar + w / tail] * right[l.offset[h - k] - !scalar + w % tail];
        }
        parents[x] = h ? sum : T(1);
    }
}

template<unsigned Lanes, typename T>
__global__ void join_block_gradients(const T* children, const T* seed, T* out, uint64_t pairs, unsigned full,
    unsigned width, unsigned first, unsigned end, unsigned degree, Degrees l, bool scalar) {
    const unsigned lane = threadIdx.x % Lanes, count = end - first;
    __shared__ T partial[2][8];
    for (uint64_t row = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Lanes; row < pairs * count;
        row += uint64_t(gridDim.x) * blockDim.x / Lanes) {
        const uint64_t pair = row / count;
        const unsigned index = first + unsigned(row % count), pos = index + !scalar, h = level(pos, l);
        const T* left = children + pair * 2 * width;
        const T* right = left + width;
        const T* grad = seed + pair * full;
        const unsigned word = pos - l.offset[h];
        T a = 0, b = 0;
        if (h) for (unsigned k = 1; k <= degree - h; ++k) {
            const unsigned other = l.offset[k] - !scalar, size = l.power[k];
            const T* g = grad + l.offset[k + h] - !scalar;
            for (unsigned w = lane; w < size; w += Lanes) {
                a += g[word * size + w] * right[other + w];
                b += g[w * l.power[h] + word] * left[other + w];
            }
        }
        if constexpr (Lanes > 1) for (unsigned k = 16; k; k /= 2) {
            a += __shfl_down_sync(0xffffffff, a, k);
            b += __shfl_down_sync(0xffffffff, b, k);
        }
        if constexpr (Lanes > 32) {
            if (!(lane % 32)) { partial[0][lane / 32] = a; partial[1][lane / 32] = b; }
            __syncthreads();
            if (!lane) for (unsigned w = 1; w < Lanes / 32; ++w) { a += partial[0][w]; b += partial[1][w]; }
            __syncthreads();
        }
        if (!lane) {
            const T direct = h ? grad[index] : T(0);
            out[2 * pair * full + index] = direct + a;
            out[(2 * pair + 1) * full + index] = direct + b;
        }
    }
}

template<typename T>
__global__ void merge_block_points(const T* vertices, T* out, uint64_t batch, uint64_t n,
    uint64_t dimension, uint64_t d, unsigned blocks, unsigned tile, bool ll) {
    const uint64_t length = (ll ? n / 2 : n) + 1;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * length * dimension;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t t = x / dimension % length, c = x % dimension, short_count = n / blocks, extra = n % blocks;
        const T* grad = vertices + x / dimension / length * blocks * (tile + 1ULL) * d;
        auto point = [&](uint64_t i, uint64_t channel) {
            const uint64_t wide = (short_count + 1) * extra;
            const uint64_t b = i == n ? blocks - 1 : i < wide ? i / (short_count + 1) : extra + (i - wide) / short_count;
            const uint64_t start = b * short_count + min(b, extra);
            const T* block = grad + b * (tile + 1ULL) * d + channel;
            T value = block[(i - start) * d];
            if (b && i == start) {
                value += block[-static_cast<int64_t>(d)];
                if (extra && b > extra) value += block[-2 * static_cast<int64_t>(d)];
            }
            if (i == n && extra) value += block[tile * d];
            return value;
        };
        if (!ll) out[x] = point(t, c);
        else {
            T value = point(2 * t, c) + point(2 * t, dimension + c);
            if (t) value += point(2 * t - 1, dimension + c);
            if (t + 1 < length) value += point(2 * t + 1, c);
            out[x] = value;
        }
    }
}

template<bool Backward, typename T>
bool blocked(const T* path, const T* seed, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream, uint64_t dimension, bool ta, bool ll, T end_time, uint64_t work) {
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX);
    if (l.m < 6 || n < 1024 || budget <= (4ULL << 20)) return false;
    const uint64_t available = budget - (4ULL << 20);
    const uint64_t target = std::min<uint64_t>(48ULL << 20, available / 3) / sizeof(T);
    unsigned blocks = 2, height = 1;
    while (work / blocks > target && blocks < 256 && n / blocks >= 256) {
        blocks *= 2;
        ++height;
    }
    const unsigned full = unsigned(l.offset[l.m + 1] - !scalar), width = unsigned(l.offset[l.m + 1 - Backward] - !scalar);
    uint64_t tile, points, elements;
    for (;;) {
        tile = (n - 1) / blocks + 1;
        if (tile > UINT_MAX - 32 || work / blocks > (180ULL << 20) / sizeof(T)) return false;
        points = mul(mul(blocks, tile + 1), l.d);
        const uint64_t states = mul(Backward ? 2 * blocks : blocks + blocks / 2, width);
        const uint64_t gradients = Backward ? mul(blocks + blocks / 2, full) : 0;
        elements = add(mul(1 + Backward, points), add(states, gradients));
        if (elements <= std::min<uint64_t>(192ULL << 20, available / 2) / sizeof(T)) break;
        if (blocks == 2) return false;
        blocks /= 2;
        --height;
    }
    const Layout child(l.d, l.m - Backward);
    Degrees levels{};
    for (unsigned k = 0; k <= l.m; ++k) levels.power[k] = unsigned(l.power[k]);
    for (unsigned k = 0; k <= l.m + 1; ++k) levels.offset[k] = unsigned(l.offset[k]);
    const uint64_t path_stride = ((ll ? n / 2 : n) + 1) * dimension;
    const unsigned lead_channels = ll ? unsigned(dimension) : 0;
    const uint64_t bytes = mul(elements, sizeof(T));
    const uint64_t paths = std::min(batch, std::max<uint64_t>(1, std::min<uint64_t>(128ULL << 20, available / 2) / bytes));
    CudaBatchWorkspace<T> storage(paths, elements, "CUDA prefix-suffix blocks", (4ULL << 20) + work / blocks * sizeof(T));
    const uint64_t reserved = mul(storage.capacity(), bytes);
    for (uint64_t base = 0; base < batch; base += storage.capacity()) {
        const uint64_t count = std::min(batch - base, storage.capacity()), leaves = count * blocks;
        T* packed = storage.get();
        T* states[9]{packed + count * points};
        T* root = out + base * (Backward ? path_stride : full);
        pack_block_paths<<<make_cuda_1d_grid(count * points, 256), 256, 0, stream>>>(path + base * path_stride,
            packed, count, n, dimension, l.d, blocks, unsigned(tile), ta, ll, end_time);
        CUDA_CHECK(cudaGetLastError());
        if (!compute<false>(packed, static_cast<const T*>(nullptr), states[0], child, device, leaves, tile,
            scalar, stream, l.d, false, false, T(1), reserved, lead_channels)) return false;
        for (unsigned h = 1; h <= height - unsigned(Backward); ++h) {
            const uint64_t pairs = leaves >> h;
            if (h == height) states[h] = root;
            else if constexpr (Backward) states[h] = states[h - 1] + 2 * pairs * width;
            else states[h] = states[0] + (h % 2) * leaves * width;
            join_blocks<<<make_cuda_1d_grid(pairs * width, 256), 256, 0, stream>>>(states[h - 1], states[h], pairs, width, levels, scalar);
            CUDA_CHECK(cudaGetLastError());
        }
        if constexpr (Backward) {
            T* g[2]{states[0] + 2 * leaves * width, nullptr};
            g[1] = g[0] + leaves / 2 * full;
            const T* current = seed + base * full;
            unsigned k = unsigned(l.m);
            while (k > 1 && l.power[l.m - k + 1] < 32) --k;
            const unsigned cut = unsigned(l.offset[k] - !scalar);
            for (unsigned h = height; h; --h) {
                const uint64_t pairs = leaves >> h;
                T* next = g[h % 2];
                if (cut) join_block_gradients<256><<<make_cuda_1d_grid(pairs * cut, 1), 256, 0, stream>>>(states[h - 1],
                    current, next, pairs, full, width, 0, cut, unsigned(l.m), levels, scalar);
                join_block_gradients<1><<<make_cuda_1d_grid(pairs * (full - cut), 256), 256, 0, stream>>>(states[h - 1],
                    current, next, pairs, full, width, cut, full, unsigned(l.m), levels, scalar);
                CUDA_CHECK(cudaGetLastError());
                current = next;
            }
            T* vertices = g[1] + leaves * full;
            if (!compute<true>(packed, current, vertices, l, device, leaves, tile, scalar, stream, l.d,
                false, false, T(1), reserved, lead_channels)) return false;
            merge_block_points<<<make_cuda_1d_grid(count * path_stride, 256), 256, 0, stream>>>(vertices, root,
                count, n, dimension, l.d, blocks, unsigned(tile), ll);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}
