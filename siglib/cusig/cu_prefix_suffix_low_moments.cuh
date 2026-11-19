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

// Exclusive scan of one tile of segments, with a carry between tiles.
template<typename T>
__device__ T scan_tile(T x, T& carry, unsigned lane) {
    for (unsigned shift = 1; shift < 32; shift *= 2) {
        const T previous = __shfl_up_sync(0xffffffff, x, shift);
        if (lane >= shift) x += previous;
    }
    const T previous = __shfl_up_sync(0xffffffff, x, 1);
    const T before = carry + (lane ? previous : T(0));
    carry += __shfl_sync(0xffffffff, x, 31);
    return before;
}

template<bool Quintic, typename T>
__global__ void small_factors(const T* path, T* left, T* right, T* output, uint64_t batch,
    unsigned n, unsigned d, uint64_t stride, uint64_t sig_stride, bool scalar) {
    const unsigned lane = threadIdx.x % 32, pairs = d * d, rows = pairs * d;
    const unsigned cols = d + (Quintic ? pairs : 0);
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; task < batch * pairs;
        task += uint64_t(gridDim.x) * blockDim.x / 32) {
        const uint64_t sample = task / pairs, base = sample * stride;
        const unsigned xy = unsigned(task % pairs), x = xy / d, y = xy % d;
        const T* u = path + sample * uint64_t(n + 1) * d;
        const T first = u[x], last = u[n * d + y];
        T prefix = 0, suffix = 0;
        for (unsigned t = 0; t < n; t += 32) {
            const unsigned i = t + lane, j = n - 1 - i;
            T vx = 0, vy = 0, wx = 0, wy = 0, mid = 0, qmid = 0;
            if (i < n) {
                vx = u[(i + 1) * d + x] - u[i * d + x]; vy = u[(i + 1) * d + y] - u[i * d + y];
                mid = (u[i * d + x] - first) + vx / 2;
                wx = u[(j + 1) * d + x] - u[j * d + x]; wy = u[(j + 1) * d + y] - u[j * d + y];
                qmid = (last - u[(j + 1) * d + y]) + wy / 2;
            }
            const T h = scan_tile(mid * vy, prefix, lane) + (mid / 2 - vx / 12) * vy;
            T s = 0;
            if constexpr (Quintic) s = scan_tile(wx * qmid, suffix, lane) + wx * (qmid / 2 - wy / 12);
            if (i < n) {
                for (unsigned z = 0; z < d; ++z) {
                    const T v = u[(i + 1) * d + z] - u[i * d + z];
                    const unsigned row = xy * d + z;
                    left[base + row * n + i] = h * v;
                    left[base + (rows + row) * n + i] = ((-mid / 12) * vy) * v;
                    if constexpr (Quintic) left[base + (2 * rows + row) * n + i] = ((vx / 720) * vy) * v;
                }
                if (!x) {
                    right[base + y * n + j] = qmid; right[base + (cols + y) * n + j] = wy;
                }
                if constexpr (Quintic) {
                    right[base + (d + xy) * n + j] = s;
                    right[base + (cols + d + xy) * n + j] = wx * qmid;
                    right[base + (2 * cols + d + xy) * n + j] = wx * wy;
                }
            }
        }
        if (!lane) {
            T* result = output + sample * sig_stride;
            result[scalar + d + xy] = prefix;
            if (!x) {
                result[scalar + y] = last - u[y];
                if (!y && scalar) result[0] = 1;
            }
        }
    }
}

template<bool Quintic, typename T>
__global__ void small_products(const T* left, const T* right, T* output, uint64_t batch,
    unsigned n, unsigned d, uint64_t stride, uint64_t sig_stride, bool scalar) {
    const unsigned lane = threadIdx.x % 32, rows = d * d * d, cols = d + (Quintic ? d * d : 0);
    const unsigned size = rows * (1 + cols);
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; task < batch * size;
        task += uint64_t(gridDim.x) * blockDim.x / 32) {
        const uint64_t sample = task / size, base = sample * stride;
        const unsigned word = unsigned(task % size);
        unsigned row = word, col = 0;
        if (word >= rows) { row = (word - rows) / d; col = (word - rows) % d; }
        const bool fifth = Quintic && word >= rows * (1 + d);
        if (fifth) { row = (word - rows * (1 + d)) / (d * d); col = d + (word - rows * (1 + d)) % (d * d); }
        const T* a = left + base + row * n; const T* b = right + base + col * n;
        T sum = 0;
        for (unsigned i = lane; i < n; i += 32) {
            if (word < rows) sum += a[i];
            else {
                sum += a[i] * b[i] + a[rows * n + i] * b[cols * n + i];
                if (fifth) sum += a[2 * rows * n + i] * b[2 * cols * n + i];
            }
        }
        for (unsigned s = 16; s; s /= 2) sum += __shfl_down_sync(0xffffffff, sum, s);
        if (!lane) output[sample * sig_stride + scalar + d + d * d + word] = sum;
    }
}

template<bool Quintic, typename T>
void small_forward(const T* path, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream) {
    const uint64_t rows = l.power[3], cols = l.d + (Quintic ? l.power[2] : 0), terms = Quintic ? 3 : 2;
    const uint64_t stride = (n * terms * (rows + cols) + 63) / 64 * 64, pitch = l.offset[l.m + 1] - !scalar;
    const uint64_t paths = std::clamp<uint64_t>((256ULL << 20) / (stride * sizeof(T)), 1, 64);
    with_factor_workspace<T, false>(l, device, batch, paths, stride, 0, 0, false, "prefix-suffix moments", stream, false,
        [&](cublasHandle_t, Fp64Emulation*, uint64_t capacity, T* left, uint64_t*, uint64_t*) {
        T* right = left + n * terms * rows;
        for (uint64_t i = 0; i < batch; i += capacity) {
            const uint64_t count = std::min(batch - i, capacity);
            small_factors<Quintic><<<make_cuda_1d_grid(count * l.d * l.d, 2), 64, 0, stream>>>(
                path + i * (n + 1) * l.d, left, right, out + i * pitch, count, unsigned(n), unsigned(l.d), stride, pitch, scalar);
            CUDA_CHECK(cudaGetLastError());
            small_products<Quintic><<<make_cuda_1d_grid(count * rows * (1 + cols), 4), 128, 0, stream>>>(
                left, right, out + i * pitch, count, unsigned(n), unsigned(l.d), stride, pitch, scalar);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}
