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

namespace gpu_midpoint {

template<typename T>
bool fits(const Layout& l, uint64_t n) {
    const uint64_t width = l.m == 2 ? l.d : l.d + l.power[2];
    return n <= (32ULL << 20) / sizeof(T) / width && l.power[l.m] <= (16ULL << 20) / sizeof(T);
}

template<typename T>
__global__ void direct(const T* path, T* out, uint64_t batch, uint64_t n, unsigned d,
    uint64_t sig_stride, bool scalar) {
    for (uint64_t task = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32; task < batch * sig_stride;
        task += uint64_t(gridDim.x) * 4) {
        const uint64_t sample = task / sig_stride, word = task % sig_stride;
        const unsigned lane = threadIdx.x % 32;
        const T* start = path + sample * (n + 1) * d;
        if (word < scalar + d) {
            if (!lane) out[task] = word < scalar ? T(1) : start[n * d + word - scalar] - start[word - scalar];
            continue;
        }
        const unsigned index = unsigned(word - scalar - d), square = d * d;
        const bool cubic = index >= square;
        const unsigned a = cubic ? (index - square) / square : index / d;
        const unsigned b = cubic ? (index - square) / d % d : index % d;
        const unsigned c = cubic ? (index - square) % d : 0;
        T sum = 0;
        for (uint64_t t = lane; t < n; t += 32) {
            const T* x = start + t * d;
            const T va = x[d + a] - x[a], vb = x[d + b] - x[b];
            T factor = (x[a] - start[a]) + T(0.5) * va;
            if (cubic) {
                const T vc = x[d + c] - x[c];
                const T q = (start[n * d + c] - x[d + c]) + T(0.5) * vc;
                factor = factor * q - (va / T(12)) * vc;
            }
            sum += factor * vb;
        }
        sum = cuda_detail::warp_sum(sum);
        if (!lane) out[task] = sum;
    }
}

template<unsigned D, typename T>
__global__ void cubic_tiles(const T* path, T* partial, uint64_t batch, uint64_t n, unsigned chunks, uint64_t stride) {
    constexpr unsigned width = D + D * D, columns = D <= 8 ? 8 : 16, row_step = 128 / columns;
    constexpr unsigned row_tiles = (D * D + 63) / 64, col_tiles = (D + columns - 1) / columns;
    __shared__ T v[8][D], p[8][D], q[8][D], features[8][65];
    const unsigned x = threadIdx.x % columns, y = threadIdx.x / columns;
    const uint64_t block_length = (n + chunks - 1) / chunks;
    for (uint64_t task = blockIdx.x; task < batch * chunks * row_tiles * col_tiles; task += gridDim.x) {
        const uint64_t group = task / (row_tiles * col_tiles), sample = group / chunks;
        const unsigned chunk = unsigned(group % chunks), row0 = unsigned(task / col_tiles % row_tiles) * 64;
        const unsigned b = unsigned(task % col_tiles) * columns + x;
        const T* start = path + sample * (n + 1) * D;
        T sum[64 / row_step]{}, second[(D + row_step - 1) / row_step]{};
        const uint64_t end = min(n, (chunk + 1) * block_length);
        for (uint64_t first = chunk * block_length; first < end; first += 8) {
            for (unsigned z = threadIdx.x; z < 8 * D; z += 128) {
                const unsigned t = z / D, c = z % D;
                T inc = 0, left = 0, right = 0;
                if (first + t < end) {
                    const T* current = start + (first + t) * D;
                    inc = current[D + c] - current[c];
                    left = (current[c] - start[c]) + T(0.5) * inc;
                    right = (start[n * D + c] - current[D + c]) + T(0.5) * inc;
                }
                v[t][c] = inc; p[t][c] = left; q[t][c] = right;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned z = threadIdx.x; z < 8 * 64; z += 128) {
                const unsigned t = z / 64, row = row0 + z % 64;
                T factor = 0;
                if (row < D * D) {
                    const unsigned a = row / D, c = row % D;
                    factor = p[t][a] * q[t][c] - (v[t][a] / T(12)) * v[t][c];
                }
                features[t][z % 64] = factor;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned t = 0; t < 8; ++t) {
                const T inc = b < D ? v[t][b] : T(0);
                #pragma unroll
                for (unsigned r = 0; r < 64 / row_step; ++r) sum[r] += features[t][y + row_step * r] * inc;
                if (!row0) {
                    #pragma unroll
                    for (unsigned r = 0; r < (D + row_step - 1) / row_step; ++r)
                        if (y + row_step * r < D) second[r] += p[t][y + row_step * r] * inc;
                }
            }
            __syncthreads();
        }
        T* result = partial + sample * stride + uint64_t(chunk) * width * D;
        #pragma unroll
        for (unsigned r = 0; r < 64 / row_step; ++r) {
            const unsigned row = row0 + y + row_step * r;
            if (row < D * D && b < D) result[(D + row) * D + b] = sum[r];
        }
        if (!row0 && b < D) {
            #pragma unroll
            for (unsigned r = 0; r < (D + row_step - 1) / row_step; ++r)
                if (y + row_step * r < D) result[(y + row_step * r) * D + b] = second[r];
        }
    }
}

template<typename T>
__global__ void cubic_reduce(const T* path, const T* partial, T* output, uint64_t batch, uint64_t n,
    unsigned d, unsigned chunks, uint64_t stride, uint64_t sig_stride, bool scalar) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * sig_stride;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / sig_stride, word = x % sig_stride;
        const T* start = path + sample * (n + 1) * d;
        if (word < scalar + d) {
            output[x] = word < scalar ? T(1) : start[n * d + word - scalar] - start[word - scalar];
            continue;
        }
        const unsigned square = d * d, index = unsigned(word - scalar - d);
        unsigned column = index;
        if (index >= square) {
            const unsigned w = index - square;
            column = square + (w / square * d + w % d) * d + w / d % d;
        }
        const T* src = partial + sample * stride + column;
        T sum = 0;
        for (unsigned t = 0; t < chunks; ++t) sum += src[uint64_t(t) * square * (d + 1)];
        output[x] = sum;
    }
}

template<typename T>
void cubic_forward(const T* path, T* out, const Layout& l, int device, uint64_t batch, uint64_t n,
    bool scalar, cudaStream_t stream) {
    const unsigned chunks = unsigned(std::min<uint64_t>(128, (n - 1) / 64 + 1)), d = unsigned(l.d);
    const uint64_t width = d + d * d, stride = (mul(chunks, width * d) + 63) / 64 * 64;
    const uint64_t sig_stride = l.offset.back() - !scalar;
    with_factor_workspace<T, false>(l, device, batch, 64, stride, 0,
        0, false, "prefix-suffix", stream, false, [&](cublasHandle_t, Fp64Emulation*, uint64_t capacity,
            T* partial, uint64_t*, uint64_t*) {
        for (uint64_t first = 0; first < batch; first += capacity) {
            const uint64_t count = std::min(capacity, batch - first);
            const T* input = path + first * (n + 1) * d;
            const unsigned columns = d <= 8 ? 8 : 16;
            const unsigned grid = make_cuda_1d_grid(count * chunks * ((d * d + 63) / 64) * ((d + columns - 1) / columns), 1);
            auto run = [&]<unsigned D>() { cubic_tiles<D><<<grid, 128, 0, stream>>>(input, partial, count, n, chunks, stride); };
            switch (d) {
            case 8: run.template operator()<8>(); break;
            case 12: run.template operator()<12>(); break;
            case 16: run.template operator()<16>(); break;
            case 20: run.template operator()<20>(); break;
            case 24: run.template operator()<24>(); break;
            }
            CUDA_CHECK(cudaGetLastError());
            cubic_reduce<<<make_cuda_1d_grid(mul(count, sig_stride), 128), 128, 0, stream>>>(input, partial,
                out + first * sig_stride, count, n, d, chunks, stride, sig_stride, scalar);
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

template<typename T>
__global__ void factors(const T* path, T* features, T* increments, uint64_t count,
    uint64_t n, uint64_t d, uint64_t width, uint64_t stride) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count * n * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = x / width, sample = row / n, i = row % n, word = x % width;
        const uint64_t a = word < d ? word : (word - d) / d;
        const T* start = path + sample * (n + 1) * d;
        const T* current = start + i * d;
        const T va = current[d + a] - current[a];
        const T p = (current[a] - start[a]) + T(0.5) * va;
        T value = p;
        if (word < d) increments[sample * stride + i * d + a] = va;
        else {
            const uint64_t c = (word - d) % d;
            const T vc = current[d + c] - current[c];
            const T q = (start[n * d + c] - current[d + c]) + T(0.5) * vc;
            value = p * q - (va / T(12)) * vc;
        }
        features[sample * stride + i * width + word] = value;
    }
}

template<bool Backward, typename T>
__global__ void transpose_levels(const T* path, const T* input, T* output, uint64_t count, uint64_t n,
    uint64_t d, uint64_t matrix_size, uint64_t sig_stride, uint64_t stride, bool scalar) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count * matrix_size;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / matrix_size, word = x % matrix_size, square = d * d;
        uint64_t index = word;
        if (word >= square) {
            const uint64_t third = word - square;
            index = square + third / square * square + third % d * d + third / d % d;
        }
        const uint64_t si = sample * sig_stride + scalar + d + word, mi = sample * stride + index;
        if constexpr (Backward) output[mi] = input[si];
        else {
            output[si] = input[mi];
            if (word < d) {
                const T* start = path + sample * (n + 1) * d;
                output[sample * sig_stride + scalar + word] = start[n * d + word] - start[word];
            }
            if (!word && scalar) output[sample * sig_stride] = 1;
        }
    }
}

template<typename T>
__global__ void factor_adjoints(const T* path, const T* df, T* dp, T* dq, T* dv,
    uint64_t count, uint64_t n, uint64_t d, uint64_t width, uint64_t stride) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count * n * d;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = x / d, sample = row / n, i = row % n, c = x % d;
        const uint64_t index = sample * stride + i * d + c;
        const T* g = df + sample * stride + i * width;
        T left = g[c], right = 0, increment = dv[index];
        if (width != d) {
            const T* start = path + sample * (n + 1) * d;
            const T* current = start + i * d;
            for (uint64_t a = 0; a < d; ++a) {
                const T v = current[d + a] - current[a];
                const T half = T(0.5) * v;
                const T gc = g[d + c * d + a], ga = g[d + a * d + c];
                left += gc * ((start[n * d + a] - current[d + a]) + half);
                right += ga * ((current[a] - start[a]) + half);
                const T scaled = v / T(12);
                increment -= gc * scaled + ga * scaled;
            }
        }
        dp[index] = left;
        dq[index] = right;
        dv[index] = increment;
    }
}

template<typename T>
__global__ void vertices(const T* dp, const T* dq, const T* dv, T* out,
    uint64_t count, uint64_t n, uint64_t d, uint64_t stride) {
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count * (n - 1) * d;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t sample = x / ((n - 1) * d), pos = x % ((n - 1) * d);
        const uint64_t prev = sample * stride + pos, next = prev + d;
        out[sample * (n + 1) * d + pos + d] = (dv[prev] - dv[next])
            + T(0.5) * (dp[prev] - dq[prev]) + T(0.5) * (dp[next] - dq[next]);
    }
}

template<typename T>
__global__ void endpoints(const T* dp, const T* dq, const T* dv, const T* deriv, T* out,
    uint64_t count, uint64_t n, uint64_t d, uint64_t sig_stride, uint64_t stride, bool scalar) {
    for (uint64_t task = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32; task < count * d;
        task += uint64_t(gridDim.x) * 4) {
        const uint64_t sample = task / d, c = task % d, first = sample * stride + c, last = first + (n - 1) * d;
        T left = 0, right = 0;
        for (uint64_t i = threadIdx.x % 32; i < n; i += 32) {
            left += dp[first + i * d];
            right += dq[first + i * d];
        }
        left = cuda_detail::warp_sum(left);
        right = cuda_detail::warp_sum(right);
        if (!(threadIdx.x % 32)) {
            const T g = deriv[sample * sig_stride + scalar + c];
            T* start = out + sample * (n + 1) * d + c;
            start[0] = T(0.5) * (dp[first] - dq[first]) - dv[first] - left - g;
            start[n * d] = T(0.5) * (dp[last] - dq[last]) + dv[last] + right + g;
        }
    }
}

template<bool Backward, typename T>
void compute(const T* path, const T* deriv, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream) {
    const uint64_t d = l.d, width = l.m == 2 ? d : add(d, l.power[2]);
    const uint64_t matrix_size = mul(width, d), sig_stride = l.offset.back() - !scalar;
    const uint64_t nd = mul(n, d), nf = mul(n, width), path_stride = mul(n + 1, d);
    const uint64_t splits = Backward ? 0 : gpu_split_k::chunks(n, width, d);
    const uint64_t used = add(add(matrix_size, nf), mul(Backward ? 3 : 1, nd));
    const uint64_t stride = mul(add(add(used, mul(splits, matrix_size)), 63) / 64, 64);
    with_factor_workspace<T>(l, device, batch, 64, stride, 0, 4ULL << 20, true,
        "prefix-suffix", stream, false, [&](cublasHandle_t handle, Fp64Emulation* emulation,
            uint64_t capacity, T* memory, uint64_t*, uint64_t*) {
        T* matrix = memory;
        T* f = matrix + matrix_size;
        T* v = f + nf;
        for (uint64_t first = 0; first < batch; first += capacity) {
            const uint64_t count = std::min(capacity, batch - first);
            const T* input = path + first * path_stride;
            factors<<<make_cuda_1d_grid(mul(count, nf), 128), 128, 0, stream>>>(input, f, v, count, n, d, width, stride);
            CUDA_CHECK(cudaGetLastError());
            CudaMatrices<T> matrices{handle, count, emulation};
            const unsigned grid = make_cuda_1d_grid(mul(count, matrix_size), 128);
            if constexpr (!Backward) {
                T* result = out + first * sig_stride;
                if (splits) gpu_split_k::gemm(f, v, v + nd, matrix, count, n, width, d, d, splits, stride, stride, true, stream);
                else matrices.gemm({f, width, n, 1, width, stride}, {v, n, d, d, 1, stride},
                    {matrix, width, d, d, 1, stride}, true);
                transpose_levels<false><<<grid, 128, 0, stream>>>(input, matrix, result, count, n, d, matrix_size,
                    sig_stride, stride, scalar);
            } else {
                T* dv = v + nd;
                T* dq = dv + nd;
                T* result = out + first * path_stride;
                const T* seeds = deriv + first * sig_stride;
                transpose_levels<true><<<grid, 128, 0, stream>>>(input, seeds, matrix, count, n, d, matrix_size,
                    sig_stride, stride, scalar);
                CUDA_CHECK(cudaGetLastError());
                matrices.gemm({f, n, width, width, 1, stride}, {matrix, width, d, d, 1, stride},
                    {dv, n, d, d, 1, stride}, true);
                matrices.gemm({v, n, d, d, 1, stride}, {matrix, d, width, 1, d, stride},
                    {f, n, width, width, 1, stride}, true);
                factor_adjoints<<<make_cuda_1d_grid(mul(count, nd), 128), 128, 0, stream>>>(
                    input, f, v, dq, dv, count, n, d, width, stride);
                CUDA_CHECK(cudaGetLastError());
                if (n > 1) {
                    vertices<<<make_cuda_1d_grid(mul(count, (n - 1) * d), 128), 128, 0, stream>>>(
                        v, dq, dv, result, count, n, d, stride);
                    CUDA_CHECK(cudaGetLastError());
                }
                endpoints<<<make_cuda_1d_grid(mul(count, d), 4), 128, 0, stream>>>(
                    v, dq, dv, seeds, result, count, n, d, sig_stride, stride, scalar);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    });
}
} // namespace gpu_midpoint
