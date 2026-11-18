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

template<unsigned K, typename T>
__device__ T word_product(const T* path, unsigned n, unsigned d, unsigned word, unsigned lane) {
    unsigned c[K];
    #pragma unroll
    for (unsigned j = K; j-- > 0;) { c[j] = word % d; word /= d; }
    const T start = path[c[0]], last = path[n * d + c[K - 1]];
    T sum = 0, p2 = 0, px = 0;
    for (unsigned first = 0; first < n; first += 32) {
        const unsigned i = first + lane;
        T v[K]{}, p = 0, q = 0;
        if (i < n) {
            const T* u = path + i * d;
            #pragma unroll
            for (unsigned j = 0; j < K; ++j) v[j] = u[d + c[j]] - u[c[j]];
            p = (u[c[0]] - start) + v[0] / 2;
            q = (last - u[d + c[K - 1]]) + v[K - 1] / 2;
        }
        if constexpr (K == 2) sum += p * v[1];
        if constexpr (K == 3) sum += (p * q - (v[0] / 12) * v[2]) * v[1];
        if constexpr (K >= 4) {
            const T before = scan_tile(p * v[1], p2, lane);
            const T x = (before + (p / 2 - v[0] / 12) * v[1]) * v[2];
            const T triple = ((v[1] / 12) * v[2]) * v[3];
            if constexpr (K == 4) sum += x * q - p * triple;
            if constexpr (K == 5) {
                // Swap the segment sums in X_i * Q2_{i+1}.
                const T prior = scan_tile(x, px, lane);
                sum += (prior * v[3] - p * triple) * q
                    + x * v[3] * (q / 2 - v[4] / 12) + ((v[0] / 60) * triple) * v[4];
            }
        }
    }
    for (unsigned shift = 16; shift; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
    return sum;
}

template<typename T>
__global__ void small_words(const T* path, T* output, uint64_t batch, unsigned n, unsigned d,
    uint64_t sig_stride, bool scalar) {
    const unsigned lane = threadIdx.x % 32;
    for (uint64_t i = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; i < batch * sig_stride;
        i += uint64_t(gridDim.x) * blockDim.x / 32) {
        unsigned word = unsigned(i % sig_stride) + !scalar, level = 0, width = 1;
        while (word >= width) { word -= width; width *= d; ++level; }
        const T* data = path + i / sig_stride * uint64_t(n + 1) * d;
        T value = 1;
        switch (level) {
        case 1: value = data[n * d + word] - data[word]; break;
        case 2: value = word_product<2>(data, n, d, word, lane); break;
        case 3: value = word_product<3>(data, n, d, word, lane); break;
        case 4: value = word_product<4>(data, n, d, word, lane); break;
        case 5: value = word_product<5>(data, n, d, word, lane); break;
        }
        if (!lane) output[i] = value;
    }
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

template<bool Adjoint, typename T>
__global__ void quadratic_scan(const T* path, T* inc, T* pre, T* suf, T* output, const T* seed,
    uint64_t batch, unsigned n, unsigned d, uint64_t stride, uint64_t out_stride, bool scalar, bool second_seed) {
    const unsigned lane = threadIdx.x % 32, pairs = d * d, q = 1 + d + pairs;
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; task < batch * pairs;
        task += uint64_t(gridDim.x) * blockDim.x / 32) {
        const uint64_t sample = task / pairs, base = sample * stride;
        const unsigned pair = unsigned(task % pairs), x = pair / d, y = pair % d;
        const T* u = path + sample * uint64_t(n + 1) * d;
        const T first = u[x], last = u[n * d + y];
        T prefix = 0, suffix = 0;
        if constexpr (Adjoint) if (second_seed) prefix = seed[sample * out_stride + scalar + d + pair];
        for (unsigned t = 0; t < n; t += 32) {
            const unsigned i = t + lane, j = n - 1 - i;
            T a = 0, b = 0;
            if (i < n) {
                if constexpr (Adjoint) {
                    a = pre[base + j * q + 1 + d + pair];
                    b = suf[base + (i + 1) * q + 1 + d + pair];
                } else {
                    const T vx = u[(i + 1) * d + x] - u[i * d + x], vy = u[(i + 1) * d + y] - u[i * d + y];
                    a = ((u[i * d + x] - first) + vx / 2) * vy;
                    const T wx = u[(j + 1) * d + x] - u[j * d + x], wy = u[(j + 1) * d + y] - u[j * d + y];
                    b = wx * ((last - u[(j + 1) * d + y]) + wy / 2);
                    if (!x) {
                        inc[base + i * d + y] = vy;
                        pre[base + i * q + 1 + y] = u[i * d + y] - u[y];
                        suf[base + (j + 1) * q + 1 + y] = last - u[(j + 1) * d + y];
                        if (!y) { pre[base + i * q] = 1; suf[base + (j + 1) * q] = 1; }
                    }
                }
            }
            const T p = scan_tile(a, prefix, lane), s = scan_tile(b, suffix, lane);
            if (i < n) {
                pre[base + (Adjoint ? j : i) * q + 1 + d + pair] = p;
                suf[base + ((Adjoint ? i : j) + 1) * q + 1 + d + pair] = s;
            }
        }
        if constexpr (!Adjoint) if (output && !lane) {
            T* res = output + sample * out_stride;
            res[scalar + d + pair] = prefix;
            if (!x) {
                res[scalar + y] = last - u[y];
                if (!y && scalar) res[0] = 1;
            }
        }
    }
}

template<typename T>
__global__ void quadratic_gradient(const T* path, const T* inc, const T* dp, const T* dq, const T* dv,
    const T* seed, T* output, uint64_t batch, unsigned n, unsigned d, uint64_t stride, uint64_t sig_stride, bool scalar) {
    const unsigned lane = threadIdx.x % 32, q = 1 + d + d * d;
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; task < batch * d;
        task += uint64_t(gridDim.x) * blockDim.x / 32) {
        const unsigned c = unsigned(task % d);
        const uint64_t sample = task / d, base = sample * stride, path_base = sample * uint64_t(n + 1) * d;
        const T* u = path + path_base;
        T start = 0, end = 0, carry = 0, final = 0;
        for (unsigned first = 0; first < n; first += 32) {
            const unsigned i = first + lane;
            T left = 0, right = 0;
            if (i < n) {
                const T* p = dp + base + i * q; const T* s = dq + base + (i + 1) * q;
                T g = dv[base + i * d + c], mp = 0, mq = 0;
                for (unsigned x = 0; x < d; ++x) {
                    const T v = inc[base + i * d + x];
                    const T pm = (u[i * d + x] - u[x]) + v / 2;
                    const T qm = (u[n * d + x] - u[(i + 1) * d + x]) + v / 2;
                    g += p[1 + d + x * d + c] * pm + s[1 + d + c * d + x] * qm;
                    mp += p[1 + d + c * d + x] * v; mq += s[1 + d + x * d + c] * v;
                }
                const T mid = (mp - mq) / 2;
                start -= p[1 + c] + mp; end += s[1 + c] + mq;
                left = mid - g + p[1 + c]; right = mid + g - s[1 + c];
                if (i + 1 == n) final = right;
            }
            const T previous = __shfl_up_sync(0xffffffff, right, 1);
            if (i < n) output[path_base + i * d + c] = left + (lane ? previous : carry);
            carry = __shfl_sync(0xffffffff, right, 31);
        }
        end += final;
        for (unsigned shift = 16; shift; shift /= 2) {
            start += __shfl_down_sync(0xffffffff, start, shift);
            end += __shfl_down_sync(0xffffffff, end, shift);
        }
        if (!lane) {
            const T g = seed[sample * sig_stride + scalar + c];
            output[path_base + c] += start - g;
            output[path_base + n * d + c] = end + g;
        }
    }
}
