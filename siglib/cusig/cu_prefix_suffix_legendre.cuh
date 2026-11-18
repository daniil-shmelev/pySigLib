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


namespace gpu_legendre {

#include "cu_prefix_suffix_low_moments.cuh"

struct Degrees {
    unsigned d, a, b, groups;
    unsigned power[17], offset[18];
    double scale[8];
};

template<typename T> struct Mode {
    unsigned head, groups, cols, h, depth, first;
    T weight[9];
};

__device__ unsigned level(unsigned index, const Degrees& l) {
    unsigned k = 0;
    while (index >= l.offset[k + 1]) ++k;
    return k;
}

template<unsigned Threads>
__device__ void tile_sync() {
    if constexpr (Threads == 32) __syncwarp();
    else __syncthreads();
}

template<unsigned Threads, typename T>
__global__ void powers_kernel(const T* inc, T* powers, T* centres, const uint4* words,
    Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    extern __shared__ unsigned char raw[];
    constexpr unsigned tiles = 128 / Threads;
    const unsigned lane = threadIdx.x % Threads, tile = threadIdx.x / Threads, q = l.offset[l.b + 1];
    T* v = reinterpret_cast<T*>(raw) + tile * q;
    for (uint64_t i = uint64_t(blockIdx.x) * tiles + tile; i < batch * n; i += uint64_t(gridDim.x) * tiles) {
        const uint64_t base = i / n * stride;
        const unsigned seg = unsigned(i % n);
        if (!lane) v[0] = 1;
        if (lane < l.d) v[1 + lane] = inc[base + seg * l.d + lane];
        tile_sync<Threads>();
        for (unsigned k = 2; k <= l.b; ++k) {
            for (unsigned t = lane; t < l.power[k]; t += Threads)
                v[l.offset[k] + t] = v[l.offset[k - 1] + t / l.d] * v[1 + t % l.d];
            tile_sync<Threads>();
        }
        for (unsigned t = lane; t < q; t += Threads) powers[base + seg * q + t] = v[t];
        for (unsigned t = lane; t < l.groups; t += Threads) {
            const uint4 entry = words[t];
            centres[base + seg * l.groups + t] = ((v[entry.x] * T(l.scale[entry.w])) * v[1 + entry.y]) * v[entry.z];
        }
        tile_sync<Threads>();
    }
}

template<bool Suffix, typename T>
__device__ T moment(const T* state, const T* powers, unsigned k, unsigned word,
    const Mode<T>& m, const Degrees& l) {
    T sum = state[l.offset[k] + word];
    for (unsigned s = 1; s <= k; ++s) {
        const unsigned p = l.power[Suffix ? k - s : s];
        const unsigned left = word / p, right = word % p;
        sum += m.weight[s] * powers[l.offset[s] + (Suffix ? left : right)]
            * state[l.offset[k - s] + (Suffix ? right : left)];
    }
    return sum;
}

template<bool Backward, typename T>
__global__ void factors_kernel(const T* pre, const T* suf, const T* powers, const T* centres,
    T* left, T* right, T* heads, Mode<T> m, Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    const unsigned rows = m.head * m.groups, cols = rows + m.cols, q = l.offset[l.b + 1];
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * n * cols;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t i = x / cols, base = i / n * stride;
        const unsigned seg = unsigned(i % n), col = unsigned(x % cols);
        const T* v = powers + base + seg * q;
        if (col < rows) {
            const unsigned head = col / m.groups, group = col % m.groups;
            const T h = moment<false>(pre + base + seg * q, v, m.h, head, m, l);
            left[base + seg * rows + col] = h * centres[base + seg * l.groups + m.first + group];
            if constexpr (Backward) if (!group) heads[base + seg * m.head + head] = h;
        } else {
            const unsigned idx = col - rows, k = level(idx, l);
            right[base + seg * m.cols + idx] = moment<true>(suf + base + (seg + 1) * q, v,
                k, idx - l.offset[k], m, l);
        }
    }
}

template<unsigned D, typename T>
__device__ T packed_moment(const T* state, const T* v, unsigned k, unsigned word,
    const Mode<T>& m, const Degrees& l, bool suffix) {
    const unsigned d = D ? D : l.d;
    unsigned head = word, tail = 0, place = 1;
    T sum = 0;
    for (unsigned j = 0; j <= k; ++j) {
        const unsigned s = suffix ? k - j : j;
        const T prefix = suffix ? v[l.offset[k - j] + head] : state[l.offset[k - j] + head];
        const T rest = suffix ? state[l.offset[j] + tail] : v[l.offset[j] + tail];
        sum += m.weight[s] * prefix * rest;
        tail += (head % d) * place; head /= d; place *= d;
    }
    return sum;
}

template<bool Backward, unsigned D, unsigned Threads, typename T>
__global__ void tile_factors(const T* pre, const T* suf, const T* powers, const T* centres,
    T* left, T* right, T* heads, Mode<T> m, Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    extern __shared__ unsigned char raw[];
    constexpr unsigned tiles = 128 / Threads;
    const unsigned lane = threadIdx.x % Threads, tile = threadIdx.x / Threads;
    const unsigned q = l.offset[l.b + 1], p = l.offset[m.h + 1], rows = m.head * m.groups;
    T* local = reinterpret_cast<T*>(raw) + tile * (2 * m.cols + p + m.groups + m.head);
    T* pre_tile = local + m.cols; T* suf_tile = pre_tile + p;
    T* mon = suf_tile + m.cols; T* h = mon + m.groups;
    for (uint64_t i = uint64_t(blockIdx.x) * tiles + tile; i < batch * n; i += uint64_t(gridDim.x) * tiles) {
        const uint64_t base = i / n * stride;
        const unsigned seg = unsigned(i % n);
        for (unsigned x = lane; x < m.cols; x += Threads) {
            local[x] = powers[base + seg * q + x]; suf_tile[x] = suf[base + (seg + 1) * q + x];
        }
        for (unsigned x = lane; x < p; x += Threads) pre_tile[x] = pre[base + seg * q + x];
        for (unsigned x = lane; x < m.groups; x += Threads) mon[x] = centres[base + seg * l.groups + m.first + x];
        tile_sync<Threads>();
        for (unsigned x = lane; x < m.head; x += Threads) {
            const T value = packed_moment<D>(pre_tile, local, m.h, x, m, l, false);
            h[x] = value;
            if constexpr (Backward) heads[base + seg * m.head + x] = value;
        }
        for (unsigned x = lane; x < m.cols; x += Threads) {
            const unsigned k = level(x, l);
            right[base + seg * m.cols + x] = packed_moment<D>(suf_tile, local, k, x - l.offset[k], m, l, true);
        }
        tile_sync<Threads>();
        for (unsigned x = lane; x < rows; x += Threads)
            left[base + seg * rows + x] = h[x / m.groups] * mon[x % m.groups];
        tile_sync<Threads>();
    }
}

template<bool Backward, typename T>
__global__ void map_levels(T* product, T* output, const T* seed, Degrees l,
    uint64_t batch, uint64_t sig_stride, uint64_t stride, bool scalar,
    const T* upper, const unsigned* grow, unsigned groups, unsigned cols, int* unsafe, T limit) {
    const unsigned q = l.offset[l.b + 1], width = l.power[l.a] * q;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * width;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const unsigned idx = unsigned(x % width), col = idx % q, k = level(col, l);
        const uint64_t sample = x / width, pos = sample * sig_stride
            + l.offset[l.a + k] - !scalar + idx / q * l.power[k] + col - l.offset[k];
        if constexpr (Backward) {
            const T value = seed[pos];
            product[sample * stride + idx] = value;
            if (!(abs(value) <= limit) || (idx < l.offset[l.a] - !scalar && !(abs(seed[sample * sig_stride + idx]) <= limit)))
                atomicExch(unsafe, 1);
        } else {
            T value = product[sample * stride + idx];
            if (k) {
                const unsigned row = idx / q, letters = l.d * l.d, width_right = l.power[k - 1];
                const unsigned word = col - l.offset[k], id = grow[row % letters * l.d + word / width_right];
                value += upper[sample * stride + (row / letters * groups + id) * cols + l.offset[k - 1] + word % width_right];
            }
            output[pos] = value;
        }
    }
}

// A block owns one factor row. Neighbouring lanes copy consecutive suffix words.
template<bool Top, bool Backward, typename T>
__global__ void map_rows(T* product, T* upper, const T* seed, T* output, const unsigned* grow,
    Mode<T> mode, Mode<T> next, Degrees l, uint64_t batch, uint64_t sig_stride,
    uint64_t stride, bool scalar, int* flag, T limit) {
    const unsigned rows = mode.head * mode.groups, letters = l.d * mode.groups;
    for (uint64_t task = blockIdx.x; task < batch * rows; task += gridDim.x) {
        const uint64_t sample = task / rows, base = sample * stride;
        const unsigned row = unsigned(task % rows);
        T* packed = product + base + row * mode.cols;
        if constexpr (Top) {
            const uint64_t pos = sample * sig_stride + l.offset[l.a] - !scalar + row;
            if (!threadIdx.x) {
                if constexpr (Backward) {
                    const T value = seed[pos];
                    packed[0] = value;
                    if (!(abs(value) <= limit)) atomicExch(flag, 1);
                    if (row < l.offset[l.a] - !scalar && !(abs(seed[sample * sig_stride + row]) <= limit)) atomicExch(flag, 1);
                } else output[pos] = packed[0];
            }
        }
        for (unsigned k = 1; k <= mode.depth; ++k) {
            const unsigned width = l.power[k - 1], off = l.offset[k];
            const uint64_t pos = sample * sig_stride + l.offset[l.a + k] - !scalar + row * l.power[k];
            for (unsigned y = 0; y < l.d; ++y) {
                const unsigned col = y * width;
                if constexpr (Backward) {
                    for (unsigned x = threadIdx.x; x < width; x += blockDim.x) {
                        const T value = seed[pos + col + x];
                        packed[off + col + x] = value;
                        if (!(abs(value) <= limit)) atomicExch(flag, 1);
                    }
                } else {
                    const unsigned group = grow[(row % letters) * l.d + y];
                    const T* higher = upper + base + (row / letters * next.groups + group) * next.cols + l.offset[k - 1];
                    for (unsigned x = threadIdx.x; x < width; x += blockDim.x) {
                        const T value = packed[off + col + x] + higher[x];
                        if constexpr (Top) output[pos + col + x] = value;
                        else packed[off + col + x] = value;
                    }
                }
            }
        }
    }
}

template<bool Fold, typename T>
__global__ void expand(T* current, T* upper, const unsigned* grow, const unsigned* offsets,
    const unsigned* members, Mode<T> m, Mode<T> next, Degrees l, uint64_t batch, uint64_t stride) {
    const unsigned cols = Fold ? next.cols : m.cols;
    const unsigned size = cols * (Fold ? next.head * next.groups : m.head * m.groups);
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * size;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t base = x / size * stride;
        const unsigned idx = unsigned(x % size), row = idx / cols, col = idx % cols;
        const unsigned k = level(col, l), w = col - l.offset[k];
        if constexpr (Fold) {
            const unsigned id = row % next.groups, first = row / next.groups * l.d * m.groups;
            T sum = 0;
            for (unsigned j = offsets[id]; j < offsets[id + 1]; ++j) {
                const unsigned ex = members[j];
                sum += current[base + (first + ex / l.d) * m.cols + l.offset[k + 1] + (ex % l.d) * l.power[k] + w];
            }
            upper[base + idx] = sum;
        } else if (k) {
            const unsigned size_right = l.power[k - 1], letters = l.d * m.groups;
            const unsigned group = grow[row % letters * l.d + w / size_right];
            current[base + idx] += upper[base + (row / letters * next.groups + group) * next.cols
                + l.offset[k - 1] + w % size_right];
        }
    }
}

template<typename T>
__global__ void head_adjoints(const T* dl, const T* centre, const T* heads, T* dh, T* dc,
    Mode<T> m, Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    const unsigned cols = m.head + m.groups, rows = m.head * m.groups;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * n * cols;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t i = x / cols, base = i / n * stride;
        const unsigned seg = unsigned(i % n), col = unsigned(x % cols);
        const T* grad = dl + base + seg * rows;
        T sum = 0;
        if (col < m.head) {
            const T* c = centre + base + seg * l.groups + m.first;
            for (unsigned g = 0; g < m.groups; ++g) sum += grad[col * m.groups + g] * c[g];
            dh[base + seg * m.head + col] = sum;
        } else {
            const T* h = heads + base + seg * m.head;
            for (unsigned a = 0; a < m.head; ++a) sum += grad[a * m.groups + col - m.head] * h[a];
            dc[base + seg * l.groups + m.first + col - m.head] = sum;
        }
    }
}

template<typename T>
__global__ void moment_adjoints(const T* pre, const T* suf, const T* powers, const T* dh, const T* dr,
    T* dp, T* dq, T* dpoly, Mode<T> m, Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    const unsigned q = l.offset[l.b + 1], cols = m.cols - 1;
    for (uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; x < batch * n * cols;
        x += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t i = x / cols, base = i / n * stride;
        const unsigned seg = unsigned(i % n), idx = unsigned(x % cols) + 1;
        const unsigned k = level(idx, l), w = idx - l.offset[k];
        const T* v = powers + base + seg * q;
        const T* head = dh + base + seg * m.head;
        const T* right = dr + base + seg * m.cols;
        T grad = 0;
        if (k <= m.h) {
            const unsigned s = m.h - k, terms = l.power[s];
            T sum = 0;
            for (unsigned t = 0; t < terms; ++t) sum += head[w * terms + t] * v[l.offset[s] + t];
            dp[base + seg * q + idx] += m.weight[s] * sum;
            const T* state = pre + base + seg * q + l.offset[s];
            sum = 0;
            for (unsigned t = 0; t < terms; ++t) sum += head[t * l.power[k] + w] * state[t];
            grad = m.weight[k] * sum;
        }
        T adj = 0;
        for (unsigned s = 0; s <= m.depth - k; ++s) {
            const unsigned terms = l.power[s];
            const T* state = suf + base + (seg + 1) * q + l.offset[s];
            const T* g = right + l.offset[k + s];
            T sum = 0, other = 0;
            for (unsigned t = 0; t < terms; ++t) {
                sum += g[t * l.power[k] + w] * v[l.offset[s] + t];
                other += g[w * terms + t] * state[t];
            }
            adj += m.weight[s] * sum;
            grad += m.weight[k] * other;
        }
        dq[base + (seg + 1) * q + idx] += adj;
        dpoly[base + seg * q + idx] += grad;
    }
}

// A differentiated centre is a previous centre times one increment, or a
// product of two ordered powers when those are already available.
struct PowerDerivatives {
    std::vector<uint4> terms;
    std::vector<unsigned> offsets{0};
    PowerDerivatives(const Layout& l, const LegendrePlan& plan, bool centres_only) {
        const unsigned q = plan.modes.front().cols;
        std::vector<std::unordered_map<unsigned, unsigned>> centre_index;
        if (centres_only) for (const auto& m : plan.modes) {
            auto& ids = centre_index.emplace_back();
            for (unsigned g = 0; g < m.groups; ++g) ids.emplace(m.words[g], m.first_group + g);
        }
        for (unsigned c = 0; c < l.d; ++c) {
            auto append = [&](unsigned word, unsigned k, unsigned col, unsigned mode) {
                unsigned count = 0, remaining = 0, power = 1;
                for (unsigned j = 0; j < k; ++j) {
                    const unsigned digit = unsigned(word % l.d); word = unsigned(word / l.d);
                    if (digit == c && ++count == 1) continue;
                    remaining += digit * power; power *= unsigned(l.d);
                }
                if (!count) return;
                if (centres_only) {
                    const unsigned parent = centre_index[mode - 1].at(unsigned(remaining / l.d));
                    terms.push_back(make_uint4(col - q, parent, unsigned(remaining % l.d), mode * 32 + count));
                    return;
                }
                const unsigned tail = (k - 1) / 2, head = k - 1 - tail, right = unsigned(l.power[tail]);
                terms.push_back(make_uint4(col, unsigned(l.offset[head]) + remaining / right,
                    unsigned(l.offset[tail]) + remaining % right, mode * 32 + count));
            };
            if (!centres_only) for (unsigned k = 1; k <= plan.b; ++k) for (unsigned w = 0; w < l.power[k]; ++w)
                append(w, k, unsigned(l.offset[k]) + w, 0);
            for (unsigned r = centres_only ? 1 : 0; r < plan.modes.size(); ++r) {
                const auto& m = plan.modes[r];
                for (unsigned g = 0; g < m.groups; ++g)
                    append(m.words[g], 2 * r + 1, q + m.first_group + g, r);
            }
            offsets.push_back(unsigned(terms.size()));
        }
    }
};

template<typename T>
__global__ void increment_adjoints(const T* powers, const T* dpoly, const T* dc, T* dv,
    const uint4* terms, const unsigned* offsets, Degrees l, unsigned n, uint64_t batch, uint64_t stride) {
    const unsigned lane = threadIdx.x % 32, q = l.offset[l.b + 1];
    for (uint64_t i = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32; i < batch * n * l.d;
        i += uint64_t(gridDim.x) * blockDim.x / 32) {
        const unsigned channel = unsigned(i % l.d), seg = unsigned(i / l.d % n);
        const uint64_t base = i / l.d / n * stride;
        const T* v = powers + base + seg * q;
        T sum = 0;
        const unsigned end = offsets[channel + 1];
        for (unsigned t = offsets[channel] + lane; t < end; t += 32) {
            const uint4 ex = terms[t];
            const T seed = ex.x < q ? dpoly[base + seg * q + ex.x] : dc[base + seg * l.groups + ex.x - q];
            sum += ((seed * T(l.scale[ex.w / 32])) * v[ex.y]) * v[ex.z] * T(ex.w % 32);
        }
        for (unsigned step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
        if (!lane) dv[base + seg * l.d + channel] = sum;
    }
}

#include "cu_prefix_suffix_moment_scan.cuh"

template<bool Backward, typename T>
bool compute(const T* path, const T* seed, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream) {
    if (l.m < 4 || l.m > 16 || l.d > 24 || l.power[l.m] > (1ULL << 25) || n < 8
        || l.offset[l.h + 1] * sizeof(T) > 48 * 1024 || n > (192ULL << 20) / (sizeof(T) * l.offset[l.h + 1]))
        return false;
    if (l.d >= 6 && (l.m <= 5 || (l.m == 6 && std::is_same_v<T, float>))) return false;
    if (n < 32 && ((l.m >= 6 && l.m <= 8) || (l.d == 2 && std::is_same_v<T, float>))) return false;
    if (l.d >= 4 && l.m >= 7 && n <= 64 && (!Backward || n < 32)) return false;
    if constexpr (std::is_same_v<T, float>) {
        if (n < 32 && (l.d >= 4 || l.m >= 9)) return false;
    }
    if constexpr (!Backward) {
        if (l.m <= 5 && l.d <= 4 && n <= 32 && std::is_same_v<T, float>) {
            const uint64_t pitch = l.offset[l.m + 1] - !scalar;
            small_words<<<make_cuda_1d_grid(batch * pitch, 4), 128, 0, stream>>>(path, out, batch,
                unsigned(n), unsigned(l.d), pitch, scalar);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
            return true;
        }
        if (l.m <= 5 && l.d <= 4
            && n * (l.m - 2) * (l.power[3] + l.offset[l.m - 2]) <= (128ULL << 20) / sizeof(T)) {
            if (l.m == 5) small_forward<true>(path, out, l, device, batch, n, scalar, stream);
            else small_forward<false>(path, out, l, device, batch, n, scalar, stream);
            return true;
        }
    }
    const LegendrePlan plan(l);
    const uint64_t q = l.offset[plan.b + 1], sig_stride = l.offset[l.m + 1] - !scalar;
    const uint64_t path_stride = (n + 1) * l.d, centre_size = n * plan.groups;
    const uint64_t chunk_length = plan.b == 2 ? n : Backward ? 16 : 32, chunks = (n - 1) / chunk_length + 1;
    const uint64_t summary_size = chunks * q, scan_size = (n + 1) * q;
    const uint64_t temp = Backward && q > 32768 / (3 * sizeof(T)) ? 3 * summary_size : 0;
    const uint64_t factors = n * (plan.max_rows + q), inc_size = n * l.d;
    const uint64_t head_size = n * plan.modes[0].head;
    uint64_t split_size = 0;
    if constexpr (!Backward)
        for (const auto& m : plan.modes) split_size = std::max(split_size,
            gpu_split_k::chunks(n, m.rows(), m.cols) * m.size());
    const uint64_t zsize = Backward ? 2 * scan_size + n * q : 0;
    uint64_t stride = inc_size + 2 * scan_size + 4 * summary_size + temp + n * q + centre_size
        + factors + plan.buffers[0] + plan.buffers[1] + split_size;
    if constexpr (Backward) stride += summary_size + q + zsize + inc_size + factors + centre_size + 2 * head_size;
    const ScanShape scan(l, plan, unsigned(n));
    const bool fuse_products = !Backward && (sizeof(T) == sizeof(float) || batch * n >= 1024);
    uint64_t left_rows = 0;
    for (const auto& m : plan.modes)
        if (m.head > 1 && !(fuse_products && m.size() <= 16384)) left_rows = std::max(left_rows, m.rows());
    const uint64_t packed = uint64_t(scan.pitch) * (scan.p.size + scan.q.size + plan.groups + left_rows);
    const uint64_t scan_stride = plan.buffers[0] + plan.buffers[1] + split_size + (1 + Backward) * packed
        + (Backward ? inc_size + 2 * uint64_t(scan.pitch) * (l.offset[scan.p.depth + 1] + q) : 0);
    const bool scanned = n >= 16 && scan_stride <= (192ULL << 20) / sizeof(T);
    if (scanned) stride = scan_stride;
    stride = (stride + 63) / 64 * 64;
    if (stride > (192ULL << 20) / sizeof(T)) return false;
    Degrees levels{};
    levels.a = plan.a; levels.b = plan.b; levels.d = unsigned(l.d); levels.groups = plan.groups;
    for (unsigned k = 0; k <= l.m; ++k) levels.power[k] = unsigned(l.power[k]);
    for (unsigned k = 0; k <= l.m + 1; ++k) levels.offset[k] = unsigned(l.offset[k]);
    std::vector<uint4> words;
    std::vector<unsigned> maps;
    struct Indices { size_t grow, offsets, members; };
    std::vector<Indices> starts;
    std::vector<Mode<T>> modes;
    for (unsigned r = 0; r < plan.modes.size(); ++r) {
        const auto& m = plan.modes[r];
        levels.scale[r] = r ? -(scanned ? 1.0 : levels.scale[r - 1]) / (4 * (2 * r - 1) * (2 * r + 1)) : 1;
        const unsigned width = levels.power[r], first = levels.offset[r];
        if (scanned && r) {
            const auto& prev = plan.modes[r - 1];
            for (auto ex : m.parent) {
                const unsigned pair = ex / levels.d;
                words.push_back(make_uint4(prev.first_group + pair % prev.groups, pair / prev.groups,
                    ex % levels.d, m.first_group + m.groups));
            }
        } else for (auto word : m.words) words.push_back(make_uint4(first + word / width / levels.d,
            word / width % levels.d, first + word % width, r));
        auto append = [&](const std::vector<unsigned>& data) {
            const auto pos = maps.size();
            maps.insert(maps.end(), data.begin(), data.end());
            return pos;
        };
        starts.push_back({append(m.grow), append(m.offsets), append(m.members)});
        Mode<T> mode{m.head, m.groups, m.cols, plan.a - 1 - r, m.depth, m.first_group};
        for (unsigned k = 0; k <= m.depth; ++k) mode.weight[k] = T(m.weight[k]);
        modes.push_back(mode);
    }
    std::vector<uint4> derivatives;
    const size_t derivative_offsets = maps.size();
    if constexpr (Backward) {
        PowerDerivatives table(l, plan, scanned);
        derivatives = std::move(table.terms);
        maps.insert(maps.end(), table.offsets.begin(), table.offsets.end());
    }
    const uint64_t meta64 = scanned ? 0 : (l.power.size() + l.offset.size() + 1) / 2 * 2;
    const uint64_t metadata = meta64 * sizeof(uint64_t) + derivatives.size() * sizeof(uint4)
        + words.size() * sizeof(uint4) + maps.size() * sizeof(unsigned) + sizeof(int);
    std::vector<std::byte> packed_metadata((metadata + 255) / 256 * 256);
    if (!scanned) {
        std::memcpy(packed_metadata.data(), l.power.data(), l.power.size() * sizeof(uint64_t));
        std::memcpy(packed_metadata.data() + l.power.size() * sizeof(uint64_t), l.offset.data(), l.offset.size() * sizeof(uint64_t));
    }
    auto* term_bytes = packed_metadata.data() + meta64 * sizeof(uint64_t);
    auto* word_bytes = term_bytes + derivatives.size() * sizeof(uint4);
    if (!derivatives.empty()) std::memcpy(term_bytes, derivatives.data(), derivatives.size() * sizeof(uint4));
    std::memcpy(word_bytes, words.data(), words.size() * sizeof(uint4));
    std::memcpy(word_bytes + words.size() * sizeof(uint4), maps.data(), maps.size() * sizeof(unsigned));
    const uint64_t paths = std::clamp<uint64_t>((512ULL << 20) / (stride * sizeof(T)), 1, 32);
    int unsafe = 0;
    with_factor_workspace<T>(l, device, batch, paths, stride, packed_metadata.size(),
        4ULL << 20, true, "prefix-suffix moments", stream, false,
        [&](cublasHandle_t handle, Fp64Emulation*, uint64_t capacity, T* memory, uint64_t* power, uint64_t* offset) {
        auto* terms = reinterpret_cast<uint4*>(power + meta64);
        auto* ids = terms + derivatives.size();
        auto* indices = reinterpret_cast<unsigned*>(ids + words.size());
        auto* flag = reinterpret_cast<int*>(indices + maps.size());
        T* cursor = memory;
        auto take = [&](uint64_t size) { T* ptr = cursor; cursor += size; return ptr; };
        const uint64_t pitch = scan.pitch;
        T* inc = take(scanned && !Backward ? 0 : inc_size); T* pre = take(scanned ? pitch * scan.p.size : scan_size);
        T* suf = take(scanned ? pitch * scan.q.size : scan_size);
        T* summary = take(scanned ? 0 : summary_size); T* pscan = take(scanned ? 0 : summary_size);
        T* qscan = take(scanned ? 0 : summary_size); T* swap = take(scanned ? 0 : summary_size);
        T* work = take(scanned ? 0 : temp); T* poly = take(scanned ? 0 : n * q);
        T* centres = take(scanned ? pitch * plan.groups : centre_size);
        T* left = take(scanned ? pitch * left_rows : n * plan.max_rows); T* right = take(scanned ? 0 : n * q);
        T* buf[]{take(plan.buffers[0]), take(plan.buffers[1])};
        T* partial = take(split_size);
        T *boundary = nullptr, *low = nullptr, *dp = nullptr, *dq = nullptr, *dpoly = nullptr, *dv = nullptr;
        T *dl = nullptr, *dr = nullptr, *dc = nullptr, *heads = nullptr, *dh = nullptr;
        T *zp = nullptr, *zq = nullptr, *cp = nullptr, *cq = nullptr;
        if constexpr (Backward) {
            if (scanned) {
                dp = take(pitch * scan.p.size); dq = take(pitch * scan.q.size);
                dc = take(pitch * plan.groups); dl = take(pitch * left_rows);
                zp = take(pitch * l.offset[scan.p.depth + 1]); zq = take(pitch * q);
                cp = take(pitch * l.offset[scan.p.depth + 1]); cq = take(pitch * q);
            } else {
                boundary = take(summary_size); low = take(q);
                dp = take(scan_size); dq = take(scan_size); dpoly = take(n * q); dv = take(inc_size);
                dl = take(n * plan.max_rows); dr = take(n * q); dc = take(centre_size);
                heads = take(head_size); dh = take(head_size);
            }
        }
        for (uint64_t base = 0; base < batch; base += capacity) {
            const uint64_t count = std::min(capacity, batch - base);
            const T* input = path + base * path_stride;
            const T* grad = Backward ? seed + base * sig_stride : nullptr;
            T* result = out + base * (Backward ? path_stride : sig_stride);
            auto grid = [&](uint64_t size) { return make_cuda_1d_grid(count * size, 256); };
            auto map_final = [&] {
                const T limit = (std::numeric_limits<T>::max)() / T(l.power[l.m] * l.m);
                const auto& m = plan.modes[0];
                if (m.cols >= 256) {
                    map_rows<true, Backward><<<make_cuda_1d_grid(count * m.rows(), 1), 128, 0, stream>>>(buf[0],
                        buf[1], grad, result, indices + starts[0].grow, modes[0], modes[1], levels, count,
                        sig_stride, stride, scalar, flag, limit);
                } else map_levels<Backward><<<grid(m.size()), 256, 0, stream>>>(buf[0], result, grad, levels,
                    count, sig_stride, stride, scalar, buf[1], indices + starts[0].grow, modes[1].groups,
                    modes[1].cols, flag, limit);
                CUDA_CHECK(cudaGetLastError());
            };
            const unsigned threads = l.power[plan.b] <= 64 ? 32 : 128;
            if (scanned) {
                scan_start<<<grid(l.d * pitch), 256, 0, stream>>>(input, centres, pre, suf, Backward ? nullptr : result,
                    scan, unsigned(l.d), count, stride, sig_stride, scalar);
                CUDA_CHECK(cudaGetLastError());
            } else if (plan.b == 2) {
                quadratic_scan<false><<<make_cuda_1d_grid(count * l.d * l.d, 2), 64, 0, stream>>>(input, inc,
                    pre, suf, Backward ? nullptr : result, grad, count, unsigned(n), unsigned(l.d),
                    stride, sig_stride, scalar, false);
                CUDA_CHECK(cudaGetLastError());
            } else prepare_scans(input, inc, pre, suf, Backward ? low : result, summary, pscan, qscan, swap, work,
                count, l.d, n + 1, l, n, q, q, chunks, chunk_length, stride, Backward ? stride : sig_stride,
                Backward || scalar, false, false, T(1), power, offset, Backward, false, threads, 0, stream);
            if (scanned) scan_centres<<<make_cuda_1d_grid(count * (pitch / 32), 1), 128, 0, stream>>>(
                centres, ids, levels, scan, count, stride);
            else if (q <= 256 && plan.groups <= 256)
                powers_kernel<32><<<make_cuda_1d_grid(count * n, 4), 128, 4 * q * sizeof(T), stream>>>(
                    inc, poly, centres, ids, levels, unsigned(n), count, stride);
            else powers_kernel<128><<<make_cuda_1d_grid(count * n, 1), 128, q * sizeof(T), stream>>>(
                inc, poly, centres, ids, levels, unsigned(n), count, stride);
            CUDA_CHECK(cudaGetLastError());
            if (scanned) for (unsigned k = 2; k <= scan.q.depth; ++k) {
                const uint64_t tasks = count * l.power[k] * (1 + unsigned(k <= scan.p.depth));
                if (n >= 512) scan_level<128><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf, centres,
                    Backward ? nullptr : result, scan, levels, k, count, stride, sig_stride, scalar);
                else scan_level<32><<<make_cuda_1d_grid(tasks, 4), 128, 0, stream>>>(pre, suf, centres,
                    Backward ? nullptr : result, scan, levels, k, count, stride, sig_stride, scalar);
                CUDA_CHECK(cudaGetLastError());
            }
            if constexpr (Backward) {
                if (!scanned) {
                    zero_adjoints_kernel<<<grid(zsize), 256, 0, stream>>>(dp, count, zsize, stride);
                    CUDA_CHECK(cudaGetLastError());
                }
                map_final();
            }
            CudaMatrices<T> matrices{handle, count};
            auto transfer = [&]<bool Fold>(unsigned r) {
                const auto& m = plan.modes[Fold ? r + 1 : r];
                if (!Fold && m.cols >= 256) {
                    map_rows<false, false><<<make_cuda_1d_grid(count * m.rows(), 1), 128, 0, stream>>>(
                        buf[r % 2], buf[(r + 1) % 2], grad, result, indices + starts[r].grow,
                        modes[r], modes[r + 1], levels, count, sig_stride, stride, scalar, flag, T(0));
                } else expand<Fold><<<grid(m.size()), 256, 0, stream>>>(buf[r % 2], buf[(r + 1) % 2],
                    indices + starts[r].grow, indices + starts[r + 1].offsets, indices + starts[r + 1].members,
                    modes[r], modes[r + 1], levels, count, stride);
                CUDA_CHECK(cudaGetLastError());
            };
            for (unsigned step = 0; step < modes.size(); ++step) {
                const unsigned r = Backward ? step : unsigned(modes.size()) - 1 - step;
                const auto& m = modes[r];
                const uint64_t rows = plan.modes[r].rows(), cols = m.cols;
                if (scanned) {
                    T* product = buf[r % 2];
                    T* head = pre + (scan.p.first[r] + levels.offset[m.h]) * pitch;
                    T* centre = centres + m.first * pitch;
                    T* rhs = suf + scan.q.first[r] * pitch;
                    if constexpr (!Backward) {
                        if (fuse_products && m.head > 1 && rows * cols <= 16384) {
                            const uint64_t splits = std::max<uint64_t>(1, gpu_split_k::chunks(n, rows, cols));
                            gpu_split_k::gemm<true, true>(head, rhs, partial, product, count, n, rows, cols,
                                pitch, splits, stride, stride, true, stream, centre, m.groups);
                            if (step && r) transfer.template operator()<false>(r);
                            continue;
                        }
                    }
                    T* lhs = m.head == 1 ? centre : left;
                    if (m.head != 1) {
                        scan_left<false><<<grid(rows * pitch), 256, 0, stream>>>(left, dp, dc, head, centre, m, scan, count, stride);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    if (cols == 1) {
                        scan_constant_mode<Backward><<<make_cuda_1d_grid(count * rows, 4), 128, 0, stream>>>(
                            Backward ? dc + m.first * pitch : nullptr, product, centre, unsigned(rows), scan, count, stride);
                        CUDA_CHECK(cudaGetLastError());
                        continue;
                    }
                    if constexpr (Backward) {
                        T* dleft = m.head == 1 ? dc + m.first * pitch : dl;
                        matrices.gemm({product, rows, cols, cols, 1, stride}, {rhs, cols, n, pitch, 1, stride},
                            {dleft, rows, n, pitch, 1, stride}, true);
                        matrices.gemm({product, cols, rows, 1, cols, stride}, {lhs, rows, n, pitch, 1, stride},
                            {dq + scan.q.first[r] * pitch, cols, n, pitch, 1, stride}, true);
                        if (m.head != 1) {
                            scan_left<true><<<grid((m.head + m.groups) * pitch), 256, 0, stream>>>(dl, dp + (head - pre),
                                dc + m.first * pitch, head, centre, m, scan, count, stride);
                            CUDA_CHECK(cudaGetLastError());
                        }
                        if (r + 1 < modes.size()) transfer.template operator()<true>(r);
                    } else {
                        const uint64_t splits = gpu_split_k::chunks(n, rows, cols);
                        if (splits) gpu_split_k::gemm<true>(lhs, rhs, partial, product, count, n, rows, cols,
                            pitch, splits, stride, stride, true, stream);
                        else matrices.gemm({lhs, rows, n, pitch, 1, stride}, {rhs, n, cols, 1, pitch, stride},
                            {product, rows, cols, cols, 1, stride}, true);
                        if (step && r) transfer.template operator()<false>(r);
                    }
                    continue;
                }
                const uint64_t bytes = sizeof(T) * (2 * cols + l.offset[m.h + 1] + m.groups + m.head);
                if (bytes <= 32768) {
                    auto fill = [&]<unsigned D, unsigned Threads>() {
                        constexpr unsigned tiles = 128 / Threads;
                        tile_factors<Backward, D, Threads><<<make_cuda_1d_grid(count * n, tiles), 128, bytes * tiles, stream>>>(
                            pre, suf, poly, centres, left, right, heads, m, levels, unsigned(n), count, stride);
                    };
                    auto run = [&]<unsigned D>() {
                        if (std::max(rows, cols) <= 256 && 4 * bytes <= 32768) fill.template operator()<D, 32>();
                        else fill.template operator()<D, 128>();
                    };
                    switch (l.d) {
                    case 2: run.template operator()<2>(); break;
                    case 3: run.template operator()<3>(); break;
                    case 4: run.template operator()<4>(); break;
                    case 5: run.template operator()<5>(); break;
                    case 6: run.template operator()<6>(); break;
                    case 8: run.template operator()<8>(); break;
                    default: run.template operator()<0>(); break;
                    }
                } else factors_kernel<Backward><<<grid(n * (rows + cols)), 256, 0, stream>>>(pre, suf, poly,
                    centres, left, right, heads, m, levels, unsigned(n), count, stride);
                CUDA_CHECK(cudaGetLastError());
                T* product = buf[r % 2];
                if constexpr (Backward) {
                    matrices.gemm({right, n, cols, cols, 1, stride}, {product, cols, rows, 1, cols, stride},
                        {dl, n, rows, rows, 1, stride}, true);
                    matrices.gemm({left, n, rows, rows, 1, stride}, {product, rows, cols, cols, 1, stride},
                        {dr, n, cols, cols, 1, stride}, true);
                    head_adjoints<<<grid(n * (m.head + m.groups)), 256, 0, stream>>>(dl, centres, heads,
                        dh, dc, m, levels, unsigned(n), count, stride);
                    CUDA_CHECK(cudaGetLastError());
                    if (cols > 1) {
                        moment_adjoints<<<grid(n * (cols - 1)), 256, 0, stream>>>(pre, suf, poly, dh, dr,
                            dp, dq, dpoly, m, levels, unsigned(n), count, stride);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    if (r + 1 < modes.size()) transfer.template operator()<true>(r);
                } else {
                    const uint64_t splits = gpu_split_k::chunks(n, rows, cols);
                    if (splits) gpu_split_k::gemm(left, right, partial, product, count, n, rows, cols,
                        cols, splits, stride, stride, true, stream);
                    else matrices.gemm({left, rows, n, 1, rows, stride}, {right, n, cols, cols, 1, stride},
                        {product, rows, cols, cols, 1, stride}, true);
                    if (step && r) transfer.template operator()<false>(r);
                }
            }
            if constexpr (Backward) {
                if (scanned) {
                    auto reverse = [&]<unsigned D>() {
                        for (unsigned k = scan.q.depth; k > 0; --k) {
                            const uint64_t tasks = count * l.power[k] * (1 + unsigned(k <= scan.p.depth));
                            if (n >= 512) scan_reverse_level<D, 128><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf,
                                dp, dq, centres, zp, zq, cp, cq, grad, scan, levels, k, count, stride, sig_stride, scalar);
                            else scan_reverse_level<D, 32><<<make_cuda_1d_grid(tasks, 4), 128, 0, stream>>>(pre, suf,
                                dp, dq, centres, zp, zq, cp, cq, grad, scan, levels, k, count, stride, sig_stride, scalar);
                            CUDA_CHECK(cudaGetLastError());
                        }
                        scan_increment_gradient<D><<<make_cuda_1d_grid(count * l.d * (pitch / 32), 1), 256, 0, stream>>>(
                            pre, suf, zp, zq, cp, cq, centres, dc, inc, terms, indices + derivative_offsets,
                            scan, levels, count, stride);
                    };
                    switch (l.d) {
                    case 2: reverse.template operator()<2>(); break;
                    case 3: reverse.template operator()<3>(); break;
                    case 4: reverse.template operator()<4>(); break;
                    default: reverse.template operator()<0>(); break;
                    }
                    CUDA_CHECK(cudaGetLastError());
                    cuda_detail::path_backprop_kernel<<<grid(path_stride), 256, 0, stream>>>(inc, result, count,
                        l.d, n + 1, l.d, n, stride, false);
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }
                increment_adjoints<<<grid(32 * inc_size), 256, 0, stream>>>(poly, dpoly, dc, dv, terms,
                    indices + derivative_offsets, levels, unsigned(n), count, stride);
                CUDA_CHECK(cudaGetLastError());
                if (plan.b == 2) {
                    quadratic_scan<true><<<make_cuda_1d_grid(count * l.d * l.d, 2), 64, 0, stream>>>(input, inc,
                        dp, dq, result, grad, count, unsigned(n), unsigned(l.d), stride, sig_stride, scalar, plan.a > 2);
                    CUDA_CHECK(cudaGetLastError());
                    quadratic_gradient<<<make_cuda_1d_grid(count * l.d, 1), 32, 0, stream>>>(input, inc, dp, dq,
                        dv, grad, result, count, unsigned(n), unsigned(l.d), stride, sig_stride, scalar);
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }
                reverse_factor_scan<false>(l, plan.a - 1, true, pre, inc, dp, grad, dv, summary,
                    pscan, qscan, swap, boundary, work, count, n, q, chunks, chunk_length, stride,
                    sig_stride, scalar, power, offset, threads, stream);
                reverse_factor_scan<true>(l, plan.b, false, suf, inc, dq, grad, dv, summary,
                    pscan, qscan, swap, boundary, work, count, n, q, chunks, chunk_length, stride,
                    sig_stride, scalar, power, offset, threads, stream);
                cuda_detail::path_backprop_kernel<<<grid(path_stride), 256, 0, stream>>>(dv, result, count,
                    l.d, n + 1, l.d, n, stride, false);
            } else {
                map_final();
            }
            CUDA_CHECK(cudaGetLastError());
        }
        if constexpr (Backward) CUDA_CHECK(cudaMemcpyAsync(&unsafe, flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
    }, packed_metadata.data());
    return !unsafe;
}

} // namespace gpu_legendre
