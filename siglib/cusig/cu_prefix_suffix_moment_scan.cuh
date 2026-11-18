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

struct ScanShape {
    MomentRows p, q;
    unsigned n, pitch;
    ScanShape(const Layout& l, const LegendrePlan& plan, unsigned segments)
        : p(l, plan.a - 1), q(l, plan.b), n(segments), pitch((segments + 31) / 32 * 32) {}
};

template<bool Augmented, typename T>
__global__ void scan_start(const T* path, T* centres, T* pre, T* suf, T* low, ScanShape s, unsigned d,
    uint64_t batch, uint64_t stride, uint64_t sig_stride, bool scalar,
    uint64_t dimension, uint64_t path_stride, bool ta, bool ll, T end_time) {
    for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < batch * d * s.pitch;
        index += uint64_t(gridDim.x) * blockDim.x) {
        const unsigned i = unsigned(index % s.pitch), c = unsigned(index / s.pitch % d);
        if (i >= s.n) continue;
        const uint64_t sample = index / s.pitch / d, base = sample * stride;
        const T* x = path + sample * path_stride;
        auto point = [&](unsigned t) {
            if constexpr (Augmented) return point_value(x, t, c, dimension, s.n + 1, ta, ll, end_time);
            else return x[t * d + c];
        };
        const T u = point(i), next = point(i + 1), v = next - u;
        centres[base + uint64_t(c) * s.pitch + i] = v;
        const T first = point(0), last = point(s.n);
        const T pm = (u - first) + v / 2, qm = (last - next) + v / 2;
        for (unsigned r = 0; r <= s.q.depth; ++r) {
            if (r <= s.p.depth) {
                T* a = pre + base + uint64_t(s.p.first[r]) * s.pitch + i;
                if (!c) *a = 1;
                if (r < s.p.depth) a[(1 + c) * s.pitch] = pm;
            }
            T* b = suf + base + uint64_t(s.q.first[r]) * s.pitch + i;
            if (!c) *b = 1;
            if (r < s.q.depth) b[(1 + c) * s.pitch] = qm;
        }
        if (low && !i) {
            low[sample * sig_stride + scalar + c] = last - first;
            if (scalar && !c) low[sample * sig_stride] = 1;
        }
    }
}

template<typename T>
__global__ void scan_centres(T* centres, const uint4* words, Degrees l, ScanShape s,
    uint64_t batch, uint64_t stride) {
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, tiles = s.pitch / 32;
    for (uint64_t task = blockIdx.x; task < batch * tiles; task += gridDim.x) {
        const unsigned i = unsigned(task % tiles) * 32 + lane;
        T* v = centres + task / tiles * stride + i;
        for (unsigned r = 1, first = l.d; r < l.a && r <= l.b; ++r) {
            const unsigned end = words[first].w;
            if (i < s.n) for (unsigned g = first + warp; g < end; g += blockDim.x / 32) {
                const uint4 e = words[g];
                v[uint64_t(g) * s.pitch] = ((v[uint64_t(e.x) * s.pitch] * T(l.scale[r]))
                    * v[uint64_t(e.y) * s.pitch]) * v[uint64_t(e.z) * s.pitch];
            }
            __syncthreads();
            first = end;
        }
    }
}

template<unsigned Threads, typename T>
__device__ T scan_segments(T x, T& carry, unsigned lane, T* totals) {
    if constexpr (Threads == 32) return scan_tile(x, carry, lane);
    const unsigned warp = threadIdx.x / 32;
    for (unsigned k = 1; k < 32; k *= 2) {
        const T v = __shfl_up_sync(0xffffffff, x, k);
        if (lane >= k) x += v;
    }
    if (lane == 31) totals[warp] = x;
    __syncthreads();
    T sum = lane < Threads / 32 ? totals[lane] : T(0);
    for (unsigned k = 1; k < Threads / 32; k *= 2) {
        const T v = __shfl_up_sync(0xffffffff, sum, k);
        if (lane >= k) sum += v;
    }
    const T prefix = __shfl_sync(0xffffffff, sum, (warp + 31) % 32);
    const T prev = __shfl_up_sync(0xffffffff, x, 1);
    const T before = (carry + (warp ? prefix : T(0))) + (lane ? prev : T(0));
    carry += __shfl_sync(0xffffffff, sum, Threads / 32 - 1);
    __syncthreads();
    return before;
}

// The integral of mode zero is an exclusive segment scan. Each higher
// Legendre moment corrects the preceding one with a word two letters shorter.
template<unsigned Threads, unsigned Items = 1, typename T>
__global__ void scan_level(T* pre, T* suf, const T* inc, T* low, ScanShape s, Degrees l,
    unsigned k, uint64_t batch, uint64_t stride, uint64_t sig_stride, bool scalar) {
    const unsigned width = l.power[k], prev = l.power[k - 1], span = l.power[k - 2];
    const unsigned tasks = width * (1 + (k <= s.p.depth)), lane = threadIdx.x % 32;
    __shared__ T totals[Threads / 32];
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Threads; task < batch * tasks;
        task += uint64_t(gridDim.x) * blockDim.x / Threads) {
        const uint64_t sample = task / tasks, base = sample * stride;
        const unsigned pos = unsigned(task % tasks), w = pos % width;
        const bool suffix = pos < width;
        const auto& side = suffix ? s.q : s.p;
        T* data = (suffix ? suf : pre) + base;
        const unsigned c = suffix ? w / prev : w % l.d, parent = suffix ? w % prev : w / l.d;
        const unsigned c2 = suffix ? parent / span : parent % l.d, stem = suffix ? parent % span : parent / l.d;
        const T* lower = data + uint64_t(l.offset[k - 2] + stem) * s.pitch;
        const T* a = data + uint64_t(l.offset[k - 1] + parent) * s.pitch;
        T* row = data + uint64_t(l.offset[k] + w) * s.pitch;
        T carry = 0;
        for (unsigned first = 0; first < s.n; first += Threads * Items) {
            const unsigned start = first + threadIdx.x % Threads * Items;
            T z[Items], subtotal = 0;
            #pragma unroll
            for (unsigned j = 0; j < Items; ++j) {
                const unsigned t = start + j, i = suffix ? s.n - 1 - t : t;
                z[j] = t < s.n ? a[i] * inc[base + uint64_t(c) * s.pitch + i] : T(0);
                subtotal += z[j];
            }
            T before = scan_segments<Threads>(subtotal, carry, lane, totals);
            #pragma unroll
            for (unsigned j = 0; j < Items && start + j < s.n; ++j) {
                const unsigned i = suffix ? s.n - 1 - start - j : start + j;
                const T v = inc[base + uint64_t(c) * s.pitch + i];
                const T u = inc[base + uint64_t(c2) * s.pitch + i];
                T value = (before + z[j] / 2) - ((lower[uint64_t(side.first[1]) * s.pitch + i] / 12) * u) * v;
                row[i] = value;
                for (unsigned r = 1; r <= side.depth - k; ++r) {
                    const T weight = T(l.scale[r + 1]);
                    value += ((weight * lower[uint64_t(side.first[r + 1]) * s.pitch + i]) * u) * v;
                    row[uint64_t(side.first[r]) * s.pitch + i] = value;
                }
                before += z[j];
            }
        }
        if (low && !suffix && threadIdx.x % Threads == 0) low[sample * sig_stride + l.offset[k] - !scalar + w] = carry;
    }
}

template<bool Backward, typename T>
__global__ void scan_left(T* left, T* dh, T* dc, const T* head, const T* centres, Mode<T> m,
    ScanShape s, uint64_t batch, uint64_t stride) {
    const unsigned rows = Backward ? m.head + m.groups : m.head * m.groups;
    for (uint64_t id = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; id < batch * rows * s.pitch;
        id += uint64_t(gridDim.x) * blockDim.x) {
        const unsigned i = unsigned(id % s.pitch), row = unsigned(id / s.pitch % rows);
        if (i >= s.n) continue;
        const uint64_t base = id / s.pitch / rows * stride + i;
        if constexpr (!Backward) left[base + uint64_t(row) * s.pitch] =
            head[base + uint64_t(row / m.groups) * s.pitch] * centres[base + uint64_t(row % m.groups) * s.pitch];
        else if (row < m.head) {
            T sum = 0;
            for (unsigned g = 0; g < m.groups; ++g)
                sum += left[base + uint64_t(row * m.groups + g) * s.pitch] * centres[base + uint64_t(g) * s.pitch];
            dh[base + uint64_t(row) * s.pitch] = sum;
        } else {
            const unsigned g = row - m.head;
            T sum = 0;
            for (unsigned x = 0; x < m.head; ++x)
                sum += left[base + uint64_t(x * m.groups + g) * s.pitch] * head[base + uint64_t(x) * s.pitch];
            dc[base + uint64_t(g) * s.pitch] = sum;
        }
    }
}

template<bool Backward, typename T>
__global__ void scan_constant_mode(T* grad, T* product, const T* centre, unsigned rows,
    ScanShape s, uint64_t batch, uint64_t stride) {
    const unsigned lane = threadIdx.x % 32;
    for (uint64_t row = uint64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32; row < batch * rows;
        row += uint64_t(gridDim.x) * (blockDim.x / 32)) {
        const uint64_t base = row / rows * stride;
        const unsigned g = unsigned(row % rows);
        const uint64_t start = base + uint64_t(g) * s.pitch;
        if constexpr (Backward) {
            const T seed = product[base + g];
            for (unsigned i = lane; i < s.n; i += 32) grad[start + i] = seed;
        } else {
            T sum = 0;
            for (unsigned i = lane; i < s.n; i += 32) sum += centre[start + i];
            for (unsigned k = 16; k; k /= 2) sum += __shfl_down_sync(0xffffffff, sum, k);
            if (!lane) product[base + g] = sum;
        }
    }
}

template<unsigned D, unsigned Threads, unsigned Items = 1, typename T>
__global__ void scan_reverse_level(const T* pre, const T* suf, T* dp, T* dq, const T* inc,
    T* zp, T* zq, T* cp, T* cq, const T* seed, ScanShape s, Degrees l, unsigned k,
    uint64_t batch, uint64_t stride, uint64_t sig_stride, bool scalar) {
    const unsigned d = D ? D : l.d, width = l.power[k], lane = threadIdx.x % 32;
    const unsigned tasks = width * (1 + (k <= s.p.depth));
    __shared__ T totals[Threads / 32];
    for (uint64_t task = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Threads; task < batch * tasks;
        task += uint64_t(gridDim.x) * blockDim.x / Threads) {
        const uint64_t sample = task / tasks, base = sample * stride;
        const unsigned pos = unsigned(task % tasks), w = pos % width, index = l.offset[k] + w;
        const bool suffix = pos < width;
        const auto& side = suffix ? s.q : s.p;
        const T* state = (suffix ? suf : pre) + base;
        T* grad = (suffix ? dq : dp) + base; T* z = (suffix ? zq : zp) + base;
        T* corr = (suffix ? cq : cp) + base;
        const unsigned stem = k < 2 ? 0 : suffix ? w % l.power[k - 2] : w / (d * d);
        T carry = suffix ? 0 : seed[sample * sig_stride + index - !scalar];
        for (unsigned first = 0; first < s.n; first += Threads * Items) {
            const unsigned start = first + threadIdx.x % Threads * Items;
            T values[Items]{}, subtotal = 0;
            #pragma unroll
            for (unsigned j = 0; j < Items && start + j < s.n; ++j) {
                const unsigned t = start + j, i = suffix ? t : s.n - 1 - t;
                T sum = 0, gpair = 0;
                const T* v = inc + base + i;
                for (int mode = int(side.depth - k); mode >= 0; --mode) {
                    const unsigned r = unsigned(mode);
                    T* g = grad + uint64_t(side.first[r] + index) * s.pitch + i;
                    // Only the highest prefix degree of each mode enters the left matrix.
                    if (suffix ? r < l.a : r + k == side.depth) sum += *g;
                    if (!r && k < side.depth) {
                        for (unsigned c = 0; c < d; ++c) {
                            const unsigned child = l.offset[k + 1] + (suffix ? c * width + w : w * d + c);
                            sum += z[uint64_t(child) * s.pitch + i] * v[c * s.pitch];
                        }
                    }
                    if (r && r < side.depth - k) {
                        const T scale = T(l.scale[r]);
                        const T* upper = grad + uint64_t(side.first[r - 1] + l.offset[k + 2]) * s.pitch + i;
                        for (unsigned c = 0; c < d; ++c) {
                            T row = 0;
                            for (unsigned y = 0; y < d; ++y) {
                                const unsigned child = suffix ? (c * d + y) * width + w : (w * d + c) * d + y;
                                row += upper[uint64_t(child) * s.pitch] * v[y * s.pitch];
                            }
                            sum += (scale * row) * v[c * s.pitch];
                        }
                    }
                    *g = sum;
                    if (k >= 2) {
                        const T scale = T(l.scale[r + 1]);
                        gpair += (scale * sum) * state[uint64_t(side.first[r + 1] + l.offset[k - 2] + stem) * s.pitch + i];
                    }
                }
                if (k >= 2) corr[uint64_t(index) * s.pitch + i] = gpair;
                values[j] = sum;
                subtotal += sum;
            }
            T before = scan_segments<Threads>(subtotal, carry, lane, totals);
            #pragma unroll
            for (unsigned j = 0; j < Items && start + j < s.n; ++j) {
                const unsigned t = start + j, i = suffix ? t : s.n - 1 - t;
                z[uint64_t(index) * s.pitch + i] = before + values[j] / 2;
                before += values[j];
            }
        }
    }
}

template<unsigned D, typename T>
__global__ void scan_increment_gradient(const T* pre, const T* suf, const T* zp, const T* zq,
    const T* cp, const T* cq, const T* centres, const T* dc, T* out, const uint4* terms,
    const unsigned* offsets, ScanShape s, Degrees l, uint64_t batch, uint64_t stride) {
    constexpr unsigned warps = 8;
    __shared__ T partial[warps][32];
    const unsigned lane = threadIdx.x % 32, worker = threadIdx.x / 32, d = D ? D : l.d, tiles = s.pitch / 32;
    for (uint64_t task = blockIdx.x; task < batch * d * tiles; task += gridDim.x) {
        const uint64_t base = task / tiles / d * stride;
        const unsigned c = unsigned(task / tiles % d), i = unsigned(task % tiles) * 32 + lane;
        T sum = 0;
        if (i < s.n) {
            const T* v = centres + base + i; const T* centre_grad = dc + base + i;
            for (unsigned e = offsets[c] + worker; e < offsets[c + 1]; e += warps) {
                const uint4 term = terms[e];
                sum += ((centre_grad[uint64_t(term.x) * s.pitch] * T(l.scale[term.w / 32]))
                    * v[uint64_t(term.y) * s.pitch]) * v[uint64_t(term.z) * s.pitch] * T(term.w % 32);
            }
            const uint64_t start = base + i;
            if (!worker) sum += centre_grad[c * s.pitch] + zp[start + (1 + c) * s.pitch] + zq[start + (1 + c) * s.pitch];
            for (unsigned k = 2; k <= s.q.depth; ++k) {
                const unsigned span = l.power[k - 2], prev = l.power[k - 1];
                for (unsigned parent = worker; parent < prev; parent += warps) {
                    const uint64_t lower = start + uint64_t(l.offset[k - 1] + parent) * s.pitch;
                    const unsigned x = parent / span, stem = parent % span;
                    const uint64_t row = start + uint64_t(l.offset[k] + c * prev + parent) * s.pitch;
                    const uint64_t other = start + uint64_t(l.offset[k] + x * prev + c * span + stem) * s.pitch;
                    sum += zq[row] * suf[lower] + v[x * s.pitch] * (cq[row] + cq[other]);
                    if (k <= s.p.depth) {
                        const unsigned y = parent % d;
                        const uint64_t a = start + uint64_t(l.offset[k] + parent * d + c) * s.pitch;
                        const uint64_t b = start + uint64_t(l.offset[k] + (parent / d * d + c) * d + y) * s.pitch;
                        sum += zp[a] * pre[lower] + v[y * s.pitch] * (cp[a] + cp[b]);
                    }
                }
            }
        }
        partial[worker][lane] = sum;
        __syncthreads();
        if (!worker && i < s.n) {
            for (unsigned w = 1; w < warps; ++w) sum += partial[w][lane];
            out[base + uint64_t(i) * d + c] = sum;
        }
        __syncthreads();
    }
}
