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
    unsigned d, a, b;
    unsigned power[17], offset[18];
    double scale[8];
};

struct Mode {
    unsigned head, groups, cols, h, depth, first;
};

__device__ unsigned level(unsigned index, const Degrees& l) {
    unsigned k = 0;
    while (index >= l.offset[k + 1]) ++k;
    return k;
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
                if (id != UINT_MAX) value += upper[sample * stride + (row / letters * groups + id) * cols + l.offset[k - 1] + word % width_right];
            }
            output[pos] = value;
        }
    }
}

// One warp copies a factor row in signature level order.
template<bool Top, bool Backward, typename T>
__global__ void map_rows(T* product, T* upper, const T* seed, T* output, const unsigned* grow,
    Mode mode, Mode next, Degrees l, uint64_t batch, uint64_t sig_stride,
    uint64_t stride, bool scalar, int* flag, T limit) {
    const unsigned rows = mode.head * mode.groups, letters = l.d * mode.groups;
    const unsigned lane = threadIdx.x % 32;
    for (uint64_t task = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32; task < batch * rows; task += uint64_t(gridDim.x) * 4) {
        const uint64_t sample = task / rows, base = sample * stride;
        const unsigned row = unsigned(task % rows);
        T* packed = product + base + row * mode.cols;
        if constexpr (Top) {
            const uint64_t pos = sample * sig_stride + l.offset[l.a] - !scalar + row;
            if (!lane) {
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
                    for (unsigned x = lane; x < width; x += 32) {
                        const T value = seed[pos + col + x];
                        packed[off + col + x] = value;
                        if (!(abs(value) <= limit)) atomicExch(flag, 1);
                    }
                } else {
                    const unsigned group = grow[(row % letters) * l.d + y];
                    const T* higher = group == UINT_MAX ? nullptr : upper + base + (row / letters * next.groups + group) * next.cols + l.offset[k - 1];
                    for (unsigned x = lane; x < width; x += 32) {
                        const T value = packed[off + col + x] + (higher ? higher[x] : T(0));
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
    const unsigned* members, Mode m, Mode next, Degrees l, uint64_t batch, uint64_t stride) {
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
            if (group == UINT_MAX) continue;
            current[base + idx] += upper[base + (row / letters * next.groups + group) * next.cols
                + l.offset[k - 1] + w % size_right];
        }
    }
}

// Remove one occurrence of the differentiated letter from each symmetric centre.
struct CentreDerivatives {
    std::vector<uint4> terms;
    std::vector<unsigned> offsets{0};

    CentreDerivatives(const LegendrePlan& plan, unsigned d) {
        std::vector<std::vector<uint4>> by_channel(d);
        for (unsigned r = 1; r < plan.modes.size(); ++r) {
            const auto& m = plan.modes[r];
            const auto& prev = plan.modes[r - 1];
            for (unsigned g = 0; g < m.groups; ++g) {
                const unsigned word = m.words[g], length = 2 * r + 1;
                unsigned prefix = word, place = 1;
                for (unsigned i = 0; i < length;) {
                    const unsigned c = prefix % d, remaining = (prefix / d) * place + word % place;
                    const unsigned first = i;
                    do { prefix /= d; place *= d; ++i; } while (i < length && prefix % d == c);
                    const auto parent = std::lower_bound(prev.words.begin(), prev.words.end(), remaining / d);
                    by_channel[c].push_back(make_uint4(m.first_group + g, prev.first_group + unsigned(parent - prev.words.begin()),
                        remaining % d, r * 32 + i - first));
                }
            }
        }
        for (const auto& channel : by_channel) offsets.push_back(offsets.back() + unsigned(channel.size()));
        terms.reserve(offsets.back());
        for (const auto& channel : by_channel) terms.insert(terms.end(), channel.begin(), channel.end());
    }
};

#include "cu_prefix_suffix_moment_scan.cuh"

template<bool Backward, typename T>
bool blocked(const T* path, const T* seed, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream, uint64_t dimension, bool ta, bool ll, T end_time, uint64_t work);

template<bool Backward, typename T>
bool compute(const T* path, const T* seed, T* out, const Layout& l, int device, uint64_t batch,
    uint64_t n, bool scalar, cudaStream_t stream, uint64_t dimension, bool ta, bool ll, T end_time,
    uint64_t reserved_bytes = 0, unsigned packed_lead_channels = 0) {
    const bool augmented = ta || ll;
    const bool leaf = reserved_bytes != 0;
    if (l.m < 4 || l.m > 16 || l.power[l.m] > (1ULL << 25) || n < 8)
        return false;
    if (n > (192ULL << 20) / (sizeof(T) * l.offset[l.h + 1]))
        return !leaf && blocked<Backward>(path, seed, out, l, device, batch, n, scalar, stream, dimension,
            ta, ll, end_time, mul(mul(n, l.offset[l.h + 1]), Backward ? 12 : 5));
    if constexpr (std::is_same_v<T, float>) {
        if (n < 32 && ((l.d >= 6 && l.m >= 7) || (!Backward && l.d == 3 && l.m >= 8))) return false;
        if (l.m != 5 && l.d >= 10 && l.power[l.m] >= (1u << 18)) return false;
    } else if (l.d >= 24 && n <= 32) return false;
    if constexpr (!Backward) {
        if (!leaf && !augmented && l.m <= 5 && l.d <= 4
            && n * (l.m - 2) * (l.power[3] + l.offset[l.m - 2]) <= (128ULL << 20) / sizeof(T)) {
            if (l.m == 5) small_forward<true>(path, out, l, device, batch, n, scalar, stream);
            else small_forward<false>(path, out, l, device, batch, n, scalar, stream);
            return true;
        }
    }
    // A shorter prefix removes the highest centre mode at odd depths.
    const bool wide = l.d >= 5 && l.m >= 5;
    const bool middle = l.m == 7 && n >= 128 && n <= 512 &&
        (l.d == 4 || (!leaf && l.d == 3 && (Backward || sizeof(T) == sizeof(double))));
    const bool deep = l.m >= 9 && (l.d >= 4 || (l.d == 3 && (sizeof(T) == sizeof(float) || l.m >= 11))
        || (l.d == 2 && !Backward && n >= 4096));
    const bool shorter = l.m % 2 && (wide || middle || deep);
    const LegendrePlan plan(l, shorter ? unsigned(l.m / 2) : 0, Backward, ll ? unsigned(dimension) : packed_lead_channels);
    const uint64_t q = l.offset[plan.b + 1], sig_stride = l.offset[l.m + 1] - !scalar;
    const uint64_t length = (ll ? n / 2 : n) + 1, path_stride = length * dimension;
    const ScanShape scan(l, plan, unsigned(n));
    const bool native_products = !Backward && sizeof(T) == sizeof(float) && batch == 1 && n >= 8192 && l.d == 3 && l.m <= 9;
    const bool fuse_products = !native_products && !Backward && (sizeof(T) == sizeof(float) || batch * n >= 1024);
    const bool alternating = ll && !ta && (sizeof(T) == sizeof(double) || n >= 1024);
    const uint64_t fused_area = alternating ? (sizeof(T) == 8 ? 1u << 20 : 65536) : 16384;
    uint64_t left_rows = 0, split_size = 0;
    for (const auto& m : plan.modes) {
        const bool fused = fuse_products && m.head > 1 && m.size() <= fused_area;
        if (m.head > 1 && !fused) left_rows = std::max(left_rows, m.rows());
        if (!Backward && !native_products) {
            const uint64_t samples = fused && alternating ? n / 2 : n;
            split_size = std::max(split_size, gpu_split_k::chunks(samples, m.rows(), m.cols) * m.size());
        }
    }
    const uint64_t packed = uint64_t(scan.pitch) * (scan.p.size + scan.q.size + plan.groups);
    const uint64_t first_buffer = (plan.buffers[0] + 31) / 32 * 32;
    const uint64_t products = first_buffer + plan.buffers[1] + split_size + scan.pitch * left_rows;
    const uint64_t reverse_size = Backward ? 2 * uint64_t(scan.pitch) * (l.offset[scan.p.depth + 1] + q) : 0;
    const uint64_t scan_stride = (1 + Backward) * packed + std::max(products, reverse_size);
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX);
    const bool long_path = batch <= 4 && n >= 16384 && scan_stride > (96ULL << 20) / sizeof(T)
        && (Backward || sizeof(T) == sizeof(float) || l.d == 2 || l.m >= 9);
    const uint64_t available = budget - std::min<uint64_t>(budget, 4ULL << 20);
    const uint64_t limit = std::min<uint64_t>(192ULL << 20, available) / sizeof(T);
    if (!leaf && (long_path || scan_stride > limit) && blocked<Backward>(path, seed, out, l, device, batch, n,
        scalar, stream, dimension, ta, ll, end_time, scan_stride)) return true;
    if (scan_stride > limit) return false;
    const uint64_t stride = (scan_stride + 63) / 64 * 64;
    Degrees levels{};
    levels.a = plan.a; levels.b = plan.b; levels.d = unsigned(l.d);
    for (unsigned k = 0; k <= l.m; ++k) levels.power[k] = unsigned(l.power[k]);
    for (unsigned k = 0; k <= l.m + 1; ++k) levels.offset[k] = unsigned(l.offset[k]);
    std::vector<uint4> words(l.d);
    std::vector<unsigned> maps;
    struct Indices { size_t grow, offsets, members; };
    std::vector<Indices> starts;
    std::vector<Mode> modes;
    for (unsigned r = 1; r < scan.q.depth || r < plan.modes.size(); ++r)
        levels.scale[r] = -1.0 / (4 * (2 * r - 1) * (2 * r + 1));
    for (unsigned r = 0; r < plan.modes.size(); ++r) {
        const auto& m = plan.modes[r];
        if (r) {
            const auto& prev = plan.modes[r - 1];
            for (auto ex : m.parent) {
                const unsigned pair = ex / levels.d;
                words.push_back(make_uint4(prev.first_group + pair % prev.groups, pair / prev.groups,
                    ex % levels.d, m.first_group + m.groups));
            }
        }
        auto append = [&](const std::vector<unsigned>& data) {
            const auto pos = maps.size();
            maps.insert(maps.end(), data.begin(), data.end());
            return pos;
        };
        starts.push_back({Backward ? 0 : append(m.grow), append(m.offsets), append(m.members)});
        modes.push_back({m.head, m.groups, m.cols, plan.a - 1 - r, m.depth, m.first_group});
    }
    std::vector<uint4> derivatives;
    const size_t derivative_offsets = maps.size();
    if constexpr (Backward) {
        CentreDerivatives table(plan, levels.d);
        derivatives = std::move(table.terms);
        maps.insert(maps.end(), table.offsets.begin(), table.offsets.end());
    }
    const uint64_t metadata = derivatives.size() * sizeof(uint4)
        + words.size() * sizeof(uint4) + maps.size() * sizeof(unsigned) + sizeof(int);
    if (add(add(reserved_bytes, mul(stride, sizeof(T))), (metadata + 255) / 256 * 256) > available)
        return false;
    std::vector<std::byte> packed_metadata((metadata + 255) / 256 * 256);
    auto* term_bytes = packed_metadata.data();
    auto* word_bytes = term_bytes + derivatives.size() * sizeof(uint4);
    if (!derivatives.empty()) std::memcpy(term_bytes, derivatives.data(), derivatives.size() * sizeof(uint4));
    std::memcpy(word_bytes, words.data(), words.size() * sizeof(uint4));
    if (!maps.empty()) std::memcpy(word_bytes + words.size() * sizeof(uint4), maps.data(), maps.size() * sizeof(unsigned));
    const uint64_t paths = std::clamp<uint64_t>((512ULL << 20) / (stride * sizeof(T)), 1, 32);
    int unsafe = 0;
    with_factor_workspace<T>(l, device, batch, paths, stride, packed_metadata.size(),
        4ULL << 20, true, "prefix-suffix moments", stream, false,
        [&](cublasHandle_t handle, Fp64Emulation*, uint64_t capacity, T* memory, uint64_t* metadata_buffer, uint64_t*) {
        auto* terms = reinterpret_cast<uint4*>(metadata_buffer);
        auto* ids = terms + derivatives.size();
        auto* indices = reinterpret_cast<unsigned*>(ids + words.size());
        auto* flag = reinterpret_cast<int*>(indices + maps.size());
        T* cursor = memory;
        auto take = [&](uint64_t size) { T* ptr = cursor; cursor += size; return ptr; };
        const uint64_t pitch = scan.pitch;
        T* pre = take(pitch * scan.p.size); T* suf = take(pitch * scan.q.size);
        T* centres = take(pitch * plan.groups);
        T *dp = nullptr, *dq = nullptr, *dc = nullptr;
        T *zp = nullptr, *zq = nullptr, *cp = nullptr, *cq = nullptr;
        if constexpr (Backward) {
            dp = take(pitch * scan.p.size); dq = take(pitch * scan.q.size);
            dc = take(pitch * plan.groups);
            const uint64_t pw = pitch * l.offset[scan.p.depth + 1], qw = pitch * q;
            // The reverse scan starts after the last matrix product on this stream.
            T* scratch = cursor;
            zp = take(pw); zq = take(qw); cp = take(pw); cq = take(qw);
            cursor = scratch;
        }
        T* left = take(pitch * left_rows);
        T* buf[]{take(first_buffer), take(plan.buffers[1])};
        T* partial = take(split_size);
        for (uint64_t base = 0; base < batch; base += capacity) {
            const uint64_t count = std::min(capacity, batch - base);
            const T* input = path + base * path_stride;
            const T* grad = Backward ? seed + base * sig_stride : nullptr;
            T* result = out + base * (Backward ? path_stride : sig_stride);
            auto grid = [&](uint64_t size) { return make_cuda_1d_grid(count * size, 256); };
            auto map_final = [&] {
                const T limit = (std::numeric_limits<T>::max)() / T(l.power[l.m] * l.m);
                const auto& m = plan.modes[0];
                if (m.cols >= 64) {
                    map_rows<true, Backward><<<make_cuda_1d_grid(count * m.rows(), 4), 128, 0, stream>>>(buf[0],
                        buf[1], grad, result, indices + starts[0].grow, modes[0], modes[1], levels, count,
                        sig_stride, stride, scalar, flag, limit);
                } else map_levels<Backward><<<grid(m.size()), 256, 0, stream>>>(buf[0], result, grad, levels,
                    count, sig_stride, stride, scalar, buf[1], indices + starts[0].grow, modes[1].groups,
                    modes[1].cols, flag, limit);
                CUDA_CHECK(cudaGetLastError());
            };
            auto start = [&]<bool Augmented>() {
                scan_start<Augmented><<<grid(l.d * pitch), 256, 0, stream>>>(input, centres, pre, suf, Backward ? nullptr : result,
                    scan, unsigned(l.d), count, stride, sig_stride, scalar, dimension, path_stride, ta, ll, end_time);
            };
            if (augmented) start.template operator()<true>();
            else start.template operator()<false>();
            CUDA_CHECK(cudaGetLastError());
            scan_centres<<<make_cuda_1d_grid(count * (pitch / 32), 1), 128, 0, stream>>>(
                centres, ids, levels, scan, count, stride);
            CUDA_CHECK(cudaGetLastError());
            for (unsigned k = 2; k <= scan.q.depth; ++k) {
                const uint64_t tasks = count * l.power[k] * (1 + (k <= scan.p.depth));
                if (n >= 1024) scan_level<128, 4><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf, centres,
                    Backward ? nullptr : result, scan, levels, k, count, stride, sig_stride, scalar);
                else if (n >= 512) scan_level<128><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf, centres,
                    Backward ? nullptr : result, scan, levels, k, count, stride, sig_stride, scalar);
                else scan_level<32><<<make_cuda_1d_grid(tasks, 4), 128, 0, stream>>>(pre, suf, centres,
                    Backward ? nullptr : result, scan, levels, k, count, stride, sig_stride, scalar);
                CUDA_CHECK(cudaGetLastError());
            }
            if constexpr (Backward) map_final();
            CudaMatrices<T> matrices{handle, count};
            auto transfer = [&]<bool Fold>(unsigned r) {
                const auto& m = plan.modes[Fold ? r + 1 : r];
                if (!Fold && m.cols >= 64) {
                    map_rows<false, false><<<make_cuda_1d_grid(count * m.rows(), 4), 128, 0, stream>>>(
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
                T* product = buf[r % 2];
                T* head = pre + (scan.p.first[r] + levels.offset[m.h]) * pitch;
                T* centre = centres + m.first * pitch;
                T* rhs = suf + scan.q.first[r] * pitch;
                if constexpr (!Backward) {
                    if (fuse_products && m.head > 1 && rows * cols <= fused_area) {
                        const uint64_t splits = std::max<uint64_t>(1, gpu_split_k::chunks(n / (1 + alternating), rows, cols));
                        auto multiply = [&]<bool Alternate>() {
                            gpu_split_k::gemm<true, true, Alternate>(head, rhs, partial, product, count, n, rows, cols,
                                pitch, splits, stride, stride, true, stream, centre, m.groups);
                        };
                        if (alternating) multiply.template operator()<true>();
                        else multiply.template operator()<false>();
                        if (step && r) transfer.template operator()<false>(r);
                        continue;
                    }
                }
                T* lhs = m.head == 1 ? centre : left;
                const bool fused_adjoint = Backward && std::is_same_v<T, double> && m.head > 1
                    && (alternating || n >= 64) && rows * cols <= (alternating ? 4u << 20 : 16384);
                if (m.head != 1 && !fused_adjoint) {
                    scan_left<false><<<make_cuda_1d_grid(count * rows * ((n + 511) / 512), 1), 128, 0, stream>>>(left, dp, dc, head, centre, m, scan, count, stride);
                    CUDA_CHECK(cudaGetLastError());
                }
                if (cols == 1) {
                    scan_constant_mode<Backward><<<make_cuda_1d_grid(count * rows, 4), 128, 0, stream>>>(
                        Backward ? dc + m.first * pitch : nullptr, product, centre, unsigned(rows), scan, count, stride);
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }
                if constexpr (Backward) {
                    T* dleft = m.head == 1 ? dc + m.first * pitch : left;
                    T* dright = dq + scan.q.first[r] * pitch;
                    if constexpr (std::is_same_v<T, double>) {
                        if (fused_adjoint) {
                            auto gradient = [&]<bool Right, bool Alternate>() {
                                const uint64_t tiles = ((Right ? cols : rows / (1 + Alternate)) + 15) / 16;
                                const uint64_t tasks = count * (1 + Alternate) * tiles * ((n / (1 + Alternate) + 127) / 128);
                                factor_gradient<Right, Alternate><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(product,
                                    Right ? head : rhs, centre, Right ? dright : dleft, m, scan, count, stride);
                                CUDA_CHECK(cudaGetLastError());
                            };
                            if (alternating) {
                                gradient.template operator()<true, true>();
                                gradient.template operator()<false, true>();
                            } else gradient.template operator()<true, false>();
                        }
                    }
                    if (!fused_adjoint) matrices.gemm({product, cols, rows, 1, cols, stride}, {lhs, rows, n, pitch, 1, stride},
                        {dright, cols, n, pitch, 1, stride}, true);
                    if (!(fused_adjoint && alternating)) matrices.gemm({product, rows, cols, cols, 1, stride},
                        {rhs, cols, n, pitch, 1, stride}, {dleft, rows, n, pitch, 1, stride}, true);
                    if (m.head != 1) {
                        scan_left<true><<<make_cuda_1d_grid(count * (m.head + m.groups) * ((n + 127) / 128), 1), 128, 0, stream>>>(dleft, dp + (head - pre),
                            dc + m.first * pitch, head, centre, m, scan, count, stride);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    if (r + 1 < modes.size()) transfer.template operator()<true>(r);
                } else {
                    const uint64_t splits = native_products ? 0 : gpu_split_k::chunks(n, rows, cols);
                    if (splits) gpu_split_k::gemm<true>(lhs, rhs, partial, product, count, n, rows, cols,
                        pitch, splits, stride, stride, true, stream);
                    else matrices.gemm({lhs, rows, n, pitch, 1, stride}, {rhs, n, cols, 1, pitch, stride},
                        {product, rows, cols, cols, 1, stride}, true);
                    if (step && r) transfer.template operator()<false>(r);
                }
            }
            if constexpr (Backward) {
                T* inc = dp;
                auto reverse = [&]<unsigned D>() {
                    for (unsigned k = scan.q.depth; k > 0; --k) {
                        const uint64_t tasks = count * l.power[k] * (1 + (k <= scan.p.depth));
                        if (n >= 1024) scan_reverse_level<D, 128, 4><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf,
                            dp, dq, centres, zp, zq, cp, cq, grad, scan, levels, k, count, stride, sig_stride, scalar);
                        else if (n >= 512) scan_reverse_level<D, 128><<<make_cuda_1d_grid(tasks, 1), 128, 0, stream>>>(pre, suf,
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
                    dimension, length, l.d, n, stride, ll);
                CUDA_CHECK(cudaGetLastError());
            } else map_final();
        }
        if constexpr (Backward) CUDA_CHECK(cudaMemcpyAsync(&unsafe, flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
    }, packed_metadata.data(), reserved_bytes);
    return !unsafe;
}

#include "cu_prefix_suffix_blocks.cuh"

} // namespace gpu_legendre
