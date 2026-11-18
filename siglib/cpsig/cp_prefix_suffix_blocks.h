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

// Included in the active Highway namespace.
struct CpuMomentScan {
    LegendrePlan plan;
    MomentRows p, q;
    uint64_t tile, size;

    CpuMomentScan(const Layout& l, uint64_t n, size_t bytes, bool backward, int jobs, unsigned split = 0, unsigned lead_channels = 0)
        : plan(l, split, backward, lead_channels), p(l, plan.a - 1), q(l, plan.b),
        tile(std::min(n, std::max<uint64_t>(128, plan.tile(n, bytes, backward) * std::min(jobs, 4)))) {
        const uint64_t moments = mul(n, p.size + q.size + plan.groups);
        size = add(plan.buffers[0] + plan.buffers[1], add(mul(1 + backward, moments), mul(tile, plan.max_rows)));
    }
};

template<bool Backward, typename T>
bool blocked_legendre_batch(const T* path, const T* seed, T* out, uint64_t batch, uint64_t n,
    bool scalar, int jobs, int inner, const Layout& l, uint64_t sig_stride, uint64_t work, uint64_t limit,
    uint64_t dimension, bool ta, bool ll, T end_time) {
    if (n < 1024 || l.m < 7) return false;
    const bool wide = (l.power[l.m] - 1) / (Backward ? 32 : 16) >= n;
    if (wide && (Backward ? work <= limit : inner < 4 || n < 2048 || (l.power[l.m] - 1) / 128 >= n)) return false;
    const bool cache_blocks = sizeof(T) == sizeof(double) && inner <= 2 && (l.power[l.m] - 1) / 8 < n;
    const uint64_t target = work > limit ? std::min<uint64_t>((32ULL << 20) / sizeof(T), limit / 2)
        : cache_blocks ? (1ULL << 20) / sizeof(T) : work;
    if (!target) return false;
    uint64_t blocks = std::bit_ceil(std::min<uint64_t>(256, (work - 1) / target + 1));
    blocks = std::max(blocks, std::bit_floor(std::min({uint64_t(inner), n / 256, uint64_t(32)})));
    if (wide) blocks = std::min<uint64_t>(4, blocks);
    if (blocks < 2 || n / blocks < 128) return false;
    const int tile_jobs = wide ? std::max(1, inner / int(blocks)) : 1;
    const uint64_t length = (n - 1) / blocks + 1, vertices = (ll ? n / 2 : n) + 1;
    const uint64_t path_stride = vertices * dimension, transformed = ta || ll ? mul(n + 1, l.d) : 0;
    // The transpose of Chen multiplication only reads child signatures through degree m-1.
    const Layout child(l.d, l.m - Backward);
    const unsigned lead_channels = ll ? unsigned(dimension) : 0;
    const CpuMomentScan reverse(l, length, sizeof(T), Backward, tile_jobs, 0, lead_channels);
    std::optional<CpuMomentScan> lower;
    if constexpr (Backward) lower.emplace(child, length, sizeof(T), false, tile_jobs, 0, lead_channels);
    const CpuMomentScan& leaf = Backward ? *lower : reverse;
    const uint64_t full = l.offset[l.m + 1], width = child.offset[child.m + 1];
    const uint64_t scratch = std::max(leaf.size, Backward ? reverse.size + (length + 1) * l.d : 0);
    const uint64_t tree_size = blocks * (width + (Backward ? width + full + l.d : 0));
    const uint64_t states_size = add(tree_size, mul(1 + Backward, transformed));
    if (states_size >= limit || scratch > limit - states_size) return false;
    const uint64_t workers = std::min({uint64_t(inner), blocks, (limit - states_size) / scratch});
    const int leaf_jobs = Backward && wide ? inner / int(workers) : tile_jobs;
    const uint64_t size = workers * scratch + states_size;
    for_paths<T>(batch, jobs, size, "CPU prefix-suffix", [&](uint64_t sample, T* memory) {
        const T* input = path + sample * path_stride;
        T* result = out + sample * (Backward ? path_stride : sig_stride);
        T* states = memory + workers * scratch;
        T* leaves = Backward ? states + blocks * width : states;
        T* packed = states + tree_size;
        if (transformed) {
            transform_path_(input, packed, dimension, vertices, ta, ll, end_time);
            input = packed;
            if constexpr (Backward) result = packed + transformed;
        }
        auto start = [&](uint64_t b) { return b * (n / blocks) + std::min(b, n % blocks); };
        parallel_range(workers, inner, [&](uint64_t first, uint64_t end) {
            for (uint64_t worker = first; worker < end; ++worker) for (uint64_t b = worker; b < blocks; b += workers) {
                const uint64_t pos = start(b), count = start(b + 1) - pos;
                scanned_legendre_path<false>(input + pos * l.d, static_cast<const T*>(nullptr),
                    leaves + b * width, true, child, leaf.plan, leaf.p, leaf.q,
                    count, std::min(count, leaf.tile), leaf_jobs, memory + worker * scratch);
            }
        });
        for (uint64_t nodes = blocks / 2; nodes > unsigned(Backward); nodes /= 2) {
            const uint64_t step = blocks / (2 * nodes);
            parallel_range(nodes, inner, [&](uint64_t first, uint64_t end) {
                for (uint64_t b = first; b < end; ++b) {
                    if constexpr (Backward) {
                        const uint64_t i = nodes + b;
                        T* parent = states + i * width;
                        std::copy_n(states + 2 * i * width, width, parent);
                        sig_combine_inplace_(parent, states + (2 * i + 1) * width, child.m, child.offset.data());
                    } else {
                        T* left = leaves + 2 * b * step * width;
                        sig_combine_inplace_(left, left + step * width, child.m, child.offset.data());
                    }
                }
            });
        }
        if constexpr (!Backward) {
            std::copy_n(leaves + !scalar, sig_stride, result);
        } else {
            T* grads = states + 2 * blocks * width;
            T* endpoints = grads + blocks * full;
            grads[0] = 0;
            std::copy_n(seed + sample * sig_stride + scalar, full - 1, grads + 1);
            for (uint64_t nodes = 1; nodes < blocks; nodes *= 2) {
                const uint64_t step = blocks / nodes;
                parallel_range(nodes, inner, [&](uint64_t first, uint64_t end) {
                    for (uint64_t b = first; b < end; ++b) {
                        const uint64_t i = nodes + b;
                        uncombine_sig_deriv(states + 2 * i * width, states + (2 * i + 1) * width,
                            grads + b * step * full, grads + (b * step + step / 2) * full, l.d, l.m, l.offset.data());
                    }
                });
            }
            parallel_range(workers, inner, [&](uint64_t first, uint64_t end) {
                for (uint64_t worker = first; worker < end; ++worker) for (uint64_t b = worker; b < blocks; b += workers) {
                    const uint64_t pos = start(b), count = start(b + 1) - pos;
                    T* work = memory + worker * scratch;
                    T* grad = work + reverse.size;
                    scanned_legendre_path<true>(input + pos * l.d, grads + b * full, grad, true, l,
                        reverse.plan, reverse.p, reverse.q, count, std::min(count, reverse.tile), leaf_jobs, work);
                    std::copy_n(grad, l.d, endpoints + b * l.d);
                    std::copy_n(grad + l.d, count * l.d, result + (pos + 1) * l.d);
                }
            });
            std::copy_n(endpoints, l.d, result);
            for (uint64_t b = 1; b < blocks; ++b)
                vec_mult_add(result + start(b) * l.d, endpoints + b * l.d, T(1), l.d);
            if (transformed) transform_path_backprop_(result, out + sample * path_stride, dimension, vertices, ta, ll, end_time);
        }
    });
    return true;
}
