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
template<bool Fold, typename T>
void legendre_expand(const LegendreMode& mode, const LegendreMode& next, const Layout& l,
    T* current, T* upper, int jobs) {
    const unsigned d = unsigned(l.d);
    if constexpr (Fold) {
        parallel_range(next.rows(), jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t row = first; row < end; ++row) {
                const unsigned id = unsigned(row % next.groups);
                T* dst = upper + row * next.cols;
                std::fill_n(dst, next.cols, T(0));
                for (unsigned t = next.offsets[id]; t < next.offsets[id + 1]; ++t) {
                    const unsigned ex = next.members[t];
                    const T* src = current + (row / next.groups * d * mode.groups + ex / d) * mode.cols;
                    for (unsigned k = 0; k <= next.depth; ++k) {
                        const uint64_t width = l.power[k];
                        vec_mult_add(dst + l.offset[k], src + l.offset[k + 1] + (ex % d) * width, T(1), width);
                    }
                }
            }
        });
    } else {
        parallel_range(mode.rows(), jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t row = first; row < end; ++row) {
                const unsigned pair = unsigned(row % (d * mode.groups));
                T* dst = current + row * mode.cols;
                for (unsigned y = 0; y < d; ++y) {
                    const unsigned id = mode.grow[pair * d + y];
                    if (id == UINT_MAX) continue;
                    const T* src = upper + (row / (d * mode.groups) * next.groups + id) * next.cols;
                    for (unsigned k = 0; k <= next.depth; ++k) {
                        const uint64_t width = l.power[k];
                        vec_mult_add(dst + l.offset[k + 1] + y * width, src + l.offset[k], T(1), width);
                    }
                }
            }
        });
    }
}

template<bool Backward, typename T>
void legendre_levels(const T* seed, T* out, bool scalar, const Layout& l,
    const LegendrePlan& plan, T* const buffers[2], int jobs) {
    const auto& base = plan.modes.front();
    const unsigned d = unsigned(l.d);
    for (unsigned s = 0; s <= plan.b; ++s) {
        const uint64_t width = l.power[s], level = l.offset[plan.a + s] - !scalar;
        parallel_range(base.rows(), jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t row = first; row < end; ++row) {
                T* packed = buffers[0] + row * base.cols + l.offset[s];
                if constexpr (Backward) std::copy_n(seed + level + row * width, width, packed);
                else if (!s) out[level + row] = packed[0];
                else {
                    const auto& next = plan.modes[1];
                    const unsigned pair = unsigned(row % (d * d));
                    const uint64_t tail = l.power[s - 1];
                    for (unsigned y = 0; y < d; ++y) {
                        const unsigned id = base.grow[pair * d + y];
                        if (id == UINT_MAX) {
                            std::copy_n(packed + y * tail, tail, out + level + row * width + y * tail);
                            continue;
                        }
                        const uint64_t group = row / (d * d) * next.groups + id;
                        vec_add_scaled(out + level + row * width + y * tail, packed + y * tail,
                            buffers[1] + group * next.cols + l.offset[s - 1], T(1), tail);
                    }
                }
            }
        });
    }
}

#include "cp_prefix_suffix_moment_scan.h"
#include "cp_prefix_suffix_blocks.h"

template<bool Backward, typename T>
bool legendre_batch(const T* path, const T* seed, T* out, uint64_t batch, uint64_t n,
    bool scalar, int jobs, const Layout& l, uint64_t sig_stride,
    uint64_t dimension, bool ta, bool ll, T end_time) {
    if (l.m < 4 || l.m > 16 || l.power[l.m] > (1ULL << 25) || n < 8
        || (l.m <= 6 && n > (256ULL << 20) / (sizeof(T) * l.offset[l.h + 1]))) return false;
    if (l.m >= 7 && n < 32 && (l.d >= 4 || (l.d == 3 && sizeof(T) == sizeof(float) && l.m >= 10))) return false;
    if (l.m == 4 && l.d >= 10 && n < 128) return false;
    const int available = std::max(1, available_threads(jobs) / int(std::min<uint64_t>(batch, jobs)));
    constexpr uint64_t grain = 131072;
    const int inner = int(std::min<uint64_t>(available, 1 + (mul(n, l.power[l.m]) - 1) / grain));
    const uint64_t length = (ll ? n / 2 : n) + 1, path_stride = length * dimension;
    const bool augmented = ta || ll;
    if constexpr (Backward) {
        const T limit = std::numeric_limits<T>::max() / T(l.power[l.m] * l.m);
        const uint64_t count = batch * sig_stride;
        std::atomic<bool> unsafe{false};
        auto check = [&](uint64_t first, uint64_t end) {
            const Simd<T> tag;
            const auto bound = hn::Set(tag, limit);
            constexpr size_t lanes = vec_batch_lanes<T>;
            for (; end - first >= lanes; first += lanes)
                if (!hn::AllTrue(tag, hn::Le(hn::Abs(hn::LoadU(tag, seed + first)), bound))) {
                    unsafe.store(true, std::memory_order_relaxed); return;
                }
            for (; first < end; ++first) if (!(std::abs(seed[first]) <= limit)) {
                unsafe.store(true, std::memory_order_relaxed); return;
            }
        };
        if (count >= 262144 && jobs > 1)
            get_tbb_arena(jobs).execute([&] { parallel_range(count, jobs, check); });
        else check(0, count);
        if (unsafe.load(std::memory_order_relaxed)) return false;
    }
    if (!augmented && l.m <= 6 && (l.d <= 4 || (l.d == 8 && l.m == 5))) {
        const uint64_t rows = l.power[3], cols = l.offset[l.m - 2], terms = std::min<uint64_t>(3, l.m - 2);
        const uint64_t block = std::min(n, std::clamp<uint64_t>((256ULL << 10) /
            (terms * (rows + cols) * sizeof(T) * (1 + Backward)), 32, 256));
        const uint64_t size = rows * cols + (1 + Backward) * terms * block * (rows + cols)
            + (uint64_t(l.m >= 5) + Backward) * n * l.power[2] + (l.m == 6 ? n * rows : 0);
        auto run = [&]<unsigned D>() {
            for_paths<T>(batch, jobs, size, "CPU prefix-suffix", [&](uint64_t i, T* memory) {
                const T* input = path + i * path_stride;
                T* output = out + i * (Backward ? path_stride : sig_stride);
                auto path_at = [&]<unsigned Degree>() {
                    low_moment_path<Backward, Degree, D>(input, Backward ? seed + i * sig_stride : nullptr,
                        output, n, scalar, block, inner, memory);
                };
                if (l.m == 6) path_at.template operator()<6>();
                else if (l.m == 5) path_at.template operator()<5>();
                else path_at.template operator()<4>();
            });
        };
        switch (l.d) {
        case 2: run.template operator()<2>(); break;
        case 3: run.template operator()<3>(); break;
        case 4: run.template operator()<4>(); break;
        case 8: run.template operator()<8>(); break;
        }
        return true;
    }
    const bool shorter = (l.m == 5 && l.d >= 5) || (l.m == 7 && (l.d >= 5 ||
        (l.d >= 3 && n >= 128 && n <= 512 && (!Backward || l.d == 4))));
    const CpuMomentScan scan(l, n, sizeof(T), Backward, inner, shorter ? unsigned(l.m / 2) : 0,
        ll ? unsigned(dimension) : 0);
    const uint64_t limit = std::min<uint64_t>(256ULL << 20, option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX)) / sizeof(T);
    if (blocked_legendre_batch<Backward>(path, seed, out, batch, n, scalar, jobs, inner, l, sig_stride,
        scan.size, limit, dimension, ta, ll, end_time)) return true;
    if (scan.size > limit) return false;
    for_paths<T>(batch, jobs, scan.size, "CPU prefix-suffix", [&](uint64_t i, T* memory) {
        const T* input = path + i * path_stride;
        const Path<T> source(input, dimension, length, ta, ll, end_time);
        scanned_legendre_path<Backward>(input, Backward ? seed + i * sig_stride : nullptr,
            out + i * (Backward ? path_stride : sig_stride), scalar, l, scan.plan, scan.p, scan.q,
            n, scan.tile, inner, memory, augmented ? &source : nullptr);
    });
    return true;
}
