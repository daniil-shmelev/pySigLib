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
template<bool Suffix, bool Reverse, typename T>
void legendre_moment(const T* state, const T* powers, T* result, T* dstate, T* dpowers,
    unsigned k, const LegendreMode& mode, const Layout& l) {
    const uint64_t width = l.power[k];
    if constexpr (Reverse) vec_mult_add(dstate + l.offset[k], result, T(1), width);
    else std::copy_n(state + l.offset[k], width, result);
    for (unsigned j = 1; j <= k; ++j) {
        const uint64_t terms = l.power[Suffix ? j : k - j], cols = width / terms;
        const T* a = (Suffix ? powers : state) + l.offset[Suffix ? j : k - j];
        const T* b = (Suffix ? state : powers) + l.offset[Suffix ? k - j : j];
        const T scale = T(mode.weight[j]);
        for (uint64_t x = 0; x < terms; ++x) {
            T* row = result + x * cols;
            if constexpr (Reverse) {
                T* da = (Suffix ? dpowers : dstate) + l.offset[Suffix ? j : k - j];
                T* db = (Suffix ? dstate : dpowers) + l.offset[Suffix ? k - j : j];
                da[x] += scale * dot_product_mult_add(b, row, db, scale * a[x], cols);
            } else vec_mult_add(row, b, scale * a[x], cols);
        }
    }
}

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
                        const uint64_t group = row / (d * d) * next.groups + base.grow[pair * d + y];
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
void legendre_path(const T* path, const T* seed, T* out, bool scalar, const Layout& l,
    const LegendrePlan& plan, uint64_t n, uint64_t block, int jobs, T* storage) {
    const unsigned d = unsigned(l.d), a = plan.a, b = plan.b;
    const uint64_t plen = l.offset[a], qlen = l.offset[b + 1], temp = std::max(plen, qlen);
    auto take = [&](uint64_t size) { T* p = storage; storage += size; return p; };
    T* buffers[]{take(plan.buffers[0]), take(plan.buffers[1])};
    T* p = take((n + 1) * plen); T* q = take((n + 1) * qlen);
    T* inc = take(n * d); T* mon = take(n * plan.groups);
    T* left = take(block * plan.max_rows); T* right = take(block * qlen);
    T *dp = nullptr, *dq = nullptr, *dv = nullptr, *dm = nullptr, *dl = nullptr, *dr = nullptr;
    if constexpr (Backward) {
        dp = take((n + 1) * plen); dq = take((n + 1) * qlen);
        dv = take(n * d); dm = take(n * plan.groups);
        std::fill(dp, storage, T(0));
        dl = take(block * plan.max_rows); dr = take(block * qlen);
    }
    T* scratch = storage;
    std::fill_n(p, plen, T(0)); p[0] = 1;
    std::fill_n(q + n * qlen, qlen, T(0)); q[n * qlen] = 1;
    for (uint64_t i = 0; i < n; ++i) {
        T* v = inc + i * d;
        for (unsigned c = 0; c < d; ++c) v[c] = path[(i + 1) * d + c] - path[i * d + c];
        T* state = p + (i + 1) * plen;
        std::copy_n(state - plen, plen, state);
        signature_horner_step_(state, v, d, a - 1, l.offset.data(), scratch);
    }
    for (uint64_t i = n; i-- > 0;) {
        T* state = q + i * qlen;
        std::copy_n(state + qlen, qlen, state);
        cpu_detail::prepend_generic(state, inc + i * d, l, scratch, b);
    }
    if constexpr (Backward) std::copy_n(seed + scalar, plen - 1, dq + 1);
    else std::copy(q + !scalar, q + plen, out);
    if constexpr (Backward) legendre_levels<true>(seed, out, scalar, l, plan, buffers, jobs);
    for (uint64_t i = 0; i < n; ++i) {
        const T* v = inc + i * d;
        T* m = mon + i * plan.groups;
        std::copy_n(v, d, m);
        for (unsigned r = 1; r < plan.modes.size(); ++r) {
            const auto& prev = plan.modes[r - 1]; const auto& mode = plan.modes[r];
            const T scale = -T(1) / T(4 * (2 * r - 1) * (2 * r + 1));
            for (unsigned g = 0; g < mode.groups; ++g) {
                const unsigned ex = mode.parent[g], pair = ex / d;
                m[mode.first_group + g] = ((m[prev.first_group + pair % prev.groups] * scale)
                    * v[pair / prev.groups]) * v[ex % d];
            }
        }
    }
    CpuMatrices<T> matrices{jobs};
    for (unsigned step = 0; step < plan.modes.size(); ++step) {
        const unsigned r = Backward ? step : unsigned(plan.modes.size()) - 1 - step;
        const auto& mode = plan.modes[r];
        const uint64_t rows = mode.rows(), cols = mode.cols;
        const unsigned degree = std::max(a - 1 - r, b - r);
        auto factors = [&]<bool Reverse>(uint64_t start, uint64_t count) {
            const uint64_t owners = std::min(count, uint64_t(jobs));
            parallel_range(owners, jobs, [&](uint64_t owner, uint64_t stop) {
                const uint64_t first = count * owner / owners, end = count * stop / owners;
                T* power = scratch + 4 * owner * temp;
                T* h = power + temp; T* dh = h + temp; T* dg = dh + temp;
                for (uint64_t j = first; j < end; ++j) {
                    const uint64_t i = start + j;
                    const T* v = inc + i * d;
                    cpu_detail::tensor_powers(v, power, degree, l);
                    const T* centre = mon + i * plan.groups + mode.first_group;
                    T* lrow = left + j * rows; T* rrow = right + j * cols;
                    legendre_moment<false, false>(p + i * plen, power, h,
                        static_cast<T*>(nullptr), static_cast<T*>(nullptr), a - 1 - r, mode, l);
                    if constexpr (Reverse) {
                        std::fill_n(dg, l.offset[degree + 1], T(0));
                        for (unsigned x = 0; x < mode.head; ++x)
                            dh[x] = dot_product_mult_add(centre, dl + j * rows + x * mode.groups,
                                dm + i * plan.groups + mode.first_group, h[x], mode.groups);
                        legendre_moment<false, true>(p + i * plen, power, dh, dp + i * plen, dg, a - 1 - r, mode, l);
                        for (unsigned k = 0; k <= mode.depth; ++k)
                            legendre_moment<true, true>(q + (i + 1) * qlen, power, dr + j * cols + l.offset[k],
                                dq + (i + 1) * qlen, dg, k, mode, l);
                        for (unsigned k = degree; k > 0; --k)
                            for (uint64_t word = 0; word < l.power[k - 1]; ++word)
                                dg[l.offset[k - 1] + word] += dot_product_mult_add(v, dg + l.offset[k] + word * d,
                                    dv + i * d, power[l.offset[k - 1] + word], d);
                    } else {
                        for (unsigned x = 0; x < mode.head; ++x)
                            vec_mult_assign(lrow + x * mode.groups, centre, h[x], mode.groups);
                        for (unsigned k = 0; k <= mode.depth; ++k)
                            legendre_moment<true, false>(q + (i + 1) * qlen, power, rrow + l.offset[k],
                                static_cast<T*>(nullptr), static_cast<T*>(nullptr), k, mode, l);
                    }
                }
            });
        };
        T* product = buffers[r % 2];
        for (uint64_t start = 0; start < n; start += block) {
            const uint64_t count = std::min(n - start, block);
            factors.template operator()<false>(start, count);
            if constexpr (Backward) {
                matrices.gemm({right, count, cols, cols, 1}, {product, cols, rows, 1, cols}, {dl, count, rows, rows, 1}, true);
                matrices.gemm({left, count, rows, rows, 1}, {product, rows, cols, cols, 1}, {dr, count, cols, cols, 1}, true);
                factors.template operator()<true>(start, count);
            } else matrices.gemm({left, rows, count, 1, rows}, {right, count, cols, cols, 1},
                {product, rows, cols, cols, 1}, start == 0);
        }
        if constexpr (Backward) {
            if (r + 1 < plan.modes.size())
                legendre_expand<true>(mode, plan.modes[r + 1], l, product, buffers[(r + 1) % 2], jobs);
        } else {
            if (step && r) legendre_expand<false>(mode, plan.modes[r + 1], l, product, buffers[(r + 1) % 2], jobs);
        }
    }
    if constexpr (Backward) {
        for (uint64_t i = 0; i < n; ++i) {
            const T* v = inc + i * d; const T* m = mon + i * plan.groups;
            T* grad = dm + i * plan.groups; T* g = dv + i * d;
            for (unsigned r = unsigned(plan.modes.size()) - 1; r > 0; --r) {
                const auto& mode = plan.modes[r]; const auto& prev = plan.modes[r - 1];
                const T scale = -T(1) / T(4 * (2 * r - 1) * (2 * r + 1));
                for (unsigned id = 0; id < mode.groups; ++id) {
                    const unsigned ex = mode.parent[id], pair = ex / d, x = pair / prev.groups, y = ex % d;
                    const unsigned parent = prev.first_group + pair % prev.groups;
                    const T z = grad[mode.first_group + id] * scale;
                    grad[parent] += (z * v[x]) * v[y];
                    g[x] += (z * m[parent]) * v[y]; g[y] += (z * m[parent]) * v[x];
                }
            }
            vec_mult_add(g, grad, T(1), d);
            cpu_detail::horner_backprop<true>(q + (i + 1) * qlen, v, dq + i * qlen, g, b, l, scratch, scratch + temp);
            vec_mult_add(dq + (i + 1) * qlen + 1, dq + i * qlen + 1, T(1), qlen - 1);
        }
        for (uint64_t i = n; i-- > 0;) {
            cpu_detail::horner_backprop<false>(p + i * plen, inc + i * d, dp + (i + 1) * plen, dv + i * d,
                a - 1, l, scratch, scratch + temp);
            vec_mult_add(dp + i * plen + 1, dp + (i + 1) * plen + 1, T(1), plen - 1);
            for (unsigned c = 0; c < d; ++c)
                out[(i + 1) * d + c] = dv[i * d + c] - (i + 1 < n ? dv[(i + 1) * d + c] : T(0));
        }
        for (unsigned c = 0; c < d; ++c) out[c] = -dv[c];
    } else legendre_levels<false>(seed, out, scalar, l, plan, buffers, jobs);
}

template<bool Backward, typename T>
bool legendre_batch(const T* path, const T* seed, T* out, uint64_t batch, uint64_t n,
    bool scalar, int jobs, const Layout& l, uint64_t sig_stride,
    uint64_t dimension, bool ta, bool ll, T end_time) {
    if (l.m < 4 || l.m > 16 || l.power[l.m] > (1ULL << 25) || n < 8 || l.d > 24
        || (l.m <= 6 && n > (256ULL << 20) / (sizeof(T) * l.offset[l.h + 1]))) return false;
    if (l.m > 6 && l.d >= 3 && n < 32) return false;
    if (l.m <= 6 && l.d > 4) {
        if (l.d != 8 || (l.m != 6 && !(l.m == 5 && sizeof(T) == sizeof(float)))) return false;
        if (l.m == 6 && Backward) return false;
    }
    const int inner = std::max(1, available_threads(jobs) / int(std::min<uint64_t>(batch, jobs)));
    const uint64_t length = (ll ? n / 2 : n) + 1, path_stride = length * dimension;
    const bool augmented = ta || ll;
    if (augmented && n < 16) return false;
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
    const bool pack_lead_lag = Backward && ll && !ta && l.d == 4 && (l.m == 5 || l.m == 6)
        && n >= (sizeof(T) == sizeof(double) ? 256 : 512) && n <= 2048
        && (jobs > 1 || (l.m == 5 && n >= 512));
    if ((!augmented || pack_lead_lag) && l.m <= 6 && (l.d <= 4 || (l.d == 8 && l.m == 5))) {
        const uint64_t rows = l.power[3], cols = l.offset[l.m - 2], terms = std::min<uint64_t>(3, l.m - 2);
        const uint64_t block = std::min(n, std::clamp<uint64_t>((256ULL << 10) /
            (terms * (rows + cols) * sizeof(T) * (1 + Backward)), 32, 256));
        const uint64_t size = rows * cols + (1 + Backward) * terms * block * (rows + cols)
            + (uint64_t(l.m >= 5) + Backward) * n * l.power[2] + (l.m == 6 ? n * rows : 0);
        const uint64_t packed = pack_lead_lag ? (n + 1) * l.d : 0;
        auto run = [&]<unsigned D>() {
            for_paths<T>(batch, jobs, size + 2 * packed, "CPU prefix-suffix", [&](uint64_t i, T* memory) {
                const T* input = path + i * path_stride;
                T* output = out + i * (Backward ? path_stride : sig_stride);
                if (packed) {
                    transform_path_(input, memory + size, dimension, length, ta, ll, end_time);
                    input = memory + size;
                }
                auto path_at = [&]<unsigned Degree>() {
                    low_moment_path<Backward, Degree, D>(input, Backward ? seed + i * sig_stride : nullptr,
                        packed ? memory + size + packed : output, n, scalar, block, inner, memory);
                };
                if (l.m == 6) path_at.template operator()<6>();
                else if (l.m == 5) path_at.template operator()<5>();
                else path_at.template operator()<4>();
                if (packed) transform_path_backprop_(memory + size + packed, output, dimension, length, ta, ll, end_time);
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
    if (n >= 16) {
        const bool shorter = !Backward && l.m == 7 && l.d >= 3 && l.d <= 6 && n >= 128 && n <= 512;
        const CpuMomentScan scan(l, n, sizeof(T), Backward, inner, shorter ? 3 : 0);
        const uint64_t limit = std::min<uint64_t>(256ULL << 20, option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX)) / sizeof(T);
        if (blocked_legendre_batch<Backward>(path, seed, out, batch, n, scalar, jobs, inner, l, sig_stride,
            scan.size, limit, dimension, ta, ll, end_time))
            return true;
        if (scan.size <= limit) {
            for_paths<T>(batch, jobs, scan.size, "CPU prefix-suffix", [&](uint64_t i, T* memory) {
                const T* input = path + i * path_stride;
                const Path<T> source(input, dimension, length, ta, ll, end_time);
                scanned_legendre_path<Backward>(input, Backward ? seed + i * sig_stride : nullptr,
                    out + i * (Backward ? path_stride : sig_stride), scalar, l, scan.plan, scan.p, scan.q,
                    n, scan.tile, inner, memory, augmented ? &source : nullptr);
            });
            return true;
        }
    }
    if (augmented || n > (256ULL << 20) / (sizeof(T) * l.offset[l.h + 1])) return false;
    const LegendrePlan plan(l, 0, Backward);
    const uint64_t block = std::min(n, plan.tile(n, sizeof(T), Backward) * std::min(inner, 8));
    const uint64_t size = plan.workspace(l, n, block, Backward)
        + 4 * uint64_t(std::max(0, inner - 2)) * l.offset[plan.b + 1];
    if (size > std::min<uint64_t>(256ULL << 20, option("PYSIGLIB_PS_WORKSPACE_BYTES", UINT64_MAX)) / sizeof(T)) return false;
    for_paths<T>(batch, jobs, size, "CPU prefix-suffix", [&](uint64_t i, T* memory) {
        legendre_path<Backward>(path + i * path_stride, Backward ? seed + i * sig_stride : nullptr,
            out + i * (Backward ? path_stride : sig_stride), scalar, l, plan, n, block, inner, memory);
    });
    return true;
}
