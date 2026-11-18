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

template<bool Reverse, typename T>
HWY_INLINE hn::Vec<Simd<T>> scan_midpoints(Simd<T> d, hn::Vec<Simd<T>> z, T& carry) {
    constexpr size_t lanes = vec_batch_lanes<T>;
    auto shift = [&](auto v, size_t k) {
        if constexpr (Reverse) return hn::SlideDownLanes(d, v, k);
        else return hn::SlideUpLanes(d, v, k);
    };
    auto sum = z;
    for (size_t k = 1; k < lanes; k *= 2) sum = hn::Add(sum, shift(sum, k));
    const auto before = hn::Add(hn::Set(d, carry), shift(sum, 1));
    carry += hn::ExtractLane(sum, Reverse ? 0 : lanes - 1);
    return hn::MulAdd(z, hn::Set(d, T(0.5)), before);
}

// H_0 is the segment mean. Its integral supplies the exclusive sum at the
// next level. Integrating the shifted Legendre polynomials couples modes r-1
// and r+1, so higher moments need only a correction from two levels below.
template<bool Suffix, typename T>
void scan_moments(const T* inc, T* moments, T* low, bool scalar,
    const Layout& l, const MomentRows& s, uint64_t n, int jobs) {
    const unsigned d = unsigned(l.d);
    for (unsigned r = 0; r <= s.depth; ++r) {
        T* base = moments + s.first[r] * n;
        std::fill_n(base, n, T(1));
        if (r && r < s.depth) for (unsigned c = 1; c <= d; ++c)
            std::copy_n(moments + c * n, n, base + c * n);
    }
    for (unsigned k = 2; k <= s.depth; ++k) {
        const uint64_t width = l.power[k], prev = width / d;
        parallel_range(width, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t w = first; w < end; ++w) {
                const uint64_t c = Suffix ? w / prev : w % d, parent = Suffix ? w % prev : w / d;
                const T* v = inc + c * n;
                T* mean = moments + (l.offset[k] + w) * n;
                const uint64_t span = prev / d, c2 = Suffix ? parent / span : parent % d;
                const uint64_t stem = Suffix ? parent % span : parent / d;
                const T* u = inc + c2 * n;
                const T* prior = moments + (l.offset[k - 1] + parent) * n;
                const T* lower = moments + (l.offset[k - 2] + stem) * n;
                const T* h1 = lower + s.first[1] * n;
                T sum = 0;
                uint64_t t = 0;
                const Simd<T> tag;
                constexpr size_t lanes = vec_batch_lanes<T>;
                if constexpr (lanes > 1) for (; n - t >= lanes; t += lanes) {
                    const uint64_t i = Suffix ? n - t - lanes : t;
                    const auto v1 = hn::LoadU(tag, v + i), z = hn::Mul(hn::LoadU(tag, prior + i), v1);
                    const auto mid = scan_midpoints<Suffix, T>(tag, z, sum);
                    const auto corr = hn::Mul(hn::Mul(hn::LoadU(tag, h1 + i), hn::Set(tag, -T(1) / T(12))), hn::LoadU(tag, u + i));
                    hn::StoreU(hn::MulAdd(corr, v1, mid), tag, mean + i);
                }
                for (; t < n; ++t) {
                    const uint64_t i = Suffix ? n - 1 - t : t;
                    const T z = prior[i] * v[i];
                    mean[i] = (sum + z / 2) - ((h1[i] / 12) * u[i]) * v[i];
                    sum += z;
                }
                if (low) low[l.offset[k] - !scalar + w] = sum;
                for (unsigned r = 1; r <= s.depth - k; ++r) {
                    const T scale = -T(1) / T(4 * (2 * r + 1) * (2 * r + 3));
                    const T* source = lower + s.first[r + 1] * n;
                    const T* last = mean + s.first[r - 1] * n;
                    T* dest = mean + s.first[r] * n;
                    for (uint64_t i = 0; i < n; ++i) dest[i] = last[i] + ((scale * source[i]) * u[i]) * v[i];
                }
            }
        });
    }
}

template<bool Suffix, typename T>
void reverse_moments(const T* inc, const T* moments, T* grad, T* dv, T* dz, const T* low,
    bool scalar, const Layout& l, const MomentRows& s, uint64_t n, int jobs) {
    const unsigned d = unsigned(l.d);
    for (unsigned k = s.depth; k > 0; --k) {
        const uint64_t width = l.power[k], prev = width / d;
        parallel_range(width, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t w = first; w < end; ++w) {
                T* g = grad + (l.offset[k] + w) * n;
                for (unsigned r = s.depth - k; r > 0; --r)
                    vec_mult_add(g + s.first[r - 1] * n, g + s.first[r] * n, T(1), n);
                T carry = low ? low[l.offset[k] - !scalar + w] : T(0);
                T* z = dz + w * n;
                uint64_t t = 0;
                const Simd<T> tag;
                constexpr size_t lanes = vec_batch_lanes<T>;
                if constexpr (lanes > 1) for (; n - t >= lanes; t += lanes) {
                    const uint64_t i = Suffix ? t : n - t - lanes;
                    hn::StoreU(scan_midpoints<!Suffix, T>(tag, hn::LoadU(tag, g + i), carry), tag, z + i);
                }
                for (; t < n; ++t) {
                    const uint64_t i = Suffix ? t : n - 1 - t;
                    z[i] = carry + g[i] / 2;
                    carry += g[i];
                }
            }
        });
        parallel_range(n, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t w = 0; w < width; ++w) {
                const uint64_t c = Suffix ? w / prev : w % d, parent = Suffix ? w % prev : w / d;
                const T* z = dz + w * n;
                T* dc = dv + c * n;
                if (k == 1) { vec_mult_add(dc + first, z + first, T(1), end - first); continue; }
                const uint64_t span = prev / d, c2 = Suffix ? parent / span : parent % d;
                const uint64_t stem = Suffix ? parent % span : parent / d;
                const T* u = inc + c2 * n; const T* v = inc + c * n;
                const T* prior = moments + (l.offset[k - 1] + parent) * n;
                T* dp = grad + (l.offset[k - 1] + parent) * n;
                for (uint64_t i = first; i < end; ++i) { dp[i] += z[i] * v[i]; dc[i] += z[i] * prior[i]; }
                for (unsigned r = 0; r <= s.depth - k; ++r) {
                    const uint64_t lower = (s.first[r + 1] + l.offset[k - 2] + stem) * n;
                    const T* h = moments + lower; T* dh = grad + lower;
                    const T* g = grad + (s.first[r] + l.offset[k] + w) * n;
                    T* du = dv + c2 * n;
                    const T scale = -T(1) / T(4 * (2 * r + 1) * (2 * r + 3));
                    for (uint64_t i = first; i < end; ++i) {
                        const T t = scale * g[i];
                        if (k > 2) dh[i] += (t * u[i]) * v[i];
                        dc[i] += (t * h[i]) * u[i]; du[i] += (t * h[i]) * v[i];
                    }
                }
            }
        });
    }
}

template<bool Backward, typename T>
void scanned_legendre_path(const T* path, const T* seed, T* out, bool scalar, const Layout& l,
    const LegendrePlan& plan, const MomentRows& p, const MomentRows& q, uint64_t n,
    uint64_t block, int jobs, T* storage, const Path<T>* augmented = nullptr) {
    const unsigned d = unsigned(l.d);
    auto take = [&](uint64_t size) { T* ptr = storage; storage += size; return ptr; };
    T* buffers[]{take(plan.buffers[0]), take(plan.buffers[1])};
    T* pre = take(p.size * n); T* suf = take(q.size * n);
    T* centres = take(plan.groups * n); T* left = take(plan.max_rows * block);
    T *dp = nullptr, *dq = nullptr, *dm = nullptr, *dl = nullptr, *dz = nullptr;
    if constexpr (Backward) {
        dp = take(p.size * n); dq = take(q.size * n); dm = take(plan.groups * n);
        std::fill_n(dp, p.size * n, T(0)); std::fill_n(dm, plan.groups * n, T(0));
        dl = take(plan.max_rows * block); dz = take(l.power[q.depth] * n);
        legendre_levels<true>(seed, out, scalar, l, plan, buffers, jobs);
    } else if (scalar) out[0] = 1;
    auto start_moments = [&](auto point) {
        for (unsigned c = 0; c < d; ++c) {
            const T first = point(0, c), last = point(n, c);
            T previous = first;
            for (uint64_t i = 0; i < n; ++i) {
                const T next = point(i + 1, c), v = next - previous;
                centres[c * n + i] = v;
                pre[(1 + c) * n + i] = (previous - first) + v / 2;
                suf[(1 + c) * n + i] = (last - next) + v / 2;
                previous = next;
            }
            if constexpr (!Backward) out[scalar + c] = last - first;
        }
    };
    if (augmented) start_moments([&](uint64_t i, unsigned c) { return (*augmented)[i][c]; });
    else start_moments([&](uint64_t i, unsigned c) { return path[i * d + c]; });
    scan_moments<false>(centres, pre, Backward ? nullptr : out, scalar, l, p, n, jobs);
    scan_moments<true>(centres, suf, static_cast<T*>(nullptr), scalar, l, q, n, jobs);
    for (unsigned r = 1; r < plan.modes.size(); ++r) {
        const auto& m = plan.modes[r]; const auto& prev = plan.modes[r - 1];
        const T scale = -T(1) / T(4 * (2 * r - 1) * (2 * r + 1));
        parallel_range(m.groups, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t g = first; g < end; ++g) {
                const unsigned ex = m.parent[g], pair = ex / d;
                const T* mid = centres + (prev.first_group + pair % prev.groups) * n;
                const T* x = centres + (pair / prev.groups) * n; const T* y = centres + (ex % d) * n;
                T* dest = centres + (m.first_group + g) * n;
                for (uint64_t i = 0; i < n; ++i) dest[i] = ((scale * mid[i]) * x[i]) * y[i];
            }
        });
    }
    CpuMatrices<T> matrices{jobs};
    for (unsigned step = 0; step < plan.modes.size(); ++step) {
        const unsigned r = Backward ? step : unsigned(plan.modes.size()) - 1 - step;
        const auto& m = plan.modes[r];
        const uint64_t rows = m.rows(), cols = m.cols, head = (p.first[r] + l.offset[p.depth - r]) * n;
        T* h = pre + head; T* mid = centres + m.first_group * n;
        T* right = suf + q.first[r] * n; T* product = buffers[r % 2];
        for (uint64_t start = 0; start < n; start += block) {
            const uint64_t count = std::min(block, n - start);
            // The empty prefix has value one, so its centre is the entire left factor.
            const bool constant_head = m.head == 1;
            const uint64_t ld = constant_head ? n : block;
            T* lhs = constant_head ? mid + start : left;
            if (!constant_head) parallel_range(rows, jobs, [&](uint64_t first, uint64_t end) {
                for (uint64_t row = first; row < end; ++row) {
                    const T* x = h + row / m.groups * n + start; const T* y = mid + row % m.groups * n + start;
                    T* z = left + row * block;
                    for (uint64_t i = 0; i < count; ++i) z[i] = x[i] * y[i];
                }
            });
            if constexpr (Backward) {
                T* dleft = constant_head ? dm + m.first_group * n + start : dl;
                if (cols == 1) {
                    parallel_range(rows, jobs, [&](uint64_t first, uint64_t end) {
                        for (uint64_t g = first; g < end; ++g) std::fill_n(dleft + g * ld, count, product[g]);
                    });
                    continue;
                }
                matrices.gemm({product, rows, cols, cols, 1}, {right + start, cols, count, n, 1},
                    {dleft, rows, count, ld, 1}, true);
                matrices.gemm({product, cols, rows, 1, cols}, {lhs, rows, count, ld, 1},
                    {dq + q.first[r] * n + start, cols, count, n, 1}, true);
                if (!constant_head) parallel_range(count, jobs, [&](uint64_t first, uint64_t end) {
                    for (uint64_t row = 0; row < rows; ++row) {
                        const uint64_t x = row / m.groups * n + start, y = row % m.groups * n + start;
                        T* dh = dp + head + x; T* dc = dm + m.first_group * n + y;
                        const T* g = dl + row * block;
                        for (uint64_t i = first; i < end; ++i) { dh[i] += g[i] * mid[y + i]; dc[i] += g[i] * h[x + i]; }
                    }
                });
            } else matrices.gemm({lhs, rows, count, ld, 1}, {right + start, count, cols, 1, n},
                {product, rows, cols, cols, 1}, !start);
        }
        if constexpr (Backward) {
            if (r + 1 < plan.modes.size()) legendre_expand<true>(m, plan.modes[r + 1], l, product, buffers[(r + 1) % 2], jobs);
        } else if (step && r) legendre_expand<false>(m, plan.modes[r + 1], l, product, buffers[(r + 1) % 2], jobs);
    }
    if constexpr (Backward) {
        parallel_range(n, jobs, [&](uint64_t first, uint64_t end) {
            for (unsigned r = unsigned(plan.modes.size()) - 1; r > 0; --r) {
                const auto& m = plan.modes[r]; const auto& prev = plan.modes[r - 1];
                const T scale = -T(1) / T(4 * (2 * r - 1) * (2 * r + 1));
                for (unsigned g = 0; g < m.groups; ++g) {
                    const unsigned ex = m.parent[g], pair = ex / d;
                    const uint64_t u = (prev.first_group + pair % prev.groups) * n, x = (pair / prev.groups) * n, y = (ex % d) * n;
                    const T* grad = dm + (m.first_group + g) * n;
                    for (uint64_t i = first; i < end; ++i) {
                        const T t = scale * grad[i], h = t * centres[u + i];
                        dm[u + i] += (t * centres[x + i]) * centres[y + i];
                        dm[x + i] += h * centres[y + i]; dm[y + i] += h * centres[x + i];
                    }
                }
            }
        });
        reverse_moments<false>(centres, pre, dp, dm, dz, seed, scalar, l, p, n, jobs);
        reverse_moments<true>(centres, suf, dq, dm, dz, static_cast<const T*>(nullptr), scalar, l, q, n, jobs);
        const uint64_t channels = augmented ? augmented->data_dimension() : d;
        for (unsigned c = 0; c < channels; ++c) {
            const T* grad = dm + c * n;
            if (augmented && augmented->lead_lag()) {
                const T* lead = grad + channels * n;
                T previous = 0;
                for (uint64_t i = 0; i < n / 2; ++i) {
                    const T current = lead[2 * i] + grad[2 * i + 1];
                    out[i * channels + c] = previous - current;
                    previous = current;
                }
                out[n / 2 * channels + c] = previous;
            } else {
                out[c] = -grad[0]; out[n * channels + c] = grad[n - 1];
                for (uint64_t i = 1; i < n; ++i) out[i * channels + c] = grad[i - 1] - grad[i];
            }
        }
    } else legendre_levels<false>(seed, out, scalar, l, plan, buffers, jobs);
}
