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

template<bool Reverse, bool Dual, class T>
hn::Vec<Simd<T>> striped_moment_row(const T* prev, T* next, const T* dx, unsigned previous, unsigned count,
    unsigned n, hn::Vec<Simd<T>> bias, T seed, T& endpoint);

// Integrating one word letter needs a segment scan only for the mean.
// Higher Legendre modes depend on the two neighbouring orders on the same segment.
template<bool FromRight = false, bool Dual = false, class T>
void coefficient_moment_row(const T* HWY_RESTRICT prev, T* HWY_RESTRICT next, const T* dx,
    unsigned previous, unsigned count, unsigned size, unsigned pitch, T& carry, bool tiled) {
    const T* mode1 = previous > 1 ? prev + pitch : nullptr;
    const Simd<T> tag;
    constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
    if (tiled && size == pitch) {
        const auto bias = striped_moment_row<FromRight, Dual>(prev, next, dx, previous, count, size, hn::Zero(tag), carry, carry);
        for (unsigned i = 0; i < size; i += lanes) hn::StoreU(hn::Add(hn::LoadU(tag, next + i), bias), tag, next + i);
        return;
    }
    constexpr T correction = (FromRight ? T(1) : T(-1)) * (Dual ? T(1) : T(1) / 12);
    unsigned t = 0;
    auto prior = hn::Set(tag, carry);
    if constexpr (lanes > 1) for (; size - t >= lanes; t += lanes) {
        const unsigned i = FromRight ? size - t - lanes : t;
        const auto v = hn::LoadU(tag, dx + i);
        auto shift = [&](auto values, unsigned offset) {
            if constexpr (FromRight) return hn::SlideDownLanes(tag, values, offset);
            else return hn::SlideUpLanes(tag, values, offset);
        };
        const auto z = hn::Mul(v, hn::LoadU(tag, prev + i));
        auto sum = z;
        for (unsigned k = 1; k < lanes; k *= 2) sum = hn::Add(sum, shift(sum, k));
        auto mid = hn::Add(prior, hn::NegMulAdd(z, hn::Set(tag, T(.5)), sum));
        prior = hn::Add(prior, hn::BroadcastLane<FromRight ? 0 : lanes - 1>(sum));
        if (mode1) mid = hn::MulAdd(hn::Mul(hn::Set(tag, correction), hn::LoadU(tag, mode1 + i)), v, mid);
        hn::StoreU(mid, tag, next + i);
    }
    if constexpr (lanes > 1) carry = hn::GetLane(prior);
    for (; t < size; ++t) {
        const unsigned i = FromRight ? size - 1 - t : t;
        const T v = dx[i], z = v * prev[i];
        const T mid = carry + T(.5) * z;
        carry += z;
        next[i] = mode1 ? mid + (correction * mode1[i]) * v : mid;
    }
    for (unsigned r = 1; r < count; ++r) {
        const T* lo = prev + (r - 1) * pitch;
        const T scale = T(1) / T((4 * r + 2) * (Dual ? 4 * r - 2 : 4 * r + 6));
        T* dest = next + r * pitch;
        const T* hi = r + 1 < previous ? prev + (r + 1) * pitch : nullptr;
        unsigned i = 0;
        if constexpr (lanes > 1) for (; size - i >= lanes; i += lanes) {
            auto v = hn::LoadU(tag, lo + i);
            if constexpr (Dual) {
                v = hn::Mul(v, hn::Set(tag, scale));
                if (hi) v = hn::Sub(v, hn::LoadU(tag, hi + i));
            } else if (hi) v = hn::NegMulAdd(hn::Set(tag, scale), hn::LoadU(tag, hi + i), v);
            if constexpr (FromRight) v = hn::Neg(v);
            hn::StoreU(hn::Mul(v, hn::LoadU(tag, dx + i)), tag, dest + i);
        }
        for (; i < size; ++i) {
            const T upper = hi ? hi[i] : T(0);
            const T value = Dual ? lo[i] * scale - upper : lo[i] - scale * upper;
            dest[i] = (FromRight ? -value : value) * dx[i];
        }
    }
}

// Contract the prefix and suffix moments at the central letter of one word.
template<unsigned Capacity, class T>
bool paired_cpu_word(const uint64_t* word, unsigned m, uint64_t n, const T* dx, const T* left_mean,
    const T* right_mean, T* output, T* memory, bool checkpoint, bool tiled, bool split_time) {
    constexpr unsigned tile = 128;
    const unsigned centre = (m - 1) / 2, modes = (m + 1) / 2;
    const uint64_t tiles = (n - 1) / tile + 1;
    auto counts = [&](unsigned level) { return 1 + std::min(level, m - 1 - level); };
    if (split_time) {
        T ends[2][Capacity + 1] = {};
        // Chen's identity pairs each prefix on the first half with its complementary suffix.
        auto sweep = [&]<bool Right>() HWY_ATTR {
            T* state = ends[Right], *rows = memory + unsigned(Right) * 2 * tile * modes;
            state[0] = 1;
            const uint64_t half = tiles / 2, steps = Right ? tiles - half : half;
            for (uint64_t t = 0; t < steps; ++t) {
                const uint64_t k = Right ? tiles - 1 - t : t, first = k * tile;
                const unsigned size = unsigned(std::min<uint64_t>(tile, n - first));
                std::fill_n(rows, size, T(1));
                for (unsigned j = 1; j <= m; ++j) {
                    const unsigned letter = Right ? m - j : j - 1;
                    coefficient_moment_row<Right>(rows + ((j - 1) % 2) * tile * modes, rows + (j % 2) * tile * modes,
                        dx + word[letter] * n + first, counts(j - 1), j == m ? 1 : counts(j), size, tile, state[j], tiled);
                }
            }
        };
        spawn_batch_threads(2, 2, [&](uint64_t begin, uint64_t end) HWY_ATTR {
            for (uint64_t side = begin; side < end; ++side)
                if (side) sweep.template operator()<true>(); else sweep.template operator()<false>();
        });
        double result = 0;
        for (unsigned j = 0; j <= m; ++j) result += double(ends[0][j]) * ends[1][m - j];
        *output = T(result);
        return pc::ScalarOps<T>::finite(*output);
    }
    T* tape = memory + (checkpoint ? tiles * centre : 0);
    const uint64_t tape_size = (checkpoint ? 1 : tiles) * tile * (centre + 1);
    T* a = tape + tape_size;
    T* b = a + tile * modes;
    auto left_tile = [&](uint64_t k, T* saved, T* state) HWY_ATTR {
        const uint64_t first = k * tile;
        const unsigned size = unsigned(std::min<uint64_t>(tile, n - first));
        T* prev = centre > 1 ? a : saved;
        if (centre) {
            std::copy_n(left_mean + first, size, prev);
            std::copy_n(dx + word[0] * n + first, size, prev + tile);
        } else std::fill_n(prev, size, T(1));
        for (unsigned j = 1; j < centre; ++j) {
            T* next = j + 1 == centre ? saved : prev == a ? b : a;
            coefficient_moment_row(prev, next, dx + word[j] * n + first, counts(j), counts(j + 1), size, tile, state[j], tiled);
            prev = next;
        }
    };
    T state[Capacity + 1] = {};
    if (!checkpoint || centre > 1) for (uint64_t k = 0; k < tiles; ++k) {
        if (checkpoint) std::copy_n(state, centre, memory + k * centre);
        left_tile(k, checkpoint ? tape : tape + k * tile * (centre + 1), state);
    }
    T total = 0;
    const Simd<T> tag;
    constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
    for (uint64_t k = tiles; k-- > 0;) {
        const uint64_t first = k * tile;
        const unsigned size = unsigned(std::min<uint64_t>(tile, n - first));
        T* p = checkpoint ? tape : tape + first * (centre + 1);
        if (checkpoint) left_tile(k, p, memory + k * centre);
        T* q = a;
        std::copy_n(right_mean + first, size, q);
        if (counts(m - 2) > 1) {
            const T* v = dx + word[m - 1] * n + first;
            for (unsigned i = 0; i < size; ++i) q[tile + i] = T(-1. / 12.) * v[i];
        }
        for (unsigned j = m - 2; j > centre; --j) {
            T* next = q == a ? b : a;
            coefficient_moment_row<true, true>(q, next, dx + word[j] * n + first, counts(j), counts(j - 1), size, tile, state[j], tiled);
            q = next;
        }
        const T* v = dx + word[centre] * n + first;
        unsigned i = 0;
        auto sum = hn::Zero(tag);
        if constexpr (lanes > 1) for (; size - i >= lanes; i += lanes) {
            auto dot = hn::Zero(tag);
            for (unsigned r = 0; r <= centre; ++r)
                dot = hn::MulAdd(hn::LoadU(tag, p + r * tile + i), hn::LoadU(tag, q + r * tile + i), dot);
            sum = hn::MulAdd(dot, hn::LoadU(tag, v + i), sum);
        }
        total += hn::ReduceSum(tag, sum);
        for (; i < size; ++i) {
            T dot = 0;
            for (unsigned r = 0; r <= centre; ++r) dot += p[r * tile + i] * q[r * tile + i];
            total += dot * v[i];
        }
    }
    *output = total;
    return pc::ScalarOps<T>::finite(*output);
}

template<class T, bool Backward, unsigned Capacity>
bool scan_cpu_word(const uint64_t* word, unsigned m, uint64_t n, const T* dx, T* grad,
    const T* left_mean, const T* right_mean, const T* coefs, const T* seeds, T* output, bool prefixes,
    T* memory, bool checkpoint, unsigned owners, bool tiled) {
    constexpr unsigned tile = 128;
    using Ops = pc::ScalarOps<T>;
    if (!m) { if constexpr (!Backward) *output = 1; return true; }
    unsigned rows[Capacity + 1] = {};
    for (unsigned j = 0; j < m; ++j) rows[j + 1] = rows[j] + 1 + std::min(j, m - 1 - j);
    const unsigned modes = (m + 1) / 2;
    const uint64_t tiles = (n - 1) / tile + 1;
    T* buffers = memory + (checkpoint ? uint64_t(owners > 1 ? 2 : 1) * tiles * m : 0);
    const unsigned row_size = rows[m] * tile, stride = (rows[m] + 2 * modes) * tile;
    T* workspace = buffers + (Backward ? (checkpoint ? 1 : tiles) * row_size : 0);
    T primal[Capacity + 1] = {}, adjoint[Capacity + 1] = {};
    if constexpr (Backward) {
        T invfact = 1;
        for (unsigned j = 1; j <= m; ++j) {
            adjoint[j] = seeds[j - 1];
            invfact /= T(j);
            if ((coefs && !Ops::finite(coefs[j - 1])) || !Ops::finite(adjoint[j]) || Ops::tiny(adjoint[j], invfact * adjoint[j])) return false;
        }
    }
    auto primal_tile = [&](uint64_t k, T* state, T* saved, T* work) HWY_ATTR {
        const uint64_t first = k * tile;
        const unsigned size = unsigned(std::min<uint64_t>(n - first, tile));
        T* prev = saved ? saved : work;
        std::fill_n(prev, size, T(1));
        for (unsigned j = 1; j <= (Backward ? m - 1 : m); ++j) {
            T* next = saved ? saved + rows[j] * tile : work + (j % 2) * modes * tile;
            if (Backward && j == 1) {
                std::copy_n(left_mean + first, size, next);
                if (rows[2] - rows[1] > 1) {
                    const T* v = dx + word[0] * n + first;
                    for (unsigned i = 0; i < size; ++i) next[tile + i] = -v[i];
                }
            } else coefficient_moment_row<Backward>(prev, next, dx + word[j - 1] * n + first,
                rows[j] - rows[j - 1], j < m ? rows[j + 1] - rows[j] : 1, size, tile, state[j], tiled);
            prev = next;
        }
    };
    auto dual_tile = [&](uint64_t k, T* state, const T* saved, T* work) HWY_ATTR {
        const uint64_t first = k * tile;
        const unsigned size = unsigned(std::min<uint64_t>(n - first, tile));
        T* prev = work;
        std::fill_n(prev, size, state[m]);
        for (unsigned j = m; j > 0; --j) {
            const unsigned count = rows[j] - rows[j - 1];
            if (saved) {
                const T* p = saved + rows[j - 1] * tile;
                T* dg = grad + word[j - 1] * n + first;
                const Simd<T> tag;
                constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
                unsigned i = 0;
                if constexpr (lanes > 1) for (; size - i >= lanes; i += lanes) {
                    auto value = hn::LoadU(tag, dg + i);
                    for (unsigned r = 0; r < count; ++r)
                        value = hn::MulAdd(hn::LoadU(tag, p + r * tile + i), hn::LoadU(tag, prev + r * tile + i), value);
                    hn::StoreU(value, tag, dg + i);
                }
                for (; i < size; ++i)
                    for (unsigned r = 0; r < count; ++r) dg[i] += p[r * tile + i] * prev[r * tile + i];
            }
            if (j > 1) {
                T* next = work + ((m - j + 1) % 2) * modes * tile;
                if (j == m) {
                    const T* v = dx + word[j - 1] * n + first;
                    for (unsigned i = 0; i < size; ++i) next[i] = seeds[m - 2] + seeds[m - 1] * right_mean[first + i];
                    if (rows[j - 1] - rows[j - 2] > 1) {
                        const T scale = seeds[m - 1] * T(1. / 12.);
                        for (unsigned i = 0; i < size; ++i) next[tile + i] = scale * v[i];
                    }
                } else coefficient_moment_row<false, true>(prev, next, dx + word[j - 1] * n + first,
                    count, rows[j - 1] - rows[j - 2], size, tile, state[j - 1], tiled);
                prev = next;
            }
        }
    };
    auto endpoints = [&](uint64_t begin, uint64_t end) HWY_ATTR {
        for (uint64_t side = begin; side < end; ++side) {
            if (!side) for (uint64_t t = 0; t < tiles; ++t) {
                const uint64_t k = Backward ? tiles - 1 - t : t;
                if (checkpoint) std::copy_n(primal, m, memory + k * m);
                primal_tile(k, primal, Backward && !checkpoint ? buffers + k * row_size : nullptr, workspace);
            } else for (uint64_t k = 0; k < tiles; ++k) {
                std::copy_n(adjoint + 1, m, memory + (tiles + k) * m);
                dual_tile(k, adjoint, nullptr, workspace + stride);
            }
        }
    };
    if (owners > 1) spawn_batch_threads(2, 2, endpoints);
    else endpoints(0, 1);
    if constexpr (Backward) {
        auto reverse = [&](uint64_t begin, uint64_t end) HWY_ATTR {
            for (uint64_t owner = begin; owner < end; ++owner) {
                T left[Capacity + 1], right[Capacity + 1];
                std::copy_n(seeds, m, right + 1);
                T* work = workspace + owner * stride;
                for (uint64_t k = tiles * owner / owners; k < tiles * (owner + 1) / owners; ++k) {
                    T* saved = checkpoint ? buffers + owner * stride : buffers + k * row_size;
                    if (checkpoint) {
                        std::copy_n(memory + k * m, m, left);
                        primal_tile(k, left, saved, work);
                    }
                    if (owners > 1) std::copy_n(memory + (tiles + k) * m, m, right + 1);
                    dual_tile(k, right, saved, work);
                }
            }
        };
        if (owners > 1) spawn_batch_threads(owners, owners, reverse);
        else reverse(0, 1);
    } else for (unsigned j = 1; j <= m; ++j) {
        if (!Ops::finite(primal[j])) return false;
        if (prefixes || j == m) output[prefixes ? j - 1 : 0] = primal[j];
    }
    return true;
}

// Each SIMD lane scans a contiguous time interval. Only its boundary sum crosses lanes.
template<bool Reverse, bool Dual, class T>
hn::Vec<Simd<T>> striped_moment_row(const T* prev, T* next, const T* dx, unsigned previous, unsigned count,
    unsigned n, hn::Vec<Simd<T>> bias, T seed, T& endpoint) {
    const Simd<T> tag;
    constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
    auto sum = hn::Zero(tag);
    for (unsigned t = 0; t < n; t += lanes) {
        const unsigned i = Reverse ? n - lanes - t : t;
        const auto v = hn::LoadU(tag, dx + i);
        const auto z = hn::Mul(hn::Add(hn::LoadU(tag, prev + i), bias), v);
        auto mid = hn::MulAdd(z, hn::Set(tag, T(.5)), sum);
        sum = hn::Add(sum, z);
        if (previous > 1) {
            const auto hi = hn::LoadU(tag, prev + n + i);
            constexpr T correction = (Reverse ? T(1) : T(-1)) * (Dual ? T(1) : T(1. / 12.));
            mid = hn::MulAdd(hn::Mul(hi, hn::Set(tag, correction)), v, mid);
        }
        hn::StoreU(mid, tag, next + i);
    }
    for (unsigned r = 1; r < count; ++r) {
        const auto scale = hn::Set(tag, T(1) / T((4 * r + 2) * (Dual ? 4 * r - 2 : 4 * r + 6)));
        const T* lo = prev + (r - 1) * n;
        const T* hi = r + 1 < previous ? prev + (r + 1) * n : nullptr;
        for (unsigned i = 0; i < n; i += lanes) {
            auto low = hn::LoadU(tag, lo + i);
            if (r == 1) low = hn::Add(low, bias);
            if constexpr (Dual) {
                low = hn::Mul(scale, low);
                if (hi) low = hn::Sub(low, hn::LoadU(tag, hi + i));
            } else if (hi) low = hn::NegMulAdd(scale, hn::LoadU(tag, hi + i), low);
            if constexpr (Reverse) low = hn::Neg(low);
            hn::StoreU(hn::Mul(low, hn::LoadU(tag, dx + i)), tag, next + r * n + i);
        }
    }
    for (unsigned shift = 1; shift < lanes; shift *= 2)
        sum = hn::Add(sum, Reverse ? hn::SlideDownLanes(tag, sum, shift) : hn::SlideUpLanes(tag, sum, shift));
    endpoint = seed + hn::ExtractLane(sum, Reverse ? 0 : lanes - 1);
    return hn::Add(hn::Set(tag, seed), Reverse ? hn::SlideDownLanes(tag, sum, 1) : hn::SlideUpLanes(tag, sum, 1));
}

template<class T, bool Backward, unsigned Capacity>
bool striped_word(const uint64_t* word, unsigned m, unsigned n, const T* dx, T* grad, const T* left_mean,
    const T* right_mean, const T* coefs, const T* seeds, T* out, bool prefixes, T* memory) {
    const Simd<T> tag;
    constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
    if (!m) { if constexpr (!Backward) *out = 1; return true; }
    const unsigned centre = (m - 1) / 2, width = (m + 1) / 2;
    auto count = [m](unsigned j) { return 1 + std::min(j, m - 1 - j); };
    uint64_t offsets[Capacity + 1] = {};
    hn::Vec<Simd<T>> bias[Capacity + 1];
    bias[0] = hn::Zero(tag);
    T invfact = 1;
    for (unsigned j = 0; j < m; ++j) {
        offsets[j + 1] = offsets[j] + count(j);
        if constexpr (Backward) {
            invfact /= T(j + 1);
            if ((coefs && !pc::ScalarOps<T>::finite(coefs[j])) || !pc::ScalarOps<T>::finite(seeds[j]) ||
                pc::ScalarOps<T>::tiny(seeds[j], seeds[j] * invfact)) return false;
        }
    }
    if (m == 2 && right_mean && (Backward || !prefixes)) {
        auto sum = hn::Zero(tag);
        for (unsigned i = 0; i < n; i += lanes) {
            const auto right = hn::LoadU(tag, right_mean + i);
            if constexpr (Backward) {
                const auto weight = hn::Set(tag, seeds[1]);
                T* a = grad + word[0] * n + i;
                T* b = grad + word[1] * n + i;
                hn::StoreU(hn::Add(hn::LoadU(tag, a), hn::MulAdd(weight, right, hn::Set(tag, seeds[0]))), tag, a);
                hn::StoreU(hn::MulAdd(weight, hn::LoadU(tag, left_mean + i), hn::LoadU(tag, b)), tag, b);
            } else sum = hn::MulAdd(hn::LoadU(tag, dx + word[0] * n + i), right, sum);
        }
        if constexpr (!Backward) { *out = hn::ReduceSum(tag, sum); return pc::ScalarOps<T>::finite(*out); }
        return true;
    }
    const bool all = !Backward && prefixes;
    T* p = memory;
    std::fill_n(p, n, T(1));
    T total = 0;
    const unsigned last = Backward ? m - 1 : all ? m : centre;
    for (unsigned j = 1; j <= last; ++j) {
        T* next = Backward ? memory + offsets[j] * n : memory + (j % 2) * width * n;
        if (j == 1 && left_mean && !all) {
            std::copy_n(left_mean, n, next);
            if (count(1) > 1) std::copy_n(dx + word[0] * n, n, next + n);
            bias[j] = hn::Zero(tag);
        } else bias[j] = striped_moment_row<false, false>(p, next, dx + word[j - 1] * n, count(j - 1),
            j == m ? 1 : count(j), n, bias[j - 1], T(0), total);
        if (all) {
            if (!pc::ScalarOps<T>::finite(total)) return false;
            out[j - 1] = total;
        }
        p = next;
    }
    if (all) return true;
    T* work = Backward ? memory + offsets[m] * n : memory + 2 * width * n;
    T* q = work;
    const T terminal = Backward ? seeds[m - 1] : T(1);
    std::fill_n(q, n, terminal);
    auto right_bias = hn::Zero(tag);
    for (unsigned j = m; j > 0; --j) {
        const unsigned order = j - 1;
        if constexpr (Backward) p = memory + offsets[order] * n;
        if (Backward || order == centre) {
            T* dg = Backward ? grad + word[order] * n : nullptr;
            auto sum = hn::Zero(tag);
            for (unsigned i = 0; i < n; i += lanes) {
                auto dot = hn::Mul(hn::Add(hn::LoadU(tag, p + i), bias[order]),
                    hn::Add(hn::LoadU(tag, q + i), right_bias));
                for (unsigned r = 1; r < count(order); ++r)
                    dot = hn::MulAdd(hn::LoadU(tag, p + r * n + i), hn::LoadU(tag, q + r * n + i), dot);
                if constexpr (Backward) hn::StoreU(hn::Add(hn::LoadU(tag, dg + i), dot), tag, dg + i);
                else sum = hn::MulAdd(dot, hn::LoadU(tag, dx + word[centre] * n + i), sum);
            }
            if constexpr (!Backward) {
                *out = hn::ReduceSum(tag, sum);
                return pc::ScalarOps<T>::finite(*out);
            }
        }
        if (order) {
            T* next = q == work ? work + width * n : work;
            const T initial = Backward ? seeds[order - 1] : T(0);
            if (j == m && right_mean) {
                for (unsigned i = 0; i < n; ++i) next[i] = terminal * right_mean[i] + initial;
                if (count(order - 1) > 1) {
                    const T scale = T(-1. / 12.) * terminal;
                    for (unsigned i = 0; i < n; ++i) next[n + i] = scale * dx[word[order] * n + i];
                }
                right_bias = hn::Zero(tag);
            } else right_bias = striped_moment_row<true, true>(q, next, dx + word[order] * n, count(order),
                count(order - 1), n, right_bias, initial, total);
            q = next;
        }
    }
    return true;
}

template<class T>
constexpr unsigned normal_factorial_degree = std::is_same_v<T, float> ? 33 : 170;

// A deep axis word can have a finite endpoint even when an intermediate power overflows.
template<bool Backward, class T>
T scaled_axis_cpu(T dx, uint64_t degree, const T* seeds, T* output, bool all_prefixes) {
    auto shifted = [](double value, int64_t exponent) {
        return std::scalbn(value, int(std::clamp<int64_t>(exponent, -2048, 2048)));
    };
    int dx_exponent = 0, shift = 0;
    const double step = std::frexp(double(dx), &dx_exponent);
    double value = std::frexp(Backward ? double(seeds[all_prefixes ? degree - 1 : 0]) : 1., &shift);
    int64_t exponent = shift;
    const uint64_t count = Backward ? degree - 1 : degree;
    for (uint64_t k = 1; k <= count; ++k) {
        const uint64_t j = Backward ? degree - k : k;
        value = std::frexp(value * (step / double(j)), &shift);
        exponent += dx_exponent + shift;
        if constexpr (Backward) {
            if (all_prefixes && seeds[j - 1] != T(0)) {
                const double seed = std::frexp(double(seeds[j - 1]), &shift);
                const int64_t shared = value == 0 ? shift : std::max<int64_t>(exponent, shift);
                value = shifted(value, exponent - shared) + shifted(seed, shift - shared);
                exponent = shared;
            }
        } else if (all_prefixes) output[j - 1] = T(shifted(value, exponent));
    }
    return T(shifted(value, exponent));
}

template<class T, bool Backward>
bool axis_cpu_words(const T* path, T* out, const T* seeds, const pc::Words& words,
    uint64_t batch, uint64_t dim, uint64_t length, bool prefixes, bool terminal, bool ta, bool ll, T time, int jobs) {
    for (uint64_t w = 0; w < words.count; ++w) {
        const auto* letters = words.word(w);
        for (uint64_t j = 1; j < words.packed[w]; ++j) if (letters[j] != letters[0]) return false;
    }
    const uint64_t n = pc::mul(length - 1, ll ? 2 : 1), stride = length * dim;
    std::atomic<bool> valid{true};
    auto work = [&](uint64_t begin, uint64_t end) HWY_ATTR {
        const Simd<T> tag;
        constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
        for (uint64_t b = begin; b < end; ++b) {
            const T* x = path + b * stride;
            bool ok = true;
            auto finite = hn::Eq(hn::Zero(tag), hn::Zero(tag));
            uint64_t i = 0;
            for (; stride - i >= lanes; i += lanes)
                finite = hn::And(finite, hn::Le(hn::Abs(hn::LoadU(tag, x + i)), hn::Set(tag, pc::ScalarOps<T>::maximum)));
            for (; i < stride; ++i) ok = ok && pc::ScalarOps<T>::finite(x[i]);
            ok = ok && hn::AllTrue(tag, finite);
            if constexpr (Backward) std::fill_n(out + b * stride, stride, T(0));
            for (uint64_t w = 0; ok && w < words.count; ++w) {
                const uint64_t m = words.packed[w], c = m ? words.word(w)[0] : 0;
                const uint64_t pos = b * words.width + words.packed[2 * words.count + w];
                const T dx = pc::point(x, n, c, dim, n, ta, ll, time) - pc::point(x, 0, c, dim, n, ta, ll, time);
                if (!pc::ScalarOps<T>::finite(dx)) { ok = false; break; }
                const bool deep = m > normal_factorial_degree<T> && std::abs(dx) > T(1);
                if constexpr (Backward) {
                    if (!m || (ta && c == dim * (ll ? 2 : 1))) continue;
                    const T* weights = seeds + (terminal ? b * words.count + w : pos);
                    const T value = deep ? scaled_axis_cpu<true>(dx, m, weights, static_cast<T*>(nullptr), !terminal)
                        : pc::axis_backward(dx, weights, m, terminal);
                    const uint64_t col = ll && c >= dim ? c - dim : c;
                    T* dst = out + b * stride + col;
                    dst[0] -= value;
                    dst[(length - 1) * dim] += value;
                    ok = ok && pc::ScalarOps<T>::finite(dst[0]);
                } else if (deep) {
                    const T value = scaled_axis_cpu<false>(dx, m, static_cast<const T*>(nullptr), out + pos, prefixes);
                    if (!prefixes) out[pos] = value;
                } else pc::axis_forward(dx, m, out + pos, prefixes);
            }
            if (!ok) valid.store(false, std::memory_order_relaxed);
        }
    };
    if (jobs == 1 || batch == 1) work(0, batch);
    else spawn_batch_threads(batch, jobs, work);
    return valid.load(std::memory_order_relaxed);
}

template<class T, bool Backward, unsigned Capacity = 20>
bool prefix_suffix_cpu(const T* path, T* out, const T* coefs, const T* seeds, const pc::Words& words,
    uint64_t batch, uint64_t dim, uint64_t length, bool prefixes, bool ta, bool ll, T time, int jobs) {
    const uint64_t segments = pc::mul(length - 1, ll ? 2 : 1), m = words.max_degree;
    const uint64_t dimension = pc::add(pc::mul(dim, ll ? 2 : 1), ta);
    if constexpr (Capacity == 20) {
        if (axis_cpu_words<T, Backward>(path, out, seeds, words, batch, dim, length, prefixes, !coefs, ta, ll, time, jobs)) return true;
        // Above these degrees even the reciprocal factorial enters the subnormal range.
        if (m > Capacity && m <= normal_factorial_degree<T>)
            return prefix_suffix_cpu<T, Backward, normal_factorial_degree<T>>(path, out, coefs, seeds, words, batch,
                dim, length, prefixes, ta, ll, time, jobs);
    }
    if (!m || m > Capacity || segments == 1 || (m >= 8 && segments < (m + 3) / 4)) return false;
    uint64_t d = dimension;
    std::vector<uint64_t> channels, remapped;
    if (dimension > 4) {
        const auto* letters = words.word(0);
        channels.assign(letters, letters + words.index_count);
        std::sort(channels.begin(), channels.end());
        channels.erase(std::unique(channels.begin(), channels.end()), channels.end());
        if (channels.size() * 2 <= dimension) {
            d = channels.size();
            remapped.reserve(words.index_count);
            for (uint64_t j = 0; j < words.index_count; ++j)
                remapped.push_back(std::lower_bound(channels.begin(), channels.end(), letters[j]) - channels.begin());
        }
    }
    const bool compact = d < dimension;
    auto letters = [&](uint64_t w) {
        return compact ? static_cast<const uint64_t*>(remapped.data()) + (words.word(w) - words.word(0)) : words.word(w);
    };
    std::vector<int64_t> left(d, -1), right(d, -1);
    uint64_t end_rows = 0;
    const bool split_time = !Backward && !prefixes && batch == 1 && words.count == 1 && jobs > 1 && segments >= 8192 && m >= 6;
    if (!split_time && (Backward || !prefixes)) for (uint64_t w = 0; w < words.count; ++w) {
        const auto depth = words.packed[w];
        if (depth < 2) continue;
        const auto* word = letters(w);
        if ((Backward || depth > 2) && left[word[0]] < 0) left[word[0]] = end_rows++;
        if (right[word[depth - 1]] < 0) right[word[depth - 1]] = end_rows++;
    }
    const uint64_t modes = (m + 1) / 2;
    const uint64_t workers = std::min<uint64_t>(jobs, pc::mul(batch, words.count));
    const uint64_t budget = pc::workspace_limit() / sizeof(T);
    const uint64_t tile_tape = 128 * modes * ((m + 2) / 2);
    const uint64_t parts = std::min<uint64_t>(words.count, std::max<uint64_t>(1, workers / batch));
    const uint64_t jobs_count = pc::mul(batch, parts);
    constexpr unsigned lane_count = unsigned(vec_batch_lanes<T>);
    const uint64_t padded = pc::add(segments, lane_count - 1) / lane_count * lane_count;
    const uint64_t stripe_rows = Backward ? modes * ((m + 2) / 2) + 2 * modes : 4 * modes;
    const uint64_t rows_per_worker = pc::add(pc::add(pc::mul(d, Backward && parts == 1 ? 2 : 1), end_rows), stripe_rows);
    const uint64_t stripe_all = pc::add(pc::mul(rows_per_worker, workers), Backward && parts > 1 ? pc::mul(jobs_count, d) : 0);
    const bool striped = lane_count > 1 && segments >= 4 * lane_count && !ta && !ll &&
        segments <= 1024 && padded <= INT32_MAX / dim && padded <= budget / stripe_all;
    const uint64_t n = striped ? padded : segments, size = pc::mul(d, n), tiles = (n - 1) / 128 + 1;
    const bool tiled = !striped && !ta && !ll && lane_count > 1 && n >= 128 && dim <= INT32_MAX / 128;
    // Different word partitions have independent increment gradients.
    const uint64_t partial_size = Backward && parts > 1 ? pc::mul(jobs_count, size) : 0;
    if (partial_size > budget) return false;
    const uint64_t allowance = (budget - partial_size) / workers;
    const uint64_t endpoints_size = pc::mul(end_rows, n);
    uint64_t per_worker = pc::add(pc::mul(size, Backward && parts == 1 ? 2 : 1), endpoints_size);
    if (per_worker > allowance) return false;
    const uint64_t full = pc::add(256 * modes, Backward ? pc::mul(tile_tape, tiles) : 0);
    const bool checkpoint = Backward && tiles > 1 && (full > allowance - per_worker || full > (1u << 20) / sizeof(T));
    const uint64_t tile_work = tile_tape + 256 * modes;
    const uint64_t boundaries = pc::mul(tiles, m);
    unsigned owners = 1;
    if (checkpoint && jobs_count == 1 && m >= 8 && n >= 32768 && boundaries <= (allowance - per_worker) / 2)
        owners = unsigned(std::max<uint64_t>(1, std::min<uint64_t>({(allowance - per_worker - 2 * boundaries) / tile_work, uint64_t(jobs), tiles})));
    const uint64_t paired_size = !Backward && !prefixes ? pc::add(256 * modes, pc::mul(128 * modes, tiles)) : 0;
    const uint64_t paired_small = paired_size ? pc::add(384 * modes, pc::mul(tiles, (m - 1) / 2)) : 0;
    const bool paired_checkpoint = !split_time && paired_size > std::min<uint64_t>(allowance - per_worker, (768u << 10) / sizeof(T));
    const uint64_t paired_space = split_time ? 512 * modes : paired_checkpoint ? paired_small : paired_size;
    const bool paired = paired_space && paired_space <= allowance - per_worker;
    const uint64_t scratch_size = striped ? n * stripe_rows : paired ? paired_space :
        checkpoint ? pc::add(pc::mul(tile_work, owners), pc::mul(boundaries, owners > 1 ? 2 : 1)) : full;
    if (scratch_size > allowance - per_worker) return false;
    per_worker += scratch_size;
    std::vector<T> partial(partial_size);
    std::atomic<bool> valid{true};
    auto packed_segment = [&](uint64_t i) {
        return tiled && i < n / 128 * 128 ? i / 128 * 128 + i % (128 / lane_count) * lane_count + i % 128 / (128 / lane_count) : i;
    };
    auto write_gradient = [&](const T* sums, T* result) HWY_ATTR {
        bool finite = true;
        if (striped) {
            if (compact) std::fill_n(result, length * dim, T(0));
            for (uint64_t c = 0; c < d; ++c) {
                const T* src = sums + c * n;
                T* dst = result + (compact ? channels[c] : c);
                T prev = 0;
                for (unsigned lane = 0; lane < lane_count; ++lane) {
                    const uint64_t start = lane * (n / lane_count);
                    const uint64_t count = std::min(n / lane_count, segments - std::min(segments, start));
                    for (uint64_t i = 0; i < count; ++i) {
                        const uint64_t t = i * lane_count + lane;
                        *dst = prev - src[t];
                        finite = pc::ScalarOps<T>::finite(*dst) && finite;
                        prev = src[t];
                        dst += dim;
                    }
                }
                *dst = prev;
            }
            return finite;
        }
        if (compact) {
            std::fill_n(result, length * dim, T(0));
            for (uint64_t slot = 0; slot < d; ++slot) {
                const uint64_t c = channels[slot];
                if (ta && c + 1 == dimension) continue;
                const uint64_t column = ll && c >= dim ? c - dim : c;
                T prev = 0;
                for (uint64_t t = 0; t < length; ++t) {
                    const T next = t + 1 == length ? T(0) : sums[slot * n + packed_segment(n - 1 - (ll ? 2 * t + (c < dim) : t))];
                    T& dst = result[t * dim + column];
                    dst += prev - next;
                    finite = pc::ScalarOps<T>::finite(dst) && finite;
                    prev = next;
                }
            }
            return finite;
        }
        for (uint64_t c = 0; c < dim; ++c) {
            const T* p = sums + c * n;
            T prev = 0;
            for (uint64_t t = 0; t + 1 < length; ++t) {
                const T next = ll ? p[n - 2 - 2 * t] + p[dim * n + n - 1 - 2 * t] : p[packed_segment(n - 1 - t)];
                result[t * dim + c] = prev - next;
                finite = pc::ScalarOps<T>::finite(result[t * dim + c]) && finite;
                prev = next;
            }
            result[(length - 1) * dim + c] = prev;
        }
        return finite;
    };
    auto work = [&](uint64_t begin, uint64_t end) HWY_ATTR {
        auto space = hwy::AllocateAligned<T>(per_worker);
        if (!space) throw std::bad_alloc();
        T* dx = space.get();
        T* endpoints = dx + size * (Backward && !partial_size ? 2 : 1);
        T* scratch = endpoints + endpoints_size;
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t sample = task / parts, part = task % parts;
            T* grad = partial_size ? partial.data() + task * size : dx + size;
            const T* x = path + sample * length * dim;
            bool ok = true;
            for (uint64_t slot = 0; slot < d; ++slot) {
                const uint64_t c = compact ? channels[slot] : slot;
                T* p = left[slot] < 0 ? nullptr : endpoints + left[slot] * n;
                T* q = right[slot] < 0 ? nullptr : endpoints + right[slot] * n;
                const T start = pc::point(x, 0, c, dim, segments, ta, ll, time);
                const T finish = pc::point(x, segments, c, dim, segments, ta, ll, time);
                uint64_t t = 0;
                const Simd<T> tag;
                constexpr unsigned lanes = unsigned(vec_batch_lanes<T>);
                if constexpr (lanes > 1) if (!ta && !ll && dim <= INT32_MAX / lanes) {
                    const hn::RebindToSigned<Simd<T>> di;
                    using I = hn::TFromD<decltype(di)>;
                    const auto offsets = hn::Mul(hn::Iota(di, 0), hn::Set(di, I(dim * (striped ? n / lanes : 1))));
                    const auto tile_offsets = hn::Mul(hn::Iota(di, 0), hn::Set(di, I(dim * (128 / lanes))));
                    auto good = hn::Eq(hn::Zero(tag), hn::Zero(tag));
                    for (; n - t >= lanes; t += lanes) {
                        const bool full_tile = tiled && t < n / 128 * 128;
                        uint64_t i = striped ? t / lanes : Backward ? n - t - lanes : t;
                        if (full_tile) {
                            i = t / 128 * 128 + t % 128 / lanes;
                            if constexpr (Backward) i = n - 1 - i - (lanes - 1) * (128 / lanes);
                        }
                        const T* pos = x + i * dim + c;
                        auto a_offsets = full_tile ? tile_offsets : offsets, b_offsets = a_offsets;
                        if (striped && n != segments && i + (lanes - 1) * (n / lanes) >= segments) {
                            a_offsets = hn::Min(offsets, hn::Set(di, I((segments - i) * dim)));
                            b_offsets = hn::Min(offsets, hn::Set(di, I((segments - i - 1) * dim)));
                        }
                        const auto a = hn::GatherIndex(tag, pos, a_offsets), b = hn::GatherIndex(tag, pos + dim, b_offsets);
                        auto v = hn::Sub(b, a);
                        const auto half = hn::Mul(v, hn::Set(tag, T(.5)));
                        if (p) {
                            auto mean = hn::Add(hn::Sub(a, hn::Set(tag, start)), half);
                            if constexpr (Backward) if (!striped) mean = hn::Reverse(tag, mean);
                            hn::StoreU(mean, tag, p + t);
                        }
                        if (q) {
                            auto mean = hn::Add(hn::Sub(hn::Set(tag, finish), b), half);
                            if constexpr (Backward) if (!striped) mean = hn::Reverse(tag, mean);
                            hn::StoreU(mean, tag, q + t);
                        }
                        const auto mag = hn::Abs(v);
                        good = hn::And(good, hn::And(hn::Le(mag, hn::Set(tag, T(65536))),
                            hn::Or(hn::Eq(v, hn::Zero(tag)), hn::Ge(mag, hn::Set(tag, T(1e-20))))));
                        if constexpr (Backward) if (!striped) v = hn::Reverse(tag, v);
                        hn::StoreU(v, tag, dx + slot * n + t);
                    }
                    ok = hn::AllTrue(tag, good) && ok;
                }
                for (; t < n; ++t) {
                    const uint64_t i = Backward ? n - 1 - t : t;
                    const T a = pc::point(x, i, c, dim, n, ta, ll, time);
                    const T b = pc::point(x, i + 1, c, dim, n, ta, ll, time);
                    const T v = b - a;
                    if (p) p[t] = (a - start) + T(.5) * v;
                    if (q) q[t] = (finish - b) + T(.5) * v;
                    dx[slot * n + t] = v;
                    ok = ok && std::abs(v) <= T(65536) && (v == 0 || std::abs(v) >= T(1e-20));
                }
            }
            if constexpr (Backward) std::fill_n(grad, size, T(0));
            for (uint64_t w = words.count * part / parts; ok && w < words.count * (part + 1) / parts; ++w) {
                const uint64_t offset = sample * words.width + words.packed[2 * words.count + w];
                const uint64_t* word = letters(w);
                const unsigned depth = unsigned(words.packed[w]);
                T terminal[Capacity] = {};
                const T* saved = coefs ? coefs + offset : nullptr;
                const T* adjoints = nullptr;
                if constexpr (Backward) {
                    if (coefs) adjoints = seeds + offset;
                    else {
                        if (depth) terminal[depth - 1] = seeds[sample * words.count + w];
                        adjoints = terminal;
                    }
                }
                const T* p = depth >= 2 && left[word[0]] >= 0 ? endpoints + left[word[0]] * n : nullptr;
                const T* q = depth >= 2 && right[word[depth - 1]] >= 0 ? endpoints + right[word[depth - 1]] * n : nullptr;
                if (striped) ok = striped_word<T, Backward, Capacity>(word, depth, unsigned(n), dx, grad, p, q,
                    saved, adjoints, Backward ? nullptr : out + offset, prefixes, scratch);
                else if (paired && depth >= 2) ok = paired_cpu_word<Capacity>(word, depth, n, dx, p, q, out + offset, scratch, paired_checkpoint, tiled, split_time);
                else ok = scan_cpu_word<T, Backward, Capacity>(word, depth, n, dx, grad, p, q,
                    saved, adjoints,
                    Backward ? nullptr : out + offset, prefixes, scratch, checkpoint, owners, tiled);
            }
            if constexpr (Backward) {
                for (uint64_t t = 0; ok && t < size; ++t) ok = pc::ScalarOps<T>::finite(grad[t]);
                if (!partial_size) ok = write_gradient(grad, out + sample * length * dim) && ok;
            }
            if (!ok) valid.store(false, std::memory_order_relaxed);
        }
    };
    if (workers == 1) work(0, jobs_count);
    else spawn_batch_threads(jobs_count, jobs, work);
    if (!valid.load(std::memory_order_relaxed)) return false;
    if constexpr (Backward) if (partial_size) {
        for (uint64_t sample = 0; sample < batch; ++sample) {
            T* sums = partial.data() + sample * parts * size;
            for (uint64_t part = 1; part < parts; ++part)
                for (uint64_t t = 0; t < size; ++t) sums[t] += sums[part * size + t];
            if (!write_gradient(sums, out + sample * length * dim)) return false;
        }
    }
    return true;
}
