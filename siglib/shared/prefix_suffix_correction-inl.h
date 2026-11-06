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

// Included inside each Highway target namespace, or the CUDA namespace.

#ifdef __CUDACC__
#define PS_CORRECTION_HD __host__ __device__
#else
#define PS_CORRECTION_HD
#endif

struct Algebra {
    uint64_t d, h, size, correction_degree, raw_d;
    const uint64_t* power;
    const uint64_t* offset;
    const uint64_t* raw_offset;
};

PS_CORRECTION_HD inline uint64_t minimum(uint64_t a, uint64_t b) { return a < b ? a : b; }

PS_CORRECTION_HD inline uint64_t ranks(const Algebra& l, uint64_t a, uint64_t b) {
    uint64_t count = 0;
    for (uint64_t r = 1; r <= minimum(l.correction_degree, a + b); ++r)
        for (uint64_t s = r > b ? r - b : 1; s <= minimum(r, a); ++s)
            count += l.power[minimum(s, r - s)];
    return count;
}

struct Plan {
    Layout layout, raw;
    uint64_t n, correction_degree = 1, size, scan_count, slots, scratch_size;
    uint64_t nodes_count, segments, max_rows = 0, max_left = 0, max_right = 0;
    uint64_t increments, prefixes, suffixes, dprefixes, dsuffixes, gradient;
    uint64_t left, right, dleft, dright, work, matrix_work, side, count;
    std::vector<double> nodes, weights;

    Plan(uint64_t d, uint64_t raw_d, uint64_t degree, uint64_t points, uint64_t split,
        uint64_t correction_len, uint64_t workers, uint64_t itemsize, bool backward)
        : layout(d, degree, split), raw(raw_d, degree), n(points ? points - 1 : 0) {
        if (!correction_len) throw std::invalid_argument("missing prefix-suffix correction");
        uint64_t width = 0;
        for (uint64_t r = 2; r <= degree; ++r) {
            width = add(width, raw.power[r]);
            if (width == correction_len) { correction_degree = r; break; }
            if (width > correction_len) break;
        }
        if (correction_degree == 1) throw std::invalid_argument("invalid prefix-suffix correction length");
        if (!split) {
            const uint64_t balanced = degree - degree / 2;
            uint64_t best = balanced;
            long double minimum_work = std::numeric_limits<long double>::infinity();
            for (uint64_t candidate = 1; candidate <= degree; ++candidate) {
                const uint64_t h = std::max({degree / 2, candidate - 1, degree - candidate});
                const Algebra l{d, h, layout.offset[h + 1], correction_degree, raw_d,
                    layout.power.data(), layout.offset.data(), raw.offset.data()};
                const long double scan = (backward ? 12.L : 4.L) * h * correction_degree * l.size;
                long double cost = scan;
                for (uint64_t k = h + 1; k <= degree; ++k) {
                    const uint64_t a = candidate == balanced ? (k + 1) / 2 : candidate;
                    const long double rows = ((degree + 1) / 2) * static_cast<long double>(ranks(l, a, k - a));
                    cost += rows * ((backward ? 4.L : 2.L) * layout.power[k] + layout.power[a] + layout.power[k - a]);
                    cost += ((degree + 1) / 2) * scan;
                }
                if (cost < minimum_work || (cost == minimum_work && candidate == balanced)) {
                    minimum_work = cost; best = candidate;
                }
            }
            if (best != balanced) layout.set_split(best);
        }
        size = layout.offset[layout.scan_degree + 1];
        scan_count = mul(add(n, 1), size);
        nodes_count = (degree + 1) / 2;
        nodes.resize(nodes_count); weights.resize(nodes_count);
        for (uint64_t j = 0; j < (nodes_count + 1) / 2; ++j) {
            double z = std::cos(std::acos(-1.) * (double(j) + 0.75) / (double(nodes_count) + 0.5));
            double derivative = 0;
            bool converged = false;
            for (int iteration = 0; iteration < 64; ++iteration) {
                double previous = 0, value = 1;
                for (uint64_t k = 1; k <= nodes_count; ++k) {
                    const double next = ((2. * k - 1.) * z * value - (k - 1.) * previous) / k;
                    previous = value; value = next;
                }
                derivative = nodes_count * (z * value - previous) / (z * z - 1.);
                const double change = value / derivative;
                if (std::abs(change) <= 2e-16) { converged = true; break; }
                z -= change;
            }
            if (!converged) throw std::runtime_error("prefix-suffix quadrature did not converge");
            nodes[j] = (1. - z) / 2.; nodes[nodes_count - j - 1] = (1. + z) / 2.;
            weights[j] = weights[nodes_count - j - 1] = 1. / ((1. - z * z) * derivative * derivative);
        }
        const Algebra l = algebra();
        for (uint64_t k = layout.scan_degree + 1; k <= degree; ++k) {
            const uint64_t a = layout.left_degree(k), b = k - a;
            max_rows = std::max(max_rows, mul(nodes_count, ranks(l, a, b)));
            max_left = std::max(max_left, layout.power[a]);
            max_right = std::max(max_right, layout.power[b]);
        }
        const uint64_t factor_count = mul(max_rows, mul(backward ? 2 : 1, add(max_left, max_right)));
        const uint64_t factor_limit = option("PYSIGLIB_PS_FACTOR_BYTES", 64ULL << 20);
        segments = std::max<uint64_t>(1, std::min(n, factor_limit / mul(std::max<uint64_t>(1, factor_count), itemsize)));
        slots = std::max<uint64_t>(1, std::min(segments, workers));
        scratch_size = mul(add(layout.scan_degree, 12), size);
        increments = 0; prefixes = mul(n, d); suffixes = add(prefixes, scan_count);
        dprefixes = add(suffixes, scan_count);
        dsuffixes = backward ? add(dprefixes, scan_count) : dprefixes;
        gradient = backward ? add(dsuffixes, scan_count) : dsuffixes;
        left = backward ? add(gradient, mul(n, d)) : gradient;
        right = add(left, mul(mul(segments, max_rows), max_left));
        dleft = add(right, mul(mul(segments, max_rows), max_right));
        dright = backward ? add(dleft, mul(mul(segments, max_rows), max_left)) : dleft;
        work = backward ? add(dright, mul(mul(segments, max_rows), max_right)) : dright;
        matrix_work = add(work, mul(slots, scratch_size));
        side = tile_side(max_left, max_right, mul(segments, max_rows), option("PYSIGLIB_PS_MAX_TILE", 4096));
        count = add(matrix_work, std::max({max_left, max_right, mul(segments, max_rows)}) > INT_MAX ? mul(3, mul(side, side)) : 0);
        mul(count, itemsize);
    }

    Algebra algebra() const {
        return {layout.d, layout.scan_degree, size, correction_degree, raw.d,
            layout.power.data(), layout.offset.data(), raw.offset.data()};
    }
};

template<typename T>
PS_CORRECTION_HD T component(const Algebra& l, const T* v, const T* c, uint64_t level, uint64_t word) {
    if (level == 1) return v[word];
    if (level > l.correction_degree) return T(0);
    if (l.raw_d != l.d) {
        uint64_t index = 0, multiplier = 1;
        for (uint64_t k = 0; k < level; ++k) {
            const uint64_t letter = word % l.d;
            if (letter == l.raw_d) return T(0);
            index += letter * multiplier; multiplier *= l.raw_d; word /= l.d;
        }
        word = index;
    }
    return c[l.raw_offset[level] - l.raw_offset[2] + word];
}

template<typename T>
PS_CORRECTION_HD void zero(T* data, uint64_t count) { for (uint64_t i = 0; i < count; ++i) data[i] = T(0); }

template<typename T>
PS_CORRECTION_HD void copy(const T* input, T* output, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) output[i] = input[i];
}

template<typename T>
PS_CORRECTION_HD void multiply(const Algebra& l, const T* x, const T* y, T* out) {
    for (uint64_t k = 0; k <= l.h; ++k)
        for (uint64_t word = 0; word < l.power[k]; ++word) {
            T value = 0;
            for (uint64_t j = 0; j <= k; ++j)
                value += x[l.offset[j] + word / l.power[k - j]] * y[l.offset[k - j] + word % l.power[k - j]];
            out[l.offset[k] + word] = value;
        }
}

template<typename T>
PS_CORRECTION_HD void multiply_backprop(const Algebra& l, const T* x, const T* y, const T* adjoint, T* dx, T* dy) {
    for (uint64_t k = 0; k <= l.h; ++k)
        for (uint64_t j = 0; j <= k; ++j)
            for (uint64_t a = 0; a < l.power[j]; ++a)
                for (uint64_t b = 0; b < l.power[k - j]; ++b) {
                    const T g = adjoint[l.offset[k] + a * l.power[k - j] + b];
                    dx[l.offset[j] + a] += g * y[l.offset[k - j] + b];
                    dy[l.offset[k - j] + b] += g * x[l.offset[j] + a];
                }
}

template<typename T>
PS_CORRECTION_HD void exponential(const Algebra& l, const T* v, const T* c, T time, T* out, T* powers) {
    zero(powers, (l.h + 1) * l.size); powers[0] = 1;
    zero(out, l.size); out[0] = 1;
    for (uint64_t p = 1; p <= l.h; ++p) {
        T* current = powers + p * l.size;
        const T* previous = current - l.size;
        for (uint64_t k = p; k <= l.h; ++k)
            for (uint64_t word = 0; word < l.power[k]; ++word) {
                T value = 0;
                for (uint64_t r = 1; r <= minimum(l.correction_degree, k - p + 1); ++r)
                    value += previous[l.offset[k - r] + word / l.power[r]] * component(l, v, c, r, word % l.power[r]);
                current[l.offset[k] + word] = value * (time / T(p));
                out[l.offset[k] + word] += current[l.offset[k] + word];
            }
    }
}

template<typename T>
PS_CORRECTION_HD void exponential_backprop(const Algebra& l, const T* v, const T* c, T time,
    const T* adjoint, T* dv, T* out, T* powers, T* current, T* next) {
    exponential(l, v, c, time, out, powers);
    zero(current, l.size);
    for (uint64_t p = l.h; p > 0; --p) {
        zero(next, l.size);
        const T* previous = powers + (p - 1) * l.size;
        for (uint64_t k = p; k <= l.h; ++k)
            for (uint64_t r = 1; r <= minimum(l.correction_degree, k - p + 1); ++r)
                for (uint64_t a = 0; a < l.power[k - r]; ++a)
                    for (uint64_t b = 0; b < l.power[r]; ++b) {
                        const uint64_t index = l.offset[k] + a * l.power[r] + b;
                        const T g = (current[index] + adjoint[index]) * (time / T(p));
                        next[l.offset[k - r] + a] += g * component(l, v, c, r, b);
                        if (r == 1) dv[b] += g * previous[l.offset[k - 1] + a];
                    }
        T* swap = current; current = next; next = swap;
    }
}

// The block containing letter a determines r and s uniquely. Its integral is
// polynomial of degree at most m-1; ceil(m/2) Gauss-Legendre nodes are sufficient.
template<bool Backward, typename T>
PS_CORRECTION_HD void factors(const Algebra& l, uint64_t a, uint64_t b, const T* v, const T* c,
    const T* prefix, const T* suffix, uint64_t nodes_count, const double* nodes, const double* weights,
    T* left, T* right, const T* dleft, const T* dright, T* dprefix, T* dsuffix, T* dv, T* work) {
    const uint64_t p = l.power[a], q = l.power[b], size = l.size;
    T* e = work; T* f = e + size; T* h = f + size; T* k = h + size;
    T* dh = k + size; T* dk = dh + size; T* de = dk + size; T* df = de + size;
    T* powers = df + size; T* current = powers + (l.h + 1) * size; T* next = current + size;
    uint64_t row = 0;
    for (uint64_t node = 0; node < nodes_count; ++node) {
        const T time = T(nodes[node]), weight = T(weights[node]);
        exponential(l, v, c, time, e, powers); multiply(l, prefix, e, h);
        exponential(l, v, c, T(1) - time, f, powers); multiply(l, f, suffix, k);
        if constexpr (Backward) { zero(dh, size); zero(dk, size); }
        for (uint64_t r = 1; r <= minimum(l.correction_degree, a + b); ++r)
            for (uint64_t s = r > b ? r - b : 1; s <= minimum(r, a); ++s) {
                const uint64_t t = r - s, nleft = l.power[s], nright = l.power[t];
                const uint64_t nh = l.power[a - s], nk = l.power[b - t];
                const T* hs = h + l.offset[a - s]; const T* ks = k + l.offset[b - t];
                for (uint64_t u = 0; u < minimum(nleft, nright); ++u, ++row) {
                    if constexpr (!Backward) { zero(left + row * p, p); zero(right + row * q, q); }
                    if (s <= t) {
                        for (uint64_t x = 0; x < nh; ++x) {
                            const uint64_t index = row * p + x * nleft + u;
                            if constexpr (Backward) dh[l.offset[a - s] + x] += dleft[index];
                            else left[index] = hs[x];
                        }
                        for (uint64_t y = 0; y < nright; ++y) {
                            const T value = weight * component(l, v, c, r, u * nright + y);
                            for (uint64_t z = 0; z < nk; ++z) {
                                const uint64_t index = row * q + y * nk + z;
                                if constexpr (Backward) dk[l.offset[b - t] + z] += dright[index] * value;
                                else right[index] = value * ks[z];
                            }
                        }
                    } else {
                        for (uint64_t x = 0; x < nh; ++x)
                            for (uint64_t y = 0; y < nleft; ++y) {
                                const uint64_t index = row * p + x * nleft + y;
                                const T value = component(l, v, c, r, y * nright + u);
                                if constexpr (Backward) {
                                    dh[l.offset[a - s] + x] += dleft[index] * value;
                                    if (r == 1) dv[y] += dleft[index] * hs[x];
                                } else left[index] = hs[x] * value;
                            }
                        for (uint64_t z = 0; z < nk; ++z) {
                            const uint64_t index = row * q + u * nk + z;
                            if constexpr (Backward) dk[l.offset[b - t] + z] += dright[index] * weight;
                            else right[index] = weight * ks[z];
                        }
                    }
                }
            }
        if constexpr (Backward) {
            zero(de, size); zero(df, size);
            multiply_backprop(l, prefix, e, dh, dprefix, de);
            multiply_backprop(l, f, suffix, dk, df, dsuffix);
            exponential_backprop(l, v, c, time, de, dv, e, powers, current, next);
            exponential_backprop(l, v, c, T(1) - time, df, dv, f, powers, current, next);
        }
    }
}

template<typename T>
PS_CORRECTION_HD void scans(const Algebra& l, uint64_t n, const T* increments, const T* correction,
    uint64_t segment_stride, T* prefixes, T* suffixes, T* work) {
    zero(prefixes, (n + 1) * l.size); prefixes[0] = 1;
    zero(suffixes, (n + 1) * l.size); suffixes[n * l.size] = 1;
    T* powers = work + 8 * l.size;
    for (uint64_t i = 0; i < n; ++i) {
        exponential(l, increments + i * l.d, correction + i * segment_stride, T(1), work, powers);
        multiply(l, prefixes + i * l.size, work, prefixes + (i + 1) * l.size);
    }
    for (uint64_t i = n; i-- > 0;) {
        exponential(l, increments + i * l.d, correction + i * segment_stride, T(1), work, powers);
        multiply(l, work, suffixes + (i + 1) * l.size, suffixes + i * l.size);
    }
}

template<typename T>
PS_CORRECTION_HD void reverse_scans(const Algebra& l, uint64_t n, const T* increments, const T* correction,
    uint64_t segment_stride, const T* prefixes, const T* suffixes, T* dprefix, T* dsuffix, T* dv, T* work) {
    T* de = work + 6 * l.size; T* powers = work + 8 * l.size;
    T* current = powers + (l.h + 1) * l.size; T* next = current + l.size;
    for (uint64_t i = n; i-- > 0;) {
        const T* v = increments + i * l.d; const T* c = correction + i * segment_stride;
        exponential(l, v, c, T(1), work, powers); zero(de, l.size);
        multiply_backprop(l, prefixes + i * l.size, work, dprefix + (i + 1) * l.size, dprefix + i * l.size, de);
        exponential_backprop(l, v, c, T(1), de, dv + i * l.d, work, powers, current, next);
    }
    for (uint64_t i = 0; i < n; ++i) {
        const T* v = increments + i * l.d; const T* c = correction + i * segment_stride;
        exponential(l, v, c, T(1), work, powers); zero(de, l.size);
        multiply_backprop(l, work, suffixes + (i + 1) * l.size, dsuffix + i * l.size, de, dsuffix + (i + 1) * l.size);
        exponential_backprop(l, v, c, T(1), de, dv + i * l.d, work, powers, current, next);
    }
}

#undef PS_CORRECTION_HD
