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
template<typename T>
void quadratic_gradient(const T* path, const T* g, T* out, uint64_t n, uint64_t d, uint64_t block,
    T* skew, T* spans, uint64_t begin, uint64_t end) {
    const T* second = g + d;
    const T* last = path + n * d;
    const T* penultimate = last - d;
    bool scaled_skew = false;
    for (uint64_t c = 0; c < d; ++c) {
        T first_value = g[c], last_value = g[c];
        for (uint64_t a = 0; a < d; ++a) {
            const T row = second[c * d + a], col = second[a * d + c];
            skew[c * d + a] = row - col;
            scaled_skew |= !std::isfinite(skew[c * d + a]);
            if (!begin) {
                const T first_half = T(0.5) * (path[d + a] - path[a]);
                const T last_half = T(0.5) * (last[a] - penultimate[a]);
                first_value += row * first_half + col * first_half + row * (last[a] - path[d + a]);
                last_value += row * last_half + col * last_half + col * (penultimate[a] - path[a]);
            }
        }
        if (!begin) {
            out[c] = -first_value;
            out[n * d + c] = last_value;
        }
    }
    if (scaled_skew)
        for (uint64_t c = 0; c < d; ++c)
            for (uint64_t a = 0; a < d; ++a)
                skew[c * d + a] = T(0.5) * second[c * d + a] - T(0.5) * second[a * d + c];
    const T span_scale = scaled_skew ? T(1) : T(0.5);
    for (uint64_t i = std::max<uint64_t>(1, begin); i < end; i += block) {
        const uint64_t count = std::min(block, end - i);
        for (uint64_t x = 0; x < count * d; ++x)
            spans[x] = (path[(i + 1) * d + x] - path[(i - 1) * d + x]) * span_scale;
        factor_gemm(Matrix<T>{spans, count, d, d, 1}, Matrix<T>{skew, d, d, 1, d},
            Matrix<T>{out + i * d, count, d, d, 1}, true);
    }
}

// On a linear segment the mean of prefix[a] * suffix[c] is
// P[a] * Q[c] - v[a] * v[c] / 12, where P and Q are the midpoint displacements.
template<bool Backward, typename T>
void midpoint_path(const T* path, const T* derivative, T* output, uint64_t n, uint64_t d,
    bool cubic, bool scalar, uint64_t block, T* storage, uint64_t begin, uint64_t end, T* boundary) {
    const uint64_t width = d + (cubic ? d * d : 0);
    T* matrix = storage;
    T* features = matrix + width * d;
    if constexpr (Backward) {
        if (!cubic) {
            quadratic_gradient(path, derivative + scalar, output, n, d, block, matrix, features, begin, end);
            return;
        }
    }
    T* v = features + block * width;
    T* q = v + block * d;
    T* dp = q + d;
    T* dq = dp + d;
    T* df = dq + d;
    T* dv = df + (Backward ? block * width : 0);
    const T* last = path + n * d;
    if constexpr (Backward) {
        std::fill_n(output + begin * d, (end - begin + (boundary == nullptr)) * d, T(0));
        const T* g = derivative + scalar;
        std::copy_n(g + d, d * d, matrix);
        const T* third = g + d + d * d;
        for (uint64_t a = 0; a < d; ++a)
            for (uint64_t c = 0; c < d; ++c)
                for (uint64_t b = 0; b < d; ++b)
                    matrix[(d + a * d + c) * d + b] = third[(a * d + b) * d + c];
        if (boundary) std::fill_n(boundary, 3 * d, T(0));
        else for (uint64_t c = 0; c < d; ++c) {
            output[c] = -g[c];
            output[n * d + c] = g[c];
        }
    } else {
        if (scalar) *output = 1;
        for (uint64_t c = 0; c < d; ++c) output[scalar + c] = last[c] - path[c];
    }

    for (uint64_t first = begin; first < end; first += block) {
        const uint64_t count = std::min(block, end - first);
        for (uint64_t i = 0; i < count; ++i) {
            const T* x = path + (first + i) * d;
            T* inc = v + i * d;
            T* f = features + i * width;
            for (uint64_t c = 0; c < d; ++c) {
                inc[c] = x[d + c] - x[c];
                f[c] = (x[c] - path[c]) + T(0.5) * inc[c];
                if (cubic) q[c] = (last[c] - x[d + c]) + T(0.5) * inc[c];
            }
            if (cubic) {
                // Integral of prefix[a] * suffix[c] on the linear segment.
                for (uint64_t a = 0; a < d; ++a) {
                    T* row = f + d + a * d;
                    vec_mult_assign(row, q, f[a], d);
                    vec_mult_add(row, inc, -inc[a] / T(12), d);
                }
            }
        }
        if constexpr (!Backward) {
            factor_gemm(Matrix<T>{features, width, count, 1, width}, Matrix<T>{v, count, d, d, 1},
                Matrix<T>{matrix, width, d, d, 1}, first == begin);
        } else {
            factor_gemm(Matrix<T>{features, count, width, width, 1}, Matrix<T>{matrix, width, d, d, 1},
                Matrix<T>{dv, count, d, d, 1}, true);
            factor_gemm(Matrix<T>{v, count, d, d, 1}, Matrix<T>{matrix, d, width, 1, d},
                Matrix<T>{df, count, width, width, 1}, true);
            for (uint64_t i = 0; i < count; ++i) {
                const T* x = path + (first + i) * d;
                const T* inc = v + i * d;
                const T* g = df + i * width;
                T* grad_v = dv + i * d;
                std::copy_n(g, d, dp);
                std::fill_n(dq, d, T(0));
                const T* p = features + i * width;
                for (uint64_t c = 0; c < d; ++c) q[c] = (last[c] - x[d + c]) + T(0.5) * inc[c];
                for (uint64_t a = 0; a < d; ++a) {
                    const T* row = g + d + a * d;
                    dp[a] += dot_product_mult_add(q, row, dq, p[a], d);
                    grad_v[a] -= dot_product_mult_add(inc, row, grad_v, -inc[a] / T(12), d) / T(12);
                }
                T* before = output + (first + i) * d;
                T* after = boundary && first + i + 1 == end ? boundary + 2 * d : before + d;
                T* start_gradient = boundary ? boundary : output;
                T* end_gradient = boundary ? boundary + d : output + n * d;
                for (uint64_t c = 0; c < d; ++c) {
                    const T midpoint = T(0.5) * (dp[c] - dq[c]);
                    before[c] += midpoint - grad_v[c];
                    after[c] += midpoint + grad_v[c];
                    start_gradient[c] -= dp[c];
                    end_gradient[c] += dq[c];
                }
            }
        }
    }
    if constexpr (!Backward) {
        T* second = output + scalar + d;
        std::copy_n(matrix, d * d, second);
        if (cubic) {
            T* third = second + d * d;
            for (uint64_t a = 0; a < d; ++a)
                for (uint64_t b = 0; b < d; ++b)
                    for (uint64_t c = 0; c < d; ++c)
                        third[(a * d + b) * d + c] = matrix[(d + a * d + c) * d + b];
        }
    }
}

template<bool Backward, typename T>
void midpoint_batch(const T* path, const T* derivative, T* output, uint64_t batch, uint64_t n,
    uint64_t d, uint64_t degree, bool scalar, int jobs, uint64_t sig_stride) {
    jobs = available_threads(jobs);
    const uint64_t width = degree == 3 ? add(d, mul(d, d)) : d;
    const bool quadratic_adjoint = Backward && degree == 2;
    const uint64_t row = quadratic_adjoint ? d : mul(Backward ? 2 : 1, add(width, d));
    const uint64_t block = std::min(n, std::max<uint64_t>(1, (256ULL << 10) / mul(row, sizeof(T))));
    const uint64_t elements = add(add(mul(width, d), mul(block, row)), quadratic_adjoint ? 0 : mul(3, d));
    const uint64_t path_stride = mul(n + 1, d);
    const uint64_t partial_stride = Backward ? (quadratic_adjoint ? 0 : 3 * d) : sig_stride;
    uint64_t parts = 1;
    if (uint64_t(jobs) > batch && n >= 256) {
        parts = std::min({uint64_t(jobs) / batch, n / 64, 1 + (n - 1) / std::max<uint64_t>(1, 131072 / (width * d))});
        if (partial_stride && parts > (8ULL << 20) / sizeof(T) / partial_stride / batch) parts = 1;
    }
    const uint64_t tasks = mul(batch, parts);
    const auto cut = [n, parts](uint64_t t) { return (n / parts) * t + std::min(t, n % parts); };
    std::vector<T> partials(parts > 1 ? mul(tasks, partial_stride) : 0);
    for_paths<T>(tasks, jobs, elements, "CPU prefix-suffix", [&](uint64_t task, T* memory) {
        const uint64_t i = task / parts, part = task % parts;
        T* extra = partials.empty() ? nullptr : partials.data() + task * partial_stride;
        T* result = !Backward && extra ? extra : output + i * (Backward ? path_stride : sig_stride);
        midpoint_path<Backward>(path + i * path_stride, Backward ? derivative + i * sig_stride : nullptr, result,
            n, d, degree == 3, scalar, block, memory, cut(part), cut(part + 1), Backward ? extra : nullptr);
    });
    if (partials.empty()) return;
    for (uint64_t i = 0; i < batch; ++i) {
        const T* extra = partials.data() + i * parts * partial_stride;
        if constexpr (Backward) {
            T* result = output + i * path_stride;
            const T* g = derivative + i * sig_stride + scalar;
            std::copy_n(g, d, result + n * d);
            vec_mult_add(result, g, T(-1), d);
            for (uint64_t t = 0; t < parts; ++t, extra += partial_stride) {
                vec_mult_add(result, extra, T(1), d);
                vec_mult_add(result + n * d, extra + d, T(1), d);
                vec_mult_add(result + cut(t + 1) * d, extra + 2 * d, T(1), d);
            }
        } else {
            T* result = output + i * sig_stride;
            std::copy_n(extra, sig_stride, result);
            for (uint64_t t = 1; t < parts; ++t)
                vec_mult_add(result + scalar + d, extra + t * sig_stride + scalar + d, T(1), sig_stride - scalar - d);
        }
    }
}
