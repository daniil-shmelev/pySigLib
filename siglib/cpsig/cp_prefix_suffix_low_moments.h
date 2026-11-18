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


// Use a three-letter prefix for levels three through six. Stack the moment
// factors along the segment axis so that a tile needs one matrix product.
template<bool Backward, unsigned Degree, unsigned D, typename T>
void low_moment_path(const T* path, const T* seed, T* out, uint64_t n, bool scalar,
    uint64_t tile, int jobs, T* storage) {
    constexpr bool Quintic = Degree >= 5, Sextic = Degree == 6;
    constexpr unsigned D2 = D * D, rows = D2 * D, cols = 1 + D + (Quintic ? D2 : 0) + (Sextic ? rows : 0);
    constexpr unsigned terms = Quintic ? 3 : 2;
    auto take = [&](uint64_t size) { T* p = storage; storage += size; return p; };
    T* product = take(rows * cols); T* left = take(tile * terms * rows); T* right = take(tile * terms * cols);
    T* suffix = Quintic ? take(n * D2) : nullptr;
    T* suffix3 = Sextic ? take(n * rows) : nullptr;
    T* dl = Backward ? take(tile * terms * rows) : nullptr;
    T* dr = Backward ? take(tile * terms * cols) : nullptr;
    T* source = Backward ? take(n * D2) : nullptr;
    T pre[D2]{}, dsuf[D2]{}, start[D]{}, end[D]{};
    T carry3[Sextic && Backward ? rows : 1]{};
    const T* last = path + n * D;
    if constexpr (Quintic) {
        T acc[D2]{};
        T third[Sextic ? rows : 1]{};
        for (uint64_t i = n; i-- > 0;) {
            const T* u = path + i * D;
            T v[D], q[D];
            for (unsigned c = 0; c < D; ++c) {
                v[c] = u[D + c] - u[c]; q[c] = (last[c] - u[D + c]) + v[c] / 2;
            }
            if constexpr (Sextic) {
                std::copy_n(third, rows, suffix3 + i * rows);
                for (unsigned x = 0; x < D; ++x) for (unsigned y = 0; y < D; ++y)
                    for (unsigned z = 0; z < D; ++z)
                        third[(x * D + y) * D + z] += v[x] * (acc[y * D + z] + v[y] * (q[z] / 2 - v[z] / 12));
            }
            for (unsigned a = 0; a < D; ++a) for (unsigned b = 0; b < D; ++b) {
                const unsigned j = a * D + b;
                suffix[i * D2 + j] = acc[j];
                acc[j] += v[a] * q[b];
            }
        }
    }
    auto levels = [&](unsigned row, unsigned col) {
        if (!col) return uint64_t(scalar + D + D2 + row);
        if (col <= D) return uint64_t(scalar + D + D2 + rows + row * D + col - 1);
        if (col <= D + D2) return uint64_t(scalar + D + D2 + rows + rows * D + row * D2 + col - 1 - D);
        return uint64_t(scalar + D + D2 + rows + rows * D + rows * D2 + row * rows + col - 1 - D - D2);
    };
    if constexpr (Backward) {
        std::fill_n(out, (n + 1) * D, T(0));
        for (unsigned x = 0; x < rows; ++x) for (unsigned y = 0; y < cols; ++y)
            product[x * cols + y] = seed[levels(x, y)];
    }
    CpuMatrices<T> matrices{jobs};
    for (uint64_t first = 0; first < n; first += tile) {
        const uint64_t count = std::min(tile, n - first), axis = count * terms;
        for (uint64_t j = 0; j < count; ++j) {
            const uint64_t i = first + j;
            const T* u = path + i * D;
            T* a = left + j * terms * rows; T* b = right + j * terms * cols;
            T v[D], p[D], q[D];
            for (unsigned c = 0; c < D; ++c) {
                v[c] = u[D + c] - u[c];
                p[c] = (u[c] - path[c]) + v[c] / 2;
                q[c] = (last[c] - u[D + c]) + v[c] / 2;
            }
            b[0] = 1; b[cols] = 0;
            for (unsigned x = 0; x < D; ++x) {
                b[1 + x] = q[x]; b[cols + 1 + x] = v[x];
                if constexpr (Quintic) b[2 * cols + 1 + x] = 0;
            }
            if constexpr (Quintic) b[2 * cols] = 0;
            for (unsigned x = 0; x < D; ++x) for (unsigned y = 0; y < D; ++y) {
                const unsigned xy = x * D + y;
                const T h = pre[xy] + (p[x] / 2 - v[x] / 12) * v[y];
                pre[xy] += p[x] * v[y];
                if constexpr (Backward) source[i * D2 + xy] = h;
                const T h1 = (-p[x] / 12) * v[y], h2 = (v[x] / 720) * v[y];
                for (unsigned z = 0; z < D; ++z) {
                    const unsigned row = xy * D + z;
                    a[row] = h * v[z]; a[rows + row] = h1 * v[z];
                    if constexpr (Quintic) a[2 * rows + row] = h2 * v[z];
                    if constexpr (Sextic) {
                        const unsigned col = 1 + D + D2 + row;
                        const T s = suffix[i * D2 + y * D + z];
                        b[col] = suffix3[i * rows + row] + v[x] * (s / 2 + v[y] * (q[z] / 6 - v[z] / 24));
                        b[cols + col] = v[x] * (s + v[y] * (q[z] / 2 - v[z] / 10));
                        b[2 * cols + col] = v[x] * v[y] * q[z];
                    }
                }
                if constexpr (Quintic) {
                    b[1 + D + xy] = suffix[i * D2 + xy] + v[x] * (q[y] / 2 - v[y] / 12);
                    b[cols + 1 + D + xy] = v[x] * q[y];
                    b[2 * cols + 1 + D + xy] = v[x] * v[y];
                }
            }
        }
        if constexpr (!Backward) {
            matrices.gemm({left, rows, axis, 1, rows}, {right, axis, cols, cols, 1},
                {product, rows, cols, cols, 1}, !first);
        } else {
            matrices.gemm({right, axis, cols, cols, 1}, {product, cols, rows, 1, cols},
                {dl, axis, rows, rows, 1}, true);
            matrices.gemm({left, axis, rows, rows, 1}, {product, rows, cols, cols, 1},
                {dr, axis, cols, cols, 1}, true);
            for (uint64_t j = 0; j < count; ++j) {
                const uint64_t i = first + j;
                const T* u = path + i * D; const T* g = dl + j * terms * rows;
                const T* b = dr + j * terms * cols;
                T v[D], p[D], q[D], dv[D], dp[D]{}, dq[D];
                for (unsigned c = 0; c < D; ++c) {
                    v[c] = u[D + c] - u[c]; p[c] = u[c] - path[c] + v[c] / 2;
                    q[c] = last[c] - u[D + c] + v[c] / 2;
                    dv[c] = b[cols + 1 + c]; dq[c] = b[1 + c];
                }
                for (unsigned x = 0; x < D; ++x) for (unsigned y = 0; y < D; ++y) {
                    const unsigned xy = x * D + y, row = xy * D;
                    const T h = source[i * D2 + xy], h1 = (-p[x] / 12) * v[y], h2 = (v[x] / 720) * v[y];
                    T g0 = 0, g1 = 0, g2 = 0;
                    for (unsigned z = 0; z < D; ++z) {
                        g0 += g[row + z] * v[z]; g1 += g[rows + row + z] * v[z];
                        dv[z] += h * g[row + z] + h1 * g[rows + row + z];
                        if constexpr (Quintic) {
                            g2 += g[2 * rows + row + z] * v[z]; dv[z] += h2 * g[2 * rows + row + z];
                        }
                    }
                    source[i * D2 + xy] = g0;
                    dp[x] += v[y] * (g0 / 2 - g1 / 12);
                    dv[x] += v[y] * (g2 / 720 - g0 / 12);
                    dv[y] += (p[x] / 2 - v[x] / 12) * g0 - (p[x] / 12) * g1 + (v[x] / 720) * g2;
                    if constexpr (Quintic) {
                        const T s0 = b[1 + D + xy], s1 = b[cols + 1 + D + xy];
                        const T s2 = b[2 * cols + 1 + D + xy], carry = dsuf[xy];
                        dv[x] += s0 * (q[y] / 2 - v[y] / 12) + (s1 + carry) * q[y] + s2 * v[y];
                        dq[y] += v[x] * (s0 / 2 + s1 + carry);
                        dv[y] += v[x] * (s2 - s0 / 12);
                        dsuf[xy] += s0;
                    }
                }
                if constexpr (Sextic) {
                    for (unsigned x = 0; x < D; ++x) for (unsigned y = 0; y < D; ++y)
                        for (unsigned z = 0; z < D; ++z) {
                            const unsigned yz = y * D + z, xyz = x * D2 + yz, col = 1 + D + D2 + xyz;
                            const T s0 = b[col], s1 = b[cols + col], s2 = b[2 * cols + col], carry = carry3[xyz];
                            const T qscale = s0 / 6 + s1 / 2 + s2 + carry / 2;
                            const T vscale = s0 / 24 + s1 / 10 + carry / 12;
                            const T pair = q[z] * qscale - v[z] * vscale, ss = s0 / 2 + s1 + carry;
                            dv[x] += suffix[i * D2 + yz] * ss + v[y] * pair;
                            dv[y] += v[x] * pair;
                            dq[z] += v[x] * v[y] * qscale; dv[z] -= v[x] * v[y] * vscale;
                            dsuf[yz] += v[x] * ss; carry3[xyz] += s0;
                        }
                }
                for (unsigned c = 0; c < D; ++c) {
                    const T mid = (dp[c] - dq[c]) / 2;
                    out[i * D + c] += mid - dv[c]; out[(i + 1) * D + c] += mid + dv[c];
                    start[c] -= dp[c]; end[c] += dq[c];
                }
            }
        }
    }
    if constexpr (Backward) {
        std::copy_n(seed + scalar + D, D2, pre);
        for (uint64_t i = n; i-- > 0;) {
            const T* u = path + i * D;
            T v[D], p[D], dp[D]{}, dv[D]{};
            for (unsigned x = 0; x < D; ++x) {
                v[x] = u[D + x] - u[x]; p[x] = u[x] - path[x] + v[x] / 2;
            }
            for (unsigned x = 0; x < D; ++x) for (unsigned y = 0; y < D; ++y) {
                const unsigned xy = x * D + y;
                dp[x] += pre[xy] * v[y]; dv[y] += pre[xy] * p[x]; pre[xy] += source[i * D2 + xy];
            }
            for (unsigned c = 0; c < D; ++c) {
                start[c] -= dp[c]; out[i * D + c] += dp[c] / 2 - dv[c];
                out[(i + 1) * D + c] += dp[c] / 2 + dv[c];
            }
        }
        for (unsigned c = 0; c < D; ++c) {
            out[c] += start[c] - seed[scalar + c]; out[n * D + c] += end[c] + seed[scalar + c];
        }
    } else {
        if (scalar) out[0] = 1;
        for (unsigned c = 0; c < D; ++c) out[scalar + c] = last[c] - path[c];
        std::copy_n(pre, D2, out + scalar + D);
        for (unsigned x = 0; x < rows; ++x) for (unsigned y = 0; y < cols; ++y)
            out[levels(x, y)] = product[x * cols + y];
    }
}
