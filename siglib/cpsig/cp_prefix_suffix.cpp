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

#include "cppch.h"
#include "cpsig.h"
#include "cp_prefix_suffix_detail.h"
#include "cp_signature.h"
#include "prefix_suffix.h"
#include <array>
#include <climits>
#include <cblas.h>
static_assert(sizeof(blasint) == 4);

namespace {
using namespace prefix_suffix;
enum class PreparationStrategy : uint64_t { automatic, stored, direct, streamed };

struct StreamGroup {
    uint64_t level = 0, a = 0, b = 0, p = 0, q = 0, qnext = 0;
    uint64_t left_offset = 0, right_offset = 0, next_offset = 0;
    bool paired = false;
};

struct CpuPlan {
    uint64_t n, plen, qlen, maxp, maxq, rows, side, scratch, psize, qsize, count, workspace_count, bytes, powers;
    uint64_t stream_segments = 0, stream_factor_count = 0, stream_elements_per_segment = 0;
    bool low, specialized, fused, paired;
    PreparationStrategy strategy = PreparationStrategy::stored;
    std::array<StreamGroup, 3> stream_groups{};
    uint64_t stream_group_count = 0;
    CpuPlan(const Layout& l, uint64_t segments, uint64_t batch, int jobs, uint64_t itemsize)
        : n(segments) {
        plen = l.offset[l.prefix_degree + 1]; qlen = l.offset[l.scan_degree + 1];
        maxp = l.power[(l.m + 1) / 2]; maxq = l.power[l.h];
        const uint64_t factor_bytes = option("PYSIGLIB_PS_FACTOR_BYTES", 64ULL << 20);
        rows = std::max<uint64_t>(1, std::min(mul(n, (l.m + 1) / 2), factor_bytes / mul(add(maxp, maxq), itemsize)));
        const uint64_t stored_rows = rows;
        side = tile_side(maxp, maxq, rows, option("PYSIGLIB_PS_MAX_TILE", 4096));
        low = option("PYSIGLIB_PS_LOW_DEPTH", 1, true) != 0 && l.m <= 6;
        paired = option("PYSIGLIB_PS_PAIR_LEVELS", 1, true) != 0;
        specialized = low && (l.d == 2 || l.d == 3 || l.d == 4 || l.d == 8);
        const uint64_t requested_strategy = option("PYSIGLIB_PS_CPU_STRATEGY", 0, true);
        if (requested_strategy > uint64_t(PreparationStrategy::streamed))
            throw std::invalid_argument("PYSIGLIB_PS_CPU_STRATEGY must be 0, 1, 2, or 3");
        const bool direct_eligible = specialized && l.scan_degree == l.h && l.scan_degree == 1
            && (l.m == 2 || l.m == 3);
        const bool stream_shape_eligible = specialized && l.scan_degree == l.h && l.m >= 4 && l.m <= 6;
        const bool automatic_stream = stream_shape_eligible && (
            (l.d == 2 && l.m == 6)
            || (l.d == 3 && (l.m >= 5 || batch >= 32))
            || (l.d == 4 && (l.m >= 5 || itemsize == sizeof(double) || batch >= 32))
            || (l.d == 8 && (l.m >= 5 || itemsize == sizeof(float) || batch >= 32)));
        if (requested_strategy == uint64_t(PreparationStrategy::direct) && !direct_eligible)
            throw std::invalid_argument("forced direct CPU prefix-suffix preparation is not available for this shape");
        if (requested_strategy == uint64_t(PreparationStrategy::streamed) && !stream_shape_eligible)
            throw std::invalid_argument("forced streamed CPU prefix-suffix preparation is not available for this shape");
        if (requested_strategy == uint64_t(PreparationStrategy::stored)) strategy = PreparationStrategy::stored;
        else if (requested_strategy == uint64_t(PreparationStrategy::streamed)) strategy = PreparationStrategy::streamed;
        else if (direct_eligible) strategy = PreparationStrategy::direct;
        else if (automatic_stream) strategy = PreparationStrategy::streamed;
        const bool fusion_requested = option("PYSIGLIB_PS_FUSION", 1, true) != 0;
        fused = specialized && fusion_requested && l.power[l.m] <= 1024 && (jobs == 1 || mul(n, l.power[l.m]) < 65536);
        if (strategy != PreparationStrategy::stored) { fused = false; rows = 0; }
        if (fused) rows = 0;
        scratch = strategy != PreparationStrategy::stored ? 0
            : ((std::max({maxp, maxq, rows}) > INT_MAX) ? mul(3, mul(side, side)) : 0);
        psize = mul(add(n, 1), plen); qsize = mul(add(n, 1), qlen);
        powers = l.offset[(l.m + 1) / 2 + 1];
        if (specialized) powers = add(powers, maxq);
        if (strategy == PreparationStrategy::streamed) {
            for (uint64_t k = l.scan_degree + 1; k <= l.m;) {
                StreamGroup& group = stream_groups[stream_group_count++];
                group.level = k; group.a = (k + 1) / 2; group.b = k / 2;
                group.p = l.power[group.a]; group.q = l.power[group.b];
                group.paired = paired && k % 2 && k < l.m;
                group.qnext = group.paired ? l.power[group.b + 1] : 0;
                stream_elements_per_segment = add(stream_elements_per_segment,
                    mul(group.a, add(add(group.p, group.q), group.qnext)));
                k += group.paired ? 2 : 1;
            }
            const uint64_t target_bytes = std::min(factor_bytes,
                option("PYSIGLIB_PS_CPU_STREAM_BLOCK_BYTES", 256ULL << 10));
            stream_segments = std::min(n, target_bytes / mul(stream_elements_per_segment, itemsize));
            if (!stream_segments) {
                strategy = PreparationStrategy::stored;
                stream_group_count = stream_elements_per_segment = 0;
                rows = stored_rows;
                fused = specialized && fusion_requested && l.power[l.m] <= 1024
                    && (jobs == 1 || mul(n, l.power[l.m]) < 65536);
                if (fused) rows = 0;
                scratch = (std::max({maxp, maxq, rows}) > INT_MAX)
                    ? mul(3, mul(side, side)) : 0;
            }
            for (uint64_t index = 0; index < stream_group_count; ++index) {
                StreamGroup& group = stream_groups[index];
                const uint64_t group_rows = mul(stream_segments, group.a);
                group.left_offset = stream_factor_count;
                stream_factor_count = add(stream_factor_count, mul(group_rows, group.p));
                group.right_offset = stream_factor_count;
                stream_factor_count = add(stream_factor_count, mul(group_rows, group.q));
                if (group.paired) {
                    group.next_offset = stream_factor_count;
                    stream_factor_count = add(stream_factor_count, mul(group_rows, group.qnext));
                }
                rows = std::max(rows, group_rows);
            }
        }
        if (strategy == PreparationStrategy::direct) {
            count = workspace_count = 0;
        } else if (strategy == PreparationStrategy::streamed) {
            count = add(add(psize, mul(n, l.d)), stream_factor_count);
            workspace_count = add(count, add(add(qlen, l.power[l.scan_degree]), powers));
        } else {
            count = add(add(psize, qsize), add(mul(n, l.d), add(mul(rows, add(maxp, maxq)), scratch)));
            workspace_count = add(count, add(add(qlen, l.power[l.scan_degree]), mul(jobs, powers)));
        }
        bytes = mul(workspace_count, itemsize);
    }
};

template<typename T> struct CpuWorkspace {
    std::vector<T> values;
    void allocate(const CpuPlan& plan) {
        try { values.resize(plan.workspace_count); }
        catch (const std::bad_alloc&) {
            throw std::runtime_error("CPU prefix-suffix requires " + std::to_string(plan.bytes)
                + " workspace bytes for one path-processing task");
        }
    }
};

template<typename F> void parallel_range(uint64_t count, int jobs, F work) {
    if (jobs == 1 || count < 2) work(0, count);
    else {
        const uint64_t owners = std::min<uint64_t>(count, jobs), chunk = count / owners, remainder = count % owners;
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint64_t>(0, owners, 1), [&](const auto& range) {
            for (uint64_t i = range.begin(); i < range.end(); ++i) {
                const uint64_t first = i * chunk + std::min(i, remainder);
                work(first, first + chunk + (i < remainder));
            }
        });
    }
}

template<typename T> struct CpuMatrices {
    int jobs;
    uint64_t scratch_stride = 0;
    template<typename F> void rows(Matrix<T> c, F f) {
        parallel_range(c.rows, mul(c.rows, c.cols) >= 65536 ? jobs : 1,
            [&](uint64_t first, uint64_t end) {
                for (uint64_t i = first; i < end; ++i)
                    for (uint64_t j = 0; j < c.cols; ++j) f(i, j, c.data[i * c.rs + j * c.cs]);
            });
    }
    void clear(Matrix<T> c) { rows(c, [](uint64_t, uint64_t, T& x) { x = 0; }); }
    void pack(Matrix<T> a, Matrix<T> c) {
        parallel_range((c.rows + 31) / 32, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t tile = first; tile < end; ++tile)
                for (uint64_t col = 0; col < c.cols; col += 32)
                    for (uint64_t i = tile * 32; i < std::min(c.rows, (tile + 1) * 32); ++i)
                        for (uint64_t j = col; j < std::min(c.cols, col + 32); ++j)
                            c.data[i * c.rs + j * c.cs] = i < a.rows && j < a.cols ? a.data[i * a.rs + j * a.cs] : T(0);
        });
    }
    void accumulate(Matrix<T> a, Matrix<T> c, T sign) {
        rows(c, [&](uint64_t i, uint64_t j, T& x) { x += sign * a.data[i * a.rs + j * a.cs]; });
    }
    void gemm(Matrix<T> a, Matrix<T> b, Matrix<T> c, T beta) {
        const bool ta = a.rs == 1, tb = b.rs == 1;
        const uint64_t lda = ta ? a.cs : a.rs, ldb = tb ? b.cs : b.rs;
        if (lda > INT_MAX || ldb > INT_MAX || c.rs > INT_MAX)
            throw std::overflow_error("prefix-suffix BLAS leading dimension exceeds LP64 range");
        const uint64_t row_block = jobs == 1 ? std::min<uint64_t>(a.rows, INT_MAX)
            : std::max<uint64_t>(64, ((a.rows + jobs - 1) / jobs + 63) / 64 * 64);
        const uint64_t blocks = (a.rows + row_block - 1) / row_block;
        parallel_range(blocks, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t block = first; block < end; ++block) {
                const uint64_t i = block * row_block;
                const int nr = static_cast<int>(std::min(row_block, a.rows - i));
                for (uint64_t j = 0; j < b.cols; j += INT_MAX) {
                    const int nc = static_cast<int>(std::min<uint64_t>(INT_MAX, b.cols - j));
                    for (uint64_t k = 0; k < a.cols; k += INT_MAX) {
                        const int nk = static_cast<int>(std::min<uint64_t>(INT_MAX, a.cols - k));
                        const T* ap = a.data + i * a.rs + k * a.cs;
                        const T* bp = b.data + k * b.rs + j * b.cs;
                        T* cp = c.data + i * c.rs + j;
                        if constexpr (std::is_same_v<T, float>)
                            cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                                nr, nc, nk, 1, ap, static_cast<int>(lda), bp, static_cast<int>(ldb), k ? T(1) : beta, cp, static_cast<int>(c.rs));
                        else
                            cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                                nr, nc, nk, 1, ap, static_cast<int>(lda), bp, static_cast<int>(ldb), k ? T(1) : beta, cp, static_cast<int>(c.rs));
                    }
                }
            }
        });
    }
};

template<typename T, int D, int A>
void small_powers(const T* v, T* powers) {
    powers[0] = 1;
    for (int x = 0; x < D; ++x) powers[1 + x] = v[x];
    if constexpr (A >= 2)
        for (int x = 0; x < D * D; ++x) powers[1 + D + x] = v[x / D] * v[x % D];
    if constexpr (A >= 3)
        for (int x = 0; x < D * D * D; ++x) powers[1 + D + D * D + x] = powers[1 + D + x / D] * v[x % D];
}

template<typename T, int D>
void direct_low_path(const Path<T>& path, T* output, uint64_t n, uint64_t degree, bool scalar,
    const Layout& l) {
    if (scalar) output[0] = 1;
    T* level1 = output + (scalar ? 1 : 0);
    T* level2 = output + l.offset[2] - (scalar ? 0 : 1);
    T* level3 = degree == 3 ? output + l.offset[3] - (scalar ? 0 : 1) : nullptr;
    std::fill_n(level2, D * D, T(0));
    if (level3) std::fill_n(level3, D * D * D, T(0));
    const auto first = path[0], last = path[n];
    for (int channel = 0; channel < D; ++channel) level1[channel] = last[channel] - first[channel];
    T v[D], prefix[D], suffix[D], right0[D], right1[D];
    for (uint64_t i = 0; i < n; ++i) {
        const auto current = path[i], following = path[i + 1];
        for (int channel = 0; channel < D; ++channel) {
            v[channel] = following[channel] - current[channel];
            prefix[channel] = current[channel] - first[channel];
            suffix[channel] = last[channel] - following[channel];
            right0[channel] = suffix[channel] + v[channel] * T(0.5);
            right1[channel] = (suffix[channel] + v[channel] / T(3)) * T(0.5);
        }
        for (int x = 0; x < D; ++x)
            for (int y = 0; y < D; ++y)
                level2[x * D + y] += v[x] * right0[y];
        if (level3) {
            for (int x0 = 0; x0 < D; ++x0)
                for (int x1 = 0; x1 < D; ++x1) {
                    const T left0 = prefix[x0] * v[x1];
                    const T left1 = v[x0] * v[x1];
                    T* destination = level3 + (x0 * D + x1) * D;
                    for (int y = 0; y < D; ++y)
                        destination[y] += left0 * right0[y] + left1 * right1[y];
                }
        }
    }
}

template<typename T, int D, int B, int J>
void small_right_values(const T* suffix, const T* v, T* r) {
    constexpr T scale = J == 0 ? T(1) : (J == 1 ? T(0.5) : T(1) / T(6));
    for (int y = 0; y < small_power(D, B); ++y) {
        if constexpr (B == 1) r[y] = (suffix[1 + y] + v[y] * (T(1) / T(J + 2))) * scale;
        else if constexpr (B == 2) {
            const T inner = suffix[1 + y % D] + v[y % D] * (T(1) / T(J + 3));
            r[y] = (suffix[1 + D + y] + v[y / D] * inner * (T(1) / T(J + 2))) * scale;
        } else {
            const T first = suffix[1 + y % D] + v[y % D] * (T(1) / T(J + 4));
            const T second = suffix[1 + D + y % (D * D)] + v[y / D % D] * first * (T(1) / T(J + 3));
            r[y] = (suffix[1 + D + D * D + y] + v[y / (D * D)] * second * (T(1) / T(J + 2))) * scale;
        }
    }
}

template<typename T, int D, int A, int J>
T small_left_value(const T* prefix, const T* powers, uint64_t word) {
    constexpr int tail = small_power(D, J + 1), po = small_offset(D, A - 1 - J), vo = small_offset(D, J + 1);
    return prefix[po + word / tail] * powers[vo + word % tail];
}

template<typename T, int D, int K, bool Paired>
void stream_fill_segment(const T* prefix, const T* suffix, const T* v, const T* powers,
    T* factors, T* temporary, uint64_t local_segment, const StreamGroup& group, const Layout& l) {
    constexpr int A = (K + 1) / 2, B = K / 2, P = small_power(D, A), Q = small_power(D, B);
    T* left = factors + group.left_offset;
    T* right = factors + group.right_offset;
    auto pack = [&]<int J>() {
        const uint64_t row = local_segment * A + J;
        for (int x = 0; x < P; ++x)
            left[row * P + x] = small_left_value<T, D, A, J>(prefix, powers, x);
        small_right_values<T, D, B, J>(suffix, v, right + row * Q);
    };
    pack.template operator()<0>();
    if constexpr (A >= 2) pack.template operator()<1>();
    if constexpr (A >= 3) pack.template operator()<2>();
    if constexpr (Paired) {
        T* next = factors + group.next_offset;
        auto extend = [&]<int J>() {
            const uint64_t row = local_segment * A + J;
            const T* previous;
            if constexpr (J + 1 < A) previous = right + (row + 1) * Q;
            else {
                right_factor(temporary, suffix, v, A, B, l);
                previous = temporary;
            }
            extend_right_factor(next + row * Q * D, previous, suffix, v, J, B, l);
        };
        extend.template operator()<0>();
        if constexpr (A >= 2) extend.template operator()<1>();
        if constexpr (A >= 3) extend.template operator()<2>();
    }
}

template<typename T, int D, int M>
void streamed_low_path(const Path<T>& path, T* output, bool scalar, int jobs,
    const Layout& l, const CpuPlan& plan, CpuWorkspace<T>& workspace) {
    constexpr int Amax = (M + 1) / 2;
    T* storage = workspace.values.data();
    T* prefixes = storage;
    T* increments = prefixes + plan.psize;
    T* factors = increments + plan.n * D;
    T* state = storage + plan.count;
    T* work = state + plan.qlen;
    T* powers = work + l.power[l.scan_degree];
    T* temporary = powers + small_offset(D, Amax + 1);
    std::fill_n(state, plan.qlen, T(0)); state[0] = 1;
    for (uint64_t i = 0; i < plan.n; ++i) {
        const auto current = path[i], following = path[i + 1];
        T* v = increments + i * D;
        for (int channel = 0; channel < D; ++channel) v[channel] = following[channel] - current[channel];
        std::copy_n(state, plan.plen, prefixes + i * plan.plen);
        signature_horner_step_(state, v, D, l.scan_degree, l.offset.data(), work);
    }
    std::copy_n(state, plan.plen, prefixes + plan.n * plan.plen);
    std::copy(state + (scalar ? 0 : 1), state + plan.qlen, output);
    std::fill_n(state, plan.qlen, T(0)); state[0] = 1;
    CpuMatrices<T> matrices{jobs};
    bool first_block = true;
    for (uint64_t end = plan.n; end;) {
        const uint64_t first = end > plan.stream_segments ? end - plan.stream_segments : 0;
        for (uint64_t i = end; i-- > first;) {
            const T* v = increments + i * D;
            small_powers<T, D, Amax>(v, powers);
            for (uint64_t index = 0; index < plan.stream_group_count; ++index) {
                const StreamGroup& group = plan.stream_groups[index];
                auto fill = [&]<int K>() {
                    if (group.paired)
                        stream_fill_segment<T, D, K, true>(prefixes + i * plan.plen, state, v, powers,
                            factors, temporary, i - first, group, l);
                    else
                        stream_fill_segment<T, D, K, false>(prefixes + i * plan.plen, state, v, powers,
                            factors, temporary, i - first, group, l);
                };
                switch (group.level) {
                case 3: fill.template operator()<3>(); break;
                case 4: fill.template operator()<4>(); break;
                case 5: fill.template operator()<5>(); break;
                case 6: fill.template operator()<6>(); break;
                }
            }
            cpu_detail::prepend(state, v, l, work);
        }
        const uint64_t segment_count = end - first;
        for (uint64_t index = 0; index < plan.stream_group_count; ++index) {
            const StreamGroup& group = plan.stream_groups[index];
            const uint64_t factor_rows = mul(segment_count, group.a);
            product(matrices,
                Matrix<T>{factors + group.left_offset, group.p, factor_rows, 1, group.p},
                Matrix<T>{factors + group.right_offset, factor_rows, group.q, group.q, 1},
                Matrix<T>{output + l.offset[group.level] - (scalar ? 0 : 1), group.p, group.q, group.q, 1},
                plan.side, static_cast<T*>(nullptr), first_block);
            if (group.paired)
                product(matrices,
                    Matrix<T>{factors + group.left_offset, group.p, factor_rows, 1, group.p},
                    Matrix<T>{factors + group.next_offset, factor_rows, group.qnext, group.qnext, 1},
                    Matrix<T>{output + l.offset[group.level + 1] - (scalar ? 0 : 1), group.p, group.qnext, group.qnext, 1},
                    plan.side, static_cast<T*>(nullptr), first_block);
        }
        first_block = false;
        end = first;
    }
}

template<typename T, int D, int K, bool Paired>
void small_level(const T* increments, const T* prefixes, const T* suffixes, T* output,
    T* left, T* right, int jobs, const CpuPlan& plan, const Layout& l) {
    constexpr int A = (K + 1) / 2, B = K / 2, P = small_power(D, A), Q = small_power(D, B);
    constexpr int powers_size = small_offset(D, A + 1);
    constexpr bool paired = Paired;
    const uint64_t pitch = paired ? Q * D : Q;
    const int workers = mul(plan.n, uint64_t(P) * Q) >= 65536 ? jobs : 1;
    if (plan.fused) {
        parallel_range(P, workers, [&](uint64_t first, uint64_t end) {
            std::fill(output + first * Q, output + end * Q, T(0));
            T powers[powers_size], r[Q];
            for (uint64_t i = 0; i < plan.n; ++i) {
                const T* v = increments + i * D;
                small_powers<T, D, A>(v, powers);
                auto accumulate = [&]<int J>() {
                    small_right_values<T, D, B, J>(suffixes + (i + 1) * plan.qlen, v, r);
                    for (uint64_t x = first; x < end; ++x) {
                        const T value = small_left_value<T, D, A, J>(prefixes + i * plan.plen, powers, x);
                        for (int y = 0; y < Q; ++y) output[x * Q + y] += value * r[y];
                    }
                };
                accumulate.template operator()<0>();
                if constexpr (A >= 2) accumulate.template operator()<1>();
                if constexpr (A >= 3) accumulate.template operator()<2>();
            }
        });
        return;
    }
    CpuMatrices<T> matrices{workers};
    const uint64_t total = mul(plan.n, A);
    for (uint64_t base = 0; base < total; base += plan.rows) {
        const uint64_t count = std::min(plan.rows, total - base);
        parallel_range(count, workers, [&](uint64_t first, uint64_t end) {
            T powers[powers_size];
            uint64_t previous_segment = plan.n;
            for (uint64_t row = first; row < end; ++row) {
                const uint64_t i = (base + row) / A;
                const int j = (base + row) % A;
                const T* v = increments + i * D;
                if (i != previous_segment) {
                    small_powers<T, D, A>(v, powers);
                    previous_segment = i;
                }
                auto pack = [&]<int J>() {
                    for (int x = 0; x < P; ++x)
                        left[row * P + x] = small_left_value<T, D, A, J>(prefixes + i * plan.plen, powers, x);
                    small_right_values<T, D, B, J>(suffixes + (i + 1) * plan.qlen, v, right + row * pitch);
                };
                if (j == 0) pack.template operator()<0>();
                if constexpr (A >= 2) if (j == 1) pack.template operator()<1>();
                if constexpr (A >= 3) if (j == 2) pack.template operator()<2>();
            }
        });
        matrices.gemm({left, P, count, 1, P}, {right, count, Q, pitch, 1}, {output, P, Q, Q, 1}, base ? T(1) : T(0));
        if (paired) {
            const uint64_t first_segment = base / A, segments = (base + count - 1) / A - first_segment + 1;
            parallel_range(segments, workers, [&](uint64_t first, uint64_t end) {
                T temporary[Q];
                for (uint64_t segment = first; segment < end; ++segment) {
                    const uint64_t i = first_segment + segment;
                    const T* v = increments + i * D;
                    const T* suffix = suffixes + (i + 1) * plan.qlen;
                    const uint64_t begin = std::max(base, i * A), stop = std::min(base + count, (i + 1) * A);
                    for (uint64_t t = begin; t < stop; ++t) {
                        const uint64_t j = t - i * A, row = t - base;
                        const T* previous = right + (row + 1) * pitch;
                        if (t + 1 == stop) {
                            right_factor(temporary, suffix, v, j + 1, B, l);
                            previous = temporary;
                        }
                        extend_right_factor(right + row * pitch, previous, suffix, v, j, B, l);
                    }
                }
            });
            matrices.gemm({left, P, count, 1, P}, {right, count, pitch, pitch, 1},
                {output + P * Q, P, pitch, pitch, 1}, base ? T(1) : T(0));
        }
    }
}

template<typename T>
void one_path(const T* input, T* output, uint64_t dimension, uint64_t length, bool time_aug,
    bool lead_lag, T end_time, bool scalar, int jobs, const Layout& l, const CpuPlan& plan,
    CpuWorkspace<T>& workspace) {
    Path<T> path(input, dimension, length, time_aug, lead_lag, end_time);
    if (plan.strategy == PreparationStrategy::direct) {
        auto run = [&]<int D>() { direct_low_path<T, D>(path, output, plan.n, l.m, scalar, l); };
        switch (l.d) {
        case 2: run.template operator()<2>(); break;
        case 3: run.template operator()<3>(); break;
        case 4: run.template operator()<4>(); break;
        case 8: run.template operator()<8>(); break;
        }
        return;
    }
    if (plan.strategy == PreparationStrategy::streamed) {
        auto run = [&]<int D>() {
            switch (l.m) {
            case 4: streamed_low_path<T, D, 4>(path, output, scalar, jobs, l, plan, workspace); break;
            case 5: streamed_low_path<T, D, 5>(path, output, scalar, jobs, l, plan, workspace); break;
            case 6: streamed_low_path<T, D, 6>(path, output, scalar, jobs, l, plan, workspace); break;
            }
        };
        switch (l.d) {
        case 2: run.template operator()<2>(); break;
        case 3: run.template operator()<3>(); break;
        case 4: run.template operator()<4>(); break;
        case 8: run.template operator()<8>(); break;
        }
        return;
    }
    const uint64_t n = plan.n, plen = plan.plen, qlen = plan.qlen, maxp = plan.maxp, maxq = plan.maxq;
    const uint64_t row_count = plan.rows, maxside = plan.side, psize = plan.psize, qsize = plan.qsize;
    const uint64_t count = plan.count, bytes = plan.bytes;
    T* storage = workspace.values.data();
    T* prefixes = storage; T* suffixes = prefixes + psize; T* increments = suffixes + qsize;
    T* left = increments + n * l.d; T* right = left + row_count * maxp; T* scratch = right + row_count * maxq;
    T* state = storage + count;
    T* work = state + qlen;
    T* factor_work = work + l.power[l.scan_degree];
    std::fill_n(state, qlen, T(0)); state[0] = 1;
    if (plan.low && l.scan_degree == 1) {
        const auto first = path[0], last = path[n];
        for (uint64_t i = 0; i <= n; ++i) {
            const auto current = path[i];
            const auto following = path[i < n ? i + 1 : i];
            prefixes[i * plen] = suffixes[i * qlen] = 1;
            for (uint64_t x = 0; x < l.d; ++x) {
                if (plen > 1) prefixes[i * plen + 1 + x] = current[x] - first[x];
                suffixes[i * qlen + 1 + x] = last[x] - current[x];
                if (i < n) increments[i * l.d + x] = following[x] - current[x];
            }
        }
        if (scalar) output[0] = 1;
        for (uint64_t x = 0; x < l.d; ++x) output[(scalar ? 1 : 0) + x] = last[x] - first[x];
    } else {
        for (uint64_t i = 0; i < n; ++i) {
            auto a = path[i], b = path[i + 1];
            T* v = increments + i * l.d;
            for (uint64_t x = 0; x < l.d; ++x) v[x] = b[x] - a[x];
            std::copy_n(state, plen, prefixes + i * plen);
            signature_horner_step_(state, v, l.d, l.scan_degree, l.offset.data(), work);
        }
        std::copy_n(state, plen, prefixes + n * plen);
        std::copy(state + (scalar ? 0 : 1), state + qlen, output);
        std::fill_n(state, qlen, T(0)); state[0] = 1;
        for (uint64_t i = n; i-- > 0;) {
            std::copy_n(state, qlen, suffixes + (i + 1) * qlen);
            cpu_detail::prepend(state, increments + i * l.d, l, work);
        }
        std::copy_n(state, qlen, suffixes);
    }
    CpuMatrices<T> matrices{jobs};
    for (uint64_t k = l.scan_degree + 1; k <= l.m; ++k) {
        if (plan.specialized) {
            auto run = [&]<int D>() {
                T* result = output + l.offset[k] - (scalar ? 0 : 1);
                auto level = [&]<int K>() {
                    if constexpr (K % 2) if (plan.paired && !plan.fused && k < l.m) {
                        small_level<T, D, K, true>(increments, prefixes, suffixes, result, left, right, jobs, plan, l);
                        return;
                    }
                    small_level<T, D, K, false>(increments, prefixes, suffixes, result, left, right, jobs, plan, l);
                };
                switch (k) {
                case 2: level.template operator()<2>(); break;
                case 3: level.template operator()<3>(); break;
                case 4: level.template operator()<4>(); break;
                case 5: level.template operator()<5>(); break;
                case 6: level.template operator()<6>(); break;
                }
            };
            switch (l.d) {
            case 2: run.template operator()<2>(); break;
            case 3: run.template operator()<3>(); break;
            case 4: run.template operator()<4>(); break;
            case 8: run.template operator()<8>(); break;
            }
            if (plan.paired && !plan.fused && k % 2 && k < l.m) ++k;
            continue;
        }
        const uint64_t a = (k + 1) / 2, b = k / 2, p = l.power[a], q = l.power[b], total = mul(n, a);
        const bool paired = plan.paired && k % 2 && k < l.m;
        const uint64_t pitch = paired ? l.power[b + 1] : q;
        for (uint64_t base = 0; base < total; base += row_count) {
            const uint64_t count_rows = std::min(row_count, total - base);
            parallel_range(jobs, jobs, [&](uint64_t first_owner, uint64_t end_owner) {
                for (uint64_t owner = first_owner; owner < end_owner; ++owner) {
                const uint64_t first = count_rows * owner / jobs, end = count_rows * (owner + 1) / jobs;
                T* powers = factor_work + owner * plan.powers;
                uint64_t previous_segment = n;
                for (uint64_t t = first; t < end; ++t) {
                    const uint64_t i = (base + t) / a, j = (base + t) % a;
                    const T* v = increments + i * l.d;
                    if (i != previous_segment) {
                        powers[0] = 1;
                        for (uint64_t level = 1; level <= a; ++level)
                            for (uint64_t x = 0; x < l.power[level - 1]; ++x) {
                                const T value = powers[l.offset[level - 1] + x];
                                for (uint64_t channel = 0; channel < l.d; ++channel)
                                    powers[l.offset[level] + x * l.d + channel] = value * v[channel];
                            }
                        previous_segment = i;
                    }
                    const uint64_t tail = l.power[j + 1];
                    const T* prefix = prefixes + i * plen + l.offset[a - 1 - j];
                    for (uint64_t x = 0; x < l.power[a - 1 - j]; ++x) {
#ifdef VEC
                        vec_mult_assign(left + t * p + x * tail, powers + l.offset[j + 1], prefix[x], tail);
#else
                        for (uint64_t y = 0; y < tail; ++y)
                            left[t * p + x * tail + y] = prefix[x] * powers[l.offset[j + 1] + y];
#endif
                    }
                    right_factor(right + t * pitch, suffixes + (i + 1) * qlen, v, j, b, l);
                }
                }
            });
            product(matrices, Matrix<T>{left, p, count_rows, 1, p}, Matrix<T>{right, count_rows, q, pitch, 1},
                Matrix<T>{output + l.offset[k] - (scalar ? 0 : 1), p, q, q, 1}, maxside, scratch, base == 0);
            if (paired) {
                const uint64_t first_segment = base / a, segments = (base + count_rows - 1) / a - first_segment + 1;
                const uint64_t owners = std::min<uint64_t>(jobs, segments);
                parallel_range(owners, jobs, [&](uint64_t first_owner, uint64_t end_owner) {
                    for (uint64_t owner = first_owner; owner < end_owner; ++owner) {
                        T* temporary = factor_work + owner * plan.powers;
                        const uint64_t first = first_segment + segments * owner / owners;
                        const uint64_t end = first_segment + segments * (owner + 1) / owners;
                        for (uint64_t i = first; i < end; ++i) {
                            const T* v = increments + i * l.d;
                            const T* suffix = suffixes + (i + 1) * qlen;
                            const uint64_t begin = std::max(base, i * a), stop = std::min(base + count_rows, (i + 1) * a);
                            for (uint64_t t = begin; t < stop; ++t) {
                                const uint64_t j = t - i * a, row = t - base;
                                const T* previous = right + (row + 1) * pitch;
                                if (t + 1 == stop) {
                                    right_factor(temporary, suffix, v, j + 1, b, l);
                                    previous = temporary;
                                }
                                extend_right_factor(right + row * pitch, previous, suffix, v, j, b, l);
                            }
                        }
                    }
                });
                product(matrices, Matrix<T>{left, p, count_rows, 1, p}, Matrix<T>{right, count_rows, pitch, pitch, 1},
                    Matrix<T>{output + l.offset[k + 1] - (scalar ? 0 : 1), p, pitch, pitch, 1}, maxside, scratch, base == 0);
            }
        }
        if (paired) ++k;
    }
}

template<typename T>
void signature_prefix_suffix(const T* path, T* out, uint64_t batch, uint64_t dimension, uint64_t length,
    uint64_t degree, bool time_aug, bool lead_lag, T end_time, bool scalar, int n_jobs) {
    if (!dimension) throw std::invalid_argument("path has zero channels");
    const uint64_t d = add(mul(dimension, lead_lag ? 2 : 1), time_aug ? 1 : 0);
    const uint64_t points = lead_lag && length ? add(length, length - 1) : length;
    Layout l(d, degree);
    const uint64_t stride = l.offset[degree + 1] - (scalar ? 0 : 1), path_stride = mul(length, dimension);
    mul(mul(batch, stride), sizeof(T)); mul(mul(batch, path_stride), sizeof(T));
    const int jobs = n_jobs > 0 ? n_jobs : static_cast<int>(get_max_threads()) + 1 + n_jobs;
    if (!n_jobs || jobs < 1) throw std::invalid_argument("invalid n_jobs");
    if (!batch || !stride) return;
    if (points < 2 || degree < 2 || points == 2 || d == 1) {
        get_tbb_arena(jobs).execute([&] { parallel_range(batch, jobs, [&](uint64_t first, uint64_t end) {
            for (uint64_t i = first; i < end; ++i) {
                T* result = out + i * stride;
                if (points < 2 || !degree) {
                    std::fill_n(result, stride, T(0));
                    if (scalar) result[0] = 1;
                    continue;
                }
                if (scalar) result[0] = 1;
                Path<T> p(path + i * path_stride, dimension, length, time_aug, lead_lag, end_time);
                auto a = p[0], b = p[points - 1];
                T* increment = result + (scalar ? 1 : 0);
                for (uint64_t x = 0; x < d; ++x) increment[x] = b[x] - a[x];
                for (uint64_t k = 2; k <= degree; ++k) {
                    const T reciprocal = T(1) / T(k);
                    const T* previous = result + l.offset[k - 1] - (scalar ? 0 : 1);
                    T* level = result + l.offset[k] - (scalar ? 0 : 1);
                    for (uint64_t x = 0; x < l.power[k - 1]; ++x) {
                        const T value = previous[x] * reciprocal;
#ifdef VEC
                        vec_mult_assign(level + x * d, increment, value, d);
#else
                        for (uint64_t channel = 0; channel < d; ++channel)
                            level[x * d + channel] = value * increment[channel];
#endif
                    }
                }
            }
        }); });
        return;
    }
    if (openblas_get_parallel() != 0)
        throw std::runtime_error("prefix-suffix requires sequential OpenBLAS built with USE_THREAD=0 and USE_LOCKING=1");
    const int available_path_jobs = std::max(1, jobs / static_cast<int>(std::min<uint64_t>(batch, jobs)));
    const uint64_t work_proxy = mul(points - 1, l.power[degree]);
    const bool internal_threads = option("PYSIGLIB_PS_CPU_INTERNAL_THREADS", 1, true) != 0;
    const uint64_t parallel_threshold = option("PYSIGLIB_PS_CPU_PARALLEL_THRESHOLD", 262144, true);
    const uint64_t parallel_segments = option("PYSIGLIB_PS_CPU_MIN_PARALLEL_SEGMENTS", 256, true);
    const int path_jobs = internal_threads && points - 1 >= parallel_segments && work_proxy >= parallel_threshold
        ? available_path_jobs : 1;
    const CpuPlan plan(l, points - 1, batch, path_jobs, sizeof(T));
    const uint64_t bytes = plan.bytes;
    const uint64_t budget = option("PYSIGLIB_PS_WORKSPACE_BYTES", 2ULL << 30);
    if (bytes > budget) throw std::runtime_error("CPU prefix-suffix requires " + std::to_string(bytes) + " workspace bytes for one path; configured limit is " + std::to_string(budget));
    const uint64_t owners = std::min<uint64_t>({batch, uint64_t(jobs), bytes ? budget / bytes : uint64_t(jobs)});
    std::vector<CpuWorkspace<T>> workspaces(owners);
    get_tbb_arena(jobs).execute([&] { parallel_range(owners, jobs, [&](uint64_t first_owner, uint64_t end_owner) {
        for (uint64_t owner = first_owner; owner < end_owner; ++owner) {
            workspaces[owner].allocate(plan);
            const uint64_t first = batch * owner / owners, end = batch * (owner + 1) / owners;
            for (uint64_t i = first; i < end; ++i)
                one_path(path + i * path_stride, out + i * stride, dimension, length, time_aug, lead_lag,
                    end_time, scalar, path_jobs, l, plan, workspaces[owner]);
        }
    }); });
}
}

extern "C" {
CPSIG_API int signature_prefix_suffix_f(const float* path, float* out, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, float end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(signature_prefix_suffix(path, out, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
CPSIG_API int signature_prefix_suffix_d(const double* path, double* out, uint64_t batch, uint64_t dimension, uint64_t length, uint64_t degree, bool time_aug, bool lead_lag, double end_time, bool scalar_term, int n_jobs) noexcept {
    SAFE_CALL(signature_prefix_suffix(path, out, batch, dimension, length, degree, time_aug, lead_lag, end_time, scalar_term, n_jobs));
}
}
