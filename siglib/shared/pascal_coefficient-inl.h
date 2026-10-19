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

template<class Ops, class Access>
PASCAL_HD void single_segment_backward(Access& access, uint64_t m, typename Ops::V* scratch) {
    using T = typename Ops::Scalar;
    auto* u = scratch;
    auto* a = u + m + 1;
    auto* lambda = a + m + 1;
    auto* gradient = lambda + m + 1;
    u[0] = Ops::set(T(1));
    for (uint64_t j = 1; j <= m; ++j) {
        a[j] = access.increment(0, j - 1);
        u[j] = Ops::mul_div(a[j], u[j - 1], j);
        lambda[j] = access.seed(j - 1);
    }
    for (uint64_t j = m; j > 0; --j) {
        gradient[j] = Ops::mul_div(u[j - 1], lambda[j], j);
        if (j > 1) lambda[j - 1] += Ops::mul_div(a[j], lambda[j], j);
    }
    access.scatter(0, gradient + 1, m);
}

template<class Ops, int Mode, int Degree = 0, bool Last = true>
PASCAL_HD PASCAL_INLINE bool step(typename Ops::V* u, const typename Ops::V* a, uint64_t degree,
    bool first, typename Ops::V* tape, typename Ops::V& q) {
    using T = typename Ops::Scalar;
    using V = typename Ops::V;
    const uint64_t m = Degree ? Degree : degree;
    bool valid = true;
    if (first) {
        for (uint64_t j = 1; j <= (Last ? m : m - 1); ++j) {
            if constexpr (Mode == guarded) u[j] = Ops::mul_div(a[j], u[j - 1], j);
            else u[j] = a[j] * u[j - 1];
        }
    } else if constexpr (Mode == guarded) {
        uint64_t k = 0;
        for (uint64_t s = 1; s <= m; ++s)
            for (uint64_t j = m; j >= s; --j) {
                if (tape) tape[k++] = u[j - 1];
                if (Last || j < m) u[j] = u[j] + Ops::mul_div(a[j], u[j - 1], j);
            }
    } else {
        if (Last || tape) q = u[m - 1];
        uint64_t k = 0;
        for (uint64_t s = 1; s < m; ++s) {
            for (uint64_t j = m - 1; j >= s; --j) {
                if (tape) tape[k++] = u[j - 1];
                u[j] = Ops::madd(a[j], u[j - 1], u[j]);
            }
            if (Last || tape) q = q + u[m - 1];
        }
        if constexpr (Last) u[m] = Ops::madd(a[m], q, u[m]);
        if (Last || tape) valid = Ops::finite(q);
    }
    return Ops::all_finite(u + 1, Last ? m : m - 1) && valid;
}

template<class Ops, int Mode, class Access>
PASCAL_HD PASCAL_INLINE bool increments(Access& access, uint64_t i, uint64_t m, typename Ops::V* a) {
    using T = typename Ops::Scalar;
    bool valid = true;
    for (uint64_t j = 1; j <= m; ++j) {
        const auto raw = access.increment(i, j - 1);
        if constexpr (Mode == unscaled) {
            a[j] = raw * Ops::set(T(1) / T(j));
            valid = !Ops::tiny(raw, a[j]) && valid;
        } else a[j] = raw;
    }
    return valid;
}

template<class Ops, int Mode, int Degree = 0, class Access>
PASCAL_HD bool forward(Access& access, uint64_t n, uint64_t degree, typename Ops::V* u, typename Ops::V* a) {
    using T = typename Ops::Scalar;
    const uint64_t m = Degree ? Degree : degree;
    u[0] = Ops::set(T(1));
    for (uint64_t j = 1; j <= m; ++j) u[j] = Ops::set(T(0));
    if (!m) return true;
    bool valid = true;
    auto q = Ops::set(T(0));
    for (uint64_t i = 0; i < n; ++i) {
        valid = increments<Ops, Mode>(access, i, m, a) && valid;
        valid = step<Ops, Mode, Degree>(u, a, m, i == 0, nullptr, q) && valid;
    }
    if constexpr (Mode == scaled) {
        T factor = 1;
        for (uint64_t j = 1; j <= m; ++j) {
            factor /= T(j);
            u[j] = u[j] * Ops::set(factor);
        }
    }
    return valid;
}

template<class Ops, int Mode, int Degree = 0, class Access, class Checkpoints>
PASCAL_HD bool backward(Access& access, uint64_t n, uint64_t degree, uint64_t block,
    uint64_t blocks, Checkpoints checkpoints, typename Ops::V* scratch) {
    using T = typename Ops::Scalar;
    using V = typename Ops::V;
    const uint64_t m = Degree ? Degree : degree, stride = m + 1;
    if (!m || !n) return true;
    const uint64_t lower = m - 1;
    V* u = scratch;
    V* a = u + stride;
    V* lambda = a + stride;
    V* grad = lambda + stride;
    V* tape = grad + stride;
    auto local = checkpoints + blocks * lower;
    auto links = local + (block > 1 ? block * lower : 0);
    u[0] = Ops::set(T(1));
    bool valid = true;
    T factor = 1;
    for (uint64_t j = 1; j <= m; ++j) {
        u[j] = Ops::set(T(0));
        const V seed = access.seed(j - 1);
        if constexpr (Mode == scaled) {
            factor /= T(j);
            lambda[j] = seed * Ops::set(factor);
            valid = !Ops::tiny(seed, lambda[j]) && valid;
        } else lambda[j] = seed;
    }
    if (!valid && Mode != guarded) return false;
    V q = Ops::set(T(0));
    uint64_t next_checkpoint = 0, checkpoint = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (i == next_checkpoint) {
            for (uint64_t j = 1; j < m; ++j) checkpoints[checkpoint * lower + j - 1] = u[j];
            ++checkpoint;
            next_checkpoint += block;
        }
        valid = increments<Ops, Mode>(access, i, m, a) && valid;
        if (block == 1)
            for (uint64_t j = 1; j <= m; ++j) links[i * m + j - 1] = a[j];
        valid = step<Ops, Mode, Degree>(u, a, m, i == 0, nullptr, q) && valid;
    }
    if (!valid && Mode != guarded) return false;
    for (uint64_t b = blocks; b-- > 0;) {
        const uint64_t begin = b * block, end = n < begin + block ? n : begin + block;
        if (block > 1) {
            for (uint64_t j = 1; j < m; ++j) u[j] = checkpoints[b * lower + j - 1];
            for (uint64_t i = begin; i < end; ++i) {
                for (uint64_t j = 1; j < m; ++j) local[(i - begin) * lower + j - 1] = u[j];
                increments<Ops, Mode>(access, i, m, a);
                for (uint64_t j = 1; j <= m; ++j) links[(i - begin) * m + j - 1] = a[j];
                step<Ops, Mode, Degree, false>(u, a, m, i == 0, nullptr, q);
            }
        }
        for (uint64_t i = end; i-- > begin;) {
            const auto saved = block > 1 ? local + (i - begin) * lower : checkpoints + b * lower;
            for (uint64_t j = 1; j < m; ++j) u[j] = saved[j - 1];
            const uint64_t link_offset = (block > 1 ? i - begin : i) * m;
            for (uint64_t j = 1; j <= m; ++j) a[j] = links[link_offset + j - 1];
            step<Ops, Mode, Degree, false>(u, a, m, i == 0, tape, q);
            for (uint64_t j = 1; j <= m; ++j) grad[j] = Ops::set(T(0));
            if (!i) {
                for (uint64_t j = m; j > 0; --j) {
                    if constexpr (Mode == guarded) {
                        grad[j] = Ops::mul_div(u[j - 1], lambda[j], j);
                        if (j > 1) lambda[j - 1] = lambda[j - 1] + Ops::mul_div(a[j], lambda[j], j);
                    } else {
                        grad[j] = u[j - 1] * lambda[j];
                        if (j > 1) lambda[j - 1] = Ops::madd(a[j], lambda[j], lambda[j - 1]);
                    }
                }
            } else if constexpr (Mode == guarded) {
                uint64_t k = m * (m + 1) / 2;
                for (uint64_t s = m; s > 0; --s)
                    for (uint64_t j = s; j <= m; ++j) {
                        grad[j] = grad[j] + Ops::mul_div(tape[--k], lambda[j], j);
                        if (j > 1) lambda[j - 1] = lambda[j - 1] + Ops::mul_div(a[j], lambda[j], j);
                    }
            } else {
                grad[m] = lambda[m] * q;
                const V dq = a[m] * lambda[m];
                uint64_t k = m * (m - 1) / 2;
                for (uint64_t s = m - 1; s > 0; --s) {
                    lambda[m - 1] = lambda[m - 1] + dq;
                    for (uint64_t j = s; j < m; ++j) {
                        grad[j] = Ops::madd(tape[--k], lambda[j], grad[j]);
                        if (j > 1) lambda[j - 1] = Ops::madd(a[j], lambda[j], lambda[j - 1]);
                    }
                }
                if (m > 1) lambda[m - 1] = lambda[m - 1] + dq;
            }
            for (uint64_t j = 1; j <= m; ++j) {
                if constexpr (Mode == unscaled) grad[j] = grad[j] * Ops::set(T(1) / T(j));
            }
            valid = Ops::all_finite(grad + 1, m) && Ops::all_finite(lambda + 1, m) && valid;
            access.scatter(i, grad + 1, m);
        }
    }
    return valid;
}
