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
#include "cp_simd_config.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_pascal_coef_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "cp_simd-inl.h"
#include "cp_dispatch.h"
#include "multithreading.h"
#include "pascal_coefficient.h"
#include "pascal_selection.h"
#include "cp_coefficient_workspace.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig {
namespace HWY_NAMESPACE {
namespace pc {
using namespace pascal_coefficient;
#include "pascal_coefficient-inl.h"
}

template<class T>
struct PascalScalarOps : pc::ScalarOps<T> {
    static constexpr uint64_t lanes = 1;
    static T load(const T* p) { return *p; }
    static void store(T x, T* p) { *p = x; }
};

template<class T>
struct PascalSimdOps {
    using Scalar = T;
    using V = hn::Vec<Simd<T>>;
    static constexpr uint64_t lanes = vec_batch_lanes<T>;
    static V set(T x) { return hn::Set(Simd<T>(), x); }
    static V load(const T* p) { return hn::LoadU(Simd<T>(), p); }
    static void store(V x, T* p) { hn::StoreU(x, Simd<T>(), p); }
    static V madd(V x, V y, V z) { return hn::MulAdd(x, y, z); }
    static bool finite(V x) { return hn::AllTrue(Simd<T>(), hn::Le(hn::Abs(x), set(std::numeric_limits<T>::max()))); }
    static bool all_finite(const V* x, uint64_t count) {
        auto mask = hn::Eq(set(0), set(0));
        const auto limit = set(std::numeric_limits<T>::max());
        for (uint64_t j = 0; j < count; ++j) mask = hn::And(mask, hn::Le(hn::Abs(x[j]), limit));
        return hn::AllTrue(Simd<T>(), mask);
    }
    static bool tiny(V raw, V x) {
        return !hn::AllFalse(Simd<T>(), hn::And(hn::Ne(raw, set(0)), hn::Lt(hn::Abs(x), set(std::numeric_limits<T>::min()))));
    }
};

template<class T, class Ops>
struct PascalAccess {
    using V = typename Ops::V;
    const T* path;
    T* out;
    const T* deriv;
    const uint64_t* word;
    uint64_t dimension, length, n, count, first, seed_stride, seed_offset;
    bool time_aug, lead_lag;
    T end_time;
    PASCAL_INLINE V increment(uint64_t i, uint64_t j) const {
        T values[Ops::lanes] = {};
        if (!time_aug && !lead_lag) {
            const uint64_t coordinate = i * dimension + word[j];
            if (count == Ops::lanes) {
                for (uint64_t lane = 0; lane < Ops::lanes; ++lane) {
                    const T* p = path + (first + lane) * dimension * length + coordinate;
                    values[lane] = p[dimension] - p[0];
                }
                return Ops::load(values);
            }
            for (uint64_t lane = 0; lane < count; ++lane) {
                const T* p = path + (first + lane) * dimension * length + coordinate;
                values[lane] = p[dimension] - p[0];
            }
            return Ops::load(values);
        }
        for (uint64_t lane = 0; lane < count; ++lane) {
            const T* p = path + (first + lane) * dimension * length;
            values[lane] = pc::point(p, i + 1, word[j], dimension, n, time_aug, lead_lag, end_time)
                - pc::point(p, i, word[j], dimension, n, time_aug, lead_lag, end_time);
        }
        return Ops::load(values);
    }
    PASCAL_INLINE V seed(uint64_t j) const {
        T values[Ops::lanes] = {};
        if (count == Ops::lanes) {
            for (uint64_t lane = 0; lane < Ops::lanes; ++lane)
                values[lane] = deriv[(first + lane) * seed_stride + seed_offset + j];
            return Ops::load(values);
        }
        for (uint64_t lane = 0; lane < count; ++lane)
            values[lane] = deriv[(first + lane) * seed_stride + seed_offset + j];
        return Ops::load(values);
    }
    PASCAL_INLINE void scatter(uint64_t i, const V* gradient, uint64_t m) const {
        if (!time_aug && !lead_lag) {
            for (uint64_t j = 0; j < m; ++j) {
                T values[Ops::lanes];
                Ops::store(gradient[j], values);
                const uint64_t coordinate = i * dimension + word[j];
                if (count == Ops::lanes) {
                    for (uint64_t lane = 0; lane < Ops::lanes; ++lane) {
                        T* p = out + (first + lane) * dimension * length + coordinate;
                        p[dimension] += values[lane];
                        p[0] -= values[lane];
                    }
                    continue;
                }
                for (uint64_t lane = 0; lane < count; ++lane) {
                    T* p = out + (first + lane) * dimension * length + coordinate;
                    p[dimension] += values[lane];
                    p[0] -= values[lane];
                }
            }
            return;
        }
        for (uint64_t j = 0; j < m; ++j) {
            const uint64_t c = word[j];
            if (time_aug && c == dimension * (lead_lag ? 2 : 1)) continue;
            const uint64_t before = lead_lag ? i / 2 + ((i % 2) && c >= dimension) : i;
            const uint64_t after = lead_lag ? (i + 1) / 2 + (((i + 1) % 2) && c >= dimension) : i + 1;
            if (before == after) continue;
            T values[Ops::lanes];
            Ops::store(gradient[j], values);
            for (uint64_t lane = 0; lane < count; ++lane) {
                T* p = out + (first + lane) * dimension * length;
                const uint64_t column = lead_lag && c >= dimension ? c - dimension : c;
                p[after * dimension + column] += values[lane];
                p[before * dimension + column] -= values[lane];
            }
        }
    }
};

template<class Ops, int Mode, class Access>
bool pascal_forward_dispatch(Access& access, uint64_t n, uint64_t m, typename Ops::V* u, typename Ops::V* a) {
#define PASCAL_FORWARD_CASE(M) case M: return pc::forward<Ops, Mode, M>(access, n, m, u, a)
    switch (m) {
        PASCAL_FORWARD_CASE(1); PASCAL_FORWARD_CASE(2); PASCAL_FORWARD_CASE(3); PASCAL_FORWARD_CASE(4);
        PASCAL_FORWARD_CASE(5); PASCAL_FORWARD_CASE(6); PASCAL_FORWARD_CASE(7); PASCAL_FORWARD_CASE(8);
        PASCAL_FORWARD_CASE(9); PASCAL_FORWARD_CASE(10); PASCAL_FORWARD_CASE(11); PASCAL_FORWARD_CASE(12);
        PASCAL_FORWARD_CASE(13); PASCAL_FORWARD_CASE(14); PASCAL_FORWARD_CASE(15); PASCAL_FORWARD_CASE(16);
        PASCAL_FORWARD_CASE(17); PASCAL_FORWARD_CASE(18); PASCAL_FORWARD_CASE(19); PASCAL_FORWARD_CASE(20);
        default: return pc::forward<Ops, Mode>(access, n, m, u, a);
    }
#undef PASCAL_FORWARD_CASE
}

template<class Ops, int Mode, class Access>
bool pascal_backward_dispatch(Access& access, uint64_t n, uint64_t m, uint64_t block, uint64_t blocks,
    typename Ops::V* checkpoints, typename Ops::V* scratch) {
    if constexpr (Mode != pc::guarded) {
#define PASCAL_BACKWARD_CASE(M) case M: return pc::backward<Ops, Mode, M>(access, n, m, block, blocks, checkpoints, scratch)
        switch (m) {
            PASCAL_BACKWARD_CASE(1); PASCAL_BACKWARD_CASE(2); PASCAL_BACKWARD_CASE(3); PASCAL_BACKWARD_CASE(4);
            PASCAL_BACKWARD_CASE(5); PASCAL_BACKWARD_CASE(6); PASCAL_BACKWARD_CASE(7); PASCAL_BACKWARD_CASE(8);
            PASCAL_BACKWARD_CASE(9); PASCAL_BACKWARD_CASE(10); PASCAL_BACKWARD_CASE(11); PASCAL_BACKWARD_CASE(12);
            PASCAL_BACKWARD_CASE(13); PASCAL_BACKWARD_CASE(14); PASCAL_BACKWARD_CASE(15); PASCAL_BACKWARD_CASE(16);
            PASCAL_BACKWARD_CASE(17); PASCAL_BACKWARD_CASE(18); PASCAL_BACKWARD_CASE(19); PASCAL_BACKWARD_CASE(20);
        }
#undef PASCAL_BACKWARD_CASE
    }
    return pc::backward<Ops, Mode>(access, n, m, block, blocks, checkpoints, scratch);
}

template<class T, int Mode, class Access>
bool pascal_prefix_simd(Access& access, uint64_t n, uint64_t m, T* u, T* a) {
    using Ops = PascalScalarOps<T>;
    const Simd<T> tag;
    constexpr uint64_t lanes = vec_batch_lanes<T>;
    std::fill_n(u, m + 1, T(0));
    u[0] = T(1);
    if (!m) return true;
    bool valid = true;
    T q = 0;
    for (uint64_t i = 0; i < n; ++i) {
        pc::increments<Ops>(access, i, m, a);
        if (!i) {
            valid = pc::step<Ops, Mode>(u, a, m, true, nullptr, q) && valid;
            continue;
        }
        q = u[m - 1];
        for (uint64_t s = 1; s < m; ++s) {
            uint64_t end = m;
            for (; end >= s + lanes; end -= lanes) {
                const uint64_t begin = end - lanes;
                hn::StoreU(hn::MulAdd(hn::LoadU(tag, a + begin), hn::LoadU(tag, u + begin - 1), hn::LoadU(tag, u + begin)), tag, u + begin);
            }
            while (end > s) { --end; u[end] += a[end] * u[end - 1]; }
            q += u[m - 1];
        }
        u[m] += a[m] * q;
        valid = Ops::finite(q) && valid;
        for (uint64_t j = 1; j <= m; ++j) valid = Ops::finite(u[j]) && valid;
    }
    if constexpr (Mode == pc::scaled) {
        T factor = 1;
        for (uint64_t j = 1; j <= m; ++j) { factor /= T(j); u[j] *= factor; }
    }
    return valid;
}

template<class T, class Ops, int Mode, bool Backward>
void pascal_cpu(const T* path, T* out, const T* deriv, const pc::Words& words, uint64_t batch,
    uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, bool prefixes, int jobs, bool prefix_simd = false, uint64_t cutoff = 64) {
    using V = typename Ops::V;
    const uint64_t n = length > 1 ? pc::mul(length - 1, lead_lag ? 2 : 1) : 0;
    const uint64_t mmax = words.max_degree, stride = pc::add(mmax, 1);
    pc::Checkpoints plan(n, mmax, cutoff);
    const uint64_t small_count = Backward ? pc::add(pc::mul(4, stride), pc::mul(mmax, stride) / 2) : pc::mul(2, stride);
    uint64_t checkpoint_count = Backward ? plan.count : 0;
    uint64_t per_owner = pc::mul(pc::mul(pc::add(small_count, checkpoint_count), sizeof(V)), Mode == pc::guarded ? 1 : 2);
    const uint64_t budget = pc::workspace_limit();
    if (Backward && per_owner > budget && plan.block == 1 && n > 1) {
        plan = pc::Checkpoints(n, mmax, 0);
        checkpoint_count = plan.count;
        per_owner = pc::mul(pc::mul(pc::add(small_count, checkpoint_count), sizeof(V)), Mode == pc::guarded ? 1 : 2);
    }
    if constexpr (Ops::lanes > 1) {
        if (per_owner > budget) {
            pascal_cpu<T, PascalScalarOps<T>, Mode, Backward>(path, out, deriv, words, batch, dimension,
                length, time_aug, lead_lag, end_time, prefixes, jobs, false, cutoff);
            return;
        }
    }
    if (per_owner > budget) throw std::runtime_error("Pascal workspace limit is too small for one worker");
    const uint64_t groups = batch / Ops::lanes + (batch % Ops::lanes != 0);
    const uint64_t tasks = Backward ? groups : pc::mul(groups, words.count);
    const uint64_t owners = std::min<uint64_t>({tasks, uint64_t(jobs), budget / std::max<uint64_t>(per_owner, 1)});
    auto work = [&](uint64_t start, uint64_t end) HWY_ATTR {
        coefficient_workspace::Allocation allocation(pc::mul(pc::add(small_count, checkpoint_count), sizeof(V)));
        auto memory = hwy::AllocateAligned<V>(pc::add(small_count, checkpoint_count));
        if (!memory) throw std::bad_alloc();
        V* scratch = memory.get();
        V* checkpoints = scratch + small_count;
        for (uint64_t owner = start; owner < end; ++owner) {
            for (uint64_t task = tasks * owner / owners; task < tasks * (owner + 1) / owners; ++task) {
                const uint64_t group = Backward ? task : task / words.count;
                const uint64_t first = group * Ops::lanes, count = std::min<uint64_t>(Ops::lanes, batch - first);
                const uint64_t begin_word = Backward ? 0 : task % words.count;
                const uint64_t end_word = Backward ? words.count : begin_word + 1;
                if constexpr (Backward)
                    for (uint64_t lane = 0; lane < count; ++lane)
                        std::fill_n(out + (first + lane) * length * dimension, length * dimension, T(0));
                bool valid = true;
                for (uint64_t w = begin_word; w < end_word; ++w) {
                    const uint64_t m = words.packed[w], offset = words.packed[2 * words.count + w];
                    PascalAccess<T, Ops> access{path, out, deriv, words.word(w), dimension, length, n,
                        count, first, words.width, offset, time_aug, lead_lag, end_time};
                    if constexpr (Backward) {
                        valid = pascal_backward_dispatch<Ops, Mode>(access, n, m, plan.block, plan.blocks, checkpoints, scratch) && valid;
                    } else {
                        if constexpr (std::is_same_v<Ops, PascalScalarOps<T>> && Mode != pc::guarded) {
                            if (prefix_simd) valid = pascal_prefix_simd<T, Mode>(access, n, m, scratch, scratch + stride);
                            else valid = pascal_forward_dispatch<Ops, Mode>(access, n, m, scratch, scratch + stride);
                        } else valid = pascal_forward_dispatch<Ops, Mode>(access, n, m, scratch, scratch + stride);
                        if (valid || Mode == pc::guarded) {
                            const uint64_t outputs = prefixes ? std::max<uint64_t>(m, 1) : 1;
                            for (uint64_t j = 0; j < outputs; ++j) {
                                T values[Ops::lanes];
                                Ops::store(scratch[m ? (prefixes ? j + 1 : m) : 0], values);
                                for (uint64_t lane = 0; lane < count; ++lane)
                                    out[(first + lane) * words.width + offset + j] = values[lane];
                            }
                        }
                    }
                }
                if constexpr (Mode != pc::guarded) {
                    if (!valid) {
                        if constexpr (Backward) {
                            pascal_cpu<T, PascalScalarOps<T>, pc::guarded, true>(path + first * length * dimension,
                                out + first * length * dimension, deriv + first * words.width, words,
                                count, dimension, length, time_aug, lead_lag, end_time, true, 1, false, plan.block > 1 ? 0 : cutoff);
                        } else {
                            const uint64_t w = begin_word, m = words.packed[w], offset = words.packed[2 * words.count + w];
                            coefficient_workspace::Allocation fallback_allocation(2 * (m + 1) * sizeof(T));
                            std::vector<T> fallback(2 * (m + 1));
                            for (uint64_t lane = 0; lane < count; ++lane) {
                                PascalAccess<T, PascalScalarOps<T>> access{path, out, nullptr, words.word(w), dimension,
                                    length, n, 1, first + lane, words.width, offset, time_aug, lead_lag, end_time};
                                pc::forward<PascalScalarOps<T>, pc::guarded>(access, n, m, fallback.data(), fallback.data() + m + 1);
                                for (uint64_t j = 0; j < (prefixes ? std::max<uint64_t>(m, 1) : 1); ++j)
                                    out[(first + lane) * words.width + offset + j] = fallback[m ? (prefixes ? j + 1 : m) : 0];
                            }
                        }
                    }
                }
            }
        }
    };
    if (jobs == 1 || owners <= 1) work(0, owners);
    else spawn_batch_threads(owners, jobs, work);
}

template<class T>
struct PascalCompactAccess : PascalAccess<T, PascalScalarOps<T>> {
    const uint64_t* columns;
    uint64_t width;
    PASCAL_INLINE void scatter(uint64_t i, const T* gradient, uint64_t m) const {
        for (uint64_t j = 0; j < m; ++j) {
            const uint64_t c = this->word[j];
            if (this->time_aug && c == this->dimension * (this->lead_lag ? 2 : 1)) continue;
            const uint64_t before = this->lead_lag ? i / 2 + ((i % 2) && c >= this->dimension) : i;
            const uint64_t after = this->lead_lag ? (i + 1) / 2 + (((i + 1) % 2) && c >= this->dimension) : i + 1;
            if (before != after) {
                this->out[after * width + columns[j]] += gradient[j];
                this->out[before * width + columns[j]] -= gradient[j];
            }
        }
    }
};

template<class T, int Mode>
bool pascal_word_parallel(const T* path, T* out, const T* deriv, const pc::Words& words,
    uint64_t batch, uint64_t dimension, uint64_t length, bool ta, bool ll, T time, int jobs, uint64_t cutoff) {
    if (batch > 4 || words.count < 8 || uint64_t(jobs) < 2 * batch || length < 2) return false;
    const uint64_t n = pc::mul(length - 1, ll ? 2 : 1), m = words.max_degree;
    if (words.count < 16 && pc::mul(n, m) < 256) return false;
    const pc::Checkpoints plan(n, m, cutoff);
    const uint64_t scratch_count = pc::add(pc::mul(4, m + 1), pc::mul(m, m + 1) / 2);
    std::vector<uint64_t> coordinates, columns(words.index_count);
    for (uint64_t k = 0; k < words.index_count; ++k) {
        uint64_t c = words.packed[3 * words.count + k];
        if (ta && c == dimension * (ll ? 2 : 1)) continue;
        if (ll && c >= dimension) c -= dimension;
        coordinates.push_back(c);
    }
    std::sort(coordinates.begin(), coordinates.end());
    coordinates.erase(std::unique(coordinates.begin(), coordinates.end()), coordinates.end());
    const uint64_t width = coordinates.size(), partial_count = pc::mul(length, width);
    for (uint64_t k = 0; k < words.index_count; ++k) {
        uint64_t c = words.packed[3 * words.count + k];
        if (ll && c >= dimension) c -= dimension;
        columns[k] = std::lower_bound(coordinates.begin(), coordinates.end(), c) - coordinates.begin();
    }
    const uint64_t row = pc::add(partial_count, pc::add(scratch_count, plan.count));
    const uint64_t owners = std::min<uint64_t>({uint64_t(jobs) / batch, words.count,
        pc::workspace_limit() / std::max<uint64_t>(pc::mul(pc::mul(row, sizeof(T)), batch), 1)});
    if (owners < 2) return false;
    coefficient_workspace::Allocation allocation(pc::mul(pc::mul(pc::mul(owners, batch), row), sizeof(T)));
    std::vector<T> memory(pc::mul(pc::mul(owners, batch), row));
    auto work = [&](uint64_t start, uint64_t end) HWY_ATTR {
        for (uint64_t task = start; task < end; ++task) {
            const uint64_t sample = task / owners, owner = task % owners;
            T* partial = memory.data() + task * row;
            T* scratch = partial + partial_count;
            T* checkpoints = scratch + scratch_count;
            auto run = [&]<int CurrentMode>() HWY_ATTR {
                std::fill_n(partial, partial_count, T(0));
                bool valid = true;
                for (uint64_t w = words.count * owner / owners; w < words.count * (owner + 1) / owners; ++w) {
                    PascalCompactAccess<T> access{{path + sample * length * dimension, partial, deriv + sample * words.width,
                        words.word(w), dimension, length, n, 1, 0, words.width, words.packed[2 * words.count + w], ta, ll, time},
                        columns.data() + words.packed[words.count + w], width};
                    valid = pascal_backward_dispatch<PascalScalarOps<T>, CurrentMode>(access, n, words.packed[w], plan.block,
                        plan.blocks, checkpoints, scratch) && valid;
                }
                return valid;
            };
            const bool valid = run.template operator()<Mode>();
            if constexpr (Mode != pc::guarded) if (!valid) run.template operator()<pc::guarded>();
        }
    };
    spawn_batch_threads(owners * batch, jobs, work);
    std::fill_n(out, pc::mul(batch, pc::mul(length, dimension)), T(0));
    for (uint64_t sample = 0; sample < batch; ++sample)
        for (uint64_t owner = 0; owner < owners; ++owner) {
            const T* partial = memory.data() + (sample * owners + owner) * row;
            for (uint64_t i = 0; i < length; ++i)
                for (uint64_t c = 0; c < width; ++c)
                    out[(sample * length + i) * dimension + coordinates[c]] += partial[i * width + c];
        }
    return true;
}

template<class T, bool Backward>
void pascal_cpu_entry(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* indices,
    uint64_t num_words, const uint64_t* degrees, uint64_t batch, uint64_t dimension, uint64_t length,
    bool time_aug, bool lead_lag, T end_time, bool prefixes, int jobs, int method) {
    const uint64_t d = pc::add(pc::mul(dimension, lead_lag ? 2 : 1), time_aug ? 1 : 0);
    pc::Words words(indices, degrees, num_words, d, Backward || prefixes);
    pc::mul(pc::mul(batch, pc::mul(length, dimension)), sizeof(T));
    pc::mul(pc::mul(batch, words.width), sizeof(T));
    if (!jobs) throw std::invalid_argument("n_jobs cannot be zero");
    jobs = jobs > 0 ? jobs : int(get_max_threads()) + 1 + jobs;
    if (jobs < 1) throw std::invalid_argument("n_jobs is below the supported range");
    pc::variant(method, words.max_degree);
    if (!batch) return;
    if (method == pc::automatic && pascal_selection::cpu_region<T>(words, batch, dimension,
        length > 1 ? length - 1 : 0, jobs, time_aug, lead_lag,
        HWY_TARGET == HWY_AVX2 && pascal_selection::cpu_hardware())) method = pc::pascal_method;
    const int mode = pc::variant(method, words.max_degree);
    if (method == pc::horner || method == pc::automatic) {
        if constexpr (Backward) {
            coefficient_workspace::Allocation allocation(batch * words.width * sizeof(T));
            std::vector<T> seeds(deriv, deriv + batch * words.width);
            sig_coef_backprop_<T>(path, out, coefs, seeds.data(), indices, num_words, degrees,
                batch, dimension, length, time_aug, lead_lag, end_time, jobs);
        } else sig_coef_<T>(path, out, indices, num_words, degrees, batch, dimension, length,
            time_aug, lead_lag, end_time, prefixes, jobs);
        return;
    }
    if constexpr (Backward) {
        if (length == 2 && !lead_lag) {
            std::fill_n(out, batch * length * dimension, T(0));
            auto work = [&](uint64_t begin, uint64_t end) HWY_ATTR {
                coefficient_workspace::Allocation allocation(4 * (words.max_degree + 1) * sizeof(T));
                std::vector<T> scratch(4 * (words.max_degree + 1));
                for (uint64_t sample = begin; sample < end; ++sample)
                    for (uint64_t w = 0; w < words.count; ++w) {
                        PascalAccess<T, PascalScalarOps<T>> access{path, out, deriv, words.word(w), dimension,
                            length, 1, 1, sample, words.width, words.packed[2 * words.count + w], time_aug, false, end_time};
                        pc::single_segment_backward<PascalScalarOps<T>>(access, words.packed[w], scratch.data());
                    }
            };
            if (jobs == 1) work(0, batch);
            else spawn_batch_threads(batch, jobs, work);
            return;
        }
    }
    const uint64_t cutoff = pc::environment_option("PYSIGLIB_PASCAL_BENCH_CHECKPOINT", 256);
    if constexpr (Backward) if (pc::environment_option("PYSIGLIB_PASCAL_BENCH_WORDS", 1)) {
        bool parallel = false;
        if (mode == pc::guarded) parallel = pascal_word_parallel<T, pc::guarded>(path, out, deriv, words, batch, dimension, length, time_aug, lead_lag, end_time, jobs, cutoff);
        else if (mode == pc::scaled) parallel = pascal_word_parallel<T, pc::scaled>(path, out, deriv, words, batch, dimension, length, time_aug, lead_lag, end_time, jobs, cutoff);
        if (parallel) return;
    }
    if (mode == pc::guarded)
        pascal_cpu<T, PascalScalarOps<T>, pc::guarded, Backward>(path, out, deriv, words, batch, dimension,
            length, time_aug, lead_lag, end_time, prefixes, jobs, false, cutoff);
    else if (batch < vec_batch_lanes<T>)
        pascal_cpu<T, PascalScalarOps<T>, pc::scaled, Backward>(path, out, deriv, words, batch, dimension,
            length, time_aug, lead_lag, end_time, prefixes, jobs, pc::environment_option("PYSIGLIB_PASCAL_BENCH_ROWS", 0), cutoff);
    else
        pascal_cpu<T, PascalSimdOps<T>, pc::scaled, Backward>(path, out, deriv, words, batch, dimension,
            length, time_aug, lead_lag, end_time, prefixes, jobs, false, cutoff);
}

template<std::floating_point T>
void sig_coef_dispatch_(const T* path, T* out, const uint64_t* indices, uint64_t words, const uint64_t* degrees,
    uint64_t batch, uint64_t dimension, uint64_t length, bool time_aug, bool lead_lag, T end_time, bool prefixes, int n_jobs, int method) {
    pascal_cpu_entry<T, false>(path, out, nullptr, nullptr, indices, words, degrees, batch, dimension, length, time_aug, lead_lag, end_time, prefixes, n_jobs, method);
}

template<std::floating_point T>
void sig_coef_backprop_dispatch_(const T* path, T* out, const T* coefs, const T* deriv, const uint64_t* indices,
    uint64_t words, const uint64_t* degrees, uint64_t batch, uint64_t dimension, uint64_t length,
    bool time_aug, bool lead_lag, T end_time, int n_jobs, int method) {
    pascal_cpu_entry<T, true>(path, out, coefs, deriv, indices, words, degrees, batch, dimension, length, time_aug, lead_lag, end_time, true, n_jobs, method);
}

#define INSTANTIATE_PASCAL(T) \
template void sig_coef_dispatch_<T>(const T*, T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, bool, int, int); \
template void sig_coef_backprop_dispatch_<T>(const T*, T*, const T*, const T*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t, uint64_t, bool, bool, T, int, int)
INSTANTIATE_PASCAL(float);
INSTANTIATE_PASCAL(double);
#undef INSTANTIATE_PASCAL

} // namespace HWY_NAMESPACE
} // namespace cpsig
HWY_AFTER_NAMESPACE();
