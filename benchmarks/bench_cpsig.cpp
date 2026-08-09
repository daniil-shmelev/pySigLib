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

#include <benchmark/benchmark.h>
#include "cp_bch.h"
#include "cpsig.h"

#include <cstdint>
#include <random>
#include <vector>

// Suppress [[nodiscard]] warnings - we ignore status codes in hot loops.
#ifdef _MSC_VER
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

static void check(int rc, const char* fn) {
    if (rc != 0) {
        fprintf(stderr, "FATAL: %s returned %d\n", fn, rc);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::vector<double> random_data(uint64_t n, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static std::vector<uint64_t> branched_coef_tree_data() {
    constexpr uint64_t num_trees = 32;
    constexpr uint64_t dimension = 3;
    constexpr uint64_t degree = 4;
    std::vector<uint64_t> tree_data{num_trees};
    for (uint64_t tree = 0; tree < num_trees; ++tree) {
        uint64_t word = tree;
        tree_data.push_back(1);
        for (uint64_t node = 0; node < degree; ++node) {
            tree_data.push_back(word % dimension);
            word /= dimension;
            tree_data.push_back(node + 1 < degree ? 1 : 0);
        }
    }
    return tree_data;
}

static void clear_caches_outside_timing(benchmark::State& state) {
    state.PauseTiming();
    check(::clear_cache(false), "clear_cache");
    state.ResumeTiming();
}

// =========================================================================
// Cache construction
// =========================================================================

static void BM_prepare_log_sig_lyndon_words(benchmark::State& state) {
    for (auto _ : state) {
        clear_caches_outside_timing(state);
        check(::prepare_log_sig(3, 4, 1, false), "prepare_log_sig");
    }
}
BENCHMARK(BM_prepare_log_sig_lyndon_words)->Unit(benchmark::kMicrosecond);

static void BM_prepare_log_sig_lyndon_basis(benchmark::State& state) {
    for (auto _ : state) {
        clear_caches_outside_timing(state);
        check(::prepare_log_sig(3, 4, 2, false), "prepare_log_sig");
    }
}
BENCHMARK(BM_prepare_log_sig_lyndon_basis)->Unit(benchmark::kMicrosecond);

static void BM_prepare_log_sig_bch(benchmark::State& state) {
    for (auto _ : state) {
        clear_caches_outside_timing(state);
        set_bch_cache(3, 4);
    }
}
BENCHMARK(BM_prepare_log_sig_bch)->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_sig_nonplanar(benchmark::State& state) {
    for (auto _ : state) {
        clear_caches_outside_timing(state);
        check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    }
}
BENCHMARK(BM_prepare_branched_sig_nonplanar)->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_sig_planar(benchmark::State& state) {
    for (auto _ : state) {
        clear_caches_outside_timing(state);
        check(::prepare_branched_sig(3, 4, false, true), "prepare_branched_sig");
    }
}
BENCHMARK(BM_prepare_branched_sig_planar)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Path transforms
// =========================================================================

static void BM_transform_path_no_lead_lag(benchmark::State& state) {
    auto path = random_data(16 * 8 * 512, 1);
    std::vector<double> out(16 * 8 * 512);
    for (auto _ : state) {
        ::transform_path_d(path.data(), out.data(), 16, 8, 512, false, false, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_transform_path_no_lead_lag)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Signature forward / backward
// =========================================================================

static void BM_sig(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto path = random_data(4 * 4 * 32);
    std::vector<double> out(4 * slen);
    for (auto _ : state) {
        ::signature_d(path.data(), out.data(), 4, 4, 32, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig)->Unit(benchmark::kMicrosecond);

static void BM_sig_correction(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto path = random_data(4 * 4 * 32, 1);
    auto correction = random_data(4 * 4, 2);
    std::vector<double> out(4 * slen);
    check(::signature_d(
        path.data(), out.data(), 4, 4, 32, 5, false, false, 1.,
        true, true, 1, correction.data(), correction.size()), "signature_d");
    for (auto _ : state) {
        ::signature_d(
            path.data(), out.data(), 4, 4, 32, 5, false, false, 1.,
            true, true, 1, correction.data(), correction.size());
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_correction)->Unit(benchmark::kMicrosecond);

static void BM_sig_no_horner(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto path = random_data(4 * 4 * 32);
    std::vector<double> out(4 * slen);
    for (auto _ : state) {
        ::signature_d(path.data(), out.data(), 4, 4, 32, 5, false, false, 1., false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_no_horner)->Unit(benchmark::kMicrosecond);

static void BM_sig_backprop(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto path = random_data(4 * 4 * 32, 1);
    auto sig = random_data(4 * slen, 2);
    auto sig_deriv = random_data(4 * slen, 3);
    std::vector<double> out(4 * 4 * 32);
    for (auto _ : state) {
        ::sig_backprop_d(path.data(), out.data(), sig_deriv.data(), sig.data(), 4, 4, 32, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_backprop)->Unit(benchmark::kMicrosecond);

static void BM_sig_backprop_correction(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto path = random_data(4 * 4 * 32, 1);
    auto sig = random_data(4 * slen, 2);
    auto sig_deriv = random_data(4 * slen, 3);
    auto correction = random_data(4 * 4, 4);
    std::vector<double> out(4 * 4 * 32);
    check(::sig_backprop_d(
        path.data(), out.data(), sig_deriv.data(), sig.data(), 4, 4, 32, 5,
        false, false, 1., true, 1, correction.data(), correction.size()), "sig_backprop_d");
    for (auto _ : state) {
        ::sig_backprop_d(
            path.data(), out.data(), sig_deriv.data(), sig.data(), 4, 4, 32, 5,
            false, false, 1., true, 1, correction.data(), correction.size());
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_backprop_correction)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Sig combine / backprop
// =========================================================================

static void BM_sig_combine(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig1 = random_data(8 * slen, 1);
    auto sig2 = random_data(8 * slen, 2);
    std::vector<double> out(8 * slen);
    for (auto _ : state) {
        ::sig_combine_d(sig1.data(), sig2.data(), out.data(), 8, 4, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_combine)->Unit(benchmark::kMicrosecond);

static void BM_sig_combine_backprop(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig1 = random_data(8 * slen, 1);
    auto sig2 = random_data(8 * slen, 2);
    auto combo_deriv = random_data(8 * slen, 3);
    std::vector<double> d_sig1(8 * slen);
    std::vector<double> d_sig2(8 * slen);
    for (auto _ : state) {
        ::sig_combine_backprop_d(combo_deriv.data(), d_sig1.data(), d_sig2.data(),
                                 sig1.data(), sig2.data(), 8, 4, 5);
        benchmark::DoNotOptimize(d_sig1.data());
    }
}
BENCHMARK(BM_sig_combine_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Linear sig / sig_join / sig_join_backprop
// =========================================================================

static void BM_linear_sig(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto displacement = random_data(16 * 4, 1);
    std::vector<double> out(16 * slen);
    for (auto _ : state) {
        ::linear_sig_d(displacement.data(), out.data(), 16, 4, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_linear_sig)->Unit(benchmark::kMicrosecond);

static void BM_sig_join(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    auto displacement = random_data(8 * 4, 2);
    std::vector<double> out(8 * slen);
    for (auto _ : state) {
        ::sig_join_d(sig.data(), displacement.data(), out.data(), 8, 4, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_join)->Unit(benchmark::kMicrosecond);

static void BM_sig_join_backprop(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    auto displacement = random_data(8 * 4, 2);
    auto d_out_data = random_data(8 * slen, 3);
    std::vector<double> d_sig(8 * slen);
    std::vector<double> d_disp(8 * 4);
    for (auto _ : state) {
        ::sig_join_backprop_d(d_out_data.data(), d_sig.data(), d_disp.data(),
                              sig.data(), displacement.data(), 8, 4, 5);
        benchmark::DoNotOptimize(d_sig.data());
    }
}
BENCHMARK(BM_sig_join_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Sig coef / backprop
// =========================================================================

static void BM_sig_coef(benchmark::State& state) {
    auto path = random_data(8 * 3 * 32, 1);
    std::vector<uint64_t> degrees(32, 4);
    std::vector<uint64_t> multi_idx(32 * 4, 0);
    std::vector<double> out(8 * 32);
    for (auto _ : state) {
        ::sig_coef_d(path.data(), out.data(), multi_idx.data(),
                     32, degrees.data(), 8, 3, 32,
                     false, false, 1.0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_coef)->Unit(benchmark::kMicrosecond);

static void BM_sig_coef_backprop(benchmark::State& state) {
    auto path = random_data(8 * 3 * 32, 1);
    std::vector<uint64_t> degrees(32, 4);
    std::vector<uint64_t> multi_idx(32 * 4, 0);
    const uint64_t prefix_size = 4 * 32;
    auto coefs = random_data(8 * prefix_size, 2);
    auto derivs = random_data(8 * prefix_size, 3);
    std::vector<double> out(8 * 3 * 32);
    for (auto _ : state) {
        auto derivs_copy = derivs;
        ::sig_coef_backprop_d(path.data(), out.data(), coefs.data(),
                              derivs_copy.data(), multi_idx.data(),
                              32, degrees.data(), 8, 3, 32,
                              false, false, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_coef_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Sig to log sig / backprop (method 0 = expanded)
// =========================================================================

static void BM_sig_to_log_sig(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 0, false), "prepare_log_sig");
    for (auto _ : state) {
        ::sig_to_log_sig_d(sig.data(), out.data(), 8, 4, 5, false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig)->Unit(benchmark::kMicrosecond);

static void BM_sig_to_log_sig_lyndon(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    const uint64_t ls_len = ::log_sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    std::vector<double> out(8 * ls_len);
    check(::prepare_log_sig(4, 5, 1, false), "prepare_log_sig");
    for (auto _ : state) {
        ::sig_to_log_sig_d(sig.data(), out.data(), 8, 4, 5, false, false, 1);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig_lyndon)->Unit(benchmark::kMicrosecond);

static void BM_sig_to_log_sig_brackets(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    const uint64_t ls_len = ::log_sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    std::vector<double> out(8 * ls_len);
    check(::prepare_log_sig(4, 5, 2, false), "prepare_log_sig");
    for (auto _ : state) {
        ::sig_to_log_sig_d(sig.data(), out.data(), 8, 4, 5, false, false, 2);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig_brackets)->Unit(benchmark::kMicrosecond);

static void BM_sig_to_log_sig_backprop(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto sig = random_data(8 * slen, 1);
    auto ls_derivs = random_data(8 * slen, 2);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 0, false), "prepare_log_sig");
    for (auto _ : state) {
        ::sig_to_log_sig_backprop_d(sig.data(), out.data(), ls_derivs.data(),
                                    8, 4, 5, false, false, 0, 1);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// logsig_to_sig (tensor exp) / backprop
// =========================================================================

static void BM_logsig_to_sig(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto log_sig = random_data(8 * slen, 1);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 0, false), "prepare_log_sig");
    for (auto _ : state) {
        ::logsig_to_sig_d(log_sig.data(), out.data(), 8, 4, 5, false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig)->Unit(benchmark::kMicrosecond);

static void BM_logsig_to_sig_lyndon(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    const uint64_t ls_len = ::log_sig_length(4, 5);
    auto log_sig = random_data(8 * ls_len, 1);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 1, false), "prepare_log_sig");
    for (auto _ : state) {
        ::logsig_to_sig_d(log_sig.data(), out.data(), 8, 4, 5, false, false, 1);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig_lyndon)->Unit(benchmark::kMicrosecond);

static void BM_logsig_to_sig_brackets(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    const uint64_t ls_len = ::log_sig_length(4, 5);
    auto log_sig = random_data(8 * ls_len, 1);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 2, false), "prepare_log_sig");
    for (auto _ : state) {
        ::logsig_to_sig_d(log_sig.data(), out.data(), 8, 4, 5, false, false, 2);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig_brackets)->Unit(benchmark::kMicrosecond);

static void BM_logsig_to_sig_backprop(benchmark::State& state) {
    const uint64_t slen = ::sig_length(4, 5);
    auto log_sig = random_data(8 * slen, 1);
    auto sig_derivs = random_data(8 * slen, 2);
    std::vector<double> out(8 * slen);
    check(::prepare_log_sig(4, 5, 0, false), "prepare_log_sig");
    for (auto _ : state) {
        ::logsig_to_sig_backprop_d(log_sig.data(), out.data(),
                                   sig_derivs.data(), 8, 4, 5, false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Log sig from path (BCH method) / backprop
// =========================================================================

static void BM_log_sig_from_path(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto path = random_data(4 * 3 * 32, 1);
    std::vector<double> out(4 * ls_len);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_from_path_d(path.data(), out.data(), 4, 32, 3, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_from_path)->Unit(benchmark::kMicrosecond);

static void BM_log_sig_from_path_backprop(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto path = random_data(4 * 3 * 32, 1);
    auto d_out_data = random_data(4 * ls_len, 2);
    std::vector<double> d_path(4 * 3 * 32);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_from_path_backprop_d(d_out_data.data(), d_path.data(),
                                       path.data(), 4, 32, 3, 5);
        benchmark::DoNotOptimize(d_path.data());
    }
}
BENCHMARK(BM_log_sig_from_path_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Log sig combine / backprop
// =========================================================================

static void BM_log_sig_combine(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto ls1 = random_data(64 * ls_len, 1);
    auto ls2 = random_data(64 * ls_len, 2);
    std::vector<double> out(64 * ls_len);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_combine_d(ls1.data(), ls2.data(), out.data(), 64, 3, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_combine)->Unit(benchmark::kMicrosecond);

static void BM_log_sig_combine_backprop(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto ls1 = random_data(64 * ls_len, 1);
    auto ls2 = random_data(64 * ls_len, 2);
    auto d_out_data = random_data(64 * ls_len, 3);
    std::vector<double> d_ls1(64 * ls_len);
    std::vector<double> d_ls2(64 * ls_len);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_combine_backprop_d(d_out_data.data(), d_ls1.data(),
                                     d_ls2.data(), ls1.data(), ls2.data(),
                                     64, 3, 5);
        benchmark::DoNotOptimize(d_ls1.data());
    }
}
BENCHMARK(BM_log_sig_combine_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Log sig join / backprop
// =========================================================================

static void BM_log_sig_join(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto log_sig = random_data(64 * ls_len, 1);
    auto displacement = random_data(64 * 3, 2);
    std::vector<double> out(64 * ls_len);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_join_d(log_sig.data(), displacement.data(), out.data(), 64, 3, 5);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_join)->Unit(benchmark::kMicrosecond);

static void BM_log_sig_join_backprop(benchmark::State& state) {
    const uint64_t ls_len = ::log_sig_length(3, 5);
    auto log_sig = random_data(64 * ls_len, 1);
    auto displacement = random_data(64 * 3, 2);
    auto d_out_data = random_data(64 * ls_len, 3);
    std::vector<double> d_ls(64 * ls_len);
    std::vector<double> d_disp(64 * 3);
    check(::prepare_log_sig(3, 5, 3, false), "prepare_log_sig");
    for (auto _ : state) {
        ::log_sig_join_backprop_d(d_out_data.data(), d_ls.data(),
                                  d_disp.data(), log_sig.data(),
                                  displacement.data(), 64, 3, 5);
        benchmark::DoNotOptimize(d_ls.data());
    }
}
BENCHMARK(BM_log_sig_join_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Sig kernel / backprop
// =========================================================================

static void BM_sig_kernel(benchmark::State& state) {
    auto gram = random_data(32 * 63 * 63, 1);
    std::vector<double> out(32);
    for (auto _ : state) {
        ::sig_kernel_d(gram.data(), out.data(), 32, 3, 64, 64, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_kernel)->Unit(benchmark::kMicrosecond);

static void BM_sig_kernel_backprop(benchmark::State& state) {
    auto gram = random_data(32 * 63 * 63, 1);
    auto deriv = random_data(32, 2);
    auto k_grid = random_data(32 * 64 * 64, 3);
    std::vector<double> out(32 * 63 * 63);
    for (auto _ : state) {
        ::sig_kernel_backprop_d(gram.data(), out.data(), deriv.data(),
                                k_grid.data(), 32, 3, 64, 64, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_kernel_backprop)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_kernel(benchmark::State& state) {
    auto gram = random_data(16 * 31 * 31, 1);
    for (auto& x : gram) x *= 0.01;
    std::vector<double> out(16);
    check(::branched_sig_kernel_d(
        gram.data(), out.data(), 16, 3, 32, 32, 3, 0, 0, false),
        "branched_sig_kernel_d");
    for (auto _ : state) {
        ::branched_sig_kernel_d(
            gram.data(), out.data(), 16, 3, 32, 32, 3, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_kernel)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_kernel_backprop(benchmark::State& state) {
    auto gram = random_data(16 * 31 * 31, 1);
    for (auto& x : gram) x *= 0.01;
    auto deriv = random_data(16, 2);
    std::vector<double> out(16 * 31 * 31);
    check(::branched_sig_kernel_backprop_d(
        gram.data(), out.data(), deriv.data(), nullptr,
        16, 3, 32, 32, 3, 0, 0, false),
        "branched_sig_kernel_backprop_d");
    for (auto _ : state) {
        ::branched_sig_kernel_backprop_d(
            gram.data(), out.data(), deriv.data(), nullptr,
            16, 3, 32, 32, 3, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_kernel_backprop)->Unit(benchmark::kMicrosecond);

// =========================================================================
// Branched sig / backprop / combine / combine_backprop
// =========================================================================

static void BM_branched_sig_length(benchmark::State& state) {
    uint64_t blen = 0;
    for (auto _ : state) {
        blen ^= ::branched_sig_length(3, 8, false);
        blen ^= ::branched_sig_length(3, 8, true);
        benchmark::DoNotOptimize(blen);
    }
}
BENCHMARK(BM_branched_sig_length)->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_sig_coef(benchmark::State& state) {
    const auto tree_data = branched_coef_tree_data();
    for (auto _ : state) {
        state.PauseTiming();
        check(::clear_cache(false), "clear_cache");
        state.ResumeTiming();
        check(::prepare_branched_sig_coef(
            tree_data.data(), tree_data.size(), 3, 3, 4, false),
            "prepare_branched_sig_coef");
    }
}
BENCHMARK(BM_prepare_branched_sig_coef)->Unit(benchmark::kMillisecond);

static void BM_branched_sig_coef(benchmark::State& state) {
    auto path = random_data(8 * 3 * 32, 1);
    const auto tree_data = branched_coef_tree_data();
    check(::prepare_branched_sig_coef(
        tree_data.data(), tree_data.size(), 3, 3, 4, false),
        "prepare_branched_sig_coef");
    std::vector<double> out(8 * tree_data[0]);
    check(::branched_sig_coef_d(
        path.data(), out.data(), tree_data.data(), tree_data.size(),
        8, 3, 32, 4), "branched_sig_coef_d");
    for (auto _ : state) {
        ::branched_sig_coef_d(
            path.data(), out.data(), tree_data.data(), tree_data.size(),
            8, 3, 32, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_coef)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_coef_backprop(benchmark::State& state) {
    auto path = random_data(8 * 3 * 32, 1);
    const auto tree_data = branched_coef_tree_data();
    check(::prepare_branched_sig_coef(
        tree_data.data(), tree_data.size(), 3, 3, 4, false),
        "prepare_branched_sig_coef");
    std::vector<double> coefs(8 * tree_data[0]);
    auto derivs = random_data(8 * tree_data[0], 2);
    std::vector<double> out(8 * 3 * 32);
    check(::branched_sig_coef_d(
        path.data(), coefs.data(), tree_data.data(), tree_data.size(),
        8, 3, 32, 4), "branched_sig_coef_d");
    check(::branched_sig_coef_backprop_d(
        path.data(), out.data(), coefs.data(), derivs.data(),
        tree_data.data(), tree_data.size(), 8, 3, 32, 4),
        "branched_sig_coef_backprop_d");
    for (auto _ : state) {
        ::branched_sig_coef_backprop_d(
            path.data(), out.data(), coefs.data(), derivs.data(),
            tree_data.data(), tree_data.size(), 8, 3, 32, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_coef_backprop)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto path = random_data(8 * 3 * 32, 1);
    std::vector<double> out(8 * blen);
    for (auto _ : state) {
        ::branched_sig_d(path.data(), out.data(), 8, 3, 32, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_correction(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto path = random_data(8 * 3 * 32, 1);
    auto correction = random_data(3 * 3, 2);
    std::vector<double> out(8 * blen);
    check(::branched_sig_d(
        path.data(), out.data(), 8, 3, 32, 4, 1, false, false, 1.,
        false, true, correction.data(), correction.size()), "branched_sig_d");
    for (auto _ : state) {
        ::branched_sig_d(
            path.data(), out.data(), 8, 3, 32, 4, 1, false, false, 1.,
            false, true, correction.data(), correction.size());
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_correction)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_planar(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, true), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, true);
    auto path = random_data(8 * 3 * 32, 1);
    std::vector<double> out(8 * blen);
    for (auto _ : state) {
        ::branched_sig_d(path.data(), out.data(), 8, 3, 32, 4, 1, false, false, 1., true);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_planar)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_backprop(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto path = random_data(8 * 3 * 32, 1);
    auto bsig = random_data(8 * blen, 2);
    auto bsig_derivs = random_data(8 * blen, 3);
    std::vector<double> out(8 * 3 * 32);
    for (auto _ : state) {
        ::branched_sig_backprop_d(path.data(), out.data(), bsig_derivs.data(),
                                  bsig.data(), 8, 3, 32, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_backprop)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_backprop_correction(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto path = random_data(8 * 3 * 32, 1);
    auto bsig = random_data(8 * blen, 2);
    auto bsig_derivs = random_data(8 * blen, 3);
    auto correction = random_data(3 * 3, 4);
    std::vector<double> out(8 * 3 * 32);
    check(::branched_sig_backprop_d(
        path.data(), out.data(), bsig_derivs.data(), bsig.data(),
        8, 3, 32, 4, 1, false, false, 1., false, true,
        correction.data(), correction.size()), "branched_sig_backprop_d");
    for (auto _ : state) {
        ::branched_sig_backprop_d(
            path.data(), out.data(), bsig_derivs.data(), bsig.data(),
            8, 3, 32, 4, 1, false, false, 1., false, true,
            correction.data(), correction.size());
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_backprop_correction)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_to_log_sig(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto bsig = random_data(64 * blen, 1);
    std::vector<double> out(64 * blen);
    check(::branched_sig_to_log_sig_d(
        bsig.data(), out.data(), 64, 3, 4), "branched_sig_to_log_sig_d");
    for (auto _ : state) {
        ::branched_sig_to_log_sig_d(bsig.data(), out.data(), 64, 3, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_to_log_sig)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_to_log_sig_backprop(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto bsig = random_data(64 * blen, 1);
    auto derivs = random_data(64 * blen, 2);
    std::vector<double> out(64 * blen);
    check(::branched_sig_to_log_sig_backprop_d(
        bsig.data(), derivs.data(), out.data(), 64, 3, 4),
        "branched_sig_to_log_sig_backprop_d");
    for (auto _ : state) {
        ::branched_sig_to_log_sig_backprop_d(
            bsig.data(), derivs.data(), out.data(), 64, 3, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_to_log_sig_backprop)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_combine(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto bsig1 = random_data(64 * blen, 1);
    auto bsig2 = random_data(64 * blen, 2);
    std::vector<double> out(64 * blen);
    for (auto _ : state) {
        ::branched_sig_combine_d(bsig1.data(), bsig2.data(), out.data(), 64, 3, 4);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_combine)->Unit(benchmark::kMicrosecond);

static void BM_branched_sig_combine_backprop(benchmark::State& state) {
    check(::prepare_branched_sig(3, 4, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(3, 4, false);
    auto bsig1 = random_data(64 * blen, 1);
    auto bsig2 = random_data(64 * blen, 2);
    auto derivs = random_data(64 * blen, 3);
    std::vector<double> out1(64 * blen);
    std::vector<double> out2(64 * blen);
    for (auto _ : state) {
        ::branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(),
                                          derivs.data(), out1.data(),
                                          out2.data(), 64, 3, 4);
        benchmark::DoNotOptimize(out1.data());
    }
}
BENCHMARK(BM_branched_sig_combine_backprop)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
