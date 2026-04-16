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
#include "cpsig.h"

#include <cstdint>
#include <random>
#include <vector>

// Suppress [[nodiscard]] warnings — we ignore status codes in hot loops.
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

// =========================================================================
// Path transforms
// =========================================================================

static void BM_transform_path(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t out_dim = 2 * dim;
    const uint64_t out_len = 2 * len - 1;
    std::vector<double> out(batch * out_dim * out_len);

    for (auto _ : state) {
        ::transform_path_d(path.data(), out.data(), batch, dim, len,
                           false, true, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_transform_path)->Args({128, 16, 2048})->Unit(benchmark::kMillisecond);

static void BM_transform_path_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);

    const uint64_t t_dim = 2 * dim;
    const uint64_t t_len = 2 * len - 1;
    auto derivs = random_data(batch * t_dim * t_len, 1);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        ::transform_path_backprop_d(derivs.data(), out.data(),
                                    batch, dim, len, false, true, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_transform_path_backprop)->Args({128, 16, 2048})->Unit(benchmark::kMillisecond);

// =========================================================================
// Signature forward / backward
// =========================================================================

static void BM_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len);
    const uint64_t slen = ::sig_length(dim, deg);
    std::vector<double> out(batch * slen);

    for (auto _ : state) {
        ::signature_d(path.data(), out.data(), batch, dim, len, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig)->Args({16, 5, 32, 7})->Unit(benchmark::kMillisecond);

static void BM_sig_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t slen = ::sig_length(dim, deg);
    auto sig       = random_data(batch * slen, 2);
    auto sig_deriv = random_data(batch * slen, 3);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        ::sig_backprop_d(path.data(), out.data(), sig_deriv.data(),
                         sig.data(), batch, dim, len, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_backprop)->Args({4, 5, 32, 7})->Unit(benchmark::kMillisecond);

// =========================================================================
// Sig combine / backprop
// =========================================================================

static void BM_sig_combine(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig1 = random_data(batch * slen, 1);
    auto sig2 = random_data(batch * slen, 2);
    std::vector<double> out(batch * slen);

    for (auto _ : state) {
        ::sig_combine_d(sig1.data(), sig2.data(), out.data(), batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_combine)->Args({64, 5, 7})->Unit(benchmark::kMillisecond);

static void BM_sig_combine_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig1        = random_data(batch * slen, 1);
    auto sig2        = random_data(batch * slen, 2);
    auto combo_deriv = random_data(batch * slen, 3);
    std::vector<double> d_sig1(batch * slen);
    std::vector<double> d_sig2(batch * slen);

    for (auto _ : state) {
        ::sig_combine_backprop_d(combo_deriv.data(), d_sig1.data(),
                                 d_sig2.data(), sig1.data(), sig2.data(),
                                 batch, dim, deg);
        benchmark::DoNotOptimize(d_sig1.data());
    }
}
BENCHMARK(BM_sig_combine_backprop)->Args({32, 5, 7})->Unit(benchmark::kMillisecond);

// =========================================================================
// Linear sig / sig_join / sig_join_backprop
// =========================================================================

static void BM_linear_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    auto displacement = random_data(batch * dim, 1);
    const uint64_t slen = ::sig_length(dim, deg);
    std::vector<double> out(batch * slen);

    for (auto _ : state) {
        ::linear_sig_d(displacement.data(), out.data(), batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_linear_sig)->Args({256, 5, 7})->Unit(benchmark::kMillisecond);

static void BM_sig_join(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig = random_data(batch * slen, 1);
    auto displacement = random_data(batch * dim, 2);
    std::vector<double> out(batch * slen);

    for (auto _ : state) {
        ::sig_join_d(sig.data(), displacement.data(), out.data(),
                     batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_join)->Args({64, 5, 7})->Unit(benchmark::kMillisecond);

static void BM_sig_join_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig = random_data(batch * slen, 1);
    auto displacement = random_data(batch * dim, 2);
    auto d_out_data = random_data(batch * slen, 3);
    std::vector<double> d_sig(batch * slen);
    std::vector<double> d_disp(batch * dim);

    for (auto _ : state) {
        ::sig_join_backprop_d(d_out_data.data(), d_sig.data(), d_disp.data(),
                              sig.data(), displacement.data(),
                              batch, dim, deg);
        benchmark::DoNotOptimize(d_sig.data());
    }
}
BENCHMARK(BM_sig_join_backprop)->Args({32, 5, 7})->Unit(benchmark::kMillisecond);

// =========================================================================
// Sig coef / backprop
// =========================================================================

static void BM_sig_coef(benchmark::State& state) {
    const uint64_t batch   = state.range(0);
    const uint64_t dim     = state.range(1);
    const uint64_t len     = state.range(2);
    const uint64_t deg     = state.range(3);
    const uint64_t n_words = state.range(4);

    auto path = random_data(batch * dim * len, 1);
    std::vector<uint64_t> degrees(n_words, deg);
    std::vector<uint64_t> multi_idx(n_words * deg, 0);
    std::vector<double> out(batch * n_words);

    for (auto _ : state) {
        ::sig_coef_d(path.data(), out.data(), multi_idx.data(),
                     n_words, degrees.data(), batch, dim, len,
                     false, false, 1.0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_coef)->Args({64, 3, 64, 6, 128})->Unit(benchmark::kMillisecond);

static void BM_sig_coef_backprop(benchmark::State& state) {
    const uint64_t batch   = state.range(0);
    const uint64_t dim     = state.range(1);
    const uint64_t len     = state.range(2);
    const uint64_t deg     = state.range(3);
    const uint64_t n_words = state.range(4);

    auto path = random_data(batch * dim * len, 1);
    std::vector<uint64_t> degrees(n_words, deg);
    std::vector<uint64_t> multi_idx(n_words * deg, 0);
    const uint64_t prefix_size = deg * n_words;
    auto coefs  = random_data(batch * prefix_size, 2);
    auto derivs = random_data(batch * prefix_size, 3);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        auto derivs_copy = derivs;
        ::sig_coef_backprop_d(path.data(), out.data(), coefs.data(),
                              derivs_copy.data(), multi_idx.data(),
                              n_words, degrees.data(), batch, dim, len,
                              false, false, 1.0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_coef_backprop)->Args({64, 3, 64, 6, 128})->Unit(benchmark::kMillisecond);

// =========================================================================
// Sig to log sig / backprop (method 0 = expanded)
// =========================================================================

static void BM_sig_to_log_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig = random_data(batch * slen, 1);
    std::vector<double> out(batch * slen);

    check(::prepare_log_sig(dim, deg, 0, false), "prepare_log_sig");

    for (auto _ : state) {
        ::sig_to_log_sig_d(sig.data(), out.data(), batch, dim, deg,
                           false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig)->Args({64, 5, 7})->Unit(benchmark::kMillisecond);

static void BM_sig_to_log_sig_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto sig = random_data(batch * slen, 1);
    auto ls_derivs = random_data(batch * slen, 2);
    std::vector<double> out(batch * slen);

    check(::prepare_log_sig(dim, deg, 0, false), "prepare_log_sig");

    for (auto _ : state) {
        ::sig_to_log_sig_backprop_d(sig.data(), out.data(), ls_derivs.data(),
                                    batch, dim, deg, false, false, 0, 1);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_to_log_sig_backprop)->Args({32, 5, 7})->Unit(benchmark::kMillisecond);

// =========================================================================
// logsig_to_sig (tensor exp) / backprop
// =========================================================================

static void BM_logsig_to_sig(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto log_sig = random_data(batch * slen, 1);
    std::vector<double> out(batch * slen);

    check(::prepare_log_sig(dim, deg, 0, false), "prepare_log_sig");

    for (auto _ : state) {
        ::logsig_to_sig_d(log_sig.data(), out.data(), batch, dim, deg,
                          false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig)->Args({32, 5, 7})->Unit(benchmark::kMillisecond);

static void BM_logsig_to_sig_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t slen = ::sig_length(dim, deg);
    auto log_sig = random_data(batch * slen, 1);
    auto sig_derivs = random_data(batch * slen, 2);
    std::vector<double> out(batch * slen);

    check(::prepare_log_sig(dim, deg, 0, false), "prepare_log_sig");

    for (auto _ : state) {
        ::logsig_to_sig_backprop_d(log_sig.data(), out.data(),
                                   sig_derivs.data(), batch, dim, deg,
                                   false, false, 0);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_logsig_to_sig_backprop)->Args({16, 5, 7})->Unit(benchmark::kMillisecond);

// =========================================================================
// Log sig from path (BCH method) / backprop
// =========================================================================

static void BM_log_sig_from_path(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t ls_len = ::log_sig_length(dim, deg);
    std::vector<double> out(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_from_path_d(path.data(), out.data(), batch, len, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_from_path)->Args({16, 3, 64, 6})->Unit(benchmark::kMillisecond);

static void BM_log_sig_from_path_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len   = state.range(2);
    const uint64_t deg   = state.range(3);

    auto path = random_data(batch * dim * len, 1);
    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto d_out_data = random_data(batch * ls_len, 2);
    std::vector<double> d_path(batch * dim * len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_from_path_backprop_d(d_out_data.data(), d_path.data(),
                                       path.data(), batch, len, dim, deg);
        benchmark::DoNotOptimize(d_path.data());
    }
}
BENCHMARK(BM_log_sig_from_path_backprop)->Args({8, 3, 64, 6})->Unit(benchmark::kMillisecond);

// =========================================================================
// Log sig combine / backprop
// =========================================================================

static void BM_log_sig_combine(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto ls1 = random_data(batch * ls_len, 1);
    auto ls2 = random_data(batch * ls_len, 2);
    std::vector<double> out(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_combine_d(ls1.data(), ls2.data(), out.data(),
                            batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_combine)->Args({512, 3, 6})->Unit(benchmark::kMillisecond);

static void BM_log_sig_combine_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto ls1 = random_data(batch * ls_len, 1);
    auto ls2 = random_data(batch * ls_len, 2);
    auto d_out_data = random_data(batch * ls_len, 3);
    std::vector<double> d_ls1(batch * ls_len);
    std::vector<double> d_ls2(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_combine_backprop_d(d_out_data.data(), d_ls1.data(),
                                     d_ls2.data(), ls1.data(), ls2.data(),
                                     batch, dim, deg);
        benchmark::DoNotOptimize(d_ls1.data());
    }
}
BENCHMARK(BM_log_sig_combine_backprop)->Args({512, 3, 6})->Unit(benchmark::kMillisecond);

// =========================================================================
// Log sig join / backprop
// =========================================================================

static void BM_log_sig_join(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto log_sig = random_data(batch * ls_len, 1);
    auto displacement = random_data(batch * dim, 2);
    std::vector<double> out(batch * ls_len);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_join_d(log_sig.data(), displacement.data(), out.data(),
                         batch, dim, deg);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_log_sig_join)->Args({512, 3, 6})->Unit(benchmark::kMillisecond);

static void BM_log_sig_join_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t deg   = state.range(2);

    const uint64_t ls_len = ::log_sig_length(dim, deg);
    auto log_sig = random_data(batch * ls_len, 1);
    auto displacement = random_data(batch * dim, 2);
    auto d_out_data = random_data(batch * ls_len, 3);
    std::vector<double> d_ls(batch * ls_len);
    std::vector<double> d_disp(batch * dim);

    check(::prepare_log_sig(dim, deg, 3, false), "prepare_log_sig");

    for (auto _ : state) {
        ::log_sig_join_backprop_d(d_out_data.data(), d_ls.data(),
                                  d_disp.data(), log_sig.data(),
                                  displacement.data(), batch, dim, deg);
        benchmark::DoNotOptimize(d_ls.data());
    }
}
BENCHMARK(BM_log_sig_join_backprop)->Args({512, 3, 6})->Unit(benchmark::kMillisecond);

// =========================================================================
// Sig kernel / backprop
// =========================================================================

static void BM_sig_kernel(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len1  = state.range(2);
    const uint64_t len2  = state.range(3);

    auto gram = random_data(batch * (len1 - 1) * (len2 - 1), 1);
    std::vector<double> out(batch);

    for (auto _ : state) {
        ::sig_kernel_d(gram.data(), out.data(), batch, dim,
                       len1, len2, 0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_kernel)->Args({256, 5, 128, 128})->Unit(benchmark::kMillisecond);

static void BM_sig_kernel_backprop(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const uint64_t dim   = state.range(1);
    const uint64_t len1  = state.range(2);
    const uint64_t len2  = state.range(3);

    const uint64_t gram_size = (len1 - 1) * (len2 - 1);
    auto gram  = random_data(batch * gram_size, 1);
    auto deriv = random_data(batch, 2);
    auto k_grid = random_data(batch * len1 * len2, 3);
    std::vector<double> out(batch * gram_size);

    for (auto _ : state) {
        ::sig_kernel_backprop_d(gram.data(), out.data(), deriv.data(),
                                k_grid.data(), batch, dim, len1, len2,
                                0, 0, false);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_sig_kernel_backprop)->Args({256, 5, 128, 128})->Unit(benchmark::kMillisecond);

// =========================================================================
// Branched sig / backprop / combine / combine_backprop
// =========================================================================

static void BM_branched_sig(benchmark::State& state) {
    const uint64_t batch     = state.range(0);
    const uint64_t dim       = state.range(1);
    const uint64_t len       = state.range(2);
    const uint64_t max_nodes = state.range(3);

    check(::prepare_branched_sig(dim, max_nodes, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(dim, max_nodes, false);

    auto path = random_data(batch * dim * len, 1);
    std::vector<double> out(batch * blen);

    for (auto _ : state) {
        ::branched_sig_d(path.data(), out.data(), batch, dim, len, max_nodes);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig)->Args({64, 3, 64, 4})->Unit(benchmark::kMillisecond);

static void BM_branched_sig_backprop(benchmark::State& state) {
    const uint64_t batch     = state.range(0);
    const uint64_t dim       = state.range(1);
    const uint64_t len       = state.range(2);
    const uint64_t max_nodes = state.range(3);

    check(::prepare_branched_sig(dim, max_nodes, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(dim, max_nodes, false);

    auto path = random_data(batch * dim * len, 1);
    auto bsig = random_data(batch * blen, 2);
    auto bsig_derivs = random_data(batch * blen, 3);
    std::vector<double> out(batch * dim * len);

    for (auto _ : state) {
        ::branched_sig_backprop_d(path.data(), out.data(), bsig_derivs.data(),
                                  bsig.data(), batch, dim, len, max_nodes);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_backprop)->Args({64, 3, 64, 4})->Unit(benchmark::kMillisecond);

static void BM_branched_sig_combine(benchmark::State& state) {
    const uint64_t batch     = state.range(0);
    const uint64_t dim       = state.range(1);
    const uint64_t max_nodes = state.range(2);

    check(::prepare_branched_sig(dim, max_nodes, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(dim, max_nodes, false);

    auto bsig1 = random_data(batch * blen, 1);
    auto bsig2 = random_data(batch * blen, 2);
    std::vector<double> out(batch * blen);

    for (auto _ : state) {
        ::branched_sig_combine_d(bsig1.data(), bsig2.data(), out.data(),
                                 batch, dim, max_nodes);
        benchmark::DoNotOptimize(out.data());
    }
}
BENCHMARK(BM_branched_sig_combine)->Args({4096, 3, 4})->Unit(benchmark::kMillisecond);

static void BM_branched_sig_combine_backprop(benchmark::State& state) {
    const uint64_t batch     = state.range(0);
    const uint64_t dim       = state.range(1);
    const uint64_t max_nodes = state.range(2);

    check(::prepare_branched_sig(dim, max_nodes, false, false), "prepare_branched_sig");
    const uint64_t blen = ::branched_sig_length(dim, max_nodes, false);

    auto bsig1 = random_data(batch * blen, 1);
    auto bsig2 = random_data(batch * blen, 2);
    auto derivs = random_data(batch * blen, 3);
    std::vector<double> out1(batch * blen);
    std::vector<double> out2(batch * blen);

    for (auto _ : state) {
        ::branched_sig_combine_backprop_d(bsig1.data(), bsig2.data(),
                                          derivs.data(), out1.data(),
                                          out2.data(), batch, dim, max_nodes);
        benchmark::DoNotOptimize(out1.data());
    }
}
BENCHMARK(BM_branched_sig_combine_backprop)->Args({4096, 3, 4})->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
