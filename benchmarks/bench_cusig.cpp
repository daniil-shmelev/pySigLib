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
#include <cuda_runtime.h>
#include "cpsig.h"
#include "cusig.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(int status) {
    if (status != 0) {
        std::fprintf(stderr, "CUDA operation returned %d: %s\n", status, cusig_last_error_message());
        std::exit(1);
    }
}

static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(uint64_t size, unsigned seed = 42) : size_(size) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-0.01, 0.01);
        std::vector<T> host(size);
        for (auto& value : host) value = static_cast<T>(dist(rng));
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
        check_cuda(cudaMemcpy(data_, host.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }

    explicit device_buffer(const std::vector<T>& host) : size_(host.size()) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
        check_cuda(cudaMemcpy(data_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    ~device_buffer() { cudaFree(data_); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    T* data() { return data_; }

    void check_finite() const {
        std::vector<T> host(size_);
        check_cuda(cudaMemcpy(host.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        for (T value : host) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "CUDA benchmark produced a non-finite result\n");
                std::exit(1);
            }
        }
    }

private:
    uint64_t size_;
    T* data_ = nullptr;
};

template <typename Function>
static void measure(benchmark::State& state, Function operation) {
    for (int i = 0; i < 3; ++i) check(operation());
    check_cuda(cudaDeviceSynchronize());
    for (auto _ : state) {
        check(operation());
        check_cuda(cudaDeviceSynchronize());
    }
}

static std::vector<uint64_t> branched_coef_tree_data() {
    constexpr uint64_t num_trees = 32, dimension = 3, degree = 4;
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

template <typename Function>
static void measure_preparation(benchmark::State& state, Function operation) {
    for (auto _ : state) {
        state.PauseTiming();
        check_cuda(cudaDeviceSynchronize());
        check(::clear_cache_cuda(false));
        if (::clear_cache(false) != 0) {
            std::fprintf(stderr, "Host cache reset failed\n");
            std::exit(1);
        }
        state.ResumeTiming();
        check(operation());
        check_cuda(cudaDeviceSynchronize());
    }
}

static void BM_prepare_log_sig_cuda(benchmark::State& state) {
    const int method = static_cast<int>(state.range(0));
    measure_preparation(state, [&]() { return ::prepare_log_sig_cuda(3, 4, method, false); });
}
BENCHMARK(BM_prepare_log_sig_cuda)->Name("CUDA/prepare_log_sig")
    ->Arg(1)->Arg(2)->Arg(3)->ArgName("method")->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_sig_cuda(benchmark::State& state) {
    const bool planar = state.range(0);
    measure_preparation(state, [&]() {
        return ::prepare_branched_sig_cuda(planar ? 3 : 4, planar ? 4 : 5, planar, false);
    });
}
BENCHMARK(BM_prepare_branched_sig_cuda)->Name("CUDA/prepare_branched_sig")
    ->Arg(0)->Arg(1)->ArgName("planar")->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_log_sig_cuda(benchmark::State& state) {
    const bool planar = state.range(0);
    const int method = static_cast<int>(state.range(1));
    measure_preparation(state, [&]() { return ::prepare_branched_log_sig_cuda(3, 4, method, planar, false); });
}
BENCHMARK(BM_prepare_branched_log_sig_cuda)->Name("CUDA/prepare_branched_log_sig")
    ->Args({0, 0})->Args({1, 0})->Args({1, 1})->Args({1, 2})->Args({1, 3})
    ->ArgNames({"planar", "method"})->UseRealTime()->Unit(benchmark::kMicrosecond);

static void BM_prepare_branched_sig_coef_cuda(benchmark::State& state) {
    const bool planar = state.range(0);
    const auto tree_data = branched_coef_tree_data();
    measure_preparation(state, [&]() {
        return ::prepare_branched_sig_coef_cuda(tree_data.data(), tree_data.size(), 3, 3, 4, planar, false);
    });
}
BENCHMARK(BM_prepare_branched_sig_coef_cuda)->Name("CUDA/prepare_branched_sig_coef")
    ->Arg(0)->Arg(1)->ArgName("planar")->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_transform_path_cuda(benchmark::State& state) {
    const bool time_aug = state.range(0), lead_lag = state.range(1), backprop = state.range(2);
    constexpr uint64_t batch = 16, dimension = 8, length = 512;
    const uint64_t transformed_length = lead_lag ? 2 * length - 1 : length;
    const uint64_t transformed_dimension = dimension * (lead_lag ? 2 : 1) + time_aug;
    device_buffer<T> path(batch * length * dimension, 1);
    device_buffer<T> transformed(batch * transformed_length * transformed_dimension, 2);
    if (backprop) {
        measure(state, [&]() {
            return Backward(transformed.data(), path.data(), batch, dimension, length, time_aug, lead_lag, T(1));
        });
        path.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(path.data(), transformed.data(), batch, dimension, length, time_aug, lead_lag, T(1));
        });
        transformed.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_transform_path_cuda, float, transform_path_cuda_f, transform_path_backprop_cuda_f)
    ->Name("CUDA/transform_path/float32")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"time_aug", "lead_lag", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_transform_path_cuda, double, transform_path_cuda_d, transform_path_backprop_cuda_d)
    ->Name("CUDA/transform_path/float64")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"time_aug", "lead_lag", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_signature_cuda(benchmark::State& state) {
    const uint64_t batch = state.range(0), length = state.range(1);
    const uint64_t dimension = state.range(2), degree = state.range(3);
    const bool time_aug = state.range(4), lead_lag = state.range(5);
    const bool scalar_term = state.range(6), backprop = state.range(7);
    const bool horner = state.range(8), corrected = state.range(9);
    const uint64_t transformed_dimension = dimension * (lead_lag ? 2 : 1) + time_aug;
    const uint64_t slen = ::sig_length(transformed_dimension, degree) - !scalar_term;
    device_buffer<T> path(batch * length * dimension, 1), sig(batch * slen);
    device_buffer<T> derivs(batch * slen, 2), out(batch * length * dimension);
    device_buffer<T> correction(transformed_dimension * transformed_dimension, 3);
    const uint64_t correction_len = corrected ? transformed_dimension * transformed_dimension : 0;
    auto forward = [&]() {
        return Forward(path.data(), sig.data(), batch, dimension, length, degree,
                       time_aug, lead_lag, T(1), horner, scalar_term,
                       corrected ? correction.data() : nullptr, correction_len, 0, 0);
    };
    check(forward());
    if (backprop) {
        measure(state, [&]() {
            return Backward(path.data(), out.data(), derivs.data(), sig.data(),
                            batch, dimension, length, degree, time_aug, lead_lag,
                            T(1), scalar_term, corrected ? correction.data() : nullptr, correction_len, 0, 0);
        });
        out.check_finite();
    } else {
        measure(state, forward);
        sig.check_finite();
    }
}

static void signature_args(benchmark::internal::Benchmark* bench) {
    for (int backprop : {0, 1}) {
        bench->Args({64, 128, 3, 4, 0, 0, 1, backprop, 1, 0});
        bench->Args({256, 32, 4, 3, 0, 0, 0, backprop, 1, 0});
        bench->Args({32, 64, 2, 3, 1, 0, 1, backprop, 1, 0});
        bench->Args({32, 64, 2, 3, 0, 1, 1, backprop, 1, 0});
        bench->Args({4, 32, 4, 5, 0, 0, 1, backprop, 1, 0});
        bench->Args({4, 32, 3, 4, 0, 0, 1, backprop, 1, 1});
    }
    bench->Args({4, 32, 4, 5, 0, 0, 1, 0, 0, 0});
    bench->ArgNames({"batch", "length", "dim", "degree", "time_aug", "lead_lag", "scalar", "backprop", "horner", "correction"});
    bench->UseRealTime()->Unit(benchmark::kMicrosecond);
}
BENCHMARK_TEMPLATE(BM_signature_cuda, float, signature_cuda_f, sig_backprop_cuda_f)
    ->Name("CUDA/signature/float32")->Apply(signature_args);
BENCHMARK_TEMPLATE(BM_signature_cuda, double, signature_cuda_d, sig_backprop_cuda_d)
    ->Name("CUDA/signature/float64")->Apply(signature_args);

template <typename T, auto Forward, auto Backward>
static void BM_sig_combine_cuda(benchmark::State& state) {
    const uint64_t batch = state.range(0);
    const bool scalar_term = state.range(1), backprop = state.range(2);
    const uint64_t slen = ::sig_length(4, 4) - !scalar_term;
    device_buffer<T> sig1(batch * slen, 1), sig2(batch * slen, 2);
    device_buffer<T> derivs(batch * slen, 3), out1(batch * slen), out2(batch * slen);
    if (backprop) {
        measure(state, [&]() {
            return Backward(derivs.data(), out1.data(), out2.data(), sig1.data(),
                            sig2.data(), batch, 4, 4, scalar_term);
        });
        out2.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(sig1.data(), sig2.data(), out1.data(), batch, 4, 4, scalar_term);
        });
    }
    out1.check_finite();
}
BENCHMARK_TEMPLATE(BM_sig_combine_cuda, float, sig_combine_cuda_f, sig_combine_backprop_cuda_f)
    ->Name("CUDA/sig_combine/float32")->ArgsProduct({{64, 1024}, {0, 1}, {0, 1}})
    ->ArgNames({"batch", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_combine_cuda, double, sig_combine_cuda_d, sig_combine_backprop_cuda_d)
    ->Name("CUDA/sig_combine/float64")->ArgsProduct({{64, 1024}, {0, 1}, {0, 1}})
    ->ArgNames({"batch", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward>
static void BM_linear_sig_cuda(benchmark::State& state) {
    const bool scalar_term = state.range(0);
    constexpr uint64_t batch = 16, dimension = 4, degree = 5;
    const uint64_t slen = ::sig_length(dimension, degree) - !scalar_term;
    device_buffer<T> displacement(batch * dimension, 1), out(batch * slen);
    measure(state, [&]() {
        return Forward(displacement.data(), out.data(), batch, dimension, degree, scalar_term);
    });
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_linear_sig_cuda, float, linear_sig_cuda_f)
    ->Name("CUDA/linear_sig/float32")->Arg(0)->Arg(1)
    ->ArgName("scalar")->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_linear_sig_cuda, double, linear_sig_cuda_d)
    ->Name("CUDA/linear_sig/float64")->Arg(0)->Arg(1)
    ->ArgName("scalar")->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_sig_join_cuda(benchmark::State& state) {
    const bool prepend = state.range(0), scalar_term = state.range(1), backprop = state.range(2);
    constexpr uint64_t batch = 8, dimension = 4, degree = 5;
    const uint64_t slen = ::sig_length(dimension, degree) - !scalar_term;
    device_buffer<T> sig(batch * slen, 1), displacement(batch * dimension, 2);
    device_buffer<T> derivs(batch * slen, 3), out(batch * slen), d_displacement(batch * dimension);
    if (backprop) {
        measure(state, [&]() {
            return Backward(derivs.data(), out.data(), d_displacement.data(), sig.data(),
                            displacement.data(), batch, dimension, degree, prepend, scalar_term);
        });
        d_displacement.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(sig.data(), displacement.data(), out.data(), batch, dimension, degree, prepend, scalar_term);
        });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_sig_join_cuda, float, sig_join_cuda_f, sig_join_backprop_cuda_f)
    ->Name("CUDA/sig_join/float32")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"prepend", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_join_cuda, double, sig_join_cuda_d, sig_join_backprop_cuda_d)
    ->Name("CUDA/sig_join/float64")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"prepend", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_sig_coef_cuda(benchmark::State& state) {
    const bool prefixes = state.range(0), backprop = state.range(1);
    constexpr uint64_t batch = 8, dimension = 3, length = 32, count = 32, degree = 4;
    device_buffer<uint64_t> degrees(std::vector<uint64_t>(count, degree));
    device_buffer<uint64_t> multi_idx(std::vector<uint64_t>(count * degree, 0));
    const uint64_t out_len = count * (prefixes ? degree : 1);
    device_buffer<T> path(batch * length * dimension, 1), coefs(batch * out_len);
    device_buffer<T> derivs(batch * out_len, 2), out(batch * length * dimension);
    auto forward = [&]() {
        return Forward(path.data(), coefs.data(), multi_idx.data(), count,
                       degrees.data(), batch, dimension, length, prefixes);
    };
    if (backprop) {
        check(forward());
        measure(state, [&]() {
            return Backward(path.data(), out.data(), coefs.data(), derivs.data(),
                            multi_idx.data(), count, degrees.data(), batch, dimension, length);
        });
        out.check_finite();
    } else {
        measure(state, forward);
        coefs.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_sig_coef_cuda, float, sig_coef_cuda_f, sig_coef_backprop_cuda_f)
    ->Name("CUDA/sig_coef/float32")->Args({0, 0})->Args({1, 0})->Args({1, 1})
    ->ArgNames({"prefixes", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_coef_cuda, double, sig_coef_cuda_d, sig_coef_backprop_cuda_d)
    ->Name("CUDA/sig_coef/float64")->Args({0, 0})->Args({1, 0})->Args({1, 1})
    ->ArgNames({"prefixes", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Signature, auto Forward, auto Backward>
static void BM_sig_to_log_sig_cuda(benchmark::State& state) {
    const int method = static_cast<int>(state.range(0));
    const bool backprop = state.range(1);
    constexpr uint64_t batch = 64, dimension = 3, degree = 4, length = 32;
    if (method != 0) check(::prepare_log_sig_cuda(dimension, degree, method, false));
    const uint64_t slen = ::sig_length(dimension, degree);
    const uint64_t llen = method ? ::log_sig_length(dimension, degree) : slen;
    device_buffer<T> path(batch * length * dimension, 1), sig(batch * slen);
    device_buffer<T> logsig(batch * llen), derivs(batch * llen, 2), out(batch * slen);
    check(Signature(path.data(), sig.data(), batch, dimension, length, degree,
                    false, false, T(1), true, true, nullptr, 0, 0, 0));
    if (backprop) {
        measure(state, [&]() {
            return Backward(sig.data(), out.data(), derivs.data(), batch, dimension, degree, method, true);
        });
        out.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(sig.data(), logsig.data(), batch, dimension, degree, method, true);
        });
        logsig.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_sig_to_log_sig_cuda, float, signature_cuda_f, sig_to_log_sig_cuda_f, sig_to_log_sig_backprop_cuda_f)
    ->Name("CUDA/sig_to_log_sig/float32")->ArgsProduct({{0, 1, 2}, {0, 1}})
    ->ArgNames({"method", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_to_log_sig_cuda, double, signature_cuda_d, sig_to_log_sig_cuda_d, sig_to_log_sig_backprop_cuda_d)
    ->Name("CUDA/sig_to_log_sig/float64")->ArgsProduct({{0, 1, 2}, {0, 1}})
    ->ArgNames({"method", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Signature, auto LogSignature, auto Forward, auto Backward>
static void BM_logsig_to_sig_cuda(benchmark::State& state) {
    const int method = static_cast<int>(state.range(0));
    const bool scalar_term = state.range(1), backprop = state.range(2);
    constexpr uint64_t batch = 8, dimension = 4, degree = 5, length = 32;
    if (method != 0) check(::prepare_log_sig_cuda(dimension, degree, method, false));
    const uint64_t slen = ::sig_length(dimension, degree) - !scalar_term;
    const uint64_t llen = method ? ::log_sig_length(dimension, degree) : slen;
    device_buffer<T> path(batch * length * dimension, 1), sig(batch * slen), logsig(batch * llen);
    device_buffer<T> derivs(batch * slen, 2), out(backprop ? batch * llen : batch * slen);
    check(Signature(path.data(), sig.data(), batch, dimension, length, degree,
                    false, false, T(1), true, scalar_term, nullptr, 0, 0, 0));
    check(LogSignature(sig.data(), logsig.data(), batch, dimension, degree, method, scalar_term));
    if (backprop) {
        measure(state, [&]() {
            return Backward(logsig.data(), out.data(), derivs.data(), batch, dimension, degree, method, scalar_term);
        });
    } else {
        measure(state, [&]() {
            return Forward(logsig.data(), out.data(), batch, dimension, degree, method, scalar_term);
        });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_logsig_to_sig_cuda, float, signature_cuda_f, sig_to_log_sig_cuda_f, logsig_to_sig_cuda_f, logsig_to_sig_backprop_cuda_f)
    ->Name("CUDA/logsig_to_sig/float32")->ArgsProduct({{0, 1, 2}, {0, 1}, {0, 1}})
    ->ArgNames({"method", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_logsig_to_sig_cuda, double, signature_cuda_d, sig_to_log_sig_cuda_d, logsig_to_sig_cuda_d, logsig_to_sig_backprop_cuda_d)
    ->Name("CUDA/logsig_to_sig/float64")->ArgsProduct({{0, 1, 2}, {0, 1}, {0, 1}})
    ->ArgNames({"method", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_log_sig_from_path_cuda(benchmark::State& state) {
    const uint64_t batch = state.range(0), length = state.range(1);
    const uint64_t dimension = state.range(2), degree = state.range(3);
    const bool backprop = state.range(4);
    check(::prepare_log_sig_cuda(dimension, degree, 3, false));
    const uint64_t llen = ::log_sig_length(dimension, degree);
    device_buffer<T> path(batch * length * dimension, 1), derivs(batch * llen, 2);
    device_buffer<T> out(backprop ? batch * length * dimension : batch * llen);
    if (backprop) {
        measure(state, [&]() {
            return Backward(derivs.data(), out.data(), path.data(), batch, length, dimension, degree);
        });
    } else {
        measure(state, [&]() {
            return Forward(path.data(), out.data(), batch, length, dimension, degree);
        });
    }
    out.check_finite();
}
static void log_sig_from_path_args(benchmark::internal::Benchmark* bench) {
    for (int backprop : {0, 1}) {
        bench->Args({4, 32, 3, 4, backprop});
        bench->Args({4, 32, 3, 5, backprop});
        bench->Args({32, 129, 2, 6, backprop});
        bench->Args({1, 129, 2, 8, backprop});
    }
    bench->ArgNames({"batch", "length", "dim", "degree", "backprop"});
    bench->UseRealTime()->Unit(benchmark::kMicrosecond);
}
BENCHMARK_TEMPLATE(BM_log_sig_from_path_cuda, float, log_sig_from_path_cuda_f, log_sig_from_path_backprop_cuda_f)
    ->Name("CUDA/log_sig_from_path/float32")->Apply(log_sig_from_path_args);
BENCHMARK_TEMPLATE(BM_log_sig_from_path_cuda, double, log_sig_from_path_cuda_d, log_sig_from_path_backprop_cuda_d)
    ->Name("CUDA/log_sig_from_path/float64")->Apply(log_sig_from_path_args);

template <typename T, auto Forward, auto Backward>
static void BM_log_sig_combine_cuda(benchmark::State& state) {
    const bool backprop = state.range(0);
    constexpr uint64_t batch = 64, dimension = 3, degree = 5;
    check(::prepare_log_sig_cuda(dimension, degree, 3, false));
    const uint64_t llen = ::log_sig_length(dimension, degree);
    device_buffer<T> ls1(batch * llen, 1), ls2(batch * llen, 2), derivs(batch * llen, 3);
    device_buffer<T> out1(batch * llen), out2(batch * llen);
    if (backprop) {
        measure(state, [&]() {
            return Backward(derivs.data(), out1.data(), out2.data(), ls1.data(), ls2.data(), batch, dimension, degree);
        });
        out2.check_finite();
    } else {
        measure(state, [&]() { return Forward(ls1.data(), ls2.data(), out1.data(), batch, dimension, degree); });
    }
    out1.check_finite();
}
BENCHMARK_TEMPLATE(BM_log_sig_combine_cuda, float, log_sig_combine_cuda_f, log_sig_combine_backprop_cuda_f)
    ->Name("CUDA/log_sig_combine/float32")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_log_sig_combine_cuda, double, log_sig_combine_cuda_d, log_sig_combine_backprop_cuda_d)
    ->Name("CUDA/log_sig_combine/float64")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_log_sig_join_cuda(benchmark::State& state) {
    const bool backprop = state.range(0);
    constexpr uint64_t batch = 64, dimension = 3, degree = 5;
    check(::prepare_log_sig_cuda(dimension, degree, 3, false));
    const uint64_t llen = ::log_sig_length(dimension, degree);
    device_buffer<T> logsig(batch * llen, 1), displacement(batch * dimension, 2), derivs(batch * llen, 3);
    device_buffer<T> out(batch * llen), d_displacement(batch * dimension);
    if (backprop) {
        measure(state, [&]() {
            return Backward(derivs.data(), out.data(), d_displacement.data(), logsig.data(),
                            displacement.data(), batch, dimension, degree);
        });
        d_displacement.check_finite();
    } else {
        measure(state, [&]() { return Forward(logsig.data(), displacement.data(), out.data(), batch, dimension, degree); });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_log_sig_join_cuda, float, log_sig_join_cuda_f, log_sig_join_backprop_cuda_f)
    ->Name("CUDA/log_sig_join/float32")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_log_sig_join_cuda, double, log_sig_join_cuda_d, log_sig_join_backprop_cuda_d)
    ->Name("CUDA/log_sig_join/float64")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_branched_sig_cuda(benchmark::State& state) {
    const bool planar = state.range(0), backprop = state.range(1);
    const bool corrected = state.range(2);
    constexpr uint64_t batch = 32, dimension = 3, degree = 4, length = 32;
    check(::prepare_branched_sig_cuda(dimension, degree, planar, false));
    const uint64_t slen = ::branched_sig_length(dimension, degree, planar);
    device_buffer<T> path(batch * length * dimension, 1), sig(batch * slen);
    device_buffer<T> derivs(batch * slen, 2), out(batch * length * dimension);
    device_buffer<T> correction(dimension * dimension, 3);
    const uint64_t correction_len = corrected ? dimension * dimension : 0;
    auto forward = [&]() {
        return Forward(path.data(), sig.data(), batch, dimension, length, degree,
                       false, false, T(1), planar, true,
                       corrected ? correction.data() : nullptr, correction_len, 0, 0);
    };
    check(forward());
    if (backprop) {
        measure(state, [&]() {
            return Backward(path.data(), out.data(), derivs.data(), sig.data(),
                            batch, dimension, length, degree, false, false, T(1),
                            planar, true, corrected ? correction.data() : nullptr, correction_len, 0, 0);
        });
        out.check_finite();
    } else {
        measure(state, forward);
        sig.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_branched_sig_cuda, float, branched_sig_cuda_f, branched_sig_backprop_cuda_f)
    ->Name("CUDA/branched_sig/float32")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"planar", "backprop", "correction"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_branched_sig_cuda, double, branched_sig_cuda_d, branched_sig_backprop_cuda_d)
    ->Name("CUDA/branched_sig/float64")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"planar", "backprop", "correction"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_branched_sig_coef_cuda(benchmark::State& state) {
    const bool planar = state.range(0), backprop = state.range(1);
    constexpr uint64_t batch = 8, dimension = 3, length = 32, degree = 4;
    const auto tree_data = branched_coef_tree_data();
    check(::prepare_branched_sig_coef_cuda(tree_data.data(), tree_data.size(),
                                          dimension, dimension, degree, planar, false));
    device_buffer<T> path(batch * length * dimension, 1), coefs(batch * tree_data[0]);
    device_buffer<T> derivs(batch * tree_data[0], 2), out(batch * length * dimension);
    auto forward = [&]() {
        return Forward(path.data(), coefs.data(), tree_data.data(), tree_data.size(),
                       batch, dimension, length, degree, false, false, T(1), planar, nullptr, 0, 0, 0);
    };
    if (backprop) {
        check(forward());
        measure(state, [&]() {
            return Backward(path.data(), out.data(), coefs.data(), derivs.data(), tree_data.data(),
                            tree_data.size(), batch, dimension, length, degree,
                            false, false, T(1), planar, nullptr, 0, 0, 0);
        });
        out.check_finite();
    } else {
        measure(state, forward);
        coefs.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_branched_sig_coef_cuda, float, branched_sig_coef_cuda_f, branched_sig_coef_backprop_cuda_f)
    ->Name("CUDA/branched_sig_coef/float32")->ArgsProduct({{0, 1}, {0, 1}})
    ->ArgNames({"planar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_branched_sig_coef_cuda, double, branched_sig_coef_cuda_d, branched_sig_coef_backprop_cuda_d)
    ->Name("CUDA/branched_sig_coef/float64")->ArgsProduct({{0, 1}, {0, 1}})
    ->ArgNames({"planar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Signature, auto Forward, auto Backward>
static void BM_branched_sig_combine_cuda(benchmark::State& state) {
    const bool planar = state.range(0), scalar_term = state.range(1), backprop = state.range(2);
    constexpr uint64_t batch = 64, dimension = 3, degree = 4, length = 32;
    check(::prepare_branched_sig_cuda(dimension, degree, planar, false));
    const uint64_t slen = ::branched_sig_length(dimension, degree, planar) - !scalar_term;
    device_buffer<T> path1(batch * length * dimension, 1), path2(batch * length * dimension, 2);
    device_buffer<T> sig1(batch * slen), sig2(batch * slen), derivs(batch * slen, 3);
    device_buffer<T> out1(batch * slen), out2(batch * slen);
    check(Signature(path1.data(), sig1.data(), batch, dimension, length, degree,
                    false, false, T(1), planar, scalar_term, nullptr, 0, 0, 0));
    check(Signature(path2.data(), sig2.data(), batch, dimension, length, degree,
                    false, false, T(1), planar, scalar_term, nullptr, 0, 0, 0));
    if (backprop) {
        measure(state, [&]() {
            return Backward(sig1.data(), sig2.data(), derivs.data(), out1.data(), out2.data(),
                            batch, dimension, degree, planar, scalar_term);
        });
        out2.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(sig1.data(), sig2.data(), out1.data(), batch, dimension, degree, planar, scalar_term);
        });
    }
    out1.check_finite();
}
BENCHMARK_TEMPLATE(BM_branched_sig_combine_cuda, float, branched_sig_cuda_f, branched_sig_combine_cuda_f, branched_sig_combine_backprop_cuda_f)
    ->Name("CUDA/branched_sig_combine/float32")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"planar", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_branched_sig_combine_cuda, double, branched_sig_cuda_d, branched_sig_combine_cuda_d, branched_sig_combine_backprop_cuda_d)
    ->Name("CUDA/branched_sig_combine/float64")->ArgsProduct({{0, 1}, {0, 1}, {0, 1}})
    ->ArgNames({"planar", "scalar", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Signature, auto Forward, auto Backward>
static void BM_branched_sig_to_log_sig_cuda(benchmark::State& state) {
    const bool planar = state.range(0), backprop = state.range(2);
    const int method = static_cast<int>(state.range(1));
    const uint64_t batch = planar ? 64 : (backprop ? 128 : 256);
    const uint64_t degree = planar ? 4 : 5;
    constexpr uint64_t dimension = 3, length = 32;
    check(::prepare_branched_log_sig_cuda(dimension, degree, method, planar, false));
    check(::prepare_branched_sig_cuda(dimension, degree, planar, false));
    const uint64_t slen = ::branched_sig_length(dimension, degree, planar);
    const uint64_t llen = method ? ::branched_log_sig_length(dimension, degree, planar) : slen;
    device_buffer<T> path(batch * length * dimension, 1), sig(batch * slen);
    device_buffer<T> derivs(batch * llen, 2), out(backprop ? batch * slen : batch * llen);
    check(Signature(path.data(), sig.data(), batch, dimension, length, degree,
                    false, false, T(1), planar, true, nullptr, 0, 0, 0));
    if (backprop) {
        measure(state, [&]() {
            return Backward(sig.data(), derivs.data(), out.data(), batch, dimension, degree, method, planar, true);
        });
    } else {
        measure(state, [&]() { return Forward(sig.data(), out.data(), batch, dimension, degree, method, planar, true); });
    }
    out.check_finite();
}
static void branched_sig_to_log_sig_args(benchmark::internal::Benchmark* bench) {
    for (int backprop : {0, 1}) {
        bench->Args({0, 0, backprop});
        for (int method : {0, 1, 2}) bench->Args({1, method, backprop});
    }
    bench->ArgNames({"planar", "method", "backprop"});
    bench->UseRealTime()->Unit(benchmark::kMicrosecond);
}
BENCHMARK_TEMPLATE(BM_branched_sig_to_log_sig_cuda, float, branched_sig_cuda_f, branched_sig_to_log_sig_cuda_f, branched_sig_to_log_sig_backprop_cuda_f)
    ->Name("CUDA/branched_sig_to_log_sig/float32")->Apply(branched_sig_to_log_sig_args);
BENCHMARK_TEMPLATE(BM_branched_sig_to_log_sig_cuda, double, branched_sig_cuda_d, branched_sig_to_log_sig_cuda_d, branched_sig_to_log_sig_backprop_cuda_d)
    ->Name("CUDA/branched_sig_to_log_sig/float64")->Apply(branched_sig_to_log_sig_args);

template <typename T, auto Forward, auto Backward>
static void BM_branched_log_sig_from_path_cuda(benchmark::State& state) {
    const bool backprop = state.range(0);
    constexpr uint64_t batch = 64, dimension = 3, length = 32, degree = 4;
    check(::prepare_branched_log_sig_cuda(dimension, degree, 3, true, false));
    const uint64_t llen = ::branched_log_sig_length(dimension, degree, true);
    device_buffer<T> path(batch * length * dimension, 1), derivs(batch * llen, 2);
    device_buffer<T> out(backprop ? batch * length * dimension : batch * llen);
    if (backprop) {
        measure(state, [&]() { return Backward(derivs.data(), out.data(), path.data(), batch, length, dimension, degree); });
    } else {
        measure(state, [&]() { return Forward(path.data(), out.data(), batch, length, dimension, degree); });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_branched_log_sig_from_path_cuda, float, branched_log_sig_from_path_cuda_f, branched_log_sig_from_path_backprop_cuda_f)
    ->Name("CUDA/branched_log_sig_from_path/float32")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_branched_log_sig_from_path_cuda, double, branched_log_sig_from_path_cuda_d, branched_log_sig_from_path_backprop_cuda_d)
    ->Name("CUDA/branched_log_sig_from_path/float64")->Arg(0)->Arg(1)
    ->ArgName("backprop")->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_sig_kernel_cuda(benchmark::State& state) {
    const uint64_t refinement = state.range(0);
    const bool backprop = state.range(1);
    constexpr uint64_t batch = 32, length = 64;
    const uint64_t grid_length = ((length - 1) << refinement) + 1;
    device_buffer<T> gram(batch * (length - 1) * (length - 1), 1);
    device_buffer<T> grid(batch * grid_length * grid_length), derivs(batch, 2);
    device_buffer<T> out(backprop ? batch * (length - 1) * (length - 1) : batch);
    if (backprop) {
        check(Forward(gram.data(), grid.data(), batch, 3, length, length, refinement, refinement, true));
        measure(state, [&]() {
            return Backward(gram.data(), out.data(), derivs.data(), grid.data(),
                            batch, 3, length, length, refinement, refinement, false);
        });
    } else {
        measure(state, [&]() {
            return Forward(gram.data(), out.data(), batch, 3, length, length, refinement, refinement, false);
        });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_sig_kernel_cuda, float, sig_kernel_cuda_f, sig_kernel_backprop_cuda_f)
    ->Name("CUDA/sig_kernel/float32")->ArgsProduct({{0, 1}, {0, 1}})
    ->ArgNames({"refinement", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_kernel_cuda, double, sig_kernel_cuda_d, sig_kernel_backprop_cuda_d)
    ->Name("CUDA/sig_kernel/float64")->ArgsProduct({{0, 1}, {0, 1}})
    ->ArgNames({"refinement", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_sig_kernel_poly_cuda(benchmark::State& state) {
    const uint64_t length = state.range(0);
    const bool backprop = state.range(1);
    constexpr uint64_t batch = 4, order = 7;
    const uint64_t cells = batch * (length - 1) * (length - 1);
    device_buffer<T> gram(cells, 1), result(batch), derivs(batch, 2);
    device_buffer<T> tape(2 * cells * (order + 1)), out(cells);
    if (backprop) {
        check(Forward(gram.data(), result.data(), tape.data(), batch, 3, length, length, order, false));
        measure(state, [&]() {
            return Backward(gram.data(), out.data(), derivs.data(), tape.data(),
                            batch, 3, length, length, order, false);
        });
        out.check_finite();
    } else {
        measure(state, [&]() {
            return Forward(gram.data(), result.data(), nullptr, batch, 3, length, length, order, false);
        });
        result.check_finite();
    }
}
BENCHMARK_TEMPLATE(BM_sig_kernel_poly_cuda, float, sig_kernel_poly_cuda_f, sig_kernel_poly_backprop_cuda_f)
    ->Name("CUDA/sig_kernel_poly/float32")->ArgsProduct({{64, 256}, {0, 1}})
    ->ArgNames({"length", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_sig_kernel_poly_cuda, double, sig_kernel_poly_cuda_d, sig_kernel_poly_backprop_cuda_d)
    ->Name("CUDA/sig_kernel_poly/float64")->ArgsProduct({{64, 256}, {0, 1}})
    ->ArgNames({"length", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

template <typename T, auto Forward, auto Backward>
static void BM_branched_sig_kernel_cuda(benchmark::State& state) {
    const uint64_t depth = state.range(0);
    const bool backprop = state.range(1);
    constexpr uint64_t batch = 16, length = 32;
    const uint64_t cells = batch * (length - 1) * (length - 1);
    device_buffer<T> gram(cells, 1), derivs(batch, 2), out(backprop ? cells : batch);
    if (backprop) {
        measure(state, [&]() {
            return Backward(gram.data(), out.data(), derivs.data(), nullptr,
                            batch, 3, length, length, depth, 0, 0, false);
        });
    } else {
        measure(state, [&]() {
            return Forward(gram.data(), out.data(), batch, 3, length, length, depth, 0, 0, false);
        });
    }
    out.check_finite();
}
BENCHMARK_TEMPLATE(BM_branched_sig_kernel_cuda, float, branched_sig_kernel_cuda_f, branched_sig_kernel_backprop_cuda_f)
    ->Name("CUDA/branched_sig_kernel/float32")->ArgsProduct({{2, 3}, {0, 1}})
    ->ArgNames({"depth", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_branched_sig_kernel_cuda, double, branched_sig_kernel_cuda_d, branched_sig_kernel_backprop_cuda_d)
    ->Name("CUDA/branched_sig_kernel/float64")->ArgsProduct({{2, 3}, {0, 1}})
    ->ArgNames({"depth", "backprop"})->UseRealTime()->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "CUDA benchmarks require a GPU\n");
        return 1;
    }
    check_cuda(cudaSetDevice(0));
    cudaDeviceProp device{};
    check_cuda(cudaGetDeviceProperties(&device, 0));
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::AddCustomContext("gpu", device.name);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
