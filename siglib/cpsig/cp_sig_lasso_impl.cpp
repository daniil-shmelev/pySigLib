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
#include "cp_simd_config.h"
#include "sig_lasso_core.h"
#include "sig_lasso_math.h"
#include "multithreading.h"
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_sig_lasso_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

class LassoCPU final : public siglasso::Backend {
    siglasso::Array paths_, responses_;
public:
    LassoCPU(const LassoCPU& source, const SigLassoOptions& options) : Backend(options, source.budget), paths_(source.paths_), responses_(source.responses_) {
        weights = source.weights;
    }
    template<typename T>
    LassoCPU(const SigLassoOptions& o, const T* paths, const T* y, const double* w) : Backend(o) {
        weights.assign(w, w + o.degree + 1);
        paths_ = alloc(siglasso::multiply(siglasso::multiply(o.samples, o.length), o.dimension));
        responses_ = alloc(siglasso::multiply(o.samples, o.targets));
        for (size_t j = 0; j < paths_.size; ++j) paths_.data[j] = siglasso::finite(paths[j]);
        for (size_t j = 0; j < responses_.size; ++j) responses_.data[j] = siglasso::finite(y[j]);
    }
    siglasso::Array alloc(size_t size) override {
        const size_t bytes = siglasso::multiply(size, sizeof(double));
        if (bytes > budget->available()) trim_caches();
        budget->acquire(bytes);
        double* data;
        try { data = new double[size]; } catch (...) { budget->release(bytes); throw; }
        return {data, size, std::shared_ptr<void>(data, [account = budget, bytes](void* p) { delete[] static_cast<double*>(p); account->release(bytes); })};
    }
    void fill(const siglasso::Array& a, double value) override { std::fill_n(a.data, a.size, value); }
    void copy(const double* from, double* to, size_t n) override { if (n) std::memcpy(to, from, n * sizeof(double)); }
    void upload(const double* from, double* to, size_t n) override { copy(from, to, n); }
    void download(const double* from, double* to, size_t n) override { copy(from, to, n); }
    double dot(const double* a, const double* b, size_t n) override {
        const hn::ScalableTag<double> d;
        const auto lanes = hn::Lanes(d);
        auto sum = hn::Zero(d);
        size_t j = 0;
        for (; j + lanes <= n; j += lanes) sum = hn::Add(sum, hn::Mul(hn::LoadU(d, a + j), hn::LoadU(d, b + j)));
        double result = hn::ReduceSum(d, sum);
        for (; j < n; ++j) result += a[j] * b[j];
        return siglasso::finite(result);
    }
    void axpy(double alpha, const double* a, double* b, size_t n) override {
        const hn::ScalableTag<double> d;
        const auto lanes = hn::Lanes(d);
        const auto factor = hn::Set(d, alpha);
        size_t j = 0;
        for (; j + lanes <= n; j += lanes) hn::StoreU(hn::Add(hn::LoadU(d, b + j), hn::Mul(factor, hn::LoadU(d, a + j))), d, b + j);
        for (; j < n; ++j) b[j] += alpha * a[j];
    }
    double absolute_sum(const siglasso::Array& values) override {
        double result = 0; for (size_t j = 0; j < values.size; ++j) result += std::abs(values.data[j]);
        return siglasso::finite(result);
    }
    void scale(double value, double* data, size_t n) override { for (size_t j = 0; j < n; ++j) data[j] *= value; }
    double center(double* data, size_t n, bool enabled) override {
        if (!enabled || !n) return 0;
        const double mean = siglasso::finite(std::accumulate(data, data + n, 0.0) / n);
        for (size_t j = 0; j < n; ++j) data[j] -= mean;
        return mean;
    }
    void round(double* data, size_t n) override { for (size_t j = 0; j < n; ++j) data[j] = static_cast<float>(data[j]); }
    void correlations(const std::vector<siglasso::Column>& columns, const siglasso::Array& v, std::vector<double>& out, const std::vector<size_t>* selected = nullptr) override {
        out.resize(selected ? selected->size() : columns.size());
        auto compute = [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) out[j] = dot(columns[selected ? (*selected)[j] : j].values.data, v.data, v.size) / v.size;
        };
        if (options.n_jobs == 1 || out.size() <= 1) compute(0, out.size());
        else spawn_batch_threads(out.size(), options.n_jobs, compute);
    }
    void sweeps(const std::vector<siglasso::Column>& columns, const std::vector<size_t>& selected,
        siglasso::Array& beta, siglasso::Array& residual, double alpha, size_t count, std::array<siglasso::Array, 6>* history, size_t& next_snapshot) override {
        for (size_t epoch = 0; epoch < count; ++epoch) {
            for (auto j : selected) {
                const auto& column = columns[j];
                if (!column.norm2) continue;
                const double next = siglasso::soft(beta.data[j] + dot(column.values.data, residual.data, residual.size) * column.inverse_norm2,
                    alpha * residual.size * column.inverse_norm2);
                const double change = next - beta.data[j];
                if (change) { beta.data[j] = next; axpy(-change, column.values.data, residual.data, residual.size); }
            }
            if (history && epoch + 6 >= count) { copy(residual.data, (*history)[next_snapshot].data, residual.size); next_snapshot = (next_snapshot + 1) % 6; }
        }
    }
    void prepare() override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        increments = alloc(siglasso::multiply(siglasso::multiply(n, l), d));
        variation = alloc(n * l); after = alloc(n * l);
        envelopes = alloc(siglasso::multiply(n * l, m * (m + 1) / 2)); fill(envelopes, 0);
        spawn_batch_threads(n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
                    increments.data[(j * d + a) * n + i] = paths_.data[(i * (l + 1) + j + 1) * d + a] - paths_.data[(i * (l + 1) + j) * d + a];
                siglasso::prepare_sample(increments.data, variation.data, after.data, envelopes.data, i, n, l, d, m);
            }
        });
        prepare_alphabet();
        spawn_batch_threads(n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t block = alphabet.size(); block-- > 0;) {
                const auto [low, high] = alphabet[block];
                const size_t right = block + 2 * ((low + high) / 2 - low);
                for (size_t j = 0; j < l; ++j) for (size_t i = begin; i < end; ++i) {
                    const size_t index = j * n + i;
                    alphabet_ratios.data[block * n * l + index] = high - low == 1
                        ? (variation.data[index] ? std::abs(increments.data[(j * d + low) * n + i]) / variation.data[index] : 0)
                        : std::max(alphabet_ratios.data[(block + 1) * n * l + index], alphabet_ratios.data[right * n * l + index]);
                }
            }
        });
        paths_ = {};
    }
    siglasso::Array factors(size_t k) override {
        const size_t n = options.samples, l = options.length - 1, m = options.degree;
        auto result = alloc(siglasso::multiply((m - k) * (k + 1), n * l));
        const size_t constant_count = siglasso::multiply(siglasso::multiply(m - k, m - k + 1), k + 1);
        auto host_constants = account(siglasso::multiply(constant_count, sizeof(double)));
        auto constants = from_host(siglasso::integration_coefficients(k, m - k));
        spawn_batch_threads(n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q <= m - k; ++q) for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j)
                for (size_t i = begin; i < end; ++i)
                    result.data[(((q - 1) * (k + 1) + a) * l + j) * n + i] = siglasso::finite(
                        siglasso::factor_value(variation.data, after.data, envelopes.data, constants.data, k, q, a, j, i, n, l) * weights[k + q] / n);
        });
        return result;
    }
    void moments(siglasso::Node& node, const siglasso::Array& factors) override {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), levels = options.degree - k;
        node.moments = alloc(levels * n * l); node.integrals = alloc(n * l);
        spawn_batch_threads(n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            const hn::ScalableTag<double> tag;
            const size_t lanes = (options.solver_flags & 32) ? n + 1 : hn::Lanes(tag);
            for (size_t j = 0; j < l; ++j) {
                size_t i = begin;
                for (; i + lanes <= end; i += lanes) {
                    auto integral = hn::Zero(tag);
                    for (size_t a = 0; a <= k; ++a) integral = hn::Add(integral, hn::LoadU(tag, node.coefficients.data + (j * (k + 1) + a) * n + i));
                    hn::StoreU(hn::Mul(integral, hn::Set(tag, 1.0 / (k + 1))), tag, node.integrals.data + j * n + i);
                    for (size_t q = 0; q < levels; ++q) {
                        auto value = hn::Zero(tag);
                        for (size_t a = 0; a <= k; ++a) value = hn::Add(value, hn::Mul(
                            hn::Abs(hn::LoadU(tag, node.coefficients.data + (j * (k + 1) + a) * n + i)),
                            hn::LoadU(tag, factors.data + ((q * (k + 1) + a) * l + j) * n + i)));
                        hn::StoreU(value, tag, node.moments.data + (q * l + j) * n + i);
                    }
                }
                for (; i < end; ++i) {
                    double integral = 0;
                    for (size_t a = 0; a <= k; ++a) integral += node.coefficients.data[(j * (k + 1) + a) * n + i];
                    node.integrals.data[j * n + i] = integral / (k + 1);
                    for (size_t q = 0; q < levels; ++q) node.moments.data[(q * l + j) * n + i] =
                        siglasso::moment_value(node.coefficients.data, factors.data, k, q, j, i, n, l);
                }
            }
        });
    }
    siglasso::Array energy_gram(size_t low, size_t high) override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        auto result = alloc(siglasso::multiply(n * l, l));
        spawn_batch_threads(n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t j = 0; j < l; ++j) for (size_t h = 0; h <= j; ++h) for (size_t i = begin; i < end; ++i) {
                double value = 0;
                for (size_t a = low; a < high; ++a) value += increments.data[(j * d + a) * n + i] * increments.data[(h * d + a) * n + i];
                result.data[(j * l + h) * n + i] = siglasso::finite(value);
            }
        });
        return result;
    }
    void bounds(const std::vector<siglasso::Probe>& tasks, const siglasso::Array& residual, double threshold, std::vector<double>& out) override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        out.assign(tasks.size(), 0);
        spawn_batch_threads(tasks.size(), options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t t = begin; t < end; ++t) {
                const auto& task = tasks[t];
                const size_t levels = options.degree - task.node->word.size();
                double best = 0;
                for (size_t q = levels; q-- > 0;) {
                    double total = 0;
                    const hn::ScalableTag<double> tag;
                    const size_t lanes = (options.solver_flags & 32) ? n + 1 : hn::Lanes(tag);
                    auto accumulator = hn::Zero(tag);
                    for (size_t j = 0; j < l; ++j) {
                        size_t i = 0;
                        for (; i + lanes <= n; i += lanes) accumulator = hn::Add(accumulator, hn::Mul(
                            hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::LoadU(tag, task.node->moments.data + (q * l + j) * n + i)),
                            hn::LoadU(tag, alphabet_ratios.data + (task.block * l + j) * n + i)));
                        for (; i < n; ++i) total += std::abs(residual.data[i]) * task.node->moments.data[(q * l + j) * n + i]
                            * alphabet_ratios.data[(task.block * l + j) * n + i];
                    }
                    total = siglasso::finite(total + hn::ReduceSum(tag, accumulator));
                    if (q == 0 && total >= threshold && task.high - task.low > 1) {
                        double energy = 0;
                        for (size_t i = 0; i < n; ++i) energy += std::abs(residual.data[i]) * siglasso::energy_sample(
                            task.node->integrals.data, task.gram.data, increments.data, task.low, task.high, i, n, l, d);
                        total = std::min(total, siglasso::finite(energy * weights[task.node->word.size() + 1] / n));
                    }
                    best = std::max(best, siglasso::finite(total));
                    if (best >= threshold) break;
                }
                out[t] = best;
            }
        });
    }
    void children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        const size_t n = options.samples, l = options.length - 1;
        out.resize(tasks.size());
        for (size_t t = 0; t < tasks.size(); ++t) {
            auto node = std::make_shared<siglasso::Node>(); node->word = tasks[t].node->word; node->word.push_back(tasks[t].low);
            node->metadata = account(512 + node->word.size() * 64);
            node->values = alloc(n);
            if (node->word.size() < options.degree) node->coefficients = alloc((node->word.size() + 1) * n * l);
            out[t] = std::move(node);
        }
        spawn_batch_threads(tasks.size() * n, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t index = begin; index < end; ++index) {
                const size_t t = index / n, i = index % n;
                siglasso::child_sample(increments.data, tasks[t].node->coefficients.data, tasks[t].node->integrals.data, out[t]->coefficients.data, out[t]->values.data,
                    tasks[t].low, tasks[t].node->word.size(), i, n, l, options.dimension);
                siglasso::finite(out[t]->values.data[i]);
            }
        });
    }
    void full_columns(std::vector<siglasso::Column>& columns) override {
        const size_t n = options.samples, d = options.dimension, m = options.degree;
        size_t count = 0, level = 1;
        for (size_t k = 1; k <= m; ++k) {
            level = siglasso::multiply(level, d);
            if (level > std::numeric_limits<size_t>::max() - count) throw std::out_of_range("Full signature size overflow");
            count += level;
        }
        auto column_metadata = account(siglasso::multiply(count, 1024 + m * 64 + options.targets * 32));
        auto matrix = alloc(siglasso::multiply(count, n));
        columns.reserve(count);
        for (size_t k = 1; k <= m; ++k) {
            siglasso::Word word(k, 0);
            do {
                siglasso::Array values{matrix.data + columns.size() * n, n, matrix.owner};
                columns.push_back({word, values, 0, 0, 0, column_metadata});
                size_t j = k;
                while (j && ++word[j - 1] == d) { word[j - 1] = 0; --j; }
                if (!j) break;
            } while (true);
        }
        const size_t tile = std::min<size_t>(n, std::max<size_t>(1, (8 * 1024 * 1024) / std::max<size_t>(1, count * sizeof(double))));
        auto temporary = alloc(tile * count);
        auto kernel_workspace = account(siglasso::multiply(32 * (count + 1) + 8 * d * m, tile * sizeof(double)));
        for (size_t begin = 0; begin < n; begin += tile) {
            const size_t rows = std::min(tile, n - begin);
            const int status = ::signature_d(paths_.data + begin * options.length * d, temporary.data, rows, d, options.length, m,
                false, false, 1.0, true, false, options.n_jobs);
            if (status == 1) throw std::bad_alloc();
            if (status) throw std::runtime_error("Full signature kernel failed in sig_lasso");
            for (size_t j = 0; j < count; ++j) for (size_t i = 0; i < rows; ++i)
                columns[j].values.data[begin + i] = siglasso::finite(temporary.data[i * count + j] * weights[columns[j].word.size()]);
        }
        spawn_batch_threads(count, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) {
                columns[j].mean = center(columns[j].values.data, n, options.intercept != 0);
                columns[j].norm2 = dot(columns[j].values.data, columns[j].values.data, n);
                columns[j].inverse_norm2 = columns[j].norm2 ? 1.0 / columns[j].norm2 : 0;
            }
        });
        stats.evaluated = stats.constructed = count;
    }
    siglasso::Array response(size_t target) override {
        auto result = alloc(options.samples);
        for (size_t i = 0; i < options.samples; ++i) result.data[i] = responses_.data[i * options.targets + target];
        return result;
    }
    void for_targets(const std::function<void(Backend&, size_t)>& action) override {
        if (!options.full || options.targets == 1 || options.n_jobs == 1) { Backend::for_targets(action); return; }
        auto started = siglasso::Clock::now();
        std::vector<SigLassoStats> counters(options.targets);
        spawn_batch_threads(options.targets, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            auto single = options; single.n_jobs = 1;
            LassoCPU worker(*this, single);
            for (size_t j = begin; j < end; ++j) { worker.stats = {}; action(worker, j); counters[j] = worker.stats; }
        });
        for (const auto& counter : counters) { stats.coordinate_sweeps += counter.coordinate_sweeps; stats.screening_count += counter.screening_count; }
        stats.solve_seconds = siglasso::elapsed(started);
    }
};

template<typename T>
void sig_lasso_(const T* path, const T* y, const double* weights, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, siglasso::Result* result) {
    LassoCPU backend(*options, path, y, weights);
    std::vector<siglasso::Word> initial;
    for (size_t j = 0; j < words; ++j) initial.emplace_back(letters + offsets[j], letters + offsets[j + 1]);
    *result = siglasso::fit(backend, std::vector<double>(alphas, alphas + options->targets), initial);
}
template void sig_lasso_<float>(const float*, const float*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, const SigLassoOptions*, siglasso::Result*);
template void sig_lasso_<double>(const double*, const double*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, const SigLassoOptions*, siglasso::Result*);

} } // namespace cpsig::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
