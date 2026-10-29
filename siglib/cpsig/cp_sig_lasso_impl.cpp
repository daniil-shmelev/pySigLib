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

class LassoCPU final : public siglasso::Backend, public siglasso::HostSearch {
    siglasso::Array paths_, responses_;
    std::vector<std::unique_ptr<LassoCPU>> search_workers_;
    std::vector<std::unique_ptr<siglasso::Search>> searches_;
    std::vector<size_t> frontier_;
    LassoCPU* factor_parent_ = nullptr;
    std::mutex factor_mutex_;
    std::mutex terminal_mutex_;
    siglasso::Array terminal_storage_, terminal_products_;
    siglasso::TerminalCache terminal_cache_;
    std::atomic<bool> terminal_needed_{false};
    double terminal_mean_ = 0;
    std::array<double, 2> terminal_caps_{};
    std::vector<siglasso::Array> shared_factors_;
    size_t factor_bytes_ = 0;
    template<typename Function> void batch(size_t count, Function action) {
        if (options.n_jobs == 1 || count <= 1) action(0, count);
        else spawn_batch_threads(count, options.n_jobs, action);
    }
    double energy_bound(const siglasso::Probe& task, const siglasso::Array& residual) {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        auto total = hn::Zero(tag); size_t i = 0;
        for (; i + lanes <= n; i += lanes) {
            auto squared = hn::Zero(tag), magnitude = hn::Zero(tag);
            if (task.gram.owner) {
                for (size_t j = 0; j < l; ++j) for (size_t h = 0; h <= j; ++h) {
                    auto term = hn::Mul(hn::Mul(hn::LoadU(tag, task.node->integrals.data + j * n + i),
                        hn::LoadU(tag, task.gram.data + (j * l + h) * n + i)), hn::LoadU(tag, task.node->integrals.data + h * n + i));
                    if (j != h) term = hn::Add(term, term);
                    squared = hn::Add(squared, term); magnitude = hn::Add(magnitude, hn::Abs(term));
                }
            } else for (size_t a = task.low; a < task.high; ++a) {
                auto value = hn::Zero(tag), absolute = hn::Zero(tag);
                for (size_t j = 0; j < l; ++j) {
                    const auto term = hn::Mul(hn::LoadU(tag, task.node->integrals.data + j * n + i), hn::LoadU(tag, increments.data + (j * d + a) * n + i));
                    value = hn::Add(value, term); absolute = hn::Add(absolute, hn::Abs(term));
                }
                squared = hn::Add(squared, hn::Mul(value, value)); magnitude = hn::Add(magnitude, hn::Mul(absolute, absolute));
            }
            const auto norm = hn::Sqrt(hn::Add(hn::Add(hn::Max(hn::Zero(tag), squared), hn::Mul(hn::Set(tag, 1e-11), magnitude)), hn::Set(tag, 1e-300)));
            total = hn::Add(total, hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), norm));
        }
        double result = hn::ReduceSum(tag, total);
        for (; i < n; ++i) result += std::abs(residual.data[i]) * siglasso::energy_sample(task.node->integrals.data, task.gram.data,
            increments.data, task.low, task.high, i, n, l, d);
        return siglasso::finite(result);
    }
public:
    LassoCPU(const LassoCPU& source, const SigLassoOptions& options) : Backend(options, source.budget), paths_(source.paths_), responses_(source.responses_) {
        weights = source.weights;
    }
    template<typename T>
    LassoCPU(const SigLassoOptions& o, const T* paths, const T* y) : Backend(o) {
        paths_ = alloc(siglasso::multiply(siglasso::multiply(o.samples, o.length), o.dimension));
        responses_ = alloc(siglasso::multiply(o.samples, o.targets));
        for (size_t j = 0; j < paths_.size; ++j) paths_.data[j] = siglasso::finite(paths[j]);
        for (size_t j = 0; j < responses_.size; ++j) responses_.data[j] = siglasso::finite(y[j]);
        batch(o.samples, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                double variation = 0;
                double* sample = paths_.data + i * o.length * o.dimension;
                for (size_t j = 1; j < o.length; ++j) {
                    double step = 0;
                    for (size_t a = 0; a < o.dimension; ++a)
                        step = std::max(step, std::abs(siglasso::finite(sample[j * o.dimension + a] - sample[(j - 1) * o.dimension + a])));
                    variation = siglasso::finite(variation + step);
                }
                for (size_t a = 0; a < o.dimension; ++a) {
                    const double origin = sample[a];
                    for (size_t j = o.length; j-- > 0;)
                        sample[j * o.dimension + a] = variation ? siglasso::finite((sample[j * o.dimension + a] - origin) / variation) : 0;
                }
            }
        });
    }
    siglasso::Array alloc(size_t size) override {
        const size_t bytes = siglasso::multiply(size, sizeof(double));
        if (bytes > budget->available()) trim_caches();
        try { budget->acquire(bytes); }
        catch (const std::bad_alloc&) { trim_caches(); budget->acquire(bytes); }
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
        auto correction = hn::Zero(d);
        size_t j = 0;
        for (; j + lanes <= n; j += lanes) {
            const auto product = hn::Mul(hn::LoadU(d, a + j), hn::LoadU(d, b + j));
            const auto next = hn::Add(sum, product);
            correction = hn::Add(correction, hn::IfThenElse(hn::Ge(hn::Abs(sum), hn::Abs(product)),
                hn::Add(hn::Sub(sum, next), product), hn::Add(hn::Sub(product, next), sum)));
            sum = next;
        }
        double result = hn::ReduceSum(d, sum), error = hn::ReduceSum(d, correction);
        for (; j < n; ++j) {
            const double value = a[j] * b[j], next = result + value;
            error += std::abs(result) >= std::abs(value) ? (result - next) + value : (value - next) + result;
            result = next;
        }
        return siglasso::finite(result + error);
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
    void correlations(const std::vector<siglasso::Column>& columns, const siglasso::Array& v, std::vector<double>& out, const std::vector<size_t>* selected = nullptr) override {
        out.resize(selected ? selected->size() : columns.size());
        auto compute = [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) out[j] = dot(columns[selected ? (*selected)[j] : j].values.data, v.data, v.size) / v.size;
        };
        if (options.n_jobs == 1 || out.size() <= 1) compute(0, out.size());
        else spawn_batch_threads(out.size(), options.n_jobs, compute);
    }
    std::unique_ptr<siglasso::ActiveSet> active_set(const siglasso::Array& y, double alpha) override { return siglasso::make_cpu_active(*this, y, alpha); }
    double correlation_max(const std::vector<siglasso::Column>& columns, const siglasso::Array& vector) override {
        double maximum = 0;
        const double eps = std::numeric_limits<double>::epsilon();
        for (const auto& column : columns) {
            double sum = 0, correction = 0, magnitude = 0;
            for (size_t i = 0; i < vector.size; ++i) {
                const double product = column.values.data[i] * vector.data[i], next = sum + product;
                correction += std::fma(column.values.data[i], vector.data[i], -product);
                correction += std::abs(sum) >= std::abs(product) ? (sum - next) + product : (product - next) + sum;
                magnitude += std::abs(product); sum = next;
            }
            const double value = std::abs(sum + correction);
            maximum = std::max(maximum, (value + 8 * eps * value + 8 * vector.size * eps * eps * magnitude) / vector.size);
        }
        return siglasso::finite(std::nextafter(maximum, std::numeric_limits<double>::infinity()));
    }
    void prepare() override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        increments = alloc(siglasso::multiply(siglasso::multiply(n, l), d));
        variation = alloc(n * l); after = alloc(n * l);
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
                    increments.data[(j * d + a) * n + i] = paths_.data[(i * (l + 1) + j + 1) * d + a] - paths_.data[(i * (l + 1) + j) * d + a];
                siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
            }
        });
        paths_ = {};
        prepare_alphabet();
        batch(n, [&](uint64_t begin, uint64_t end) {
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
    }
    siglasso::Array factors(size_t k) override {
        if (!factor_parent_) return compute_factors(k);
        std::lock_guard lock(factor_parent_->factor_mutex_);
        auto& cached = factor_parent_->shared_factors_[k];
        if (cached.owner) return cached;
        auto value = compute_factors(k);
        const size_t limit = factor_parent_->options.cache_bytes / 4;
        if (value.bytes() <= limit) {
            if (factor_parent_->factor_bytes_ + value.bytes() > limit) {
                for (auto& entry : factor_parent_->shared_factors_) entry = {};
                factor_parent_->factor_bytes_ = 0;
            }
            cached = value; factor_parent_->factor_bytes_ += value.bytes();
        }
        return value;
    }
    siglasso::Array compute_factors(size_t k) {
        const size_t n = options.samples, l = options.length - 1, m = options.degree;
        const size_t constant_count = siglasso::multiply(siglasso::multiply(m - k, m - k + 1) / 2, k + 1);
        const size_t values = siglasso::multiply((m - k) * (k + 1), n * l);
        auto result = alloc(values + constant_count);
        auto host_constants = account(siglasso::multiply(constant_count, sizeof(double)));
        auto constants = siglasso::integration_coefficients(k, m - k, weights);
        std::copy(constants.begin(), constants.end(), result.data + values);
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q <= m - k; ++q) for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j)
                for (size_t i = begin; i < end; ++i)
                    result.data[(((q - 1) * (k + 1) + a) * l + j) * n + i] = siglasso::finite(
                        siglasso::factor_value(variation.data, after.data, constants.data(), k, q, a, j, i, n) / n);
        });
        return result;
    }
    void moments(siglasso::Node& node, const siglasso::Array& factors) override {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), levels = options.degree - k;
        node.moments = alloc(levels * n * l);
        if (!options.bound_method) {
            if (k == 0 && ((options.search_method & 1) || ((options.search_method & 2) && levels <= 2))) {
                node.integrals = alloc(n * l); fill(node.integrals, 1);
                if ((options.search_method & 2) && levels <= 2) { node.linear_integrals = alloc(n * l); fill(node.linear_integrals, .5); }
            }
            batch(n, [&](uint64_t begin, uint64_t end) {
                const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                for (size_t j = 0; j < l; ++j) {
                    size_t i = begin;
                    for (; i + lanes <= end; i += lanes) {
                        const auto step = hn::LoadU(tag, variation.data + j * n + i), tail = hn::LoadU(tag, after.data + j * n + i);
                        const auto weight = hn::Set(tag, weights[k + 1] / (2 * n));
                        auto left = hn::Mul(hn::Mul(hn::Abs(hn::LoadU(tag, node.time_values.data + j * n + i)), step), weight);
                        auto right = hn::Mul(hn::Mul(hn::Abs(hn::LoadU(tag, node.time_values.data + (j + 1) * n + i)), step), weight);
                        for (size_t q = 0; q < levels; ++q) {
                            hn::StoreU(hn::Add(left, right), tag, node.moments.data + (q * l + j) * n + i);
                            const auto ratio = hn::Set(tag, (k + q + 2) * reciprocals[q + 1]);
                            left = hn::Mul(left, hn::Mul(hn::Add(tail, step), ratio)); right = hn::Mul(right, hn::Mul(tail, ratio));
                        }
                    }
                    for (; i < end; ++i) siglasso::trapezoidal_moments(node.time_values.data[j * n + i], node.time_values.data[(j + 1) * n + i],
                        variation.data[j * n + i], after.data[j * n + i], weights[k + 1] / n, k, levels, n * l, node.moments.data + j * n + i, reciprocals.data());
                }
            });
            return;
        }
        node.integrals = alloc(n * l);
        if ((options.search_method & 2) && levels <= 2) node.linear_integrals = alloc(n * l);
        batch(n, [&](uint64_t begin, uint64_t end) {
            auto scratch = alloc(siglasso::integral_scratch_size(options.degree));
            for (size_t j = 0; j < l; ++j) for (size_t i = begin; i < end; ++i) {
                    double integral = 0, linear = 0;
                    for (size_t a = 0; a <= k; ++a) {
                        const double c = node.coefficients.data[(j * (k + 1) + a) * n + i];
                        integral += c;
                        if (node.linear_integrals.owner) linear += (a + 1) * c;
                    }
                    node.integrals.data[j * n + i] = integral / (k + 1);
                    if (node.linear_integrals.owner) node.linear_integrals.data[j * n + i] = linear / ((k + 1) * (k + 2));
                    siglasso::absolute_moments(node.coefficients.data, factors.data, variation.data[j * n + i], after.data[j * n + i],
                        k, levels, j, i, n, l, scratch.data, node.moments.data + j * n + i, (options.search_method & 2) && levels <= 2);
            }
        });
        node.uncertain_moments = std::any_of(node.moments.data, node.moments.data + node.moments.size, [](double value) { return value < 0; });
    }
    siglasso::Array energy_gram(size_t low, size_t high) override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        auto result = alloc(siglasso::multiply(n * l, l));
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t j = 0; j < l; ++j) for (size_t h = 0; h <= j; ++h) for (size_t i = begin; i < end; ++i) {
                double value = 0;
                for (size_t a = low; a < high; ++a) value += increments.data[(j * d + a) * n + i] * increments.data[(h * d + a) * n + i];
                result.data[(j * l + h) * n + i] = siglasso::finite(value);
            }
        });
        return result;
    }
    siglasso::HostSearch* host_search() override { return this; }
    void begin_query(const siglasso::Array& residual) override {
        if (!(options.search_method & 2)) return;
        const size_t n = options.samples, m = options.degree;
        terminal_mean_ = options.intercept ? std::accumulate(residual.data, residual.data + n, 0.0) / n : 0;
        terminal_caps_ = {};
        if (options.length > 1) for (size_t i = 0; i < n; ++i) {
            const double r = std::abs(residual.data[i] - terminal_mean_), v = variation.data[i] + after.data[i];
            if (m > 1) terminal_caps_[0] += r * std::pow(v, static_cast<double>(m - 1));
            terminal_caps_[1] += r * std::pow(v, static_cast<double>(m));
        }
        if (!terminal_needed_) return;
        if (!terminal_storage_.owner) {
            const size_t available = std::min(options.cache_bytes / 8, budget->available() / 4);
            size_t possible = 1;
            for (size_t k = 0; k < options.degree && possible < 16384; ++k)
                possible = options.dimension >= 16383 / possible ? 16384 : 1 + options.dimension * possible;
            size_t slots = 1;
            while (slots < 4 * possible && 2 * slots * (options.degree + 3) * sizeof(double) <= available) slots *= 2;
            if (slots < 64) return;
            terminal_storage_ = alloc(slots * (options.degree + 3));
            terminal_cache_ = {terminal_storage_.data, slots, options.degree};
            for (size_t j = 0; j < slots; ++j) terminal_storage_.data[j * (options.degree + 3)] = 0;
        }
        if (++terminal_cache_.generation >= 9007199254740992.0) {
            for (size_t j = 0; j < terminal_cache_.capacity; ++j) terminal_storage_.data[j * (options.degree + 3)] = 0;
            terminal_cache_.generation = 1;
        }
    }
    void trim_caches() override {
        terminal_storage_ = {}; terminal_products_ = {}; terminal_cache_ = {};
        Backend::trim_caches();
    }
    void bounds(const std::vector<siglasso::Probe>& tasks, const siglasso::Array& residual, double threshold, std::vector<double>& out) override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        out.assign(tasks.size(), 0);
        std::atomic<uint64_t> terminal_queries{0}, terminal_pruned{0}, terminal_correlations{0};
        const double residual_mean = terminal_mean_;
        batch(tasks.size(), [&](uint64_t begin, uint64_t end) {
            siglasso::Array terminal_products = options.n_jobs == 1 ? terminal_products_ : siglasso::Array{};
            uint64_t queries = 0, pruned = 0, correlations = 0;
            for (size_t t = begin; t < end; ++t) {
                const auto& task = tasks[t];
                const size_t levels = options.degree - task.node->word.size();
                double best = 0;
                if (task.node->uncertain_moments) { out[t] = threshold; continue; }
                for (size_t q = levels; q-- > 0;) {
                    double total = 0;
                    const hn::ScalableTag<double> tag;
                    const size_t lanes = hn::Lanes(tag);
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
                        const double energy = energy_bound(task, residual);
                        total = std::min(total, siglasso::finite(energy * weights[task.node->word.size() + 1] / n));
                    }
                    best = std::max(best, siglasso::finite(total));
                    if (best >= threshold) break;
                }
                if ((options.search_method & 2) && levels <= 2 && best >= threshold && l) {
                    ++queries;
                    const size_t k = task.node->word.size();
                    const double error = siglasso::terminal_roundoff(k, n, l);
                    double signed_bound = 0;
                    bool complete = true;
                    for (size_t q = 1; q <= levels && complete; ++q) {
                        const double cap = terminal_caps_[k + q == options.degree];
                        const bool reuse = task.high - task.low >= (q == 2 ? 2 : 4)
                            && (terminal_products.owner || n * l * sizeof(double) <= budget->available());
                        if (reuse && !terminal_products.owner) {
                            terminal_products = alloc(n * l);
                            if (options.n_jobs == 1) terminal_products_ = terminal_products;
                        }
                        bool prepared = false;
                        double product_magnitude = 0;
                        const size_t width = task.high - task.low;
                        size_t tile_start = 0, tile_count = 0;
                        std::array<double, 4> tile_sums{};
                        const double rounding = reuse ? siglasso::terminal_roundoff(k, n * l, l) : error;
                        for (size_t index = 0; index < width * (q == 2 ? d : 1); ++index) {
                            const size_t a = task.low + index % width, b = index / width;
                            if (a == task.low) prepared = false;
                            double cached_bound = 0;
                            size_t cache_slot = 0;
                            bool cached = false;
                            if (terminal_cache_.capacity) {
                                std::unique_lock lock(terminal_mutex_, std::defer_lock);
                                if (options.n_jobs != 1) lock.lock();
                                cache_slot = terminal_cache_.slot(task.node->word.data(), k, a, b, q);
                                cached = terminal_cache_.get(cache_slot, task.node->word.data(), k, a, b, q, cached_bound);
                            }
                            if (cached) {
                                if (cached_bound * (1 + 1e-10) >= threshold) { complete = false; break; }
                                signed_bound = std::max(signed_bound, cached_bound);
                                continue;
                            }
                            if (reuse && !prepared) {
                                const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                                auto absolute = hn::Zero(tag);
                                // Segment variation bounds every first-letter increment.
                                product_magnitude = 0;
                                size_t i = 0;
                                for (; i + lanes <= n; i += lanes) {
                                    auto tail = hn::Zero(tag);
                                    for (size_t j = l; j-- > 0;) {
                                        const auto other = q == 2 ? hn::LoadU(tag, increments.data + (j * d + b) * n + i) : hn::Zero(tag);
                                        tail = hn::Add(tail, other);
                                        const auto integral = hn::LoadU(tag, task.node->integrals.data + j * n + i);
                                        const auto left = q == 2 ? hn::Mul(integral, tail) : integral;
                                        const auto right = q == 2 ? hn::Mul(hn::LoadU(tag, task.node->linear_integrals.data + j * n + i), other) : hn::Zero(tag);
                                        const auto r = hn::Sub(hn::LoadU(tag, residual.data + i), hn::Set(tag, residual_mean));
                                        hn::StoreU(hn::Mul(r, hn::Sub(left, right)), tag, terminal_products.data + j * n + i);
                                        absolute = hn::MulAdd(hn::Mul(hn::Abs(r), hn::Add(hn::Abs(left), hn::Abs(right))),
                                            hn::LoadU(tag, variation.data + j * n + i), absolute);
                                    }
                                }
                                for (; i < n; ++i) {
                                    double tail = 0;
                                    for (size_t j = l; j-- > 0;) {
                                        const double other = q == 2 ? increments.data[(j * d + b) * n + i] : 0; tail += other;
                                        const double integral = task.node->integrals.data[j * n + i];
                                        const double left = q == 2 ? integral * tail : integral;
                                        const double right = q == 2 ? task.node->linear_integrals.data[j * n + i] * other : 0;
                                        const double r = residual.data[i] - residual_mean;
                                        terminal_products.data[j * n + i] = r * (left - right);
                                        product_magnitude += std::abs(r) * (std::abs(left) + std::abs(right)) * variation.data[j * n + i];
                                    }
                                }
                                product_magnitude = (product_magnitude + hn::ReduceSum(tag, absolute)) * (1 + rounding);
                                prepared = true;
                            }
                            double sum = 0, correction = 0, magnitude = 0;
                            const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                            auto sums = hn::Zero(tag), corrections = hn::Zero(tag), magnitudes = hn::Zero(tag);
                            size_t i = 0;
                            if (reuse) {
                                if (index >= tile_start + tile_count) {
                                    tile_start = index; tile_count = std::min<size_t>(4, task.high - a);
                                    tile_sums = {};
                                    auto s0 = hn::Zero(tag), s1 = hn::Zero(tag), s2 = hn::Zero(tag), s3 = hn::Zero(tag);
                                    auto s4 = hn::Zero(tag), s5 = hn::Zero(tag), s6 = hn::Zero(tag), s7 = hn::Zero(tag);
                                    for (size_t j = 0; j < l; ++j) {
                                        i = 0;
                                        for (; i + 2 * lanes <= n; i += 2 * lanes) {
                                            auto value = hn::LoadU(tag, terminal_products.data + j * n + i);
                                            s0 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a) * n + i), value, s0);
                                            if (tile_count > 1) s1 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 1) * n + i), value, s1);
                                            if (tile_count > 2) s2 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 2) * n + i), value, s2);
                                            if (tile_count > 3) s3 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 3) * n + i), value, s3);
                                            value = hn::LoadU(tag, terminal_products.data + j * n + i + lanes);
                                            s4 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a) * n + i + lanes), value, s4);
                                            if (tile_count > 1) s5 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 1) * n + i + lanes), value, s5);
                                            if (tile_count > 2) s6 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 2) * n + i + lanes), value, s6);
                                            if (tile_count > 3) s7 = hn::MulAdd(hn::LoadU(tag, increments.data + (j * d + a + 3) * n + i + lanes), value, s7);
                                        }
                                        for (; i + lanes <= n; i += lanes) {
                                            const auto value = hn::LoadU(tag, terminal_products.data + j * n + i);
                                            const auto x0 = hn::LoadU(tag, increments.data + (j * d + a) * n + i);
                                            const auto x1 = tile_count > 1 ? hn::LoadU(tag, increments.data + (j * d + a + 1) * n + i) : hn::Zero(tag);
                                            const auto x2 = tile_count > 2 ? hn::LoadU(tag, increments.data + (j * d + a + 2) * n + i) : hn::Zero(tag);
                                            const auto x3 = tile_count > 3 ? hn::LoadU(tag, increments.data + (j * d + a + 3) * n + i) : hn::Zero(tag);
                                            s0 = hn::MulAdd(x0, value, s0); s1 = hn::MulAdd(x1, value, s1);
                                            s2 = hn::MulAdd(x2, value, s2); s3 = hn::MulAdd(x3, value, s3);
                                        }
                                        for (; i < n; ++i) for (size_t h = 0; h < tile_count; ++h) {
                                            const double x = increments.data[(j * d + a + h) * n + i];
                                            tile_sums[h] += x * terminal_products.data[j * n + i];
                                        }
                                    }
                                    tile_sums[0] += hn::ReduceSum(tag, hn::Add(s0, s4)); tile_sums[1] += hn::ReduceSum(tag, hn::Add(s1, s5));
                                    tile_sums[2] += hn::ReduceSum(tag, hn::Add(s2, s6)); tile_sums[3] += hn::ReduceSum(tag, hn::Add(s3, s7));
                                    correlations += tile_count;
                                }
                                sum = tile_sums[index - tile_start]; magnitude = product_magnitude;
                            } else {
                                for (; i + lanes <= n; i += lanes) {
                                    auto value = hn::Zero(tag), compensation = hn::Zero(tag), tail = hn::Zero(tag), absolute = hn::Zero(tag);
                                    for (size_t j = l; j-- > 0;) {
                                        const auto step = hn::LoadU(tag, increments.data + (j * d + a) * n + i);
                                        const auto integral = hn::LoadU(tag, task.node->integrals.data + j * n + i);
                                        auto term = hn::Mul(step, integral), bound = hn::Abs(term);
                                        if (q == 2) {
                                            const auto other = hn::LoadU(tag, increments.data + (j * d + b) * n + i);
                                            tail = hn::Add(tail, other);
                                            const auto left = hn::Mul(integral, tail);
                                            const auto right = hn::Mul(hn::LoadU(tag, task.node->linear_integrals.data + j * n + i), other);
                                            term = hn::Mul(step, hn::Sub(left, right));
                                            bound = hn::Mul(hn::Abs(step), hn::Add(hn::Abs(left), hn::Abs(right)));
                                        }
                                        const auto adjusted = hn::Sub(term, compensation), next = hn::Add(value, adjusted);
                                        compensation = hn::Sub(hn::Sub(next, value), adjusted); value = next;
                                        absolute = hn::Add(absolute, bound);
                                    }
                                    const auto r = hn::Sub(hn::LoadU(tag, residual.data + i), hn::Set(tag, residual_mean));
                                    const auto adjusted = hn::Sub(hn::Mul(r, value), corrections), next = hn::Add(sums, adjusted);
                                    corrections = hn::Sub(hn::Sub(next, sums), adjusted); sums = next;
                                    magnitudes = hn::Add(magnitudes, hn::Mul(hn::Abs(r), absolute));
                                }
                                sum = hn::ReduceSum(tag, sums); magnitude = hn::ReduceSum(tag, magnitudes);
                                for (; i < n; ++i) {
                                    double absolute;
                                    const double value = siglasso::terminal_sample(task.node->integrals.data, task.node->linear_integrals.data,
                                        increments.data, a, b, q, k, i, n, l, d, absolute);
                                    const double r = residual.data[i] - residual_mean;
                                    const double adjusted = r * value - correction, next = sum + adjusted;
                                    correction = (next - sum) - adjusted; sum = next;
                                    magnitude += std::abs(r) * absolute;
                                }
                            }
                            if (!reuse) ++correlations;
                            const double bound = (std::abs(sum) * weights[k + q] + rounding * (magnitude * weights[k + q] + cap)) / n + 1e-15;
                            if (terminal_cache_.capacity && std::isfinite(bound)) {
                                std::unique_lock lock(terminal_mutex_, std::defer_lock);
                                if (options.n_jobs != 1) lock.lock();
                                terminal_cache_.put(cache_slot, task.node->word.data(), k, a, b, q, bound);
                            }
                            if (!std::isfinite(bound) || bound * (1 + 1e-10) >= threshold) { terminal_needed_ = true; complete = false; break; }
                            signed_bound = std::max(signed_bound, bound);
                        }
                    }
                    if (complete && siglasso::guard(signed_bound * (1 + 1e-10)) < threshold) {
                        best = std::min(best, signed_bound); ++pruned;
                    }
                }
                out[t] = best;
            }
            terminal_queries += queries; terminal_pruned += pruned; terminal_correlations += correlations;
        });
        stats.terminal_queries += terminal_queries; stats.terminal_pruned += terminal_pruned;
        stats.terminal_correlations += terminal_correlations;
    }
    void children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        const size_t n = options.samples, l = options.length - 1;
        out.resize(tasks.size());
        for (size_t t = 0; t < tasks.size(); ++t) {
            auto node = std::make_shared<siglasso::Node>(); node->word = tasks[t].node->word; node->word.push_back(tasks[t].low);
            node->metadata = account(512 + node->word.size() * 64);
            node->values = alloc(n);
            if (node->word.size() < options.degree) {
                if (options.bound_method) node->coefficients = alloc((node->word.size() + 1) * n * l);
                else {
                    node->time_values = alloc(n * (l + 1));
                    if ((options.search_method & 1) || ((options.search_method & 2) && options.degree - node->word.size() <= 2)) node->integrals = alloc(n * l);
                    if ((options.search_method & 2) && options.degree - node->word.size() <= 2) node->linear_integrals = alloc(n * l);
                }
            }
            out[t] = std::move(node);
        }
        batch(tasks.size() * n, [&](uint64_t begin, uint64_t end) {
            const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
            for (size_t index = begin; index < end;) {
                const size_t t = index / n, i = index % n;
                if (!options.bound_method && i + lanes <= n && index + lanes <= end) {
                    auto& node = *out[t]; const size_t k = node.word.size();
                    const auto& parent = *tasks[t].node;
                    auto value = hn::Zero(tag);
                    if (node.time_values.owner) hn::StoreU(value, tag, node.time_values.data + i);
                    for (size_t j = 0; j < l; ++j) {
                        if (k == options.degree && parent.integrals.owner) {
                            value = hn::Add(value, hn::Mul(hn::LoadU(tag, increments.data + (j * options.dimension + tasks[t].low) * n + i),
                                hn::LoadU(tag, parent.integrals.data + j * n + i)));
                            continue;
                        }
                        auto product = hn::Set(tag, 1), integral = value, linear = hn::Mul(value, hn::Set(tag, .5));
                        for (size_t h = 1; h <= k; ++h) {
                            product = hn::Mul(product, hn::Mul(hn::LoadU(tag, increments.data + (j * options.dimension + node.word[k - h]) * n + i), hn::Set(tag, reciprocals[h])));
                            const auto term = hn::Mul(hn::LoadU(tag, tasks[t].ancestors->nodes[k - h]->time_values.data + j * n + i), product);
                            value = hn::Add(value, term);
                            if (node.integrals.owner) integral = hn::Add(integral, hn::Mul(term, hn::Set(tag, reciprocals[h + 1])));
                            if (node.linear_integrals.owner) linear = hn::Add(linear, hn::Mul(term, hn::Set(tag, reciprocals[h + 2])));
                        }
                        if (node.time_values.owner) hn::StoreU(value, tag, node.time_values.data + (j + 1) * n + i);
                        if (node.integrals.owner) hn::StoreU(integral, tag, node.integrals.data + j * n + i);
                        if (node.linear_integrals.owner) hn::StoreU(linear, tag, node.linear_integrals.data + j * n + i);
                    }
                    hn::StoreU(value, tag, node.values.data + i);
                    for (size_t lane = 0; lane < lanes; ++lane) siglasso::finite(node.values.data[i + lane]);
                    index += lanes;
                    continue;
                }
                if (options.bound_method) siglasso::child_sample(increments.data, tasks[t].node->coefficients.data, tasks[t].node->integrals.data, out[t]->coefficients.data, out[t]->values.data,
                    tasks[t].low, tasks[t].node->word.size(), i, n, l, options.dimension);
                else {
                    auto& node = *out[t]; const size_t k = node.word.size();
                    double value = 0;
                    if (node.time_values.owner) node.time_values.data[i] = 0;
                    for (size_t j = 0; j < l; ++j) {
                        if (k == options.degree && tasks[t].node->integrals.owner) {
                            value += increments.data[(j * options.dimension + tasks[t].low) * n + i] * tasks[t].node->integrals.data[j * n + i];
                            continue;
                        }
                        double product = 1, integral = value, linear = value / 2;
                        for (size_t h = 1; h <= k; ++h) {
                            product *= increments.data[(j * options.dimension + node.word[k - h]) * n + i] * reciprocals[h];
                            const double term = tasks[t].ancestors->nodes[k - h]->time_values.data[j * n + i] * product;
                            value += term;
                            if (node.integrals.owner) integral += term * reciprocals[h + 1];
                            if (node.linear_integrals.owner) linear += term * reciprocals[h + 2];
                        }
                        if (node.time_values.owner) node.time_values.data[(j + 1) * n + i] = value;
                        if (node.integrals.owner) node.integrals.data[j * n + i] = integral;
                        if (node.linear_integrals.owner) node.linear_integrals.data[j * n + i] = linear;
                    }
                    node.values.data[i] = value;
                }
                siglasso::finite(out[t]->values.data[i]);
                ++index;
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
                columns.push_back({word, values, 0, 0, column_metadata});
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
            ++stats.full_transform_calls;
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
            }
        });
        stats.evaluated = stats.constructed = count;
    }
    siglasso::Array response(size_t target) override {
        auto result = alloc(options.samples);
        for (size_t i = 0; i < options.samples; ++i) result.data[i] = responses_.data[i * options.targets + target];
        return result;
    }
    template<typename T> void export_typed(const siglasso::Result& result, const SigLassoOutput& output) {
        const auto convert = [](double value) { return static_cast<T>(siglasso::finite(static_cast<T>(value))); };
        const std::vector<double>* inputs[] = {&result.coefficients, &result.intercepts, &result.objectives, &result.gaps};
        void* outputs[] = {output.coefficients, output.intercepts, output.objectives, output.gaps};
        for (size_t j = 0; j < 4; ++j) std::transform(inputs[j]->begin(), inputs[j]->end(), static_cast<T*>(outputs[j]), convert);
        const size_t p = result.words.size();
        auto* signatures = static_cast<T*>(output.signatures);
        batch(options.samples, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) for (size_t j = 0; j < p; ++j)
                signatures[i * p + j] = convert(result.selected_columns[j].values.data[i] + result.selected_columns[j].mean);
        });
    }
    void export_result(const siglasso::Result& result, const SigLassoOutput& output) override {
        if (options.single_precision) export_typed<float>(result, output);
        else export_typed<double>(result, output);
    }
    bool has_native_search() const override { return options.n_jobs != 1 && options.dimension > 1 && !options.memory_limit; }
    bool reuse_search_bound() const override { return !options.single_precision; }
    bool native_search(const siglasso::Array& residual, double alpha,
        const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active, size_t capacity, siglasso::SearchResult& result) override {
        if (searches_.empty()) {
            const size_t threads = options.n_jobs > 0 ? options.n_jobs : get_max_threads() + 1 + options.n_jobs;
            if (!(options.search_method & 1)) {
                frontier_.resize(options.dimension); std::iota(frontier_.begin(), frontier_.end(), 0);
            } else frontier_.push_back(0);
            while (frontier_.size() < std::min(threads > 1 ? 4 * threads : threads, static_cast<size_t>(options.dimension))) {
                auto largest = std::max_element(frontier_.begin(), frontier_.end(), [&](size_t a, size_t b) {
                    return alphabet[a].second - alphabet[a].first < alphabet[b].second - alphabet[b].first;
                });
                const auto [low, high] = alphabet[*largest];
                const size_t block = *largest;
                *largest = block + 1; frontier_.push_back(block + 2 * ((low + high) / 2 - low));
            }
            std::sort(frontier_.begin(), frontier_.end());
            shared_factors_.resize(options.degree);
            auto single = options; single.n_jobs = 1; single.cache_bytes /= frontier_.size();
            for (size_t j = 0; j < frontier_.size(); ++j) {
                auto worker = std::make_unique<LassoCPU>(*this, single);
                worker->increments = increments; worker->variation = variation; worker->after = after;
                worker->alphabet = alphabet; worker->alphabet_ratios = alphabet_ratios;
                worker->factor_parent_ = this;
                searches_.push_back(std::make_unique<siglasso::Search>(*worker, searches_.empty() ? siglasso::NodePtr{} : searches_.front()->root()));
                search_workers_.push_back(std::move(worker));
            }
        }
        std::vector<siglasso::SearchResult> found(searches_.size());
        std::atomic<size_t> next{0};
        spawn_batch_threads(searches_.size(), options.n_jobs, [&](uint64_t, uint64_t) {
            for (size_t j = next.fetch_add(1); j < searches_.size(); j = next.fetch_add(1)) {
                const auto [low, high] = alphabet[frontier_[j]];
                search_workers_[j]->stats = {};
                found[j] = searches_[j]->query(residual, alpha, active, capacity, low, high, frontier_[j]);
            }
        });
        stats.constructed = constructed.size();
        for (size_t j = 0; j < searches_.size(); ++j) {
            const auto& worker = *search_workers_[j];
            stats.constructed += worker.constructed.size();
            for (const auto& word : constructed) if (worker.constructed.count(word)) --stats.constructed;
            stats.evaluated += worker.stats.evaluated; stats.pruned += worker.stats.pruned; stats.block_queries += worker.stats.block_queries;
            stats.cache_hits += worker.stats.cache_hits; stats.cache_misses += worker.stats.cache_misses;
            stats.terminal_queries += worker.stats.terminal_queries; stats.terminal_pruned += worker.stats.terminal_pruned;
            stats.terminal_correlations += worker.stats.terminal_correlations;
            result.omitted = std::max(result.omitted, found[j].omitted);
            for (auto& candidate : found[j].candidates) result.candidates.push_back(std::move(candidate));
        }
        if (result.candidates.size() > capacity) {
            std::nth_element(result.candidates.begin(), result.candidates.begin() + capacity, result.candidates.end(), siglasso::better);
            for (size_t j = capacity; j < result.candidates.size(); ++j) result.omitted = std::max(result.omitted, siglasso::guard(result.candidates[j].score));
            result.candidates.resize(capacity);
        }
        std::sort(result.candidates.begin(), result.candidates.end(), siglasso::better);
        return true;
    }
    std::unique_ptr<Backend> solver_worker() override {
        if (options.targets == 1 || options.n_jobs == 1) return {};
        auto single = options; single.n_jobs = 1;
        auto worker = std::make_unique<LassoCPU>(*this, single);
        worker->trim_search = [this] { std::lock_guard lock(factor_mutex_); trim_caches(); };
        return worker;
    }
    void for_targets(const std::function<void(size_t)>& action) override {
        if (options.targets == 1 || options.n_jobs == 1) { Backend::for_targets(action); return; }
        spawn_batch_threads(options.targets, options.n_jobs, [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) action(j);
        });
    }
};

template<typename T>
void sig_lasso_(const T* path, const T* y, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) {
    LassoCPU backend(*options, path, y);
    std::vector<siglasso::Word> initial;
    for (size_t j = 0; j < words; ++j) initial.emplace_back(letters + offsets[j], letters + offsets[j + 1]);
    auto result = siglasso::fit(backend, std::vector<double>(alphas, alphas + options->targets), initial);
    siglasso::export_result(backend, result, allocate, context);
}
template void sig_lasso_<float>(const float*, const float*, const double*, const uint64_t*, const uint64_t*, uint64_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_lasso_<double>(const double*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, const SigLassoOptions*, SigLassoAllocator, void*);

} } // namespace cpsig::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
