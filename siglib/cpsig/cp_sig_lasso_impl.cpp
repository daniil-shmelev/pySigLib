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
#include "cp_lasso_features.h"
#include "cp_lasso_acceleration.h"
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
    uint32_t feature_type_ = 0;
    unsigned bound_type_ = 0;
    siglasso::Array tt_suffix_;
    siglasso::Array paths_, responses_;
    std::vector<std::unique_ptr<LassoCPU>> search_workers_;
    std::vector<std::unique_ptr<siglasso::Search>> searches_;
    std::vector<size_t> frontier_;
    LassoCPU* factor_parent_ = nullptr;
    std::mutex factor_mutex_;
    std::mutex terminal_mutex_;
    siglasso::Array terminal_storage_, terminal_products_, terminal_matrix_workspace_, terminal_channel_norms_;
    siglasso::Array terminal_sample_norms_, terminal_operator_weights_, terminal_suffix_bounds_;
    siglasso::TerminalCache terminal_cache_;
    std::atomic<bool> terminal_needed_{false};
    std::array<double, 2> terminal_caps_{};
    std::unordered_map<siglasso::Word, std::array<std::vector<std::pair<uint64_t, uint64_t>>, 2>, siglasso::WordHash> terminal_active_;
    std::shared_ptr<void> terminal_active_metadata_;
    std::vector<siglasso::Array> shared_factors_;
    size_t factor_bytes_ = 0;
    struct Summary {
        siglasso::Word word;
        size_t suffix = 0, order = 0, generation = 0;
        double norm = 0, magnitude = 0;
        bool ready = false;
    };
    std::vector<Summary> summaries_;
    siglasso::Array summary_values_;
    std::shared_ptr<void> summary_metadata_;
    size_t query_generation_ = 0;
    Summary* summary(const siglasso::Word& word, size_t suffix, size_t order) {
        if (summaries_.empty()) return nullptr;
        const size_t hash = siglasso::WordHash{}(word) ^ (suffix * 0x9e3779b97f4a7c15ull) ^ (order * 0xbf58476d1ce4e5b9ull);
        auto& entry = summaries_[hash & (summaries_.size() - 1)];
        if (entry.ready && entry.word == word && entry.suffix == suffix && entry.order == order
            && (!(options.summary_flags & 4) || entry.generation == query_generation_))
            ++optimization_stats[siglasso::SummaryHits];
        else {
            ++optimization_stats[siglasso::SummaryMisses];
            entry.word = word; entry.suffix = suffix; entry.order = order;
            entry.ready = false; entry.generation = 0;
        }
        return &entry;
    }
    double* summary_values(Summary* entry) { return summary_values_.data + (entry - summaries_.data()) * 2 * options.samples; }
    double positive_dot(const double* a, const double* b, size_t n) {
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        auto sum = hn::Zero(tag);
        size_t i = 0;
        for (; i + lanes <= n; i += lanes) sum = hn::MulAdd(hn::LoadU(tag, a + i), hn::LoadU(tag, b + i), sum);
        double value = hn::ReduceSum(tag, sum);
        for (; i < n; ++i) value += a[i] * b[i];
        return siglasso::finite(value);
    }
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
    void terminal_matrix(const siglasso::Probe& task, const siglasso::Array& residual, size_t first, size_t count,
        const siglasso::Array& scratch, double* magnitudes) {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, width = task.high - task.low;
        auto packing = account(siglasso::multiply(siglasso::multiply(n, width + count), sizeof(double)) + 65536);
        double* tails = scratch.data;
        double* products = scratch.at(n * count);
        double* output = scratch.at(2 * n * count);
        std::fill_n(tails, n * count, 0.);
        std::fill_n(output, width * count, 0.);
        std::fill_n(magnitudes, count, 0.);
        const hn::ScalableTag<double> tag;
        const size_t lanes = hn::Lanes(tag);
        for (size_t j = l; j-- > 0;) {
            for (size_t b = 0; b < count; ++b) {
                auto magnitude = hn::Zero(tag);
                size_t i = 0;
                for (; i + lanes <= n; i += lanes) {
                    const auto other = hn::LoadU(tag, increments.data + (j * d + first + b) * n + i);
                    const auto tail = hn::Add(hn::LoadU(tag, tails + b * n + i), other);
                    hn::StoreU(tail, tag, tails + b * n + i);
                    const auto left = hn::Mul(hn::LoadU(tag, task.node->integrals.data + j * n + i), tail);
                    const auto right = hn::Mul(hn::LoadU(tag, task.node->linear_integrals.data + j * n + i), other);
                    const auto r = hn::LoadU(tag, residual.data + i);
                    hn::StoreU(hn::Mul(r, hn::Sub(left, right)), tag, products + b * n + i);
                    magnitude = hn::MulAdd(hn::Mul(hn::Abs(r), hn::Add(hn::Abs(left), hn::Abs(right))),
                        hn::LoadU(tag, variation.data + j * n + i), magnitude);
                }
                magnitudes[b] += hn::ReduceSum(tag, magnitude);
                for (; i < n; ++i) {
                    const double other = increments.data[(j * d + first + b) * n + i];
                    tails[b * n + i] += other;
                    const double left = task.node->integrals.data[j * n + i] * tails[b * n + i];
                    const double right = task.node->linear_integrals.data[j * n + i] * other;
                    products[b * n + i] = residual.data[i] * (left - right);
                    magnitudes[b] += std::abs(residual.data[i]) * (std::abs(left) + std::abs(right)) * variation.data[j * n + i];
                }
            }
            siglasso::terminal_matrix_product(increments.data + (j * d + task.low) * n, products, output, n, width, count);
        }
    }
public:
    LassoCPU(const LassoCPU& source, const siglasso::SearchOptions& options) : Backend(options, source.budget), paths_(source.paths_), responses_(source.responses_) {
        weights = source.weights;
        bound_type_ = source.bound_type_; tt_suffix_ = source.tt_suffix_;
        terminal_channel_norms_ = source.terminal_channel_norms_;
        terminal_sample_norms_ = source.terminal_sample_norms_;
    }
    template<typename T>
    LassoCPU(const siglasso::SearchOptions& o, const T* paths, const T* y, uint32_t feature_type = 0) : Backend(o), feature_type_(feature_type) {
        if (!feature_type && !o.selected_only) bound_type_ = siglasso::experimental_signature_bound.load();
        paths_ = alloc(siglasso::multiply(siglasso::multiply(o.samples, o.length), o.dimension));
        if (y) responses_ = alloc(siglasso::multiply(o.samples, o.targets));
        for (size_t j = 0; j < paths_.size; ++j) {
            if (!std::isfinite(paths[j])) throw std::invalid_argument("Paths must be finite");
            paths_.data[j] = paths[j];
        }
        for (size_t j = 0; j < responses_.size; ++j) {
            const double v = y[j];
            if (!std::isfinite(v)) throw std::invalid_argument("Targets must be finite");
            if (o.loss == 1 && v != 0 && v != 1) throw std::invalid_argument("Logistic targets must be binary");
            if (o.loss == 2 && (v < 0 || std::floor(v) != v)) throw std::invalid_argument("Poisson targets must be nonnegative counts");
            responses_.data[j] = v;
        }
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
    void prepare() override {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const bool segmented = options.terminal_kernel == 6 || options.terminal_kernel == 8 || options.terminal_kernel == 9;
        increments = alloc(siglasso::multiply(siglasso::multiply(n, l), d));
        if (!options.selected_only) { variation = alloc(n * l); after = alloc(n * l); }
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                for (size_t j = 0; j < l; ++j) for (size_t a = 0; a < d; ++a)
                    increments.data[(j * d + a) * n + i] = paths_.data[(i * (l + 1) + j + 1) * d + a] - paths_.data[(i * (l + 1) + j) * d + a];
                if (!options.selected_only) siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
            }
        });
        paths_ = {};
        if (feature_type_ || options.selected_only) return;
        if (options.terminal_kernel >= 5) {
            terminal_channel_norms_ = alloc(siglasso::multiply(d, segmented ? l + 1 : 1));
            const double error = siglasso::terminal_roundoff(0, n * l, l);
            for (size_t a = 0; a < d; ++a) {
                double squared = 0;
                for (size_t j = 0; j < l; ++j) {
                    const double* x = increments.data + (j * d + a) * n;
                    const double part = dot(x, x, n) + 32. * (n + 1) * std::numeric_limits<double>::min();
                    squared += part;
                    if (segmented) terminal_channel_norms_.data[(j + 1) * d + a] = std::sqrt(part) * (1 + error);
                }
                terminal_channel_norms_.data[a] = std::sqrt(squared) * (1 + error);
            }
        }
        prepare_alphabet();
        if (options.terminal_kernel >= 10) {
            terminal_sample_norms_ = alloc(siglasso::multiply(n, d + alphabet.size()));
            const double error = siglasso::terminal_roundoff(0, n * l, l);
            batch(n, [&](uint64_t begin, uint64_t end) {
                for (size_t a = 0; a < d; ++a) for (size_t i = begin; i < end; ++i) {
                    double squared = 0;
                    for (size_t j = 0; j < l; ++j) {
                        const double x = increments.data[(j * d + a) * n + i]; squared += x * x;
                    }
                    terminal_sample_norms_.data[a * n + i] = std::sqrt(squared + 32. * (l + 1) * std::numeric_limits<double>::min()) * (1 + error);
                }
                for (size_t block = alphabet.size(); block-- > 0;) {
                    const auto [low, high] = alphabet[block];
                    const size_t right = block + 2 * ((low + high) / 2 - low);
                    for (size_t i = begin; i < end; ++i) terminal_sample_norms_.data[(d + block) * n + i] = high - low == 1
                        ? terminal_sample_norms_.data[low * n + i]
                        : std::max(terminal_sample_norms_.data[(d + block + 1) * n + i], terminal_sample_norms_.data[(d + right) * n + i]);
                }
            });
        }
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
        if (bound_type_) prepare_tt();
    }
    void prepare_tt() {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        tt_suffix_ = alloc(siglasso::multiply(siglasso::multiply(m, m + 1), siglasso::multiply(n, l)));
        fill(tt_suffix_, 0);
        for (size_t a = 1; a <= m; ++a) std::fill_n(tt_suffix_.data + a * l * n, l * n, 1 / weights[a]);
        if (bound_type_ >= 3) { prepare_tt_blocks(); return; }
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q < m; ++q) for (size_t c = 0; c < d; ++c) for (size_t i = begin; i < end; ++i) {
                double tail = 0;
                for (size_t j = l; j-- > 0;) {
                    const double increment = std::abs(increments.data[(j * d + c) * n + i]);
                    for (size_t a = 1; a + q <= m; ++a) {
                        double& out = tt_suffix_.data[((q * (m + 1) + a) * l + j) * n + i];
                        const double next = tt_suffix_.data[(((q - 1) * (m + 1) + a + 1) * l + j) * n + i];
                        out = std::max(out, siglasso::finite(increment * next + tail / weights[a]));
                    }
                    tail += increment * tt_suffix_.data[(((q - 1) * (m + 1) + 1) * l + j) * n + i];
                }
            }
        });
    }
    void tt_apply(const double* source, double* target, size_t channel, size_t width, size_t begin, size_t end, bool positive) {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        size_t i = begin;
        for (; i + lanes <= end; i += lanes) {
            auto tail = hn::Zero(tag);
            for (size_t j = l; j-- > 0;) {
                auto step = hn::LoadU(tag, increments.data + (j * d + channel) * n + i);
                if (positive) step = hn::Abs(step);
                for (size_t a = 1; a <= width; ++a) hn::StoreU(hn::MulAdd(step,
                    hn::LoadU(tag, source + ((a + 1) * l + j) * n + i), hn::Mul(tail, hn::Set(tag, 1 / weights[a]))),
                    tag, target + (a * l + j) * n + i);
                tail = hn::MulAdd(step, hn::LoadU(tag, source + (l + j) * n + i), tail);
            }
        }
        for (; i < end; ++i) {
            double tail = 0;
            for (size_t j = l; j-- > 0;) {
                double step = increments.data[(j * d + channel) * n + i];
                if (positive) step = std::abs(step);
                for (size_t a = 1; a <= width; ++a)
                    target[(a * l + j) * n + i] = step * source[((a + 1) * l + j) * n + i] + tail / weights[a];
                tail += step * source[(l + j) * n + i];
            }
        }
    }
    void tt_maximum(double* target, const double* source, size_t width, size_t begin, size_t end) {
        const size_t n = options.samples, l = options.length - 1;
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        for (size_t a = 1; a <= width; ++a) for (size_t j = 0; j < l; ++j) {
            const size_t offset = (a * l + j) * n;
            size_t i = begin;
            for (; i + lanes <= end; i += lanes) hn::StoreU(hn::Max(hn::LoadU(tag, target + offset + i),
                hn::Abs(hn::LoadU(tag, source + offset + i))), tag, target + offset + i);
            for (; i < end; ++i) target[offset + i] = std::max(target[offset + i], std::abs(source[offset + i]));
        }
    }
    void prepare_tt_blocks() {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        const size_t stride = siglasso::multiply(m + 1, siglasso::multiply(n, l));
        if (bound_type_ == 4) {
            size_t words = 1;
            for (size_t q = 1; q < m; ++q) {
                words = siglasso::multiply(words, d);
                if (words > 10000000) throw std::out_of_range("Exact TT suffix enumeration exceeds the experimental work limit");
            }
        }
        auto scratch = alloc(siglasso::multiply(bound_type_ == 4 ? m : 2, stride));
        batch(n, [&](uint64_t begin, uint64_t end) {
            if (bound_type_ == 4) {
                const auto visit = [&](auto&& self, const double* source, size_t q) -> void {
                    double* value = scratch.data + q * stride;
                    for (size_t c = 0; c < d; ++c) {
                        tt_apply(source, value, c, m - q, begin, end, false);
                        tt_maximum(tt_suffix_.data + q * stride, value, m - q, begin, end);
                        if (q + 1 < m) self(self, value, q + 1);
                    }
                };
                if (m > 1) visit(visit, tt_suffix_.data, 1);
            } else {
                for (size_t q = 1; q < m; ++q) for (size_t b = 0; b < d; ++b) {
                    const size_t block = std::min<size_t>(q, 2);
                    tt_apply(tt_suffix_.data + (q - block) * stride, scratch.data, b, m - q + block - 1, begin, end, true);
                    if (block == 1) tt_maximum(tt_suffix_.data + q * stride, scratch.data, m - q, begin, end);
                    else for (size_t a = 0; a < d; ++a) {
                        tt_apply(scratch.data, scratch.data + stride, a, m - q, begin, end, true);
                        tt_maximum(tt_suffix_.data + q * stride, scratch.data + stride, m - q, begin, end);
                    }
                }
            }
        });
    }
    void export_tt_suffix(double* output) { copy(tt_suffix_.data, output, tt_suffix_.size); }
    siglasso::Array tt_moments(const siglasso::Node& node) {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), m = options.degree;
        auto result = alloc(siglasso::multiply(m - k, siglasso::multiply(n, l)));
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        batch(l, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q <= m - k; ++q) for (size_t j = begin; j < end; ++j) {
                const double scale = weights[k + q] / n;
                size_t i = 0;
                for (; i + lanes <= n; i += lanes) {
                    auto value = hn::Mul(hn::Abs(hn::LoadU(tag, node.coefficients.data + j * (k + 1) * n + i)),
                        hn::LoadU(tag, tt_suffix_.data + (((q - 1) * (m + 1) + 1) * l + j) * n + i));
                    for (size_t a = 1; a <= k; ++a) value = hn::MulAdd(
                        hn::Abs(hn::LoadU(tag, node.tt_state.data + ((a - 1) * l + j) * n + i)),
                        hn::LoadU(tag, tt_suffix_.data + (((q - 1) * (m + 1) + a + 1) * l + j) * n + i), value);
                    hn::StoreU(hn::Mul(hn::Mul(value, hn::LoadU(tag, variation.data + j * n + i)), hn::Set(tag, scale)), tag,
                        result.data + ((q - 1) * l + j) * n + i);
                }
                for (; i < n; ++i) {
                    double value = std::abs(node.coefficients.data[j * (k + 1) * n + i])
                        * tt_suffix_.data[(((q - 1) * (m + 1) + 1) * l + j) * n + i];
                    for (size_t a = 1; a <= k; ++a) value += std::abs(node.tt_state.data[((a - 1) * l + j) * n + i])
                        * tt_suffix_.data[(((q - 1) * (m + 1) + a + 1) * l + j) * n + i];
                    result.data[((q - 1) * l + j) * n + i] = siglasso::finite(value * variation.data[j * n + i] * scale);
                }
            }
        });
        return result;
    }
    siglasso::Array factors(size_t k) override {
        if (bound_type_ == 1 || bound_type_ >= 3) return {};
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
    std::unique_ptr<siglasso::FeatureSearch> make_search() override {
        return feature_type_ ? siglasso::make_cpu_feature_search(*this, feature_type_) : Backend::make_search();
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
        if (bound_type_ == 1 || bound_type_ >= 3) node.moments = tt_moments(node);
        else node.moments = alloc(levels * n * l);
        if (bound_type_ == 2) node.tt_moments = tt_moments(node);
        node.integrals = alloc(n * l);
        if ((options.search_method & 2) && levels <= 2 && (!(options.traversal_order & 4) || levels == 2)) node.linear_integrals = alloc(n * l);
        batch(n, [&](uint64_t begin, uint64_t end) {
            auto scratch = bound_type_ == 1 || bound_type_ >= 3 ? siglasso::Array{} : alloc(siglasso::integral_scratch_size(options.degree));
            for (size_t j = 0; j < l; ++j) for (size_t i = begin; i < end; ++i) {
                    double integral = 0, linear = 0;
                    for (size_t a = 0; a <= k; ++a) {
                        const double c = node.coefficients.data[(j * (k + 1) + a) * n + i];
                        integral += c;
                        if (node.linear_integrals.owner) linear += (a + 1) * c;
                    }
                    node.integrals.data[j * n + i] = integral / (k + 1);
                    if (node.linear_integrals.owner) node.linear_integrals.data[j * n + i] = linear / ((k + 1) * (k + 2));
                    if (bound_type_ == 0 || bound_type_ == 2) siglasso::absolute_moments(node.coefficients.data, factors.data, variation.data[j * n + i], after.data[j * n + i],
                        k, levels, j, i, n, l, scratch.data, node.moments.data + j * n + i, (options.search_method & 2) && levels <= 2);
            }
        });
        node.uncertain_moments = std::any_of(node.moments.data, node.moments.data + node.moments.size, [](double value) { return value < 0; });
        if ((options.traversal_order & 4) && levels == 1) { node.coefficients = {}; node.tt_state = {}; }
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
    double complete_bound(const siglasso::Probe& task, double bound) override {
        const auto found = terminal_active_.find(task.node->word);
        if (found != terminal_active_.end()) for (const auto& level : found->second) for (const auto& suffix : level)
            if (suffix.second >= task.low && suffix.second < task.high) return std::max(bound, active_corr_cap);
        return bound;
    }
    double subtree_norm(const siglasso::Probe& task) override {
        const size_t n = options.samples, l = options.length - 1, k = task.node->word.size(), levels = options.degree - k;
        const double error = siglasso::terminal_roundoff(options.degree, n, l);
        double squared = 0;
        for (size_t i = 0; i < n; ++i) {
            const double v = l ? variation.data[i] + after.data[i] : 0;
            const double cap = std::pow(v, static_cast<double>(v > 1 ? options.degree : k + 1));
            double envelope = 0;
            if (task.node->uncertain_moments) envelope = cap;
            else for (size_t q = 0; q < levels; ++q) {
                double value = 0;
                for (size_t j = 0; j < l; ++j) value += task.node->moments.data[(q * l + j) * n + i]
                    * alphabet_ratios.data[(task.block * l + j) * n + i];
                envelope = std::max(envelope, value * n);
            }
            envelope = std::min(envelope, cap) + error * cap;
            squared += envelope * envelope;
        }
        return siglasso::guard(std::sqrt(squared));
    }
    void begin_query(const siglasso::Array& residual, const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active) override {
        if (!(options.search_method & 2)) return;
        const size_t n = options.samples, m = options.degree;
        if (options.summary_flags && options.summary_bytes && options.n_jobs == 1 && summaries_.empty()) {
            const size_t stored_samples = options.summary_flags & 4 ? 0 : 2 * n;
            const size_t entry_bytes = siglasso::multiply(stored_samples + m, sizeof(double)) + sizeof(Summary) + 64;
            const size_t available = std::min(options.summary_bytes, budget->available() / 8);
            size_t count = 1;
            while (count <= available / entry_bytes / 2) count *= 2;
            if (count * entry_bytes <= available) {
                auto values = stored_samples ? alloc(siglasso::multiply(stored_samples, count)) : siglasso::Array{};
                auto metadata = account(count * (sizeof(Summary) + m * sizeof(uint64_t) + 64));
                summaries_.resize(count);
                for (auto& entry : summaries_) entry.word.reserve(m);
                summary_values_ = std::move(values); summary_metadata_ = std::move(metadata);
            }
        }
        if (++query_generation_ == 0) {
            for (auto& entry : summaries_) entry.generation = 0;
            query_generation_ = 1;
        }
        terminal_active_ = decltype(terminal_active_){}; terminal_active_metadata_.reset();
        const size_t active_count = std::count_if(active.begin(), active.end(), [m](const auto& entry) { return entry.first.size() == m; });
        const size_t active_bytes = siglasso::multiply(active_count, 384 + 16 * m);
        if (!options.memory_limit && active_bytes <= options.cache_bytes / 8 && active_bytes <= budget->available() / 4) {
            terminal_active_metadata_ = account(active_bytes);
            for (const auto& entry : active) {
                const auto& word = entry.first;
                if (word.size() != m) continue;
                for (size_t q = 1; q <= 2 && q <= word.size(); ++q) {
                    const size_t k = word.size() - q;
                    siglasso::Word prefix(word.begin(), word.begin() + k);
                    terminal_active_[prefix][q - 1].emplace_back(q == 2 ? word.back() : 0, word[k]);
                }
            }
            for (auto& entry : terminal_active_) for (auto& suffixes : entry.second) std::sort(suffixes.begin(), suffixes.end());
        }
        terminal_caps_ = {};
        if (options.length > 1) for (size_t i = 0; i < n; ++i) {
            const double r = std::abs(residual.data[i]), v = variation.data[i] + after.data[i];
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
        terminal_storage_ = {}; terminal_products_ = {}; terminal_matrix_workspace_ = {}; terminal_cache_ = {};
        summaries_ = decltype(summaries_){}; summary_values_ = {}; summary_metadata_.reset(); terminal_operator_weights_ = {}; terminal_suffix_bounds_ = {};
        Backend::trim_caches();
    }
    void bounds(const std::vector<siglasso::Probe>& tasks, const siglasso::Array& residual, double threshold, std::vector<double>& out) override {
        if (options.terminal_kernel >= 9 || options.summary_flags) bounds_impl<true>(tasks, residual, threshold, out);
        else bounds_impl<false>(tasks, residual, threshold, out);
    }
    template<bool advanced>
    void bounds_impl(const std::vector<siglasso::Probe>& tasks, const siglasso::Array& residual, double threshold, std::vector<double>& out) {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const bool segmented = options.terminal_kernel == 6 || options.terminal_kernel == 8 || (advanced && options.terminal_kernel == 9);
        const bool lazy_segments = advanced && options.terminal_kernel == 9;
        // Cover underflow before the factorial scaling of a terminal norm bound.
        const double underflow = 32. * (n + 1) * (l + 1) * std::numeric_limits<double>::min();
        out.assign(tasks.size(), 0);
        std::atomic<uint64_t> terminal_queries{0}, terminal_pruned{0}, terminal_correlations{0};
        std::atomic<uint64_t> matrix_blocks{0}, matrix_correlations{0};
        std::atomic<uint64_t> segment_refinements{0};
        std::atomic<double> matrix_seconds{0};
        constexpr double residual_mean = 0;
        const double tt_error = bound_type_ ? siglasso::terminal_roundoff(options.degree, n, l) * absolute_sum(residual) / n : 0;
        batch(tasks.size(), [&](uint64_t begin, uint64_t end) {
            siglasso::Array terminal_products = options.n_jobs == 1 ? terminal_products_ : siglasso::Array{};
            siglasso::Array matrix_workspace = options.n_jobs == 1 ? terminal_matrix_workspace_ : siglasso::Array{};
            siglasso::Array operator_weights = advanced && options.n_jobs == 1 ? terminal_operator_weights_ : siglasso::Array{};
            siglasso::Array suffix_bounds = advanced && options.n_jobs == 1 ? terminal_suffix_bounds_ : siglasso::Array{};
            if (advanced && options.terminal_kernel >= 10 && !operator_weights.owner) {
                operator_weights = alloc(n);
                if (options.n_jobs == 1) terminal_operator_weights_ = operator_weights;
            }
            uint64_t queries = 0, pruned = 0, correlations = 0;
            uint64_t blocks = 0, blocked_correlations = 0;
            double blocked_seconds = 0;
            for (size_t t = begin; t < end; ++t) {
                const auto& task = tasks[t];
                const size_t levels = options.degree - task.node->word.size();
                double best = 0;
                if (task.node->uncertain_moments) { out[t] = threshold; continue; }
                for (size_t q = levels; q-- > 0;) {
                    const auto envelope = [&](const siglasso::Array& moments) {
                    double total = 0;
                    const hn::ScalableTag<double> tag;
                    const size_t lanes = hn::Lanes(tag);
                    auto accumulator = hn::Zero(tag);
                    for (size_t j = 0; j < l; ++j) {
                        size_t i = 0;
                        for (; i + lanes <= n; i += lanes) accumulator = hn::Add(accumulator, hn::Mul(
                            hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::LoadU(tag, moments.data + (q * l + j) * n + i)),
                            hn::LoadU(tag, alphabet_ratios.data + (task.block * l + j) * n + i)));
                        for (; i < n; ++i) total += std::abs(residual.data[i]) * moments.data[(q * l + j) * n + i]
                            * alphabet_ratios.data[(task.block * l + j) * n + i];
                    }
                    return siglasso::finite(total + hn::ReduceSum(tag, accumulator));
                    };
                    double total = envelope(task.node->moments);
                    if (bound_type_ == 1 || bound_type_ >= 3) total += tt_error;
                    if (bound_type_ == 2 && total >= threshold) total = std::min(total, envelope(task.node->tt_moments) + tt_error);
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
                    const auto active = terminal_active_.find(task.node->word);
                    for (size_t q = 1; q <= levels && complete; ++q) {
                        const auto* excluded = active == terminal_active_.end() ? nullptr : &active->second[q - 1];
                        size_t excluded_index = 0;
                        const double cap = terminal_caps_[k + q == options.degree];
                        const size_t product_size = siglasso::multiply(n, l + (advanced && options.terminal_kernel >= 13)) + (segmented ? l : 0);
                        const bool reuse = task.high - task.low >= (q == 2 ? 2 : 4)
                            && (terminal_products.owner || product_size * sizeof(double) <= budget->available());
                        if (reuse && !terminal_products.owner) {
                            terminal_products = alloc(product_size);
                            if (options.n_jobs == 1) terminal_products_ = terminal_products;
                        }
                        bool prepared = false;
                        double product_magnitude = 0, product_squared = 0, sample_bound = 0;
                        const size_t width = task.high - task.low;
                        const size_t block_width = options.terminal_kernel == 1 ? 8 : options.terminal_kernel == 3 ? 32 : 16;
                        const size_t matrix_size = siglasso::multiply(2 * n + width, block_width);
                        const size_t matrix_bytes = siglasso::multiply(matrix_size, sizeof(double));
                        const size_t packing_bytes = siglasso::multiply(siglasso::multiply(n, width + block_width), sizeof(double)) + 65536;
                        const bool blocked = options.terminal_kernel && options.terminal_kernel < 5 && q == 2 && reuse && width >= 8 && d >= 8
                            && matrix_bytes + packing_bytes <= budget->available();
                        if (blocked && matrix_workspace.size < matrix_size) {
                            matrix_workspace = alloc(matrix_size);
                            if (options.n_jobs == 1) terminal_matrix_workspace_ = matrix_workspace;
                        }
                        size_t matrix_first = 0, matrix_count = 0;
                        std::array<double, 32> matrix_magnitudes{};
                        double channel_norm = 0;
                        if (options.terminal_kernel >= 5 && reuse) for (size_t a = task.low; a < task.high; ++a)
                            channel_norm = std::max(channel_norm, terminal_channel_norms_.data[a]);
                        size_t tile_start = 0, tile_count = 0;
                        std::array<double, 4> tile_sums{};
                        const double rounding = reuse ? siglasso::terminal_roundoff(k, n * l, l) : error;
                        const bool operator_screen = options.terminal_kernel >= 7 && q == 2 && reuse;
                        double operator_norm = 0;
                        if (operator_screen) {
                            auto* entry = advanced && (options.summary_flags & 1) ? summary(task.node->word, 0, 0) : nullptr;
                            double* values = entry ? summary_values(entry) : nullptr;
                            if (entry && entry->generation == query_generation_ && !operator_weights.owner) operator_norm = entry->norm;
                            else {
                            const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                            auto maximum = hn::Zero(tag);
                            size_t i = 0;
                            for (; i + lanes <= n; i += lanes) {
                                auto squared = hn::Zero(tag);
                                if (entry && entry->ready) squared = hn::LoadU(tag, values + i);
                                else for (size_t j = 0; j < l; ++j) {
                                    const auto integral = hn::LoadU(tag, task.node->integrals.data + j * n + i);
                                    const auto moment = hn::LoadU(tag, task.node->linear_integrals.data + j * n + i);
                                    const auto a = hn::Abs(integral), b = hn::Abs(moment);
                                    const auto diagonal = advanced && options.terminal_kernel >= 12 ? hn::Sub(integral, moment) : hn::Add(a, b);
                                    squared = hn::MulAdd(diagonal, diagonal, squared);
                                    squared = hn::MulAdd(hn::Set(tag, static_cast<double>(l - j - 1)), hn::Mul(a, a), squared);
                                }
                                if (entry && !entry->ready) hn::StoreU(squared, tag, values + i);
                                const auto weighted = hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::Sqrt(hn::Add(squared, hn::Set(tag, underflow))));
                                if (operator_weights.owner) hn::StoreU(hn::Mul(weighted,
                                    hn::LoadU(tag, terminal_sample_norms_.data + (d + task.block) * n + i)), tag, operator_weights.data + i);
                                maximum = hn::Max(maximum, weighted);
                            }
                            operator_norm = hn::ReduceMax(tag, maximum);
                            for (; i < n; ++i) {
                                double squared = 0;
                                if (entry && entry->ready) squared = values[i];
                                else for (size_t j = 0; j < l; ++j) {
                                    const double a = std::abs(task.node->integrals.data[j * n + i]);
                                    const double b = std::abs(task.node->linear_integrals.data[j * n + i]);
                                    const double diagonal = advanced && options.terminal_kernel >= 12
                                        ? task.node->integrals.data[j * n + i] - task.node->linear_integrals.data[j * n + i] : a + b;
                                    squared += diagonal * diagonal + (l - j - 1) * a * a;
                                }
                                if (entry && !entry->ready) values[i] = squared;
                                const double weighted = std::abs(residual.data[i]) * std::sqrt(squared + underflow);
                                if (operator_weights.owner) operator_weights.data[i] = weighted * terminal_sample_norms_.data[(d + task.block) * n + i];
                                operator_norm = std::max(operator_norm, weighted);
                            }
                            operator_norm *= 1 + rounding;
                            if (entry) { entry->ready = true; entry->generation = query_generation_; entry->norm = operator_norm; }
                            }
                        }
                        const bool suffix_groups = advanced && options.terminal_kernel >= 15 && operator_screen;
                        if (suffix_groups) {
                            if (!suffix_bounds.owner) {
                                suffix_bounds = alloc(d);
                                if (options.n_jobs == 1) terminal_suffix_bounds_ = suffix_bounds;
                            }
                            std::fill_n(suffix_bounds.data, d, -1.);
                            std::function<void(size_t)> group = [&](size_t block) {
                                const auto [low, high] = alphabet[block];
                                if (high - low == 1) return;
                                double total = std::numeric_limits<double>::infinity();
                                try { total = positive_dot(operator_weights.data, terminal_sample_norms_.data + (d + block) * n, n); }
                                catch (const siglasso::NonfiniteError&) {}
                                const double bound = ((total + underflow) * (1 + rounding) * weights[k + q] + rounding * cap) / n + 1e-15;
                                if (std::isfinite(bound) && bound * (1 + 1e-10) < threshold)
                                    std::fill(suffix_bounds.data + low, suffix_bounds.data + high, bound);
                                else { group(block + 1); group(block + 2 * ((low + high) / 2 - low)); }
                            };
                            group(0);
                        }
                        for (size_t index = 0; index < width * (q == 2 ? d : 1); ++index) {
                            const size_t a = task.low + index % width, b = index / width;
                            const bool matrix_entry = blocked && (options.terminal_kernel != 4 || b >= 2);
                            if (a == task.low) prepared = false;
                            if (suffix_groups && a == task.low && suffix_bounds.data[b] >= 0) {
                                signed_bound = std::max(signed_bound, suffix_bounds.data[b]);
                                index = (b + 1) * width - 1;
                                continue;
                            }
                            if (operator_screen && a == task.low) {
                                double bound = ((operator_norm * channel_norm * terminal_channel_norms_.data[b] + underflow)
                                    * (1 + rounding) * weights[k + q] + rounding * cap) / n + 1e-15;
                                if (operator_weights.owner && bound * (1 + 1e-10) >= threshold) {
                                    double total = std::numeric_limits<double>::infinity();
                                    try { total = options.terminal_kernel >= 14
                                        ? positive_dot(operator_weights.data, terminal_sample_norms_.data + b * n, n)
                                        : dot(operator_weights.data, terminal_sample_norms_.data + b * n, n); }
                                    catch (const siglasso::NonfiniteError&) {}
                                    const double sample = ((total + underflow) * (1 + rounding) * weights[k + q] + rounding * cap) / n + 1e-15;
                                    bound = std::min(bound, sample);
                                }
                                if (std::isfinite(bound) && bound * (1 + 1e-10) < threshold) {
                                    signed_bound = std::max(signed_bound, bound);
                                    index = (b + 1) * width - 1;
                                    continue;
                                }
                            }
                            // Retain the cheap one-letter exit before a two-letter scan.
                            if (excluded && (q == 2 || levels == 1)) {
                                const std::pair<uint64_t, uint64_t> suffix{b, a};
                                while (excluded_index < excluded->size() && (*excluded)[excluded_index] < suffix) ++excluded_index;
                                if (excluded_index < excluded->size() && (*excluded)[excluded_index] == suffix) continue;
                            }
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
                            auto* energy = advanced && reuse && a == task.low && (options.summary_flags & 6) ? summary(task.node->word, b, q) : nullptr;
                            double* energy_values = energy && !(options.summary_flags & 4) ? summary_values(energy) : nullptr;
                            if (energy && energy->ready) {
                                if (energy->generation != query_generation_) {
                                    double squared = 0, magnitude = 0;
                                    for (size_t i = 0; i < n; ++i) {
                                        const double r = residual.data[i];
                                        squared += (r * energy_values[i]) * r;
                                        magnitude += std::abs(r) * energy_values[n + i];
                                    }
                                    energy->norm = std::sqrt(squared + underflow) * (1 + rounding);
                                    energy->magnitude = magnitude * (1 + rounding);
                                    energy->generation = query_generation_;
                                }
                                const double bound = ((channel_norm * energy->norm * (1 + rounding)
                                    + rounding * energy->magnitude) * weights[k + q] + rounding * cap) / n + 1e-15;
                                if (std::isfinite(bound) && bound * (1 + 1e-10) < threshold) {
                                    ++optimization_stats[siglasso::SummaryPruned];
                                    signed_bound = std::max(signed_bound, bound);
                                    index = (b + 1) * width - 1;
                                    continue;
                                }
                            }
                            if (reuse && !matrix_entry && !prepared) {
                                const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                                auto absolute = hn::Zero(tag);
                                auto squares = hn::Zero(tag);
                                auto sample_norm = hn::Zero(tag);
                                // Segment variation bounds every first-letter increment.
                                product_magnitude = 0; product_squared = 0; sample_bound = 0;
                                size_t i = 0;
                                for (; i + lanes <= n; i += lanes) {
                                    auto tail = hn::Zero(tag);
                                    auto energy_sum = hn::Zero(tag), energy_magnitude = hn::Zero(tag);
                                    for (size_t j = l; j-- > 0;) {
                                        const auto other = q == 2 ? hn::LoadU(tag, increments.data + (j * d + b) * n + i) : hn::Zero(tag);
                                        tail = hn::Add(tail, other);
                                        const auto integral = hn::LoadU(tag, task.node->integrals.data + j * n + i);
                                        const auto left = q == 2 ? hn::Mul(integral, tail) : integral;
                                        const auto right = q == 2 ? hn::Mul(hn::LoadU(tag, task.node->linear_integrals.data + j * n + i), other) : hn::Zero(tag);
                                        const auto r = hn::Sub(hn::LoadU(tag, residual.data + i), hn::Set(tag, residual_mean));
                                        const auto product = hn::Mul(r, hn::Sub(left, right));
                                        hn::StoreU(product, tag, terminal_products.data + j * n + i);
                                        if (options.terminal_kernel >= 5 && (!segmented || lazy_segments || (advanced && (options.summary_flags & 4)))) squares = hn::MulAdd(product, product, squares);
                                        if ((energy_values && !energy->ready) || (advanced && options.terminal_kernel >= 11)) {
                                            const auto h = hn::Sub(left, right);
                                            energy_sum = hn::MulAdd(h, h, energy_sum);
                                            if (energy_values) energy_magnitude = hn::MulAdd(hn::Add(hn::Abs(left), hn::Abs(right)),
                                                hn::LoadU(tag, variation.data + j * n + i), energy_magnitude);
                                        }
                                        absolute = hn::MulAdd(hn::Mul(hn::Abs(r), hn::Add(hn::Abs(left), hn::Abs(right))),
                                            hn::LoadU(tag, variation.data + j * n + i), absolute);
                                    }
                                    if (energy_values && !energy->ready) {
                                        hn::StoreU(hn::Add(energy_sum, hn::Set(tag, underflow)), tag, energy_values + i);
                                        hn::StoreU(energy_magnitude, tag, energy_values + n + i);
                                    }
                                    if (advanced && options.terminal_kernel >= 11) {
                                        const auto weighted = hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::Sqrt(hn::Add(energy_sum, hn::Set(tag, underflow))));
                                        sample_norm = hn::MulAdd(weighted, hn::LoadU(tag, terminal_sample_norms_.data + (d + task.block) * n + i), sample_norm);
                                        if (options.terminal_kernel >= 13) hn::StoreU(weighted, tag, terminal_products.data + n * l + i);
                                    }
                                }
                                for (; i < n; ++i) {
                                    double tail = 0;
                                    double energy_sum = 0, energy_magnitude = 0;
                                    for (size_t j = l; j-- > 0;) {
                                        const double other = q == 2 ? increments.data[(j * d + b) * n + i] : 0; tail += other;
                                        const double integral = task.node->integrals.data[j * n + i];
                                        const double left = q == 2 ? integral * tail : integral;
                                        const double right = q == 2 ? task.node->linear_integrals.data[j * n + i] * other : 0;
                                        const double r = residual.data[i] - residual_mean;
                                        terminal_products.data[j * n + i] = r * (left - right);
                                        if (options.terminal_kernel >= 5 && (!segmented || lazy_segments || (advanced && (options.summary_flags & 4))))
                                            product_squared += terminal_products.data[j * n + i] * terminal_products.data[j * n + i];
                                        if ((energy_values && !energy->ready) || (advanced && options.terminal_kernel >= 11)) {
                                            energy_sum += (left - right) * (left - right);
                                            if (energy_values) energy_magnitude += (std::abs(left) + std::abs(right)) * variation.data[j * n + i];
                                        }
                                        product_magnitude += std::abs(r) * (std::abs(left) + std::abs(right)) * variation.data[j * n + i];
                                    }
                                    if (energy_values && !energy->ready) { energy_values[i] = energy_sum + underflow; energy_values[n + i] = energy_magnitude; }
                                    if (advanced && options.terminal_kernel >= 11) {
                                        const double weighted = std::abs(residual.data[i]) * std::sqrt(energy_sum + underflow);
                                        sample_bound += weighted * terminal_sample_norms_.data[(d + task.block) * n + i];
                                        if (options.terminal_kernel >= 13) terminal_products.data[n * l + i] = weighted;
                                    }
                                }
                                if (energy) energy->ready = true;
                                product_magnitude = (product_magnitude + hn::ReduceSum(tag, absolute)) * (1 + rounding);
                                product_squared += hn::ReduceSum(tag, squares);
                                if constexpr (advanced) sample_bound += hn::ReduceSum(tag, sample_norm);
                                if (energy && (options.summary_flags & 4)) {
                                    energy->norm = std::sqrt(product_squared + underflow) * (1 + rounding);
                                    energy->magnitude = product_magnitude; energy->generation = query_generation_;
                                }
                                prepared = true;
                                if (options.terminal_kernel >= 5) {
                                    double squared = std::numeric_limits<double>::infinity();
                                    double* segment_norms = terminal_products.at(n * l);
                                    try {
                                        if (segmented && !lazy_segments) {
                                            squared = 0;
                                            for (size_t j = 0; j < l; ++j) {
                                                const double part = dot(terminal_products.at(j * n), terminal_products.at(j * n), n) + underflow;
                                                segment_norms[j] = std::sqrt(part) * (1 + rounding);
                                                squared += part;
                                            }
                                        } else squared = product_squared + underflow;
                                    }
                                    catch (const siglasso::NonfiniteError&) { squared = std::numeric_limits<double>::infinity(); }
                                    double product_bound = channel_norm * std::sqrt(squared) * (1 + rounding);
                                    if (advanced && options.terminal_kernel >= 11) product_bound = std::min(product_bound, (sample_bound + underflow) * (1 + rounding));
                                    if (segmented && std::isfinite(squared)
                                        && product_bound * weights[k + q] / n >= threshold) {
                                        if (lazy_segments) {
                                            ++segment_refinements;
                                            for (size_t j = 0; j < l; ++j) {
                                                double part = std::numeric_limits<double>::infinity();
                                                try { part = dot(terminal_products.at(j * n), terminal_products.at(j * n), n) + underflow; }
                                                catch (const siglasso::NonfiniteError&) {}
                                                segment_norms[j] = std::sqrt(part) * (1 + rounding);
                                            }
                                        }
                                        double maximum = 0;
                                        for (size_t a = task.low; a < task.high; ++a) {
                                            double sum = 0;
                                            for (size_t j = 0; j < l; ++j)
                                                sum += terminal_channel_norms_.data[(j + 1) * d + a] * segment_norms[j];
                                            maximum = std::max(maximum, sum);
                                        }
                                        product_bound = std::min(product_bound, maximum * (1 + rounding));
                                    }
                                    const double norm_bound = ((product_bound
                                        + rounding * product_magnitude) * weights[k + q] + rounding * cap) / n + 1e-15;
                                    if (std::isfinite(norm_bound) && norm_bound * (1 + 1e-10) < threshold) {
                                        signed_bound = std::max(signed_bound, norm_bound);
                                        index = (b + 1) * width - 1;
                                        continue;
                                    }
                                }
                            }
                            if (advanced && options.terminal_kernel >= 13 && reuse && prepared) {
                                double norm = std::numeric_limits<double>::infinity();
                                try { norm = options.terminal_kernel >= 14
                                    ? positive_dot(terminal_products.data + n * l, terminal_sample_norms_.data + a * n, n)
                                    : dot(terminal_products.data + n * l, terminal_sample_norms_.data + a * n, n); }
                                catch (const siglasso::NonfiniteError&) {}
                                const double bound = (((norm + underflow) * (1 + rounding) + rounding * product_magnitude)
                                    * weights[k + q] + rounding * cap) / n + 1e-15;
                                if (std::isfinite(bound) && bound * (1 + 1e-10) < threshold) {
                                    signed_bound = std::max(signed_bound, bound);
                                    continue;
                                }
                            }
                            double sum = 0, correction = 0, magnitude = 0;
                            const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
                            auto sums = hn::Zero(tag), corrections = hn::Zero(tag), magnitudes = hn::Zero(tag);
                            size_t i = 0;
                            if (matrix_entry) {
                                if (b >= matrix_first + matrix_count) {
                                    matrix_first = b; matrix_count = std::min(block_width, d - b);
                                    const auto started = siglasso::Clock::now();
                                    terminal_matrix(task, residual, b, matrix_count, matrix_workspace, matrix_magnitudes.data());
                                    blocked_seconds += siglasso::elapsed(started);
                                    ++blocks;
                                    blocked_correlations += width * matrix_count;
                                    correlations += width * matrix_count;
                                }
                                sum = matrix_workspace.data[2 * n * matrix_count + (b - matrix_first) * width + a - task.low];
                                magnitude = matrix_magnitudes[b - matrix_first] * (1 + rounding);
                            } else if (reuse) {
                                if (index >= tile_start + tile_count) {
                                    tile_start = index; tile_count = std::min<size_t>(advanced && options.terminal_kernel >= 13 ? 1 : 4, task.high - a);
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
            matrix_blocks += blocks; matrix_correlations += blocked_correlations; matrix_seconds += blocked_seconds;
        });
        stats.terminal_queries += terminal_queries; stats.terminal_pruned += terminal_pruned;
        stats.terminal_correlations += terminal_correlations;
        optimization_stats[siglasso::MatrixBlocks] += matrix_blocks;
        optimization_stats[siglasso::MatrixCorrelations] += matrix_correlations;
        optimization_stats[siglasso::MatrixSeconds] += matrix_seconds;
        optimization_stats[siglasso::SegmentRefinements] += segment_refinements;
    }
    void children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        const size_t n = options.samples, l = options.length - 1;
        out.resize(tasks.size());
        for (size_t t = 0; t < tasks.size(); ++t) {
            auto node = std::make_shared<siglasso::Node>(); node->word = tasks[t].node->word; node->word.push_back(tasks[t].low);
            node->metadata = account(512 + node->word.size() * 64);
            node->values = alloc(n);
            if (node->word.size() < options.degree) node->coefficients = alloc((node->word.size() + 1) * n * l);
            if (bound_type_ && node->word.size() < options.degree) node->tt_state = alloc(node->word.size() * n * l);
            out[t] = std::move(node);
        }
        batch(tasks.size() * n, [&](uint64_t begin, uint64_t end) {
            for (size_t index = begin; index < end; ++index) {
                const size_t t = index / n, i = index % n;
                siglasso::child_sample(increments.data, tasks[t].node->coefficients.data, tasks[t].node->integrals.data,
                    out[t]->coefficients.data, out[t]->values.data, tasks[t].low, tasks[t].node->word.size(), i, n, l, options.dimension);
                siglasso::finite(out[t]->values.data[i]);
            }
        });
        if (bound_type_) {
            const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
            batch(tasks.size(), [&](uint64_t begin, uint64_t end) {
                for (size_t t = begin; t < end; ++t) {
                    if (!out[t]->tt_state.owner) continue;
                    const auto& parent = *tasks[t].node;
                    const size_t k = parent.word.size();
                    for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j) {
                        const double* source = a ? parent.tt_state.data + ((a - 1) * l + j) * n : parent.coefficients.data + j * (k + 1) * n;
                        const double* increment = increments.data + (j * options.dimension + tasks[t].low) * n;
                        double* target = out[t]->tt_state.data + (a * l + j) * n;
                        size_t i = 0;
                        for (; i + lanes <= n; i += lanes) hn::StoreU(hn::Mul(hn::LoadU(tag, source + i), hn::LoadU(tag, increment + i)), tag, target + i);
                        for (; i < n; ++i) target[i] = source[i] * increment[i];
                    }
                }
            });
        }
    }
    siglasso::Array response(size_t target) override {
        auto result = alloc(options.samples);
        for (size_t i = 0; i < options.samples; ++i) result.data[i] = responses_.data[i * options.targets + target];
        return result;
    }
    template<typename T> void export_typed(const siglasso::Result& result, const SigLassoOutput& output) {
        const auto convert = [](double value) { return static_cast<T>(siglasso::finite(static_cast<T>(value))); };
        const std::vector<double>* inputs[] = {&result.coefficients, &result.intercepts, &result.objectives, &result.gaps, &result.kkt};
        void* outputs[] = {output.coefficients, output.intercepts, output.objectives, output.gaps, output.kkt};
        for (size_t j = 0; j < 5; ++j) {
            std::transform(inputs[j]->begin(), inputs[j]->end(), static_cast<T*>(outputs[j]), [&](double value) {
                T output = convert(value);
                if (j >= 3 && output < value) output = std::nextafter(output, std::numeric_limits<T>::infinity());
                return output;
            });
        }
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
    void query_benchmark(const double* residuals, const double* thresholds, size_t response_batch, double* scores) {
        const size_t targets = options.targets, n = options.samples, d = options.dimension;
        const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash> active;
        if (!response_batch) {
            auto search = make_search();
            for (size_t t = 0; t < targets; ++t) {
                auto residual = alloc(n); copy(residuals + t * n, residual.data, n);
                auto found = search->query(residual, thresholds[t], active, 16);
                scores[2 * t] = found.candidates.empty() ? 0 : found.candidates.front().score;
                scores[2 * t + 1] = found.omitted;
            }
            return;
        }
        std::vector<size_t> frontier{0};
        const size_t threads = options.n_jobs > 0 ? options.n_jobs : get_max_threads() + 1 + options.n_jobs;
        const size_t parts = std::min<size_t>(d, threads > 1 ? 4 * threads : 1);
        while (frontier.size() < parts) {
            auto largest = std::max_element(frontier.begin(), frontier.end(), [&](size_t a, size_t b) {
                return alphabet[a].second - alphabet[a].first < alphabet[b].second - alphabet[b].first;
            });
            const auto [low, high] = alphabet[*largest];
            const size_t block = *largest;
            *largest = block + 1; frontier.push_back(block + 2 * ((low + high) / 2 - low));
        }
        std::sort(frontier.begin(), frontier.end());
        auto single = options; single.n_jobs = 1; single.summary_bytes /= parts;
        siglasso::Search root_search(*this);
        auto metadata = account(siglasso::multiply(parts * targets, 2048 + 16 * (256 + 16 * options.degree)));
        std::vector<std::vector<siglasso::SearchResult>> results(parts, std::vector<siglasso::SearchResult>(targets));
        std::vector<SigLassoStats> worker_stats(parts);
        std::atomic<size_t> next{0};
        batch(parts, [&](uint64_t, uint64_t) {
            for (size_t p = next.fetch_add(1); p < parts; p = next.fetch_add(1)) {
                std::vector<siglasso::Array> factors(options.degree);
                for (size_t first = 0; first < targets; first += response_batch) {
                    const size_t count = std::min(response_batch, targets - first);
                    std::vector<std::unique_ptr<LassoCPU>> workers;
                    std::vector<siglasso::Array> gradients;
                    for (size_t t = 0; t < count; ++t) {
                        auto worker = std::make_unique<LassoCPU>(*this, single);
                        worker->increments = increments; worker->variation = variation; worker->after = after;
                        worker->alphabet = alphabet; worker->alphabet_ratios = alphabet_ratios;
                        auto gradient = alloc(n); copy(residuals + (first + t) * n, gradient.data, n);
                        worker->begin_query(gradient, active);
                        gradients.push_back(std::move(gradient)); workers.push_back(std::move(worker));
                    }
                    auto& maker = *workers.front();
                    struct Task { siglasso::Probe probe; uint64_t mask; };
                    std::vector<Task> stack{{{root_search.root(), alphabet[frontier[p]].first, alphabet[frontier[p]].second, frontier[p], {}},
                        (uint64_t{1} << count) - 1}};
                    while (!stack.empty()) {
                        auto task = std::move(stack.back()); stack.pop_back();
                        const auto& probe = task.probe;
                        const size_t k = probe.node->word.size();
                        uint64_t mask = 0;
                        for (size_t t = 0; t < count; ++t) if (task.mask & (uint64_t{1} << t)) {
                            auto& result = results[p][first + t];
                            double threshold = thresholds[first + t];
                            if (result.candidates.size() == 16) threshold = std::max(threshold, result.candidates.front().score);
                            std::vector<double> bound;
                            workers[t]->bounds({probe}, gradients[t], threshold, bound);
                            ++workers[t]->stats.block_queries;
                            const double estimate = bound.front() * (1 + 1e-10);
                            if (std::isfinite(estimate) && estimate >= 0 && siglasso::guard(estimate) < threshold) {
                                result.omitted = std::max(result.omitted, siglasso::guard(estimate));
                                ++workers[t]->stats.pruned;
                            } else mask |= uint64_t{1} << t;
                        }
                        if (!mask) continue;
                        if (probe.high - probe.low > 1) {
                            const size_t middle = (probe.low + probe.high) / 2;
                            stack.push_back({{probe.node, middle, probe.high, probe.block + 2 * (middle - probe.low), {}}, mask});
                            stack.push_back({{probe.node, probe.low, middle, probe.block + 1, {}}, mask});
                            continue;
                        }
                        std::vector<siglasso::NodePtr> children;
                        maker.children({probe}, children); ++maker.stats.constructed;
                        auto node = children.front();
                        if (k + 1 < options.degree) {
                            if (!factors[k + 1].owner) factors[k + 1] = maker.factors(k + 1);
                            maker.moments(*node, factors[k + 1]);
                        }
                        for (size_t t = 0; t < count; ++t) if (mask & (uint64_t{1} << t)) {
                            auto& result = results[p][first + t];
                            const double score = std::abs(maker.dot(node->values.data, gradients[t].data, n)) * weights[k + 1] / n;
                            result.omitted = std::max(result.omitted, siglasso::guard(score));
                            if (score > thresholds[first + t]) {
                                siglasso::Candidate candidate{node->word, node->values, score};
                                if (result.candidates.size() < 16) {
                                    result.candidates.push_back(std::move(candidate));
                                    std::push_heap(result.candidates.begin(), result.candidates.end(), siglasso::better);
                                } else if (siglasso::better(candidate, result.candidates.front())) {
                                    std::pop_heap(result.candidates.begin(), result.candidates.end(), siglasso::better);
                                    result.candidates.back() = std::move(candidate);
                                    std::push_heap(result.candidates.begin(), result.candidates.end(), siglasso::better);
                                }
                            }
                        }
                        if (k + 1 < options.degree) stack.push_back({{node, 0, d, 0, {}}, mask});
                    }
                    for (auto& worker : workers) {
                        auto& out = worker_stats[p];
                        out.constructed += worker->stats.constructed; out.block_queries += worker->stats.block_queries;
                        out.terminal_correlations += worker->stats.terminal_correlations; out.pruned += worker->stats.pruned;
                    }
                }
            }
        });
        for (size_t p = 0; p < parts; ++p) {
            stats.constructed += worker_stats[p].constructed; stats.block_queries += worker_stats[p].block_queries;
            stats.terminal_correlations += worker_stats[p].terminal_correlations; stats.pruned += worker_stats[p].pruned;
            for (size_t t = 0; t < targets; ++t) {
                scores[2 * t + 1] = std::max(scores[2 * t + 1], results[p][t].omitted);
                for (const auto& c : results[p][t].candidates) scores[2 * t] = std::max(scores[2 * t], c.score);
            }
        }
    }
    bool has_native_search() const override { return options.n_jobs != 1 && options.dimension > 1 && !options.memory_limit; }
    void native_new_target() override { for (auto& search : searches_) search->new_target(); }
    void native_screen(double scale, double radius, double stamp, double alpha) override {
        for (size_t j = 0; j < searches_.size(); ++j) {
            search_workers_[j]->experiment_stats = {};
            searches_[j]->screen(scale, radius, stamp, alpha);
            for (size_t k = 0; k < experiment_stats.size(); ++k) experiment_stats[k] += search_workers_[j]->experiment_stats[k];
        }
    }
    bool native_search(const siglasso::Array& residual, double alpha,
        const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active, size_t capacity, siglasso::SearchResult& result) override {
        if (searches_.empty()) {
            const size_t threads = options.n_jobs > 0 ? options.n_jobs : get_max_threads() + 1 + options.n_jobs;
            const size_t parts = options.partitions ? options.partitions : (threads > 1 ? 4 * threads : threads);
            if (!(options.search_method & 1)) {
                const size_t count = options.partitions ? std::min<size_t>(parts, options.dimension) : options.dimension;
                frontier_.resize(count);
                for (size_t j = 0; j < count; ++j) frontier_[j] = j * options.dimension / count;
            } else frontier_.push_back(0);
            while ((options.search_method & 1) && frontier_.size() < std::min(parts, static_cast<size_t>(options.dimension))) {
                auto largest = std::max_element(frontier_.begin(), frontier_.end(), [&](size_t a, size_t b) {
                    return alphabet[a].second - alphabet[a].first < alphabet[b].second - alphabet[b].first;
                });
                const auto [low, high] = alphabet[*largest];
                const size_t block = *largest;
                *largest = block + 1; frontier_.push_back(block + 2 * ((low + high) / 2 - low));
            }
            std::sort(frontier_.begin(), frontier_.end());
            shared_factors_.resize(options.degree);
            auto single = options; single.n_jobs = 1; single.cache_bytes /= frontier_.size(); single.summary_bytes /= frontier_.size();
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
        std::shared_ptr<siglasso::CandidateCutoff> cutoff;
        if ((options.traversal_order & 2) && capacity && searches_.size() > 1) {
            const size_t rank = options.pool_rank ? std::min<size_t>(capacity, options.pool_rank) : capacity;
            cutoff = std::make_shared<siglasso::CandidateCutoff>(rank, account(256 + siglasso::multiply(rank, sizeof(double))));
        }
        std::atomic<size_t> next{0};
        spawn_batch_threads(searches_.size(), options.n_jobs, [&](uint64_t, uint64_t) {
            for (size_t j = next.fetch_add(1); j < searches_.size(); j = next.fetch_add(1)) {
                const size_t low = options.search_method & 1 ? alphabet[frontier_[j]].first : frontier_[j];
                const size_t high = options.search_method & 1 ? alphabet[frontier_[j]].second
                    : (j + 1 < frontier_.size() ? frontier_[j + 1] : options.dimension);
                search_workers_[j]->stats = {};
                search_workers_[j]->experiment_stats = {};
                search_workers_[j]->optimization_stats = {};
                search_workers_[j]->active_corr_cap = active_corr_cap;
                search_workers_[j]->verify_search = verify_search;
                search_workers_[j]->shared_cutoff = cutoff;
                found[j] = searches_[j]->query(residual, alpha, active, capacity, low, high, frontier_[j]);
                search_workers_[j]->shared_cutoff.reset();
            }
        });
        for (size_t j = 0; j < searches_.size(); ++j) {
            const auto& worker = *search_workers_[j];
            for (size_t k = 0; k < experiment_stats.size(); ++k) experiment_stats[k] += worker.experiment_stats[k];
            for (size_t k = 0; k < optimization_stats.size(); ++k) optimization_stats[k] += worker.optimization_stats[k];
            stats.constructed += worker.stats.constructed;
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
};

template<typename T>
void sig_working_set_query_benchmark_(const T* path, const double* residuals, const double* thresholds,
    uint32_t response_batch, const SigLassoOptions* options, double* scores, double* diagnostics) {
    if (!path || !residuals || !thresholds || !options || !scores || !diagnostics || !options->samples
        || options->length < 2 || !options->dimension || !options->degree || !options->targets || response_batch > 32)
        throw std::invalid_argument("Invalid shared query benchmark input");
    auto settings = siglasso::SearchOptions(*options, 0); settings.search_method = 3;
    LassoCPU backend(settings, path, static_cast<const T*>(nullptr));
    backend.prepare();
    diagnostics[0] = siglasso::elapsed(backend.started);
    std::fill_n(scores, 2 * options->targets, 0.);
    const auto started = siglasso::Clock::now();
    backend.query_benchmark(residuals, thresholds, response_batch, scores);
    diagnostics[1] = siglasso::elapsed(started); diagnostics[2] = static_cast<double>(backend.budget->peak());
    diagnostics[3] = static_cast<double>(backend.stats.constructed); diagnostics[4] = static_cast<double>(backend.stats.terminal_correlations);
    diagnostics[5] = static_cast<double>(backend.stats.block_queries); diagnostics[6] = static_cast<double>(backend.stats.pruned);
}

template<typename T>
void sig_working_set_tt_envelopes_(const T* path, const SigLassoOptions* options, double* output) {
    if (!path || !options || !output || !options->degree || !options->samples || !options->length || !options->dimension)
        throw std::invalid_argument("Invalid TT envelope audit input");
    LassoCPU backend(siglasso::SearchOptions(*options, 0), path, static_cast<const T*>(nullptr));
    backend.prepare(); backend.export_tt_suffix(output);
}

template<typename T>
void sig_working_set_(const T* path, const T* y, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) {
    if (!options || !path || !y || !alphas || !allocate || (words && (!letters || !offsets)))
        throw std::invalid_argument("Missing working-set input or output allocator");
    if (!options->length || !options->samples || !options->dimension || !options->targets || !options->degree ||
        !options->max_iter || !std::isfinite(options->tolerance) || options->tolerance <= 0 || options->loss > 3)
        throw std::invalid_argument("Invalid working-set size, loss or tolerance");
    for (size_t t = 0; t < options->targets; ++t)
        if (!std::isfinite(alphas[t]) || alphas[t] <= 0) throw std::invalid_argument("Expected a positive finite penalty");
    for (size_t j = 0; j < words; ++j) {
        if (offsets[j+1] <= offsets[j] || offsets[j+1]-offsets[j] > options->degree)
            throw std::invalid_argument("Invalid initial word length");
        for (size_t k = offsets[j]; k < offsets[j+1]; ++k)
            if (letters[k] >= options->dimension) throw std::invalid_argument("Invalid initial word letter");
    }
    if (feature_type > 2) throw std::invalid_argument("Invalid Lasso feature type");
    LassoCPU backend(siglasso::SearchOptions(*options, feature_type), path, y, feature_type);
    std::vector<siglasso::Word> initial;
    for (size_t j = 0; j < words; ++j) initial.emplace_back(letters + offsets[j], letters + offsets[j + 1]);
    auto result = siglasso::fit(backend, feature_type, std::vector<double>(alphas, alphas + options->targets), initial);
    siglasso::experimental_last_stats = backend.experiment_stats;
    siglasso::experimental_optimization_stats = backend.optimization_stats;
    siglasso::export_result(backend, result, allocate, context);
}
template<typename T>
void sig_working_set_predict_(const T* path, const T* coefficients, const T* intercept, T* output,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options) {
    if (feature_type > 2) throw std::invalid_argument("Invalid Lasso feature type");
    LassoCPU backend(siglasso::SearchOptions(*options, feature_type, true), path, static_cast<const T*>(nullptr), feature_type);
    backend.prepare(); auto search = feature_type ? siglasso::make_cpu_feature_search(backend, feature_type, true) : backend.make_search();
    auto prediction = backend.alloc(siglasso::multiply(options->samples, options->targets));
    for (size_t i = 0; i < options->samples; ++i) for (size_t t = 0; t < options->targets; ++t) prediction.data[i * options->targets + t] = siglasso::finite(intercept[t]);
    for (size_t j = 0; j < words; ++j) {
        const siglasso::Word word(letters + offsets[j], letters + offsets[j + 1]);
        const auto node = search->selected(word);
        for (size_t i = 0; i < options->samples; ++i) for (size_t t = 0; t < options->targets; ++t)
            prediction.data[i * options->targets + t] += node->values.data[i] * backend.weights[word.size()] * siglasso::finite(coefficients[j * options->targets + t]);
    }
    for (size_t i = 0; i < prediction.size; ++i) output[i] = static_cast<T>(siglasso::finite(prediction.data[i]));
}
template void sig_working_set_<float>(const float*, const float*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_tt_envelopes_<double>(const double*, const SigLassoOptions*, double*);
template void sig_working_set_query_benchmark_<double>(const double*, const double*, const double*, uint32_t, const SigLassoOptions*, double*, double*);
template void sig_working_set_<double>(const double*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_predict_<float>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);
template void sig_working_set_predict_<double>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);

} } // namespace cpsig::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
