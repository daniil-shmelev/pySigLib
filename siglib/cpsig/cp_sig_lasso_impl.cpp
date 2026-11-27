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
#include "multithreading.h"
#include <bit>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_sig_lasso_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
using siglasso::SignatureOrder;
using siglasso::TerminalBound;

class LassoCPU final : public siglasso::Backend {
    const bool log_features_;
    bool terminal_only_ = false;
    size_t proposal_degree = 0;
    siglasso::Array tt_suffix_;
    siglasso::Array responses_, endpoint_variation_;
    const void* endpoint_input_ = nullptr;
    bool endpoint_float_ = false;
    const void* original_paths_ = nullptr;
    bool input_float_ = false;
    siglasso::Array normalization_;
    bool joint_search_enabled() const {
        const auto& o = options;
        return !o.selected_only && !o.memory_limit && !o.loss && !o.single_precision &&
            o.targets >= 2 && o.dimension >= 2 && o.degree >= 2 && o.degree <= 4;
    }
    bool channels_ready_ = false;
    std::vector<siglasso::Array> channels_;
    std::vector<size_t> ordered_letters_, rank_of_letter_;
    std::vector<bool> pinned_;
    struct ColdChannels {
        std::mutex mutex;
        size_t first = SIZE_MAX;
        std::array<siglasso::Array, 2> values;
    };
    std::array<ColdChannels, 8> cold_;
    size_t cold_shards_ = 1;
    siglasso::Array terminal_workspace_, terminal_child_storage_;
    static double normalized_position(double delta, double total, double reciprocal) {
        // A subnormal variation can have an infinite reciprocal.
        return siglasso::finite(std::isfinite(reciprocal) ? delta * reciprocal : delta / total);
    }
    template<bool adjacent, bool reverse, size_t width, class T>
    void load_channels(const T* paths, const std::array<size_t, width>& letters, const std::array<double*, width>& output) const {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        for (size_t i = 0; i < n; ++i) {
            const T* x = paths + i * (l + 1) * d + (adjacent ? letters[0] : 0);
            const double total = normalization_.data[i];
            const double reciprocal = total ? 1. / total : 0.;
            std::array<double, width> previous{};
            for (size_t j = 0; j < l; ++j) for (size_t c = 0; c < width; ++c) {
                const size_t a = adjacent ? c : letters[c];
                const double delta = double(x[(j + 1) * d + a]) - x[a];
                const double position = normalized_position(delta, total, reciprocal);
                const size_t segment = reverse ? l - 1 - j : j;
                output[c][segment * n + i] = position - previous[c];
                previous[c] = position;
            }
        }
    }
    template<size_t width, bool adjacent = true>
    void read_channels(const std::array<size_t, width>& letters, const std::array<double*, width>& output) const {
        const auto read = [&](const auto* paths) {
            if (options.order == SignatureOrder::suffix) load_channels<adjacent, true>(paths, letters, output);
            else load_channels<adjacent, false>(paths, letters, output);
        };
        if (input_float_) read(static_cast<const float*>(original_paths_));
        else read(static_cast<const double*>(original_paths_));
    }
    template<class T>
    static double step_variation(const T* previous, const T* next, size_t dimension) {
        double maximum = 0.;
        size_t a = 0;
        if constexpr (std::is_same_v<T, double>) {
            const hn::ScalableTag<double> tag;
            const size_t lanes = hn::Lanes(tag);
            auto largest = hn::Zero(tag);
            auto valid = hn::Eq(largest, largest);
            for (; a + lanes <= dimension; a += lanes) {
                const auto delta = hn::Sub(hn::LoadU(tag, next + a), hn::LoadU(tag, previous + a));
                valid = hn::And(valid, hn::IsFinite(delta));
                largest = hn::Max(largest, hn::Abs(delta));
            }
            if (hn::AllTrue(tag, valid)) maximum = hn::ReduceMax(tag, largest);
            else a = 0; // Distinguish a nonfinite input from finite inputs whose difference overflows.
        }
        for (; a < dimension; ++a) {
            if (!std::isfinite(next[a])) throw std::invalid_argument("Paths must be finite");
            maximum = std::max(maximum, std::abs(siglasso::finite(double(next[a]) - previous[a])));
        }
        return maximum;
    }
    siglasso::Channel channel(size_t a) override {
        if (channel_source_) return channel_source_->channel(a);
        if (!channels_ready_) return Backend::channel(a);
        if (pinned_[a]) return {channels_[a], 1};
        return rank_of_letter_.empty() ? cold_channel<false>(a) : cold_channel<true>(a);
    }
    template<bool permuted>
    siglasso::Channel cold_channel(size_t a) {
        // Terminal loops visit neighbouring letters. Decode the pair in one
        // pass through the caller's sample-major data and recycle idle buffers.
        const size_t rank = permuted ? rank_of_letter_[a] : a, first = rank - rank % 2;
        auto& shard = cold_[(first / 2) % cold_shards_];
        std::lock_guard lock(shard.mutex);
        if (shard.first != first) {
            const size_t count = std::min<size_t>(2, options.dimension - first);
            const auto source = [&](size_t j) { return permuted ? ordered_letters_[first + j] : first + j; };
            for (size_t j = 0; j < count; ++j) if (!pinned_[source(j)]) {
                auto& buffer = shard.values[j];
                if (buffer.owner.use_count() != 1) buffer = alloc(options.samples * (options.length - 1));
            }
            if (count == 2 && !pinned_[source(0)] && !pinned_[source(1)])
                read_channels<2, !permuted>({source(0), source(1)}, {shard.values[0].data, shard.values[1].data});
            else read_channels<1>({a}, {shard.values[rank - first].data});
            shard.first = first;
        }
        return {shard.values[rank - first], 1};
    }
    template<class T>
    void channel_geometry(const T* paths, double* energy, double* previous, size_t begin, size_t end) {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        std::fill_n(energy, d, 0.);
        for (size_t i = begin; i < end; ++i) {
            const T* origin = paths + i * (l + 1) * d;
            const double total = normalization_.data[i], reciprocal = total ? 1. / total : 0.;
            std::fill_n(previous, d, 0.);
            for (size_t j = 0; j < l; ++j) {
                const T* next = origin + (j + 1) * d;
                double maximum = 0.;
                for (size_t a = 0; a < d; ++a) {
                    const double delta = double(next[a]) - origin[a];
                    const double position = normalized_position(delta, total, reciprocal);
                    const double step = position - previous[a];
                    previous[a] = position;
                    maximum = std::max(maximum, std::abs(step));
                    energy[a] += step * step;
                }
                const size_t segment = options.order == SignatureOrder::suffix ? l - 1 - j : j;
                variation.data[segment * n + i] = maximum;
            }
            double tail = 0.;
            for (size_t j = l; j-- > 0;) {
                after.data[j * n + i] = tail;
                tail += variation.data[j * n + i];
            }
        }
    }
    void prepare_channels() {
        if (!original_paths_ || channels_ready_) return;
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const size_t workers = std::min({n, static_cast<size_t>(options.n_jobs), size_t{8}});
        std::vector<std::pair<double, size_t>> rank(d);
        variation = alloc(n * l);
        after = alloc(n * l);
        // Read successive frames contiguously. Each worker retains one frame
        // and one energy per input channel, then reduces its disjoint samples.
        auto scratch = alloc(siglasso::multiply(2 * workers, d));
        batch(workers, [&](size_t begin, size_t end) {
            for (size_t t = begin; t < end; ++t) {
                double* energy = scratch.data + 2 * t * d;
                const size_t first = n * t / workers, last = n * (t + 1) / workers;
                if (input_float_) channel_geometry(static_cast<const float*>(original_paths_), energy, energy + d, first, last);
                else channel_geometry(static_cast<const double*>(original_paths_), energy, energy + d, first, last);
            }
        });
        for (size_t a = 0; a < d; ++a) {
            double energy = 0.;
            for (size_t t = 0; t < workers; ++t) energy += scratch.data[2 * t * d + a];
            rank[a] = {-energy, a};
        }
        scratch = {};
        std::sort(rank.begin(), rank.end());
        channels_.resize(d);
        pinned_.resize(d);
        cold_shards_ = std::clamp<size_t>(d / 16, 1, cold_.size());
        const size_t hot_channels = log_features_ ? std::min<size_t>(8, d / 4) : std::min<size_t>(48, d / 2);
        const size_t keep = std::max<size_t>(1, hot_channels);
        auto bank = alloc(siglasso::multiply(keep, n * l));
        for (size_t pos = 0; pos < keep; ++pos) {
            const size_t a = rank[pos].second;
            pinned_[a] = true;
            channels_[a] = {bank.data + pos * n * l, n * l, bank.owner};
        }
        batch(keep, [&](size_t begin, size_t end) {
            for (size_t pos = begin; pos < end; ++pos) {
                const size_t a = rank[pos].second;
                read_channels<1>({a}, {channels_[a].data});
            }
        });
        channels_ready_ = true;
    }
    class EndpointSearch final : public siglasso::FeatureSearch {
        LassoCPU& b_;
        template<class T> void column(const T* paths, size_t a, double* out) const {
            const auto& o = b_.options;
            for (size_t i = 0; i < o.samples; ++i) {
                const T* x = paths + i * o.length * o.dimension + a;
                const double v = b_.endpoint_variation_.data[i];
                out[i] = v ? (double(x[(o.length - 1) * o.dimension]) - x[0]) / v : 0.;
            }
        }
        void column(size_t a, double* out) const {
            if (b_.endpoint_float_) column(static_cast<const float*>(b_.endpoint_input_), a, out);
            else column(static_cast<const double*>(b_.endpoint_input_), a, out);
        }
    public:
        explicit EndpointSearch(LassoCPU& b) : b_(b) {}
        siglasso::NodePtr selected(const siglasso::Word& word, bool) override {
            if (word.size() != 1 || word.front() >= b_.options.dimension) throw std::invalid_argument("Invalid endpoint word");
            auto node = std::make_shared<siglasso::Node>();
            node->word = word; node->values = b_.alloc(b_.options.samples);
            column(word.front(), node->values.data);
            return node;
        }
        siglasso::SearchResult query(const siglasso::Array& residual, double alpha,
            const siglasso::ActiveWords& active, size_t capacity) override {
            siglasso::SearchResult result;
            auto values = b_.alloc(residual.size);
            for (size_t a = 0; a < b_.options.dimension; ++a) {
                siglasso::Word word{a};
                if (active.count(word)) continue;
                column(a, values.data);
                const double score = std::abs(b_.dot(values.data, residual.data, residual.size)) / residual.size;
                result.omitted = std::max(result.omitted, siglasso::guard(score));
                ++b_.stats.block_queries; ++b_.stats.evaluated;
                if (!capacity || score <= alpha) { ++b_.stats.pruned; continue; }
                if (siglasso::shortlist(result.candidates, capacity, word, score, [&] { return b_.clone(values); }))
                    ++b_.stats.constructed;
            }
            ++b_.stats.search_passes;
            std::sort(result.candidates.begin(), result.candidates.end(), siglasso::better);
            return result;
        }
    };
    std::vector<std::unique_ptr<LassoCPU>> search_workers_;
    std::vector<std::unique_ptr<siglasso::Search>> searches_;
    std::vector<size_t> frontier_;
    LassoCPU* channel_source_ = nullptr;
    siglasso::Array terminal_sample_norms_, panel_scratch_;
    siglasso::Array channel_group_norms_;
    siglasso::Array primitive_norms_, difference_norms_;
    siglasso::Array projection_boxes_, projection_workspace_;
    static constexpr size_t projection_panels = 4;
    static constexpr size_t box_planes = 2 * projection_panels + 1;
    static constexpr size_t sample_groups = 8;
    struct EndpointGeometry {
        size_t first_degree;
        std::vector<float> left, right;
        siglasso::Array groups;
        std::shared_ptr<void> storage;
    };
    struct EndpointMask {
        size_t anchor = SIZE_MAX;
        double threshold = 0., omitted = 0.;
        std::vector<size_t> counts;
        std::shared_ptr<void> storage;
        bool any(size_t low, size_t high) const { return counts[high] != counts[low]; }
    };
    std::shared_ptr<EndpointGeometry> endpoints_;
    std::array<EndpointMask, 3> endpoint_masks_;
    siglasso::Array endpoint_weights_;
    std::vector<double> root_work_;
    std::array<double, 2> terminal_caps_{};
    // Independent SIMD accumulators shorten the dependency chain. Terminal
    // bounds include their reduction error. Exact correlations still use dot().
    template<bool square = false, class T> double reduction(const T* a, const double* b, size_t n) {
        const hn::ScalableTag<double> tag;
        const size_t lanes = hn::Lanes(tag);
        auto s0 = hn::Zero(tag), s1 = s0, s2 = s0, s3 = s0;
        const auto add = [&](auto sum, size_t i) {
            const auto x = [&] {
                if constexpr (std::is_same_v<T, float>) return hn::PromoteTo(tag, hn::LoadU(hn::Rebind<float, decltype(tag)>(), a + i));
                else return hn::LoadU(tag, a + i);
            }();
            const auto y = hn::LoadU(tag, b + i);
            if constexpr (square) {
                const auto product = hn::Mul(x, y);
                return hn::MulAdd(product, product, sum);
            } else return hn::MulAdd(x, y, sum);
        };
        size_t i = 0;
        for (; i + 4 * lanes <= n; i += 4 * lanes) {
            s0 = add(s0, i);
            s1 = add(s1, i + lanes);
            s2 = add(s2, i + 2 * lanes);
            s3 = add(s3, i + 3 * lanes);
        }
        auto sum = hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
        for (; i + lanes <= n; i += lanes) sum = add(sum, i);
        double value = hn::ReduceSum(tag, sum);
        for (; i < n; ++i) {
            const double product = a[i] * b[i];
            value += square ? product * product : product;
        }
        return std::isfinite(value) ? value : std::numeric_limits<double>::infinity();
    }
    double bound_dot(const double* a, const double* b, size_t n) override { return reduction(a, b, n); }
    double bound_dot(const float* a, const double* b, size_t n) override { return reduction(a, b, n); }
    double weighted_square(const double* a, const double* b, size_t n) override { return reduction<true>(a, b, n); }
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
            for (size_t a = task.low; a < task.high; ++a) {
                const size_t c = letter(a);
                auto value = hn::Zero(tag), absolute = hn::Zero(tag);
                for (size_t j = 0; j < l; ++j) {
                    const auto term = hn::Mul(hn::LoadU(tag, task.node->integrals.data + j * n + i), hn::LoadU(tag, increments.data + (j * d + c) * n + i));
                    value = hn::Add(value, term); absolute = hn::Add(absolute, hn::Abs(term));
                }
                squared = hn::Add(squared, hn::Mul(value, value)); magnitude = hn::Add(magnitude, hn::Mul(absolute, absolute));
            }
            const auto norm = hn::Sqrt(hn::Add(hn::Add(hn::Max(hn::Zero(tag), squared), hn::Mul(hn::Set(tag, 1e-11), magnitude)), hn::Set(tag, 1e-300)));
            total = hn::Add(total, hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), norm));
        }
        double result = hn::ReduceSum(tag, total);
        for (; i < n; ++i) result += std::abs(residual.data[i]) * siglasso::energy_sample(task.node->integrals.data,
            increments.data, task.low, task.high, i, n, l, d, letter_order);
        return siglasso::finite(result);
    }
public:
    LassoCPU(const LassoCPU& source, const siglasso::SearchOptions& options) : Backend(options, source.budget), log_features_(source.log_features_), responses_(source.responses_) {
        weights = source.weights;
        tt_suffix_ = source.tt_suffix_;
        terminal_only_ = source.terminal_only_;
        channel_group_norms_ = source.channel_group_norms_;
        primitive_norms_ = source.primitive_norms_;
        letter_order = source.letter_order;
        endpoints_ = source.endpoints_;
        projection_boxes_ = source.projection_boxes_;
        terminal_sample_norms_ = source.terminal_sample_norms_;
        compact_panels_ = source.compact_panels_;
    }
    template<typename T>
    LassoCPU(const siglasso::SearchOptions& o, const T* paths, const T* y, bool log_features,
             std::shared_ptr<siglasso::Budget> budget = {}) : Backend(o, std::move(budget)), log_features_(log_features) {
        terminal_only_ = y && !log_features && o.degree == 2;
        const size_t segments = o.length - 1;
        if (y && joint_search_enabled() && o.length > 1 && o.dimension >= (log_features_ ? 8u : 24u)) {
            original_paths_ = paths; input_float_ = std::is_same_v<T, float>; normalization_ = alloc(o.samples);
        }
        if (o.degree == 1 && (y || o.selected_only)) {
            endpoint_input_ = paths;
            endpoint_float_ = std::is_same_v<T, float>;
            endpoint_variation_ = alloc(o.samples);
        } else if (!original_paths_)
            increments = alloc(siglasso::multiply(siglasso::multiply(o.samples, segments), o.dimension));
        if (y) responses_ = alloc(siglasso::multiply(o.samples, o.targets));
        for (size_t j = 0; j < responses_.size; ++j) {
            const double v = y[j];
            if (!std::isfinite(v)) throw std::invalid_argument("Targets must be finite");
            if (o.loss == 1 && v != 0 && v != 1) throw std::invalid_argument("Logistic targets must be binary");
            if (o.loss == 2 && (v < 0 || std::floor(v) != v)) throw std::invalid_argument("Poisson targets must be nonnegative counts");
            responses_.data[j] = v;
        }
        batch(o.samples, [&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                const T* sample = paths + i * o.length * o.dimension;
                for (size_t a = 0; a < o.dimension; ++a)
                    if (!std::isfinite(sample[a])) throw std::invalid_argument("Paths must be finite");
                double total = 0;
                for (size_t j = 0; j < segments; ++j) {
                    const T* previous = sample + j * o.dimension;
                    const double step = step_variation(previous, previous + o.dimension, o.dimension);
                    total = siglasso::finite(total + step);
                }
                if (normalization_.owner) normalization_.data[i] = total;
                if (endpoint_input_) { endpoint_variation_.data[i] = total; continue; }
                if (!increments.owner) continue;
                for (size_t a = 0; a < o.dimension; ++a) {
                    double previous = 0;
                    for (size_t j = 0; j < segments; ++j) {
                        const double position = total ? siglasso::finite(
                            (double(sample[(j + 1) * o.dimension + a]) - sample[a]) / total) : 0;
                        const size_t segment = o.order == SignatureOrder::suffix ? segments - 1 - j : j;
                        increments.data[(segment * o.dimension + a) * o.samples + i] = position - previous;
                        previous = position;
                    }
                }
            }
        });
    }
    siglasso::Array alloc(size_t size) override {
        const size_t bytes = siglasso::multiply(size, sizeof(double));
        budget->acquire(bytes);
        double* data;
        try { data = new double[size]; } catch (...) { budget->release(bytes); throw; }
        return {data, size, std::shared_ptr<void>(data, [account = budget, bytes](void* p) { delete[] static_cast<double*>(p); account->release(bytes); })};
    }
    void fill(const siglasso::Array& a, double value) override { std::fill_n(a.data, a.size, value); }
    void copy(const double* from, double* to, size_t n) override { if (n) std::memcpy(to, from, n * sizeof(double)); }
    double dot(const double* a, const double* b, size_t n) override {
        const hn::ScalableTag<double> d;
        const auto lanes = hn::Lanes(d);
        auto sum0 = hn::Zero(d), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        auto err0 = sum0, err1 = sum0, err2 = sum0, err3 = sum0;
        const auto accumulate = [&](size_t j, auto& sum, auto& correction) {
            const auto product = hn::Mul(hn::LoadU(d, a + j), hn::LoadU(d, b + j));
            const auto next = hn::Add(sum, product);
            correction = hn::Add(correction, hn::IfThenElse(hn::Ge(hn::Abs(sum), hn::Abs(product)),
                hn::Add(hn::Sub(sum, next), product), hn::Add(hn::Sub(product, next), sum)));
            sum = next;
        };
        size_t j = 0;
        for (; j + 4 * lanes <= n; j += 4 * lanes) {
            accumulate(j, sum0, err0); accumulate(j + lanes, sum1, err1);
            accumulate(j + 2 * lanes, sum2, err2); accumulate(j + 3 * lanes, sum3, err3);
        }
        for (; j + lanes <= n; j += lanes) accumulate(j, sum0, err0);
        // Compensate the combination of the four long partial sums as well.
        const auto merge = [&](auto other) {
            const auto next = hn::Add(sum0, other);
            err0 = hn::Add(err0, hn::IfThenElse(hn::Ge(hn::Abs(sum0), hn::Abs(other)),
                hn::Add(hn::Sub(sum0, next), other), hn::Add(hn::Sub(other, next), sum0)));
            sum0 = next;
        };
        merge(sum1); merge(sum2); merge(sum3);
        double result = hn::ReduceSum(d, sum0), error = hn::ReduceSum(d, hn::Add(hn::Add(err0, err1), hn::Add(err2, err3)));
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
    double gram_dot(const double* a, const double* b, size_t n) override {
        return options.tolerance >= 1e-10 && n <= 65536 ? bound_dot(a, b, n) : dot(a, b, n);
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
    void correlations(const std::vector<siglasso::Column>& columns, const siglasso::Array& v, std::vector<double>& out) override {
        out.resize(columns.size());
        auto compute = [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) {
                const auto& column = columns[j].values;
                out[j] = dot(column.data, v.data, v.size) / v.size;
            }
        };
        if (options.n_jobs == 1 || out.size() <= 1) compute(0, out.size());
        else spawn_batch_threads(out.size(), options.n_jobs, compute);
    }
    static constexpr size_t panel_width = 6;
    size_t panels() const { return (options.length - 1 + panel_width - 1) / panel_width; }
    size_t norm_index(size_t block) const {
        const auto [low, high] = alphabet[block];
        return high - low == 1 ? letter(low) : options.dimension + block - low;
    }
    // Nonnegative bfloat16 values are ordered by their unsigned encoding.
    // Round each leaf upward once. Every internal maximum then stays exact
    // in the encoded representation without accumulating quantisation error.
    struct PanelEnvelopes {
        std::vector<uint16_t> codes;
        std::shared_ptr<void> storage;
    };
    std::shared_ptr<PanelEnvelopes> compact_panels_;
    static double panel_value(uint16_t bits) {
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
    static uint16_t upper_panel(double value) {
        const auto code = static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(value)) >> 16);
        return code + (panel_value(code) < value);
    }
    double compact_bound(size_t block, const double* moments, const siglasso::Array& residual, double* majorant) {
        const size_t n = options.samples;
        // Sum the nonnegative panel majorants once per sample. The
        // same geometry bounds every response in the joint query.
        for (size_t j = 0; j < panels(); ++j) {
            const double* moment = moments + j * n;
            const auto* step = compact_panels_->codes.data() + (block * panels() + j) * n;
            if (!j) for (size_t i = 0; i < n; ++i) majorant[i] = moment[i] * panel_value(step[i]);
            else for (size_t i = 0; i < n; ++i) majorant[i] += moment[i] * panel_value(step[i]);
        }
        double maximum = 0.;
        if (batch_absolute.empty()) {
            for (size_t i = 0; i < n; ++i) maximum += std::abs(residual.data[i]) * majorant[i];
        } else for (const auto& r : batch_absolute)
            maximum = std::max(maximum, bound_dot(r.data, majorant, n));
        return maximum;
    }
    void prepare_compact_panels() {
        const size_t n = options.samples, l = options.length - 1, count = panels();
        const size_t stride = siglasso::multiply(count, n);
        compact_panels_ = std::make_shared<PanelEnvelopes>();
        auto& table = *compact_panels_;
        const size_t size = siglasso::multiply(alphabet.size(), stride);
        table.storage = account(siglasso::multiply(size, sizeof(uint16_t)));
        table.codes.resize(size);
        batch(alphabet.size(), [&](size_t begin, size_t end) {
            for (size_t block = begin; block < end; ++block) {
                const auto [low, high] = alphabet[block];
                if (high - low != 1) continue;
                const auto steps = channel(letter(low));
                for (size_t g = 0; g < count; ++g) {
                    auto* out = table.codes.data() + (block * count + g) * n;
                    for (size_t j = g * panel_width; j < std::min(l, (g + 1) * panel_width); ++j) {
                        const double* x = steps.values.data + j * steps.dimension * n;
                        for (size_t i = 0; i < n; ++i) out[i] = std::max(out[i], upper_panel(std::abs(x[i])));
                    }
                }
            }
        });
        for (size_t block = alphabet.size(); block-- > 0;) {
            const auto [low, high] = alphabet[block];
            if (high - low == 1) continue;
            const size_t right = block + 2 * ((low + high) / 2 - low);
            const auto* left_child = table.codes.data() + (block + 1) * stride;
            const auto* right_child = table.codes.data() + right * stride;
            auto* out = table.codes.data() + block * stride;
            for (size_t i = 0; i < stride; ++i) out[i] = std::max(left_child[i], right_child[i]);
        }
    }
    void order_channel_groups() {
        const size_t d = options.dimension;
        const bool adaptive = options.order == SignatureOrder::adaptive;
        const bool permuted = adaptive || options.order == SignatureOrder::energy || options.order == SignatureOrder::profile;
        if (permuted) {
            ordered_letters_.resize(d);
            std::iota(ordered_letters_.begin(), ordered_letters_.end(), 0);
            letter_order = ordered_letters_.data();
        }
        const auto* profile = channel_group_norms_.data;
        auto energy = alloc(d);
        for (size_t a = 0; a < d; ++a)
            energy.data[a] = bound_dot(profile + a * sample_groups, profile + a * sample_groups, sample_groups);
        std::vector<size_t> best_order;
        double best_cost = std::numeric_limits<double>::infinity();
        const auto test_order = [&] {
            double cost = 0.;
            for (const auto& [lo, hi] : alphabet) {
                if (hi - lo == 1) continue;
                std::array<double, sample_groups> largest{};
                double total = 0.;
                for (size_t p = lo; p < hi; ++p) {
                    const auto* v = profile + ordered_letters_[p] * sample_groups;
                    for (size_t g = 0; g < sample_groups; ++g) {
                        largest[g] = std::max(largest[g], v[g]); total += v[g];
                    }
                }
                cost += (hi - lo) * std::accumulate(largest.begin(), largest.end(), 0.) - total;
            }
            if (best_order.empty() || cost < best_cost) { best_cost = cost; best_order = ordered_letters_; }
        };
        if (adaptive) test_order();
        if (adaptive || options.order == SignatureOrder::energy) {
            std::sort(ordered_letters_.begin(), ordered_letters_.end(), [&](size_t a, size_t b) {
                return energy.data[a] == energy.data[b] ? a < b : energy.data[a] > energy.data[b];
            });
            if (adaptive) test_order();
        }
        if (adaptive || options.order == SignatureOrder::profile) {
            // Partition the channel norm profiles at the median on their widest axis.
            for (const auto& [low, high] : alphabet) {
                if (high - low <= 1) continue;
                size_t axis = 0;
                double widest = -1.;
                for (size_t g = 0; g < sample_groups; ++g) {
                    double lower = std::numeric_limits<double>::infinity(), upper = 0.;
                    for (size_t p = low; p < high; ++p) {
                        const double value = profile[ordered_letters_[p] * sample_groups + g];
                        lower = std::min(lower, value); upper = std::max(upper, value);
                    }
                    if (upper - lower > widest) { widest = upper - lower; axis = g; }
                }
                auto first = ordered_letters_.begin();
                std::nth_element(first + low, first + low + (high - low) / 2, first + high, [&](size_t a, size_t b) {
                    const double x = profile[a * sample_groups + axis], y = profile[b * sample_groups + axis];
                    return x == y ? a < b : x > y;
                });
            }
            if (adaptive) {
                test_order();
                ordered_letters_.swap(best_order);
                letter_order = ordered_letters_.data();
            }
        }
        if (permuted && channels_ready_) {
            rank_of_letter_.resize(d);
            for (size_t rank = 0; rank < d; ++rank) rank_of_letter_[letter(rank)] = rank;
            for (auto& shard : cold_) shard.first = SIZE_MAX;
        }
        root_work_.resize(alphabet.size());
        for (size_t block = alphabet.size(); block-- > 0;) {
            const auto [lo, hi] = alphabet[block];
            root_work_[block] = hi - lo == 1 ? energy.data[letter(lo)]
                : root_work_[block + 1] + root_work_[block + 2 * ((hi - lo) / 2)];
        }
    }
    // First and last order-statistic marginals on the variation simplex.
    void prepare_endpoint_geometry() {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        endpoints_ = std::make_shared<EndpointGeometry>();
        auto& e = *endpoints_;
        e.first_degree = std::max<size_t>(3, m - 2);
        const size_t levels = m - e.first_degree + 1;
        const size_t size = siglasso::multiply(levels, siglasso::multiply(d, n));
        e.storage = account(siglasso::multiply(2 * sizeof(float), size));
        e.left.resize(size); e.right.resize(size);
        e.groups = alloc(siglasso::multiply(levels * sample_groups, alphabet.size()));
        const double error = siglasso::terminal_roundoff(m, n, l);
        const auto upward = [](double x) {
            const auto v = static_cast<float>(x);
            return v < x ? std::nextafter(v, std::numeric_limits<float>::infinity()) : v;
        };
        batch(d, [&](size_t low, size_t high) {
            for (size_t c = low; c < high; ++c) {
                const auto dx = channel(c);
                for (size_t i = 0; i < n; ++i) {
                    double end = 0.;
                    for (size_t j = 0; j < l; ++j) end += dx.values.data[j * dx.dimension * n + i];
                    double x = 0., v = 0.;
                    std::array<double, 3> left{}, right{};
                    for (size_t j = 0; j < l; ++j) {
                        const double next = x + dx.values.data[j * dx.dimension * n + i];
                        const double y = end - x, y1 = end - next;
                        const double dv = variation.data[j * n + i], tail = after.data[j * n + i];
                        const double sx = (x + next) * (x + next), sy = (y + y1) * (y + y1);
                        // Integrals of (1-t) and t against the squared linear excursion.
                        const double x0 = dv / 12 * (sx + 2 * x * x), x1 = dv / 12 * (sx + 2 * next * next);
                        const double y0 = dv / 12 * (sy + 2 * y * y), yr = dv / 12 * (sy + 2 * y1 * y1);
                        const size_t power = e.first_degree - 3;
                        double wl = power ? std::pow(tail + dv, power) : 1., wr = power ? std::pow(tail, power) : 1.;
                        double vl = power ? std::pow(v, power) : 1., vr = power ? std::pow(v + dv, power) : 1.;
                        for (size_t s = 0; s < levels; ++s) {
                            left[s] += wl * x0 + wr * x1; right[s] += vl * y0 + vr * yr;
                            wl *= tail + dv; wr *= tail; vl *= v; vr *= v + dv;
                        }
                        x = next; v += dv;
                    }
                    for (size_t s = 0; s < levels; ++s) {
                        const size_t k = e.first_degree + s - 2, at = (s * d + c) * n + i;
                        const double pad = error * std::pow(v, 1. + .5 * k);
                        e.left[at] = upward(std::sqrt(k * left[s]) * (1 + error) + pad);
                        e.right[at] = upward(std::sqrt(k * right[s]) * (1 + error) + pad);
                    }
                }
            }
        });
        for (size_t s = 0; s < levels; ++s) for (size_t block = alphabet.size(); block-- > 0;) {
            const auto [lo, hi] = alphabet[block];
            auto* dst = e.groups.data + (s * alphabet.size() + block) * sample_groups;
            if (hi - lo == 1) {
                const float* right = e.right.data() + (s * d + lo) * n;
                for (size_t g = 0; g < sample_groups; ++g) {
                    double v = 0.;
                    for (size_t i = n * g / sample_groups; i < n * (g + 1) / sample_groups; ++i)
                        v += double(right[i]) * right[i];
                    dst[g] = std::sqrt(v);
                }
            } else {
                const size_t offset = 2 * ((hi - lo) / 2) * sample_groups;
                for (size_t g = 0; g < sample_groups; ++g) dst[g] = std::max(dst[g + sample_groups], dst[g + offset]);
            }
        }
    }
    const EndpointMask* endpoint_mask(size_t a, size_t degree, const siglasso::Array& r, double threshold) {
        if (!endpoints_ || degree < endpoints_->first_degree) return nullptr;
        const auto& e = *endpoints_;
        const size_t n = options.samples, d = options.dimension, s = degree - e.first_degree;
        auto& mask = endpoint_masks_[s];
        if (mask.anchor == a && mask.threshold <= threshold) return &mask;
        mask.anchor = a; mask.threshold = threshold; mask.omitted = 0.;
        if (!mask.storage) mask.storage = account(siglasso::multiply(d + 1, sizeof(size_t)));
        mask.counts.assign(d + 1, 0);
        const size_t targets = std::max<size_t>(1, batch_absolute.size());
        if (endpoint_weights_.size < targets * n) endpoint_weights_ = alloc(targets * n);
        const float* left = e.left.data() + (s * d + a) * n;
        std::array<std::array<double, sample_groups>, siglasso::max_response_batch> groups{};
        for (size_t t = 0; t < targets; ++t) {
            const auto* residual = batch_absolute.empty() ? r.data : batch_absolute[t].data;
            double* out = endpoint_weights_.data + t * n;
            for (size_t i = 0; i < n; ++i) out[i] = std::abs(residual[i]) * left[i];
            for (size_t g = 0; g < sample_groups; ++g) {
                const size_t first = n * g / sample_groups, end = n * (g + 1) / sample_groups;
                groups[t][g] = std::sqrt(bound_dot(out + first, out + first, end - first));
            }
        }
        const double error = siglasso::terminal_roundoff(degree, n, options.length - 1);
        const double scale = double(degree * (degree - 1)) / n * (1. + error);
        const double slack = error * absolute_sum(r) / n + 1e-15;
        if (!std::isfinite(scale) || !std::isfinite(slack)) { mask.anchor = SIZE_MAX; return nullptr; }
        for (size_t block = 0; block < alphabet.size();) {
            const auto [lo, hi] = alphabet[block];
            const auto* radii = e.groups.data + (s * alphabet.size() + block) * sample_groups;
            double bound = 0.;
            for (size_t t = 0; t < targets; ++t) {
                bound = std::max(bound, bound_dot(groups[t].data(), radii, sample_groups) * scale + slack);
                if (bound >= threshold) break;
            }
            if (bound < threshold) {
                mask.omitted = std::max(mask.omitted, bound);
                block += 2 * (hi - lo) - 1;
                continue;
            }
            if (hi - lo == 1) {
                const float* right = e.right.data() + (s * d + lo) * n;
                bound = 0.;
                for (size_t t = 0; t < targets; ++t) {
                    bound = std::max(bound, bound_dot(right, endpoint_weights_.data + t * n, n) * scale + slack);
                    if (bound >= threshold) break;
                }
                if (bound < threshold) mask.omitted = std::max(mask.omitted, bound);
                else mask.counts[hi] = 1;
            }
            ++block;
        }
        std::partial_sum(mask.counts.begin(), mask.counts.end(), mask.counts.begin());
        return &mask;
    }
    void prepare() override {
        if (endpoint_input_) return;
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        prepare_channels();
        if (!options.selected_only && increments.owner) {
            variation = alloc(n * l); after = alloc(n * l);
            batch(n, [&](uint64_t begin, uint64_t end) {
                for (size_t i = begin; i < end; ++i)
                    siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
            });
        }
        if (log_features_ || options.selected_only) return;
        prepare_alphabet();
        if (options.order == SignatureOrder::outside && options.degree >= 3 && l) prepare_endpoint_geometry();
        terminal_sample_norms_ = alloc(siglasso::multiply(n, alphabet.size()));
        if (options.terminal == TerminalBound::projected && l)
            projection_boxes_ = alloc(siglasso::multiply(box_planes * alphabet.size(), n));
        if (options.terminal == TerminalBound::primitive && l)
            primitive_norms_ = alloc(siglasso::multiply(2, terminal_sample_norms_.size));
        const double error = siglasso::terminal_roundoff(0, n * l, l);
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t a = 0; a < d; ++a) {
                const auto steps = channel(a);
                double* norms = terminal_sample_norms_.data + a * n;
                std::fill(norms + begin, norms + end, 0.);
                for (size_t j = 0; j < l; ++j) {
                    const double* dx = steps.values.data + j * steps.dimension * n;
                    for (size_t i = begin; i < end; ++i) norms[i] += dx[i] * dx[i];
                }
                for (size_t i = begin; i < end; ++i)
                    norms[i] = std::sqrt(norms[i] + 32. * (l + 1) * std::numeric_limits<double>::min()) * (1 + error);
                if (primitive_norms_.owner) {
                    double* endpoint = primitive_norms_.data + 2 * a * n;
                    double* norm = endpoint + n;
                    std::fill(endpoint + begin, endpoint + end, 0.);
                    std::fill(norm + begin, norm + end, 0.);
                    for (size_t j = l; j-- > 0;) {
                        const double* dx = steps.values.data + j * steps.dimension * n;
                        for (size_t i = begin; i < end; ++i) {
                            endpoint[i] += dx[i];
                            if (j) norm[i] += endpoint[i] * endpoint[i];
                        }
                    }
                    for (size_t i = begin; i < end; ++i) {
                        // A tail sum can cancel. Bound its absolute summation error by the increment norm.
                        const double pad = error * std::sqrt(static_cast<double>(l)) * norms[i];
                        endpoint[i] = std::abs(endpoint[i]) + pad;
                        norm[i] = std::sqrt(norm[i]) * (1 + error) + std::sqrt(static_cast<double>(l)) * pad;
                    }
                }
                if (projection_boxes_.owner) {
                    auto* box = projection_boxes_.data + box_planes * a * n;
                    auto* remainder = box + (box_planes - 1) * n;
                    std::fill(remainder + begin, remainder + end, 0.);
                    for (size_t p = 0; p < projection_panels; ++p) {
                        const auto first = p * l / projection_panels, stop = (p + 1) * l / projection_panels;
                        auto* mean = box + 2 * p * n;
                        std::fill(mean + begin, mean + end, 0.);
                        std::fill(mean + n + begin, mean + n + end, 0.);
                        for (size_t j = first; j < stop; ++j) {
                            const auto* dx = steps.values.data + j * steps.dimension * n;
                            for (size_t i = begin; i < end; ++i) mean[i] += dx[i];
                        }
                        for (size_t i = begin; i < end; ++i) mean[i] /= std::max<size_t>(1, stop - first);
                        for (size_t j = first; j < stop; ++j) {
                            const auto* dx = steps.values.data + j * steps.dimension * n;
                            for (size_t i = begin; i < end; ++i) {
                                const auto delta = dx[i] - mean[i];
                                remainder[i] += delta * delta;
                            }
                        }
                    }
                    for (size_t i = begin; i < end; ++i)
                        remainder[i] = std::sqrt(remainder[i] + std::numeric_limits<double>::min()) * (1 + error);
                }
            }
        });
        const bool grouped = options.grouped_terminals();
        channel_group_norms_ = alloc((grouped ? alphabet.size() : d) * sample_groups);
        for (size_t a = 0; a < d; ++a) for (size_t g = 0; g < sample_groups; ++g) {
            double sum = 0;
            for (size_t i = n * g / sample_groups; i < n * (g + 1) / sample_groups; ++i) {
                const double value = terminal_sample_norms_.data[a * n + i];
                sum += value * value;
            }
            channel_group_norms_.data[a * sample_groups + g] = std::sqrt(sum + 32. * (n + 1) * std::numeric_limits<double>::min());
        }
        if (options.order == SignatureOrder::energy || options.order == SignatureOrder::profile ||
            options.order == SignatureOrder::balanced || options.order == SignatureOrder::outside ||
            options.order == SignatureOrder::adaptive) order_channel_groups();
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t block = alphabet.size(); block-- > 0;) {
                const auto [low, high] = alphabet[block];
                const size_t right = block + 2 * ((low + high) / 2 - low);
                if (high - low == 1) continue;
                const size_t dst = norm_index(block), left = norm_index(block + 1), other = norm_index(right);
                for (size_t i = begin; i < end; ++i) terminal_sample_norms_.data[dst * n + i] = std::max(
                    terminal_sample_norms_.data[left * n + i], terminal_sample_norms_.data[other * n + i]);
                if (primitive_norms_.owner) for (size_t p = 0; p < 2; ++p) {
                    auto* table = primitive_norms_.data + p * n;
                    for (size_t i = begin; i < end; ++i)
                        table[2 * dst * n + i] = std::max(table[2 * left * n + i], table[2 * other * n + i]);
                }
                if (projection_boxes_.owner) {
                    const auto* x = projection_boxes_.data + box_planes * left * n;
                    const auto* y = projection_boxes_.data + box_planes * other * n;
                    auto* box = projection_boxes_.data + box_planes * dst * n;
                    for (size_t p = 0; p < projection_panels; ++p) for (size_t i = begin; i < end; ++i) {
                        const size_t j = 2 * p * n + i;
                        const double lower = std::min(x[j] - x[n + j], y[j] - y[n + j]);
                        const double upper = std::max(x[j] + x[n + j], y[j] + y[n + j]);
                        box[j] = std::midpoint(lower, upper);
                        box[n + j] = std::max(upper - box[j], box[j] - lower);
                    }
                    for (size_t i = begin; i < end; ++i) {
                        const size_t j = (box_planes - 1) * n + i;
                        box[j] = std::max(x[j], y[j]);
                    }
                }
            }
        });
        if (grouped) for (size_t block = alphabet.size(); block-- > 0;) {
            const auto [lo, hi] = alphabet[block];
            if (hi - lo == 1) continue;
            auto* out = channel_group_norms_.data + norm_index(block) * sample_groups;
            const auto* left = channel_group_norms_.data + norm_index(block + 1) * sample_groups;
            const auto* right = channel_group_norms_.data + norm_index(block + 2 * ((lo + hi) / 2 - lo)) * sample_groups;
            for (size_t g = 0; g < sample_groups; ++g) out[g] = std::max(left[g], right[g]);
        }
        if (terminal_only_) return;
        if (responses_.owner && joint_search_enabled() && l) {
            prepare_compact_panels();
            prepare_tt();
            return;
        }
        alphabet_ratios = alloc(siglasso::multiply(alphabet.size(), siglasso::multiply(n, panels())));
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t block = alphabet.size(); block-- > 0;) {
                const auto [low, high] = alphabet[block];
                const size_t right = block + 2 * ((low + high) / 2 - low);
                const auto steps = high - low == 1 ? channel(letter(low)) : siglasso::Channel{};
                const size_t count = panels();
                for (size_t g = 0; g < count; ++g) {
                    double* out = alphabet_ratios.data + (block * count + g) * n;
                    if (high - low > 1) {
                        const double* left = alphabet_ratios.data + ((block + 1) * count + g) * n;
                        const double* other = alphabet_ratios.data + (right * count + g) * n;
                        for (size_t i = begin; i < end; ++i) out[i] = std::max(left[i], other[i]);
                    } else {
                        std::fill(out + begin, out + end, 0.);
                        for (size_t j = g * panel_width; j < std::min(l, (g + 1) * panel_width); ++j) {
                            const double* step = steps.values.data + j * steps.dimension * n;
                            for (size_t i = begin; i < end; ++i) out[i] = std::max(out[i], std::abs(step[i]));
                        }
                    }
                }
            }
        });
        prepare_tt();
    }
    size_t suffix_plane(size_t q, size_t a) const {
        const size_t m = options.degree;
        return q * (2 * m - q + 1) / 2 + a - 1;
    }
    void prepare_tt() {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        const size_t planes = m * (m + 1) / 2;
        tt_suffix_ = alloc(siglasso::multiply(planes, siglasso::multiply(n, l)));
        fill(tt_suffix_, 0);
        for (size_t a = 1; a <= m; ++a) std::fill_n(tt_suffix_.data + suffix_plane(0, a) * l * n, l * n, 1 / weights[a]);
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q < m; ++q) for (size_t c = 0; c < d; ++c) {
                const auto steps = channel(letter(c));
                for (size_t i = begin; i < end; ++i) {
                    double tail = 0.;
                    for (size_t j = l; j-- > 0;) {
                        const double increment = std::abs(steps.values.data[j * steps.dimension * n + i]);
                        for (size_t a = 1; a + q <= m; ++a) {
                            double& out = tt_suffix_.data[(suffix_plane(q, a) * l + j) * n + i];
                            const double next = tt_suffix_.data[(suffix_plane(q - 1, a + 1) * l + j) * n + i];
                            out = std::max(out, siglasso::finite(increment * next + tail / weights[a]));
                        }
                        tail += increment * tt_suffix_.data[(suffix_plane(q - 1, 1) * l + j) * n + i];
                    }
                }
            }
        });
    }
    const double* prefix_start(const siglasso::Node& node, size_t j) const {
        return node.coefficients.data + j * options.samples;
    }
    siglasso::Array tt_moments(const siglasso::Node& node) {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), m = options.degree;
        const size_t count = panels();
        auto result = alloc(siglasso::multiply(m - k, siglasso::multiply(n, count)));
        fill(result, 0.);
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        batch(count, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q <= m - k; ++q) for (size_t g = begin; g < end; ++g)
            for (size_t j = g * panel_width; j < std::min(l, (g + 1) * panel_width); ++j) {
                double* out = result.data + ((q - 1) * count + g) * n;
                const double scale = weights[k + q] / n;
                size_t i = 0;
                for (; i + lanes <= n; i += lanes) {
                    auto value = hn::Mul(hn::Abs(hn::LoadU(tag, prefix_start(node, j) + i)),
                        hn::LoadU(tag, tt_suffix_.data + (suffix_plane(q - 1, 1) * l + j) * n + i));
                    for (size_t a = 1; a <= k; ++a) value = hn::MulAdd(
                        hn::Abs(hn::LoadU(tag, node.tt_state.data + ((a - 1) * l + j) * n + i)),
                        hn::LoadU(tag, tt_suffix_.data + (suffix_plane(q - 1, a + 1) * l + j) * n + i), value);
                    const auto scaled = hn::Mul(value, hn::Set(tag, scale));
                    hn::StoreU(hn::Add(hn::LoadU(tag, out + i), scaled), tag, out + i);
                }
                for (; i < n; ++i) {
                    double value = std::abs(prefix_start(node, j)[i])
                        * tt_suffix_.data[(suffix_plane(q - 1, 1) * l + j) * n + i];
                    for (size_t a = 1; a <= k; ++a) value += std::abs(node.tt_state.data[((a - 1) * l + j) * n + i])
                        * tt_suffix_.data[(suffix_plane(q - 1, a + 1) * l + j) * n + i];
                    out[i] += siglasso::finite(value * scale);
                }
            }
        });
        return result;
    }
    // Chen's identity on a segment updates each prefix from shorter prefixes
    // at the previous endpoint. Tensor log additionally needs contiguous
    // subwords of this one selected word for the finite logarithm series.
    template<size_t K, bool Log, typename D>
    void selected_block(D tag, const std::array<siglasso::Channel, 4>& channels, size_t i, double* output) {
        using V = decltype(hn::Zero(tag));
        constexpr size_t starts = Log ? K : 1;
        V signature[starts][K + 1];
        for (size_t start = 0; start < starts; ++start) {
            signature[start][start] = hn::Set(tag, 1.);
            for (size_t end = start + 1; end <= K; ++end) signature[start][end] = hn::Zero(tag);
        }
        const size_t n = options.samples;
        constexpr double inverse_factorial[] = {1., 1., .5, 1. / 6, 1. / 24};
        for (size_t j = 0; j + 1 < options.length; ++j) {
            V steps[K];
            for (size_t a = 0; a < K; ++a)
                steps[a] = hn::LoadU(tag, channels[a].values.data + j * channels[a].dimension * n + i);
            for (size_t start = 0; start < starts; ++start) for (size_t end = K; end > start; --end) {
                auto product = hn::Set(tag, 1.);
                auto value = signature[start][end];
                for (size_t cut = end; cut-- > start;) {
                    product = hn::Mul(product, steps[cut]);
                    value = hn::MulAdd(hn::Mul(product, hn::Set(tag, inverse_factorial[end - cut])), signature[start][cut], value);
                }
                signature[start][end] = value;
            }
        }
        auto value = signature[0][K];
        if constexpr (Log) for (size_t mask = 1; mask < (size_t{1} << (K - 1)); ++mask) {
            size_t start = 0, parts = 1;
            auto product = hn::Set(tag, 1.);
            for (size_t cut = 1; cut < K; ++cut) if (mask & (size_t{1} << (cut - 1))) {
                product = hn::Mul(product, signature[start][cut]); start = cut; ++parts;
            }
            product = hn::Mul(product, signature[start][K]);
            value = hn::MulAdd(product, hn::Set(tag, (parts % 2 ? 1. : -1.) / parts), value);
        }
        hn::StoreU(value, tag, output + i);
    }
    template<size_t K, bool Log>
    void selected_values(const std::array<siglasso::Channel, 4>& channels, double* output) {
        const hn::ScalableTag<double> tag;
        const size_t width = hn::Lanes(tag), n = options.samples;
        size_t i = 0;
        for (; i + width <= n; i += width) selected_block<K, Log>(tag, channels, i, output);
        for (; i < n; ++i) selected_block<K, Log>(hn::CappedTag<double, 1>(), channels, i, output);
    }
    template<bool Log> siglasso::NodePtr selected_word(const siglasso::Word& word) {
        std::array<siglasso::Channel, 4> channels;
        if (word.empty() || word.size() > channels.size())
            throw std::logic_error("Unexpected selected word degree");
        auto node = std::make_shared<siglasso::Node>();
        node->word = word; node->values = alloc(options.samples);
        for (size_t a = 0; a < word.size(); ++a) channels[a] = channel(word[a]);
        switch (word.size()) {
        case 1: selected_values<1, Log>(channels, node->values.data); break;
        case 2: selected_values<2, Log>(channels, node->values.data); break;
        case 3: selected_values<3, Log>(channels, node->values.data); break;
        case 4: selected_values<4, Log>(channels, node->values.data); break;
        }
        return node;
    }
    class BatchedSearch final : public siglasso::FeatureSearch {
        LassoCPU& b_;
        std::unique_ptr<siglasso::FeatureSearch> tree_;
        bool lower_degree_;
        void complete_dictionary() {
            if (!lower_degree_) return;
            if (b_.log_features_) {
                tree_.reset();
                tree_ = siglasso::make_log_search(b_);
            }
            b_.proposal_degree = 0;
            lower_degree_ = false;
        }
        std::shared_ptr<const siglasso::QuantizedColumn> pack(const double* values, size_t n) const {
            auto packed = std::make_shared<siglasso::QuantizedColumn>();
            packed->storage = b_.account(128 + n * sizeof(int16_t));
            packed->samples.resize(n);
            const auto [lo, hi] = std::minmax_element(values, values + n);
            packed->offset = std::midpoint(*lo, *hi);
            const double radius = std::max(*hi - packed->offset, packed->offset - *lo);
            packed->unit = radius / 32767.;
            if (packed->unit) for (size_t i = 0; i < n; ++i) {
                const auto code = static_cast<int16_t>(std::clamp(std::nearbyint((values[i] - packed->offset) / packed->unit), -32767., 32767.));
                packed->samples[i] = code;
                packed->error = std::max(packed->error, std::abs((values[i] - packed->offset) - code * packed->unit));
            }
            else packed->error = radius;
            return packed;
        }
    public:
        BatchedSearch(LassoCPU& b, std::unique_ptr<siglasso::FeatureSearch> tree, bool lower) : b_(b), tree_(std::move(tree)), lower_degree_(lower) {}
        size_t response_batch() const override {
            if (b_.options.degree == 2 && b_.options.dimension < 96) return b_.log_features_ ? 6 : 5;
            if (!b_.log_features_ && b_.options.degree == 3 && b_.options.dimension < 96) return 5;
            return siglasso::max_response_batch;
        }
        // A lower-degree violation gives a valid descent direction for the full
        // problem. On exhaustion, check the missing level at the same residual
        // and combine the two bounds. All later requests cover the full tree.
        void query_batch(const std::vector<siglasso::Separation*>& queries) override {
            if (!lower_degree_ || std::any_of(queries.begin(), queries.end(),
                    [](const auto* q) { return !q->allow_lower_degree; })) {
                complete_dictionary();
                separate(queries);
                return;
            }
            separate(queries);
            std::vector<siglasso::Separation*> remaining;
            std::vector<double> lower_bounds;
            for (auto* q : queries) if (q->found.candidates.empty()) {
                remaining.push_back(q);
                lower_bounds.push_back(q->found.omitted);
            }
            if (remaining.empty()) return;
            complete_dictionary();
            b_.highest_degree_only = true;
            try { separate(remaining); }
            catch (...) { b_.highest_degree_only = false; throw; }
            b_.highest_degree_only = false;
            for (size_t i = 0; i < remaining.size(); ++i)
                remaining[i]->found.omitted = std::max(remaining[i]->found.omitted, lower_bounds[i]);
        }
        // With s_t = min(alpha)/alpha_t, the common envelope dominates every
        // |s_t r_t|. Exact signed scores decide inclusion in each target's heap.
        void separate(const std::vector<siglasso::Separation*>& queries) {
            const size_t n = b_.options.samples;
            auto envelope = b_.alloc(n); b_.fill(envelope, 0.);
            const auto clear_batch = [&] {
                b_.visit_column = {}; b_.batch_absolute.clear(); b_.batch_ratios.clear(); b_.batch_signed.clear();
            };
            // The dual residual needs the maximum above alpha, even for a word
            // inside the stationarity tolerance that will not enter the fit.
            double threshold = queries.front()->penalty;
            for (const auto* q : queries) threshold = std::min(threshold, q->penalty);
            if (std::any_of(queries.begin(), queries.end(), [&](const auto* q) {
                return !std::isfinite(q->penalty) || threshold < 1e-12 * q->penalty || q->penalty < 1e-240;
            })) { tree_->query_batch(queries); return; }
            for (auto* q : queries) {
                auto magnitude = b_.alloc(n), signed_weights = b_.alloc(n);
                const double scale = threshold / q->penalty * (1. + 1e-12);
                for (size_t i = 0; i < n; ++i) {
                    signed_weights.data[i] = q->residual.data[i] * scale;
                    magnitude.data[i] = std::abs(signed_weights.data[i]);
                    envelope.data[i] = std::max(envelope.data[i], magnitude.data[i]);
                }
                b_.batch_absolute.push_back(std::move(magnitude)); b_.batch_signed.push_back(std::move(signed_weights));
                q->found.omitted = q->penalty;
            }
            // Ratios let terminal bounds test each response against its own
            // absolute residual after computing a shared geometric majorant.
            for (const auto& r : b_.batch_absolute) {
                auto ratio = b_.alloc(n);
                for (size_t i = 0; i < n; ++i) ratio.data[i] = envelope.data[i] ? r.data[i] / envelope.data[i] : 0.;
                b_.batch_ratios.push_back(std::move(ratio));
            }
            std::vector<double> allowance;
            const double cap = b_.log_features_ ? tree_->feature_cap() : 2.;
            for (const auto* q : queries) allowance.push_back((64. * (n + 1) * std::numeric_limits<double>::epsilon()) *
                cap * (b_.absolute_sum(q->residual) / n) + std::numeric_limits<double>::min());
            std::mutex mutex;
            b_.visit_column = [&](const siglasso::Word& word, const double* values) {
                if (b_.highest_degree_only && word.size() < b_.options.degree) return false;
                const double factor = b_.weights[word.size()] / n;
                std::shared_ptr<const siglasso::QuantizedColumn> shared_values;
                bool found = false;
                for (size_t target = 0; target < queries.size(); ++target) {
                    auto* q = queries[target];
                    if (std::abs(b_.bound_dot(values, q->residual.data, n)) * factor + allowance[target] < q->penalty ||
                        q->active.count(word)) continue;
                    const double score = siglasso::finite(std::abs(b_.dot(values, q->residual.data, n)) * factor);
                    if (score <= q->penalty) continue;
                    found = true;
                    std::lock_guard lock(mutex);
                    q->found.omitted = std::max(q->found.omitted, siglasso::guard(score));
                    if (score <= q->threshold) continue;
                    siglasso::retain_candidate(q->found.candidates, q->capacity, word, score, [&](siglasso::Candidate& candidate) {
                        if (!shared_values) shared_values = pack(values, n);
                        candidate.proposal = shared_values;
                    });
                }
                return found;
            };
            try { tree_->query(envelope, threshold, {}, 0); }
            catch (...) { clear_batch(); throw; }
            clear_batch();
            for (auto* q : queries) std::sort(q->found.candidates.begin(), q->found.candidates.end(), siglasso::better);
        }
        double feature_cap() const override { return tree_->feature_cap(); }
        siglasso::NodePtr selected(const siglasso::Word& word, bool keep) override {
            if (keep && lower_degree_ && word.size() >= b_.options.degree) complete_dictionary();
            if (keep) return tree_->selected(word, true);
            return b_.log_features_ ? b_.selected_word<true>(word) : b_.selected_word<false>(word);
        }
        siglasso::SearchResult query(const siglasso::Array& r, double alpha,
                const siglasso::ActiveWords& active, size_t capacity) override {
            complete_dictionary();
            return tree_->query(r, alpha, active, capacity);
        }
    };
    std::unique_ptr<siglasso::FeatureSearch> make_search() override {
        if (endpoint_input_) return std::make_unique<EndpointSearch>(*this);
        const bool joint = joint_search_enabled();
        const bool lower = joint && ((options.degree == 4 && (log_features_ || options.dimension >= 24)) ||
            (!log_features_ && options.degree == 3 && options.dimension >= 96));
        if (lower && !log_features_) proposal_degree = options.degree - 1;
        auto tree = log_features_ ? siglasso::make_log_search(*this, lower ? options.degree - 1 : 0) : Backend::make_search();
        if (joint) return std::make_unique<BatchedSearch>(*this, std::move(tree), lower);
        return tree;
    }
    void moments(siglasso::Node& node) override {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), levels = options.degree - k;
        if (!terminal_only_) node.moments = tt_moments(node);
        // For the empty word the polynomial and its segment integral both equal one.
        node.integrals = !k ? node.coefficients : alloc(n * l);
        if (levels == 2) node.linear_integrals = alloc(n * l);
        // P(s) = sum F_a s^a/a!. Integrating P or sP gives the two
        // terminal moments directly from the same state used by the TT bound.
        batch(l, [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) {
                double* integral = node.integrals.data + j * n;
                const double* start = prefix_start(node, j);
                if (k) std::copy_n(start, n, integral);
                double* linear = node.linear_integrals.owner ? node.linear_integrals.data + j * n : nullptr;
                if (linear) for (size_t i = 0; i < n; ++i) linear[i] = .5 * start[i];
                for (size_t a = 1; a <= k; ++a) {
                    const double* plane = node.tt_state.data + ((a - 1) * l + j) * n;
                    const double scale = 1. / weights[a] / (a + 1);
                    for (size_t i = 0; i < n; ++i) integral[i] += scale * plane[i];
                    if (linear) {
                        const double factor = 1. / weights[a] / (a + 2);
                        for (size_t i = 0; i < n; ++i) linear[i] += factor * plane[i];
                    }
                }
            }
        });
        if (levels == 1) { node.coefficients = {}; node.tt_state = {}; }
    }
    void begin_query(const siglasso::Array& residual) override {
        if (endpoints_) for (auto& mask : endpoint_masks_) mask.anchor = SIZE_MAX;
        const size_t n = options.samples, m = options.degree;
        terminal_caps_ = {};
        if (options.length > 1) for (size_t i = 0; i < n; ++i) {
            const double r = std::abs(residual.data[i]), v = variation.data[i] + after.data[i];
            if (m > 1) terminal_caps_[0] += r * std::pow(v, static_cast<double>(m - 1));
            terminal_caps_[1] += r * std::pow(v, static_cast<double>(m));
        }
    }
    size_t search_degree() const override { return proposal_degree ? proposal_degree : options.degree; }
    void contract_terminal(const siglasso::Channel& channel, const double* factors, double* values) const {
        const size_t n = options.samples, l = options.length - 1, stride = channel.dimension * n;
        const hn::ScalableTag<double> tag;
        const size_t lanes = hn::Lanes(tag);
        size_t i = 0;
        // Keep four sample vectors in registers over all path segments.
        for (; i + 4 * lanes <= n; i += 4 * lanes) {
            auto v0 = hn::Zero(tag), v1 = v0, v2 = v0, v3 = v0;
            for (size_t j = 0; j < l; ++j) {
                const auto* d = channel.values.data + j * stride + i;
                const auto* h = factors + j * n + i;
                v0 = hn::Add(v0, hn::Mul(hn::LoadU(tag, d), hn::LoadU(tag, h)));
                v1 = hn::Add(v1, hn::Mul(hn::LoadU(tag, d + lanes), hn::LoadU(tag, h + lanes)));
                v2 = hn::Add(v2, hn::Mul(hn::LoadU(tag, d + 2 * lanes), hn::LoadU(tag, h + 2 * lanes)));
                v3 = hn::Add(v3, hn::Mul(hn::LoadU(tag, d + 3 * lanes), hn::LoadU(tag, h + 3 * lanes)));
            }
            hn::StoreU(v0, tag, values + i); hn::StoreU(v1, tag, values + i + lanes);
            hn::StoreU(v2, tag, values + i + 2 * lanes); hn::StoreU(v3, tag, values + i + 3 * lanes);
        }
        for (; i < n; ++i) {
            double value = 0.;
            for (size_t j = 0; j < l; ++j) value += channel.values.data[j * stride + i] * factors[j * n + i];
            values[i] = value;
        }
    }
    void project_terminal(const double* factors, const siglasso::Array& residual, double rounding, double tiny) {
        const size_t n = options.samples, l = options.length - 1, targets = std::max<size_t>(1, batch_signed.size());
        constexpr size_t stride = projection_panels + 1;
        const size_t size = stride * (1 + targets) * n;
        if (projection_workspace_.size < size) projection_workspace_ = alloc(size);
        auto* sum = projection_workspace_.data;
        auto* perpendicular = sum + projection_panels * n;
        std::fill_n(sum, stride * n, 0.);
        std::array<double, projection_panels> inverse_length;
        for (size_t p = 0; p < projection_panels; ++p) {
            const size_t first = p * l / projection_panels, end = (p + 1) * l / projection_panels;
            inverse_length[p] = end > first ? 1. / (end - first) : 0.;
            auto* part = sum + p * n;
            for (size_t j = first; j < end; ++j) {
                const auto* h = factors + j * n;
                for (size_t i = 0; i < n; ++i) { part[i] += h[i]; perpendicular[i] += h[i] * h[i]; }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const double squared = perpendicular[i];
            double project = 0.;
            for (size_t p = 0; p < projection_panels; ++p)
                project += sum[p * n + i] * sum[p * n + i] * inverse_length[p];
            perpendicular[i] = std::sqrt(std::max(0., squared - project) + rounding * squared + tiny);
        }
        for (size_t t = 0; t < targets; ++t) {
            const auto* r = batch_signed.empty() ? residual.data : batch_signed[t].data;
            auto* v = sum + stride * (1 + t) * n;
            for (size_t p = 0; p < projection_panels; ++p)
                for (size_t i = 0; i < n; ++i) v[p * n + i] = r[i] * sum[p * n + i];
            for (size_t i = 0; i < n; ++i) v[projection_panels * n + i] = std::abs(r[i]) * perpendicular[i];
        }
    }
    // The group intervals enclose the signed projection on panel-constant
    // vectors. Cauchy-Schwarz bounds the orthogonal remainder on each path.
    double projected_radius(size_t index, size_t target) {
        const size_t n = options.samples;
        const double* box = projection_boxes_.data + box_planes * index * n;
        const double* v = projection_workspace_.data + (projection_panels + 1) * (1 + target) * n;
        const auto* remainder = box + (box_planes - 1) * n;
        const auto* perpendicular = v + projection_panels * n;
        if (index < options.dimension) {
            double center = 0.;
            for (size_t p = 0; p < projection_panels; ++p) center += bound_dot(v + p * n, box + 2 * p * n, n);
            return std::abs(center) + bound_dot(perpendicular, remainder, n);
        }
        const hn::ScalableTag<double> tag;
        const size_t lanes = hn::Lanes(tag);
        auto signed_sum = hn::Zero(tag), widths = hn::Zero(tag);
        size_t i = 0;
        for (; i + lanes <= n; i += lanes) {
            for (size_t p = 0; p < projection_panels; ++p) {
                const auto s = hn::LoadU(tag, v + p * n + i);
                signed_sum = hn::MulAdd(s, hn::LoadU(tag, box + 2 * p * n + i), signed_sum);
                widths = hn::MulAdd(hn::Abs(s), hn::LoadU(tag, box + (2 * p + 1) * n + i), widths);
            }
            widths = hn::MulAdd(hn::LoadU(tag, perpendicular + i), hn::LoadU(tag, remainder + i), widths);
        }
        double center = hn::ReduceSum(tag, signed_sum), width = hn::ReduceSum(tag, widths);
        for (; i < n; ++i) {
            for (size_t p = 0; p < projection_panels; ++p) {
                center += v[p * n + i] * box[2 * p * n + i];
                width += std::abs(v[p * n + i]) * box[(2 * p + 1) * n + i];
            }
            width += perpendicular[i] * remainder[i];
        }
        return std::abs(center) + width;
    }
    void terminal_query(const siglasso::Probe& task, const siglasso::Array& residual, double alpha,
        const siglasso::ActiveWords& active,
        size_t capacity, siglasso::SearchResult& result) override {
        const bool restricted_pair = (options.order == SignatureOrder::adaptive || options.order == SignatureOrder::outside) &&
            task.high - task.low < options.dimension;
        if (options.order == SignatureOrder::forward || restricted_pair)
            terminal_search<true, true>(task, residual, alpha, active, capacity, result);
        else if (options.grouped_terminals())
            terminal_search<true>(task, residual, alpha, active, capacity, result);
        else terminal_search<false>(task, residual, alpha, active, capacity, result);
    }
    template<bool Grouped, bool ForwardPair = false>
    void terminal_search(const siglasso::Probe& task, const siglasso::Array& residual, double alpha,
        const siglasso::ActiveWords& active, size_t capacity, siglasso::SearchResult& result) {
        const auto& node = *task.node;
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const size_t k = node.word.size(), levels = (proposal_degree ? proposal_degree : options.degree) - k;
        ++stats.terminal_queries; ++stats.block_queries;
        const auto correlations_before = stats.terminal_correlations;
        if (!l) { ++stats.terminal_pruned; return; }
        const double rounding = siglasso::terminal_roundoff(options.degree, n, l);
        const double slack = rounding * std::max(terminal_caps_[0], terminal_caps_[1]) / n + 1e-15;
        const double tiny = 32. * (n + 1) * (l + 1) * std::numeric_limits<double>::min();
        const size_t size = (levels == 2 ? l + 4 : 2) * n;
        if (visit_column && terminal_workspace_.size < size) terminal_workspace_ = alloc(size);
        auto buffer = visit_column ? terminal_workspace_ : alloc(size);
        double* values = buffer.data;
        double* norms = values + n;
        double* operators = norms + n;
        double* products = levels == 2 ? operators + n : nullptr;
        double* tail = levels == 2 ? products + n * l : nullptr;
        const auto cutoff = [&] {
            return capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
        };
        const auto excluded = [&](double estimate, size_t q) {
            const double bound = (estimate + tiny) * (1 + rounding) * weights[k + q] / n + slack;
            if (!(bound < cutoff())) return false;
            result.omitted = std::max(result.omitted, bound);
            return true;
        };
        std::fill_n(norms, n, 0.);
        if (levels == 2) std::fill_n(operators, n, 0.);
        for (size_t j = 0; j < l; ++j) {
            const double* integral = node.integrals.data + j * n;
            for (size_t i = 0; i < n; ++i) norms[i] += integral[i] * integral[i];
            if (levels == 2) {
                const double* linear = node.linear_integrals.data + j * n;
                for (size_t i = 0; i < n; ++i) {
                    const double diag = integral[i] - linear[i];
                    operators[i] += diag * diag + (l - j - 1) * integral[i] * integral[i];
                }
            }
        }
        const double* first_norm = terminal_sample_norms_.data + norm_index(task.block) * n;
        const double* inner_norm = ForwardPair ? terminal_sample_norms_.data + norm_index(0) * n : first_norm;
        for (size_t i = 0; i < n; ++i) {
            const double r = std::abs(residual.data[i]);
            norms[i] = r * std::sqrt(norms[i] + tiny);
            if (levels == 2) operators[i] = r * std::sqrt(operators[i] + tiny) * inner_norm[i];
        }
        const auto grouped_norm = [&](const double* v) {
            std::array<double, sample_groups> norms{};
            for (size_t g = 0; g < sample_groups; ++g) {
                const size_t first = n * g / sample_groups, end = n * (g + 1) / sample_groups;
                const double sum = bound_dot(v + first, v + first, end - first);
                norms[g] = std::sqrt(sum + tiny);
            }
            return norms;
        };
        using ResponseGroups = std::array<std::array<double, sample_groups>, siglasso::max_response_batch>;
        const auto response_groups = [&](const double* v) {
            ResponseGroups groups{};
            for (size_t t = 0; t < batch_ratios.size(); ++t) for (size_t g = 0; g < sample_groups; ++g) {
                const size_t first = n * g / sample_groups, end = n * (g + 1) / sample_groups;
                const double squared = weighted_square(v + first, batch_ratios[t].data + first, end - first);
                groups[t][g] = std::sqrt(squared + tiny);
            }
            return groups;
        };
        const auto coarse = [&](const auto& group, size_t a) {
            double sum = 0;
            for (size_t g = 0; g < sample_groups; ++g) sum += group[g] * channel_group_norms_.data[a * sample_groups + g];
            return sum;
        };
        const auto separated = [&](const ResponseGroups& groups, size_t a, size_t q) {
            if (batch_ratios.size() < 2) return false;
            double largest = 0.;
            for (size_t t = 0; t < batch_ratios.size(); ++t) {
                largest = std::max(largest, coarse(groups[t], a));
                if ((largest + tiny) * (1 + rounding) * weights[k + q] / n + slack >= cutoff()) return false;
            }
            return excluded(largest, q);
        };
        // The union envelope can be loose when different samples have their
        // largest residual in different responses. Refine the signed leaf's
        // Cauchy radius before running its full time contraction.
        auto individual = [&](const double* v, const double* channel_norm, size_t q) {
            if (batch_ratios.size() < 2) return false;
            for (size_t i = 0; i < n; ++i) values[i] = v[i] * channel_norm[i];
            double maximum = 0.;
            for (const auto& ratio : batch_ratios) {
                maximum = std::max(maximum, bound_dot(values, ratio.data, n));
                if ((maximum + tiny) * (1 + rounding) * weights[k + q] / n + slack >= cutoff()) return false;
            }
            return excluded(maximum, q);
        };
        siglasso::Word word = node.word; word.push_back(0);
        const auto& floor = task.range.floor;
        const auto& ceiling = task.range.ceiling;
        const size_t floor_a = Grouped && floor.length ? floor.letter(node.word) : 0;
        const size_t ceiling_a = Grouped && ceiling.length ? ceiling.letter(node.word) : d;
        const size_t floor_b = Grouped && floor.length ? floor.letter(node.word, 1) : 0;
        const size_t ceiling_b = Grouped && ceiling.length ? ceiling.letter(node.word, 1) : d;
        const auto cut = Grouped && options.order == SignatureOrder::lyndon && k && !floor.length ? siglasso::last_factor(node.word) : siglasso::PeriodicWord{};
        const EndpointMask* last[2]{};
        if (Grouped && endpoints_ && k) for (size_t q = 1; q <= levels; ++q) {
            last[q - 1] = endpoint_mask(node.word.front(), k + q, residual, alpha);
            if (last[q - 1]) result.omitted = std::max(result.omitted, last[q - 1]->omitted);
        }
        // Test a dyadic alphabet block using the largest channel norm in each
        // sample group. Leaves share the exact signed contraction below.
        const auto screened_channels = [&](size_t first, size_t end, size_t block, const auto& group,
                const auto& separate, size_t q, auto&& refine, auto&& visit_range) {
            if (!Grouped || end <= first + 3) {
                visit_range(first, end);
                return;
            }
            while (block < alphabet.size()) {
                const auto [lo, hi] = alphabet[block];
                if (lo >= end) break;
                const size_t span = hi - lo;
                if (hi <= first) { block += 2 * span - 1; continue; }
                if (span >= 4) {
                    const auto index = norm_index(block);
                    if (excluded(coarse(group, index), q) || separated(separate, index, q) || refine(index)) {
                        block += 2 * span - 1; continue;
                    }
                }
                if (span <= 3) {
                    visit_range(std::max(first, lo), std::min(end, hi));
                    block += 2 * span - 1;
                } else ++block;
            }
        };
        const auto visit = [&](size_t q, const double* factors) {
            const bool pair = q == 2;
            const size_t group_block = ForwardPair && pair ? 0 : task.block;
            size_t first = ForwardPair && pair ? 0 : task.low, end = ForwardPair && pair ? d : task.high;
            if (Grouped && floor.length && (!pair || word[k + 1] <= floor_b)) first = std::max(first, floor_a + 1);
            if (Grouped && ceiling.length && pair && word[k + 1] > ceiling_b) end = std::min(end, ceiling_a);
            if (first >= end) return;
            const auto* inner_mask = Grouped && (ForwardPair || !pair) ? last[q - 1] : nullptr;
            if (inner_mask && !inner_mask->any(first, end)) return;
            if (excluded(bound_dot(norms, ForwardPair && pair ? inner_norm : first_norm, n), q)) return;
            const auto group = grouped_norm(norms);
            const auto separate = response_groups(norms);
            bool projected = false;
            const auto projection = [&](size_t index) {
                if (primitive_norms_.owner) {
                    if (!projected) {
                        if (!difference_norms_.owner) difference_norms_ = alloc(2 * n);
                        double* diff = difference_norms_.data + n;
                        std::fill_n(diff, n, 0.);
                        for (size_t j = 1; j < l; ++j) for (size_t i = 0; i < n; ++i) {
                            const double delta = factors[j * n + i] - factors[(j - 1) * n + i];
                            diff[i] += delta * delta;
                        }
                        for (size_t i = 0; i < n; ++i) {
                            difference_norms_.data[i] = std::abs(residual.data[i] * factors[i]);
                            diff[i] = std::abs(residual.data[i]) * std::sqrt(diff[i] + tiny);
                        }
                        projected = true;
                    }
                    const double* endpoint = primitive_norms_.data + 2 * index * n;
                    const double* direct = terminal_sample_norms_.data + index * n;
                    for (size_t i = 0; i < n; ++i) {
                        const double primitive = difference_norms_.data[i] * endpoint[i]
                            + difference_norms_.data[n + i] * endpoint[n + i];
                        values[i] = std::min(norms[i] * direct[i], primitive);
                    }
                    if (batch_ratios.empty()) return excluded(absolute_sum({values, n, {}}), q);
                    double maximum = 0.;
                    for (const auto& ratio : batch_ratios) {
                        maximum = std::max(maximum, bound_dot(values, ratio.data, n));
                        if (!((maximum + tiny) * (1 + rounding) * weights[k + q] / n + slack < cutoff())) return false;
                    }
                    return excluded(maximum, q);
                }
                if (!projection_boxes_.owner) return false;
                const size_t targets = std::max<size_t>(1, batch_signed.size());
                // At a single channel a full signed column costs one dot per
                // path. Several panel tests per target can exceed that work.
                if (index < d && targets * (projection_panels + 1) > l) return false;
                if (!projected) {
                    project_terminal(factors, residual, rounding, tiny);
                    projected = true;
                }
                double maximum = 0.;
                for (size_t t = 0; t < targets; ++t) {
                    const double value = projected_radius(index, t);
                    if (!std::isfinite(value) || !((value + tiny) * (1 + rounding) * weights[k + q] / n + slack < cutoff())) return false;
                    maximum = std::max(maximum, value);
                }
                return excluded(maximum, q);
            };
            const auto visit_range = [&](size_t low, size_t high) {
                for (size_t rank = low; rank < high; ++rank) {
                    const size_t a = Grouped ? letter(rank) : rank;
                    if (inner_mask && !inner_mask->any(rank, rank + 1)) continue;
                    word[k + (ForwardPair && pair)] = a;
                    if (excluded(coarse(group, a), q) || separated(separate, a, q) || active.count(word) ||
                        excluded(bound_dot(norms, terminal_sample_norms_.data + a * n, n), q) ||
                        individual(norms, terminal_sample_norms_.data + a * n, q) || (Grouped && projection(a))) continue;
                    contract_terminal(channel(a), factors, values);
                    ++stats.terminal_correlations; ++stats.evaluated;
                    if (visit_column) { visit_column(word, values); continue; }
                    const double score = siglasso::finite(std::abs(dot(values, residual.data, n)) / n * weights[k + q]);
                    result.omitted = std::max(result.omitted, siglasso::guard(score));
                    if (score > alpha && siglasso::shortlist(result.candidates, capacity, word, score, [&] {
                        auto column = alloc(n); copy(values, column.data, n); return column;
                    })) ++stats.constructed;
                }
            };
            if constexpr (!Grouped) {
                visit_range(first, end);
            } else if (cut.length) {
                const auto split = std::clamp<size_t>(cut.letter(node.word) +
                    (!pair || word[k + 1] <= cut.letter(node.word, 1)), first, end);
                if (!cut.start) screened_channels(split, end, task.block, group, separate, q, projection, visit_range);
                screened_channels(first, split, task.block, group, separate, q, projection, visit_range);
                if (cut.start) screened_channels(split, end, task.block, group, separate, q, projection, visit_range);
            } else screened_channels(first, end, group_block, group, separate, q, projection, visit_range);
        };
        if (!highest_degree_only || !visit_column || k + 1 == options.degree) visit(1, node.integrals.data);
        if (levels == 2 && (!Grouped || !last[1] || last[1]->any(0, d))) {
            word.push_back(0);
            const auto group = grouped_norm(operators);
            const auto separate = response_groups(operators);
            const size_t first_b = Grouped && floor.length && task.high == floor_a + 1 ? floor_b + 1 : 0;
            const size_t end_b = Grouped && ceiling.length && task.low == ceiling_a ? ceiling_b + 1 : d;
            const auto second_letters = [&](size_t low, size_t high) {
                for (size_t rank = low; rank < high; ++rank) {
                    const size_t b = Grouped ? letter(rank) : rank;
                    if (Grouped && !ForwardPair && last[1] && !last[1]->any(rank, rank + 1)) continue;
                    if (excluded(coarse(group, b), 2) || separated(separate, b, 2)) continue;
                    if (excluded(bound_dot(operators, terminal_sample_norms_.data + b * n, n), 2) ||
                        individual(operators, terminal_sample_norms_.data + b * n, 2)) continue;
                    word[k + (ForwardPair ? 0 : 1)] = b;
                    const auto dx = channel(b);
                    std::fill_n(tail, n, 0.); std::fill_n(norms, n, 0.);
                    for (size_t s = 0; s < l; ++s) {
                        const size_t j = ForwardPair ? s : l - 1 - s;
                        const double* step = dx.values.data + j * dx.dimension * n;
                        const double* integral = node.integrals.data + j * n;
                        const double* linear = node.linear_integrals.data + j * n;
                        for (size_t i = 0; i < n; ++i) {
                            double h;
                            if constexpr (ForwardPair) {
                                h = tail[i] + (integral[i] - linear[i]) * step[i];
                                tail[i] += integral[i] * step[i];
                            } else {
                                tail[i] += step[i];
                                h = integral[i] * tail[i] - linear[i] * step[i];
                            }
                            products[j * n + i] = h; norms[i] += h * h;
                        }
                    }
                    for (size_t i = 0; i < n; ++i) norms[i] = std::abs(residual.data[i]) * std::sqrt(norms[i] + tiny);
                    visit(2, products);
                }
            };
            if constexpr (!Grouped) {
                second_letters(0, d);
            } else if constexpr (ForwardPair) {
                screened_channels(task.low, task.high, task.block, group, separate, 2, [](size_t) { return false; }, second_letters);
            } else screened_channels(first_b, end_b, 0, group, separate, 2, [](size_t) { return false; }, second_letters);
        }
        if (correlations_before == stats.terminal_correlations) ++stats.terminal_pruned;
    }
    // Contract the nonnegative suffix envelope with the residual over each time panel.
    double panel_bound(const siglasso::Probe& task, const double* moments, const siglasso::Array& residual) {
        const size_t n = options.samples;
        if (compact_panels_) {
            if (!panel_scratch_.owner) panel_scratch_ = alloc(n);
            return compact_bound(task.block, moments, residual, panel_scratch_.data);
        }
        const hn::ScalableTag<double> tag;
        const size_t lanes = hn::Lanes(tag);
        auto accumulator = hn::Zero(tag);
        double total = 0.;
        for (size_t j = 0; j < panels(); ++j) {
            const double* step = alphabet_ratios.data + (task.block * panels() + j) * n;
            const double* moment = moments + j * n;
            size_t i = 0;
            for (; i + lanes <= n; i += lanes) {
                auto weighted = hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::LoadU(tag, moment + i));
                accumulator = hn::Add(accumulator, hn::Mul(weighted, hn::LoadU(tag, step + i)));
            }
            for (; i < n; ++i) total += std::abs(residual.data[i]) * moment[i] * step[i];
        }
        return siglasso::finite(total + hn::ReduceSum(tag, accumulator));
    }

    double bound(const siglasso::Probe& task, const siglasso::Array& residual, double threshold) override {
        const size_t n = options.samples, k = task.node->word.size();
        const size_t first = highest_degree_only && visit_column ? options.degree - k - 1 : 0;
        const size_t levels = (proposal_degree ? proposal_degree : options.degree) - k;
        const double slack = siglasso::terminal_roundoff(options.degree, n, options.length - 1) * absolute_sum(residual) / n;
        double best = 0.;
        for (size_t q = levels; q-- > first;) {
            if (endpoints_ && k) {
                const auto* mask = endpoint_mask(task.node->word.front(), k + q + 1, residual, threshold);
                if (mask && !mask->any(q ? 0 : task.low, q ? options.dimension : task.high)) {
                    best = std::max(best, mask->omitted);
                    continue;
                }
            }
            double total = panel_bound(task, task.node->moments.at(q * panels() * n), residual) + slack;
            if (q == 0 && total >= threshold && task.high - task.low > 1 && increments.owner)
                total = std::min(total, siglasso::finite(energy_bound(task, residual) * weights[k + 1] / n));
            best = std::max(best, siglasso::finite(total));
            if (best >= threshold) break;
        }
        return best;
    }
    // Only I=integral(P) and J=integral(sP) are consumed by a terminal query.
    // For the child Q(s)=carry+D*integral_0^s P(t)dt, a parent monomial
    // F_a*s^a/a! contributes D*F_a/(a+1)! times 1/(a+2) or 1/(a+3).
    siglasso::NodePtr terminal_child(const siglasso::Probe& task) override {
        const size_t n = options.samples, l = options.length - 1;
        const auto& parent = *task.node;
        const size_t k = parent.word.size(), letter = this->letter(task.low);
        auto child = std::make_shared<siglasso::Node>();
        child->word = parent.word; child->word.push_back(letter);
        child->metadata = account(512 + child->word.size() * 64);
        const bool two = (proposal_degree ? proposal_degree : options.degree) - k - 1 == 2;
        if (visit_column) {
            const size_t size = n + (two ? 2 : 1) * n * l;
            if (terminal_child_storage_.size < size || terminal_child_storage_.owner.use_count() != 1)
                terminal_child_storage_ = alloc(size);
            auto& slab = terminal_child_storage_;
            child->values = {slab.data, n, slab.owner};
            child->integrals = {slab.data + n, n * l, slab.owner};
            if (two) child->linear_integrals = {slab.data + n + n * l, n * l, slab.owner};
        } else {
            child->values = alloc(n); child->integrals = alloc(n * l);
            if (two) child->linear_integrals = alloc(n * l);
        }
        fill(child->values, 0.);
        const auto steps = channel(letter);
        for (size_t j = 0; j < l; ++j) {
            const double* dx = steps.values.data + j * steps.dimension * n;
            double* I = child->integrals.data + j * n;
            double* J = child->linear_integrals.owner ? child->linear_integrals.data + j * n : nullptr;
            const double* start = prefix_start(parent, j);
            for (size_t i = 0; i < n; ++i) I[i] = start[i] * .5;
            if (J) for (size_t i = 0; i < n; ++i) J[i] = start[i] / 3.;
            for (size_t a = 1; a <= k; ++a) {
                const double* F = parent.tt_state.data + ((a - 1) * l + j) * n;
                const double scale = 1. / weights[a + 1];
                axpy(scale / (a + 2), F, I, n);
                if (J) axpy(scale / (a + 3), F, J, n);
            }
            if (J) for (size_t i = 0; i < n; ++i) J[i] = .5 * child->values.data[i] + dx[i] * J[i];
            for (size_t i = 0; i < n; ++i) {
                I[i] = child->values.data[i] + dx[i] * I[i];
                child->values.data[i] += dx[i] * parent.integrals.data[j * n + i];
            }
        }
        return child;
    }
    siglasso::NodePtr prediction_child(const siglasso::NodePtr& parent, size_t letter, bool extendible) override {
        const size_t n = options.samples, l = options.length - 1, h = parent->word.size() + 1;
        auto node = std::make_shared<siglasso::Node>();
        node->word = parent->word; node->word.push_back(letter);
        node->metadata = account(512 + h * 64);
        node->values = alloc(n); fill(node->values, 0.);
        if (extendible) node->coefficients = alloc(siglasso::multiply(h + 1, l * n));
        const auto steps = channel(letter);
        for (size_t j = 0; j < l; ++j) {
            const double* dx = steps.values.data + j * steps.dimension * n;
            if (extendible) copy(node->values.data, node->coefficients.data + j * (h + 1) * n, n);
            for (size_t a = 0; a < h; ++a) {
                const double* plane = parent->coefficients.data + (j * h + a) * n;
                for (size_t i = 0; i < n; ++i) node->values.data[i] += (dx[i] / h) * plane[i];
                if (extendible) copy(node->values.data, node->coefficients.data + (j * (h + 1) + a + 1) * n, n);
            }
        }
        return node;
    }
    siglasso::NodePtr child(const siglasso::NodePtr& parent, size_t letter) override {
        const size_t n = options.samples, l = options.length - 1, k = parent->word.size();
        auto node = std::make_shared<siglasso::Node>();
        node->word = parent->word;
        node->word.push_back(letter);
        node->metadata = account(512 + node->word.size() * 64);
        node->values = alloc(n);
        fill(node->values, 0.);
        if (k + 1 < options.degree) {
            node->coefficients = alloc(n * l);
            node->tt_state = alloc((k + 1) * n * l);
        }
        const auto steps = channel(letter);
        for (size_t j = 0; j < l; ++j) {
            if (node->coefficients.owner) copy(node->values.data, node->coefficients.data + j * n, n);
            const double* step = steps.values.data + j * steps.dimension * n;
            const double* integral = parent->integrals.data + j * n;
            for (size_t i = 0; i < n; ++i) node->values.data[i] += step[i] * integral[i];
        }
        if (node->tt_state.owner) {
            const hn::ScalableTag<double> tag;
            const size_t lanes = hn::Lanes(tag);
            for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j) {
                const double* source = a ? parent->tt_state.data + ((a - 1) * l + j) * n : prefix_start(*parent, j);
                const double* increment = steps.values.data + j * steps.dimension * n;
                double* target = node->tt_state.data + (a * l + j) * n;
                size_t i = 0;
                for (; i + lanes <= n; i += lanes) hn::StoreU(hn::Mul(hn::LoadU(tag, source + i), hn::LoadU(tag, increment + i)), tag, target + i);
                for (; i < n; ++i) target[i] = source[i] * increment[i];
            }
        }
        return node;
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
    void release_fit_workspace() {
        searches_.clear(); search_workers_.clear(); channels_.clear();
        for (auto& shard : cold_) shard.values = {};
        increments = {}; variation = {}; after = {}; alphabet_ratios = {}; compact_panels_.reset(); tt_suffix_ = {};
        responses_ = {}; normalization_ = {}; endpoint_variation_ = {}; terminal_workspace_ = {}; terminal_child_storage_ = {}; terminal_sample_norms_ = {}; channel_group_norms_ = {};
        panel_scratch_ = {};
        projection_boxes_ = {}; projection_workspace_ = {};
        primitive_norms_ = {}; difference_norms_ = {};
        endpoints_.reset(); endpoint_weights_ = {}; endpoint_masks_ = {};
    }
    std::unique_ptr<siglasso::Backend> terminal_worker() override {
        auto single = options;
        single.n_jobs = 1;
        auto worker = std::make_unique<LassoCPU>(*this, single);
        worker->channel_source_ = this;
        return worker;
    }
    bool parallel_search(const siglasso::NodePtr& root, const siglasso::Array& residual, double alpha,
        const siglasso::ActiveWords& active, size_t capacity, siglasso::SearchResult& result) override {
        if (options.n_jobs == 1 || options.dimension == 1 || options.memory_limit) return false;
        if (searches_.empty()) {
            const size_t threads = options.n_jobs;
            const size_t parts = threads > 1 ? 4 * threads : threads;
            frontier_.push_back(0);
            while (frontier_.size() < std::min(parts, static_cast<size_t>(options.dimension))) {
                auto largest = std::max_element(frontier_.begin(), frontier_.end(), [&](size_t a, size_t b) {
                    const size_t x = alphabet[a].second - alphabet[a].first, y = alphabet[b].second - alphabet[b].first;
                    if (root_work_.empty() || x == 1 || y == 1 || root_work_[a] == root_work_[b]) return x < y;
                    return root_work_[a] < root_work_[b];
                });
                const auto [low, high] = alphabet[*largest];
                const size_t block = *largest;
                *largest = block + 1; frontier_.push_back(block + 2 * ((low + high) / 2 - low));
            }
            if (root_work_.empty()) std::sort(frontier_.begin(), frontier_.end());
            else std::sort(frontier_.begin(), frontier_.end(), [&](size_t a, size_t b) {
                return root_work_[a] == root_work_[b] ? a < b : root_work_[a] > root_work_[b];
            });
            auto single = options;
            single.n_jobs = 1;
            const size_t count = std::min(frontier_.size(), threads);
            for (size_t j = 0; j < count; ++j) {
                auto worker = std::make_unique<LassoCPU>(*this, single);
                worker->increments = increments; worker->variation = variation; worker->after = after;
                worker->alphabet = alphabet; worker->alphabet_ratios = alphabet_ratios;
                worker->channel_source_ = this;
                searches_.push_back(std::make_unique<siglasso::Search>(*worker, root));
                search_workers_.push_back(std::move(worker));
            }
        }
        std::vector<siglasso::SearchResult> found(frontier_.size());
        std::atomic<size_t> next{0};
        spawn_batch_threads(searches_.size(), options.n_jobs, [&](size_t begin, size_t end) {
            for (size_t worker_id = begin; worker_id < end; ++worker_id) {
                auto& worker = *search_workers_[worker_id];
                worker.stats = {};
                worker.proposal_degree = proposal_degree;
                worker.highest_degree_only = highest_degree_only;
                worker.visit_column = visit_column; worker.batch_absolute = batch_absolute; worker.batch_ratios = batch_ratios; worker.batch_signed = batch_signed;
                for (size_t j = next.fetch_add(1); j < frontier_.size(); j = next.fetch_add(1))
                    found[j] = searches_[worker_id]->query_block(residual, alpha, active, capacity, frontier_[j]);
                worker.visit_column = {}; worker.batch_absolute.clear(); worker.batch_ratios.clear(); worker.batch_signed.clear();
            }
        });
        for (size_t j = 0; j < searches_.size(); ++j) {
            const auto& worker = *search_workers_[j];
            stats.constructed += worker.stats.constructed;
            stats.evaluated += worker.stats.evaluated; stats.pruned += worker.stats.pruned; stats.block_queries += worker.stats.block_queries;
            stats.cache_hits += worker.stats.cache_hits; stats.cache_misses += worker.stats.cache_misses;
            stats.terminal_queries += worker.stats.terminal_queries; stats.terminal_pruned += worker.stats.terminal_pruned;
            stats.terminal_correlations += worker.stats.terminal_correlations;
        }
        for (size_t j = 0; j < found.size(); ++j) {
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

// Fit and prediction share the packed word representation and worker convention.
static siglasso::SearchOptions search_options(const SigLassoOptions* options, uint32_t feature_type,
        const uint64_t* letters, const uint64_t* offsets, uint64_t words, bool predict) {
    if (!options || !options->samples || !options->length || !options->dimension || !options->targets ||
        !options->degree || options->degree > 170 || (predict && feature_type > 1))
        throw std::invalid_argument("Invalid Lasso dimensions, degree or feature type");
    if (words && (!letters || !offsets || offsets[0] != 0))
        throw std::invalid_argument("Invalid packed Lasso words");
    for (size_t j = 0; j < words; ++j) {
        if (offsets[j + 1] <= offsets[j] || offsets[j + 1] - offsets[j] > options->degree)
            throw std::invalid_argument("Invalid Lasso word length");
        for (size_t k = offsets[j]; k < offsets[j + 1]; ++k)
            if (letters[k] >= options->dimension) throw std::invalid_argument("Invalid Lasso word letter");
    }
    siglasso::SearchOptions result(*options, predict);
    if (feature_type <= 4) {
        if (feature_type == 2 || feature_type == 4) result.order = SignatureOrder::lyndon;
        if (feature_type == 2 || feature_type == 3) result.terminal = TerminalBound::projected;
    } else {
        if (feature_type < 16 || (feature_type - 16) / 4 > static_cast<uint32_t>(SignatureOrder::adaptive) ||
            feature_type % 4 > static_cast<uint32_t>(TerminalBound::primitive))
            throw std::invalid_argument("Invalid signature search mode");
        result.order = static_cast<SignatureOrder>((feature_type - 16) / 4);
        result.terminal = static_cast<TerminalBound>(feature_type % 4);
    }
    const int64_t jobs = options->n_jobs < 0 ? int64_t(get_max_threads()) + 1 + options->n_jobs : options->n_jobs;
    if (jobs < 1 || jobs > std::numeric_limits<int>::max()) throw std::invalid_argument("Invalid n_jobs");
    result.n_jobs = static_cast<int>(jobs);
    return result;
}

template<typename T>
void sig_working_set_(const T* path, const T* y, const double* alphas, const uint64_t* letters,
    const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options, SigLassoAllocator allocate, void* context) {
    if (!path || !y || !alphas || !allocate)
        throw std::invalid_argument("Missing working-set input or output allocator");
    const auto settings = search_options(options, feature_type, letters, offsets, words, false);
    if (!options->max_iter || !std::isfinite(options->tolerance) || options->tolerance <= 0 || options->loss > 3)
        throw std::invalid_argument("Invalid working-set size, loss or tolerance");
    for (size_t t = 0; t < options->targets; ++t)
        if (!std::isfinite(alphas[t]) || alphas[t] <= 0) throw std::invalid_argument("Expected a positive finite penalty");
    LassoCPU backend(settings, path, y, feature_type == 1);
    std::vector<siglasso::Word> initial;
    for (size_t j = 0; j < words; ++j) {
        auto& word = initial.emplace_back(letters + offsets[j], letters + offsets[j + 1]);
        if (settings.order == SignatureOrder::suffix) std::reverse(word.begin(), word.end());
    }
    auto result = siglasso::fit(backend, feature_type == 1, std::vector<double>(alphas, alphas + options->targets), initial);
    backend.release_fit_workspace();
    siglasso::export_result(backend, result, allocate, context);
}
template<typename T>
void sig_working_set_predict_(const T* path, const T* coefficients, const T* intercept, T* output,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options) {
    if (!path || !intercept || !output || (words && !coefficients))
        throw std::invalid_argument("Missing Lasso prediction input or output");
    const auto settings = search_options(options, feature_type, letters, offsets, words, true);
    const size_t n = options->samples, targets = options->targets, stride = siglasso::multiply(options->length, options->dimension);
    for (size_t t = 0; t < targets; ++t) siglasso::finite(intercept[t]);
    for (size_t j = 0; j < siglasso::multiply(words, targets); ++j) siglasso::finite(coefficients[j]);
    auto budget = std::make_shared<siglasso::Budget>(options->memory_limit);
    std::vector<size_t> order;
    std::shared_ptr<void> indices;
    if (options->degree > 1 && words * sizeof(size_t) <= budget->available() / 4) {
        indices = budget->reserve(siglasso::multiply(words, sizeof(size_t)), budget);
        order.resize(words); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return std::lexicographical_compare(letters + offsets[a], letters + offsets[a + 1], letters + offsets[b], letters + offsets[b + 1]);
        });
    }
    // Bound the transposed path buffer independently of the input batch size.
    // Under a tighter user limit, retry a smaller tile after releasing its workspace.
    size_t tile = std::clamp<size_t>((2u << 20) / sizeof(double) / std::max<size_t>(1, stride), 1, 256);
    for (size_t start = 0; start < n;) {
        const size_t count = std::min(tile, n - start);
        try {
            auto opt = settings;
            opt.samples = count;
            LassoCPU backend(opt, path + start * stride, static_cast<const T*>(nullptr), feature_type == 1, budget);
            backend.prepare(); auto search = backend.make_search();
            auto prediction = backend.alloc(siglasso::multiply(count, targets));
            for (size_t i = 0; i < count; ++i) std::copy_n(intercept, targets, prediction.data + i * targets);
            for (size_t k = 0; k < words; ++k) {
                const size_t j = order.empty() ? k : order[k];
                const siglasso::Word word(letters + offsets[j], letters + offsets[j + 1]);
                const size_t next = k + 1 < words ? (order.empty() ? k + 1 : order[k + 1]) : j;
                const bool extendible = k + 1 < words && offsets[next + 1] - offsets[next] > word.size() &&
                    std::equal(word.begin(), word.end(), letters + offsets[next]);
                const auto node = search->selected(word, extendible);
                const double weight = backend.weights[word.size()];
                for (size_t t = 0; t < targets; ++t) {
                    const double beta = coefficients[j * targets + t];
                    if (beta) for (size_t i = 0; i < count; ++i) prediction.data[i * targets + t] += (node->values.data[i] * weight) * beta;
                }
            }
            for (size_t i = 0; i < prediction.size; ++i) output[start * targets + i] = static_cast<T>(siglasso::finite(static_cast<T>(prediction.data[i])));
        } catch (const std::bad_alloc&) {
            if (!options->memory_limit || count == 1) throw;
            tile = (count + 1) / 2;
            continue;
        }
        start += count;
    }
}
template void sig_working_set_<float>(const float*, const float*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_<double>(const double*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_predict_<float>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);
template void sig_working_set_predict_<double>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);

} } // namespace cpsig::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
