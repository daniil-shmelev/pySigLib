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
#include <bit>
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
        if (o.selected_only || o.experiment || o.memory_limit || o.loss || o.single_precision || o.targets < 2 ||
            o.degree < 2 || o.degree > 4 || o.dimension < 2 || o.cache_bytes || o.summary_flags) return false;
        if (feature_type_ == 1) return o.feature_kernel == 2039;
        return feature_type_ == 0 && bound_type_ == 1 && o.terminal_kernel == siglasso::signed_terminal_kernel &&
            o.search_method == 3 && o.traversal_order == 4;
    }
    bool channels_ready_ = false;
    std::vector<siglasso::Array> channels_;
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
    template<size_t width, class T>
    void load_channels(const T* paths, size_t first, const std::array<double*, width>& output) const {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        for (size_t i = 0; i < n; ++i) {
            const T* x = paths + i * (l + 1) * d + first;
            const double total = normalization_.data[i];
            const double reciprocal = total ? 1. / total : 0.;
            std::array<double, width> previous{};
            for (size_t j = 0; j < l; ++j) for (size_t c = 0; c < width; ++c) {
                const double delta = double(x[(j + 1) * d + c]) - x[c];
                const double position = normalized_position(delta, total, reciprocal);
                output[c][j * n + i] = position - previous[c];
                previous[c] = position;
            }
        }
    }
    template<size_t width>
    void read_channels(size_t first, const std::array<double*, width>& output) const {
        if (input_float_) load_channels(static_cast<const float*>(original_paths_), first, output);
        else load_channels(static_cast<const double*>(original_paths_), first, output);
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
        if (factor_parent_) return factor_parent_->channel(a);
        if (!channels_ready_) return Backend::channel(a);
        if (pinned_[a]) return {channels_[a], 1};
        // Terminal loops visit neighbouring letters. Decode the pair in one
        // pass through the caller's sample-major data and recycle idle buffers.
        const size_t first = a - a % 2;
        auto& shard = cold_[(first / 2) % cold_shards_];
        std::lock_guard lock(shard.mutex);
        if (shard.first != first) {
            const size_t last = std::min<size_t>(first + 2, options.dimension);
            for (size_t c = first; c < last; ++c) if (!pinned_[c]) {
                auto& buffer = shard.values[c - first];
                if (buffer.owner.use_count() != 1) buffer = alloc(options.samples * (options.length - 1));
            }
            if (last - first == 2 && !pinned_[first] && !pinned_[first + 1])
                read_channels<2>(first, {shard.values[0].data, shard.values[1].data});
            else read_channels<1>(a, {shard.values[a - first].data});
            shard.first = first;
        }
        return {shard.values[a - first], 1};
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
                variation.data[j * n + i] = maximum;
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
        const int jobs = options.n_jobs > 0 ? options.n_jobs : static_cast<int>(get_max_threads()) + 1 + options.n_jobs;
        if (jobs < 1) throw std::invalid_argument("Invalid n_jobs");
        const size_t workers = std::min({n, static_cast<size_t>(jobs), size_t{8}});
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
        const size_t hot_channels = feature_type_ == 1 ? std::min<size_t>(8, d / 4) : std::min<size_t>(48, d / 2);
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
                read_channels<1>(a, {channels_[a].data});
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
            const std::unordered_map<siglasso::Word,size_t,siglasso::WordHash>& active, size_t capacity,
            size_t, size_t, size_t) override {
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
    LassoCPU* factor_parent_ = nullptr;
    std::mutex factor_mutex_;
    std::mutex terminal_mutex_;
    siglasso::Array terminal_storage_, terminal_products_, terminal_matrix_workspace_, terminal_channel_norms_;
    siglasso::Array terminal_sample_norms_, terminal_operator_weights_, terminal_suffix_bounds_;
    siglasso::TerminalCache terminal_cache_;
    siglasso::Array channel_group_norms_;
    static constexpr size_t sample_groups = 8;
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
    LassoCPU(const LassoCPU& source, const siglasso::SearchOptions& options) : Backend(options, source.budget), responses_(source.responses_) {
        weights = source.weights;
        bound_type_ = source.bound_type_; tt_suffix_ = source.tt_suffix_; terminal_only_ = source.terminal_only_;
        terminal_channel_norms_ = source.terminal_channel_norms_; channel_group_norms_ = source.channel_group_norms_;
        terminal_sample_norms_ = source.terminal_sample_norms_;
        compact_panels_ = source.compact_panels_;
    }
    template<typename T>
    LassoCPU(const siglasso::SearchOptions& o, const T* paths, const T* y, uint32_t feature_type = 0,
             std::shared_ptr<siglasso::Budget> budget = {}) : Backend(o, std::move(budget)), feature_type_(feature_type) {
        if (!feature_type && !o.selected_only) bound_type_ = siglasso::experimental_signature_bound.load();
        terminal_only_ = y && o.degree == 2 && bound_type_ == 1 && (o.search_method & 2) &&
            o.terminal_kernel == siglasso::signed_terminal_kernel && !o.experiment && !o.summary_flags;
        const size_t segments = o.length - 1;
        if (y && joint_search_enabled() && o.length > 1 && o.dimension >= (feature_type_ == 1 ? 8u : 24u)) {
            original_paths_ = paths; input_float_ = std::is_same_v<T, float>; normalization_ = alloc(o.samples);
        }
        if (o.degree == 1 && !o.experiment && (y || o.selected_only)) {
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
                        increments.data[(j * o.dimension + a) * o.samples + i] = position - previous;
                        previous = position;
                    }
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
    void correlations(const std::vector<siglasso::Column>& columns, const siglasso::Array& v, std::vector<double>& out, const std::vector<size_t>* selected = nullptr) override {
        out.resize(selected ? selected->size() : columns.size());
        auto compute = [&](uint64_t begin, uint64_t end) {
            for (size_t j = begin; j < end; ++j) {
                const auto& column = columns[selected ? (*selected)[j] : j].values;
                out[j] = dot(column.data, v.data, v.size) / v.size;
            }
        };
        if (options.n_jobs == 1 || out.size() <= 1) compute(0, out.size());
        else spawn_batch_threads(out.size(), options.n_jobs, compute);
    }
    size_t panel_size() const { return bound_type_ == 1 ? 6 : 1; }
    size_t panels() const { return (options.length - 1 + panel_size() - 1) / panel_size(); }
    bool absolute_steps() const { return bound_type_ == 1 || bound_type_ >= 3; }
    const double* step_envelope(size_t block, size_t low, size_t high, size_t j) const {
        const size_t n = options.samples, l = options.length - 1;
        if (!absolute_steps() || panel_size() > 1) return alphabet_ratios.data + (block * panels() + j) * n;
        if (high - low == 1) return increments.data + (j * options.dimension + low) * n;
        // In preorder, precisely 'low' leaves precede this internal node.
        return alphabet_ratios.data + ((block - low) * l + j) * n;
    }
    size_t norm_index(size_t block) const {
        const auto [low, high] = alphabet[block];
        return high - low == 1 ? low : options.dimension + block - low;
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
                const auto steps = channel(low);
                for (size_t g = 0; g < count; ++g) {
                    auto* out = table.codes.data() + (block * count + g) * n;
                    for (size_t j = g * panel_size(); j < std::min(l, (g + 1) * panel_size()); ++j) {
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
    void prepare() override {
        if (endpoint_input_) return;
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const bool segmented = options.terminal_kernel == 6 || options.terminal_kernel == 8 || options.terminal_kernel == 9;
        prepare_channels();
        if (!options.selected_only && increments.owner) {
            variation = alloc(n * l); after = alloc(n * l);
            batch(n, [&](uint64_t begin, uint64_t end) {
                for (size_t i = begin; i < end; ++i)
                    siglasso::prepare_sample(increments.data, variation.data, after.data, i, n, l, d);
            });
        }
        if (feature_type_ || options.selected_only) return;
        if (options.terminal_kernel >= 5) {
            terminal_channel_norms_ = alloc(siglasso::multiply(d, segmented ? l + 1 : 1));
            const double error = siglasso::terminal_roundoff(0, n * l, l);
            for (size_t a = 0; a < d; ++a) {
                const auto steps = channel(a);
                double squared = 0;
                for (size_t j = 0; j < l; ++j) {
                    const double* x = steps.values.data + j * steps.dimension * n;
                    const double part = dot(x, x, n) + 32. * (n + 1) * std::numeric_limits<double>::min();
                    squared += part;
                    if (segmented) terminal_channel_norms_.data[(j + 1) * d + a] = std::sqrt(part) * (1 + error);
                }
                terminal_channel_norms_.data[a] = std::sqrt(squared) * (1 + error);
            }
        }
        prepare_alphabet();
        if (options.terminal_kernel >= 10) {
            terminal_sample_norms_ = alloc(siglasso::multiply(n, alphabet.size()));
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
                }
                for (size_t block = alphabet.size(); block-- > 0;) {
                    const auto [low, high] = alphabet[block];
                    const size_t right = block + 2 * ((low + high) / 2 - low);
                    if (high - low == 1) continue;
                    const size_t dst = norm_index(block), left = norm_index(block + 1), other = norm_index(right);
                    for (size_t i = begin; i < end; ++i) terminal_sample_norms_.data[dst * n + i] = std::max(
                        terminal_sample_norms_.data[left * n + i], terminal_sample_norms_.data[other * n + i]);
                }
            });
        }
        if (options.terminal_kernel == siglasso::signed_terminal_kernel) {
            channel_group_norms_ = alloc(d * sample_groups);
            for (size_t a = 0; a < d; ++a) for (size_t g = 0; g < sample_groups; ++g) {
                double sum = 0;
                for (size_t i = n * g / sample_groups; i < n * (g + 1) / sample_groups; ++i) {
                    const double value = terminal_sample_norms_.data[a * n + i];
                    sum += value * value;
                }
                channel_group_norms_.data[a * sample_groups + g] = std::sqrt(sum + 32. * (n + 1) * std::numeric_limits<double>::min());
            }
        }
        if (terminal_only_) return;
        if (responses_.owner && joint_search_enabled() && panel_size() > 1 && l) {
            prepare_compact_panels();
            prepare_tt();
            return;
        }
        const size_t stored = absolute_steps() && panel_size() == 1 ? alphabet.size() - d : alphabet.size();
        alphabet_ratios = alloc(siglasso::multiply(stored, siglasso::multiply(n, panels())));
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t block = alphabet.size(); block-- > 0;) {
                const auto [low, high] = alphabet[block];
                const size_t right = block + 2 * ((low + high) / 2 - low);
                if (panel_size() > 1) {
                    const auto steps = high - low == 1 ? channel(low) : siglasso::Channel{};
                    const size_t count = panels();
                    for (size_t g = 0; g < count; ++g) {
                        double* out = alphabet_ratios.data + (block * count + g) * n;
                        if (high - low > 1) {
                            const double* left = alphabet_ratios.data + ((block + 1) * count + g) * n;
                            const double* other = alphabet_ratios.data + (right * count + g) * n;
                            for (size_t i = begin; i < end; ++i) out[i] = std::max(left[i], other[i]);
                        } else {
                            std::fill(out + begin, out + end, 0.);
                            for (size_t j = g * panel_size(); j < std::min(l, (g + 1) * panel_size()); ++j) {
                                const double* step = steps.values.data + j * steps.dimension * n;
                                for (size_t i = begin; i < end; ++i) out[i] = std::max(out[i], std::abs(step[i]));
                            }
                        }
                    }
                    continue;
                }
                if (absolute_steps()) {
                    if (high - low == 1) continue;
                    const size_t mid = (low + high) / 2;
                    for (size_t j = 0; j < l; ++j) {
                        const double* left = step_envelope(block + 1, low, mid, j);
                        const double* other = step_envelope(right, mid, high, j);
                        double* out = alphabet_ratios.data + ((block - low) * l + j) * n;
                        for (size_t i = begin; i < end; ++i) out[i] = std::max(std::abs(left[i]), std::abs(other[i]));
                    }
                    continue;
                }
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
    size_t suffix_plane(size_t q, size_t a) const {
        const size_t m = options.degree;
        return bound_type_ == 1 ? q * (2 * m - q + 1) / 2 + a - 1 : q * (m + 1) + a;
    }
    void prepare_tt() {
        const size_t n = options.samples, l = options.length - 1, d = options.dimension, m = options.degree;
        const size_t planes = bound_type_ == 1 ? m * (m + 1) / 2 : m * (m + 1);
        tt_suffix_ = alloc(siglasso::multiply(planes, siglasso::multiply(n, l)));
        fill(tt_suffix_, 0);
        for (size_t a = 1; a <= m; ++a) std::fill_n(tt_suffix_.data + suffix_plane(0, a) * l * n, l * n, 1 / weights[a]);
        if (bound_type_ >= 3) { prepare_tt_blocks(); return; }
        batch(n, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q < m; ++q) for (size_t c = 0; c < d; ++c) {
                const auto steps = channel(c);
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
    void export_tt_suffix(double* output) {
        const size_t m = options.degree, stride = options.samples * (options.length - 1);
        std::fill_n(output, m * (m + 1) * stride, 0.);
        for (size_t q = 0; q < m; ++q) for (size_t a = 1; q + a <= m; ++a)
            copy(tt_suffix_.data + suffix_plane(q, a) * stride, output + (q * (m + 1) + a) * stride, stride);
    }
    bool power_prefix() const { return bound_type_ == 1; }
    const double* prefix_start(const siglasso::Node& node, size_t j) const {
        return node.coefficients.data + j * (power_prefix() ? 1 : node.word.size() + 1) * options.samples;
    }
    siglasso::Array tt_moments(const siglasso::Node& node) {
        const size_t n = options.samples, l = options.length - 1, k = node.word.size(), m = options.degree;
        const size_t count = panels();
        auto result = alloc(siglasso::multiply(m - k, siglasso::multiply(n, count)));
        if (panel_size() > 1) fill(result, 0.);
        const hn::ScalableTag<double> tag; const size_t lanes = hn::Lanes(tag);
        batch(count, [&](uint64_t begin, uint64_t end) {
            for (size_t q = 1; q <= m - k; ++q) for (size_t g = begin; g < end; ++g)
            for (size_t j = g * panel_size(); j < std::min(l, (g + 1) * panel_size()); ++j) {
                double* out = result.data + ((q - 1) * count + g) * n;
                const double scale = weights[k + q] / n;
                size_t i = 0;
                for (; i + lanes <= n; i += lanes) {
                    auto value = hn::Mul(hn::Abs(hn::LoadU(tag, prefix_start(node, j) + i)),
                        hn::LoadU(tag, tt_suffix_.data + (suffix_plane(q - 1, 1) * l + j) * n + i));
                    for (size_t a = 1; a <= k; ++a) value = hn::MulAdd(
                        hn::Abs(hn::LoadU(tag, node.tt_state.data + ((a - 1) * l + j) * n + i)),
                        hn::LoadU(tag, tt_suffix_.data + (suffix_plane(q - 1, a + 1) * l + j) * n + i), value);
                    const auto scaled = hn::Mul(value, absolute_steps() ? hn::Set(tag, scale) : hn::Mul(hn::LoadU(tag, variation.data + j * n + i), hn::Set(tag, scale)));
                    hn::StoreU(panel_size() > 1 ? hn::Add(hn::LoadU(tag, out + i), scaled) : scaled, tag, out + i);
                }
                for (; i < n; ++i) {
                    double value = std::abs(prefix_start(node, j)[i])
                        * tt_suffix_.data[(suffix_plane(q - 1, 1) * l + j) * n + i];
                    for (size_t a = 1; a <= k; ++a) value += std::abs(node.tt_state.data[((a - 1) * l + j) * n + i])
                        * tt_suffix_.data[(suffix_plane(q - 1, a + 1) * l + j) * n + i];
                    out[i] = (panel_size() > 1 ? out[i] : 0.) + siglasso::finite(value * (absolute_steps() ? 1. : variation.data[j * n + i]) * scale);
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
            if (b_.feature_type_) {
                tree_.reset();
                tree_ = siglasso::make_cpu_feature_search(b_, b_.feature_type_);
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
            if (b_.options.degree == 2 && b_.options.dimension < 96) return b_.feature_type_ ? 6 : 5;
            if (!b_.feature_type_ && b_.options.degree == 3 && b_.options.dimension < 96) return 5;
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
            const double cap = b_.feature_type_ == 1 ? tree_->feature_cap() : 2.;
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
        void new_target() override { tree_->new_target(); }
        siglasso::NodePtr selected(const siglasso::Word& word, bool keep) override {
            if (keep && lower_degree_ && word.size() >= b_.options.degree) complete_dictionary();
            if (keep) return tree_->selected(word, true);
            return b_.feature_type_ == 1 ? b_.selected_word<true>(word) : b_.selected_word<false>(word);
        }
        siglasso::SearchResult query(const siglasso::Array& r, double alpha,
                const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active, size_t capacity,
                size_t, size_t, size_t) override {
            complete_dictionary();
            return tree_->query(r, alpha, active, capacity);
        }
    };
    std::unique_ptr<siglasso::FeatureSearch> make_search() override {
        if (endpoint_input_) return std::make_unique<EndpointSearch>(*this);
        const bool joint = joint_search_enabled();
        const bool lower = joint && ((options.degree == 4 && (feature_type_ || options.dimension >= 24)) ||
            (!feature_type_ && options.degree == 3 && options.dimension >= 96));
        if (lower && !feature_type_) proposal_degree = options.degree - 1;
        auto tree = feature_type_ ? siglasso::make_cpu_feature_search(*this, feature_type_, options.selected_only, lower ? options.degree - 1 : 0) : Backend::make_search();
        if (joint) return std::make_unique<BatchedSearch>(*this, std::move(tree), lower);
        return tree;
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
        if (!terminal_only_) node.moments = bound_type_ == 1 || bound_type_ >= 3 ? tt_moments(node) : alloc(levels * n * l);
        if (bound_type_ == 2) node.tt_moments = tt_moments(node);
        // For the empty word the polynomial and its segment integral both equal one.
        node.integrals = !k && power_prefix() ? node.coefficients : alloc(n * l);
        if ((options.search_method & 2) && levels <= 2 && (!(options.traversal_order & 4) || levels == 2)) node.linear_integrals = alloc(n * l);
        if (power_prefix()) {
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
        } else batch(n, [&](uint64_t begin, uint64_t end) {
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
        node.uncertain_moments = node.moments.size && std::any_of(node.moments.data, node.moments.data + node.moments.size, [](double value) { return value < 0; });
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
                for (size_t j = 0; j < panels(); ++j) value += task.node->moments.data[(q * panels() + j) * n + i]
                    * std::abs(step_envelope(task.block, task.low, task.high, j)[i]);
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
    bool terminal_query(const siglasso::Probe& task, const siglasso::Array& residual, double alpha,
        const std::unordered_map<siglasso::Word, size_t, siglasso::WordHash>& active,
        size_t capacity, siglasso::SearchResult& result) override {
        const auto& node = *task.node;
        const size_t n = options.samples, l = options.length - 1, d = options.dimension;
        const size_t k = node.word.size(), levels = (proposal_degree ? proposal_degree : options.degree) - k;
        if ((levels == 2 && !node.linear_integrals.owner) || !terminal_sample_norms_.owner) return false;
        ++stats.terminal_queries; ++stats.block_queries;
        const auto correlations_before = stats.terminal_correlations;
        if (!l) { ++stats.terminal_pruned; return true; }
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
        for (size_t i = 0; i < n; ++i) {
            const double r = std::abs(residual.data[i]);
            norms[i] = r * std::sqrt(norms[i] + tiny);
            if (levels == 2) operators[i] = r * std::sqrt(operators[i] + tiny) * first_norm[i];
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
        const auto visit = [&](size_t q, const double* factors) {
            if (excluded(bound_dot(norms, first_norm, n), q)) return;
            const auto group = grouped_norm(norms);
            const auto separate = response_groups(norms);
            for (size_t a = task.low; a < task.high; ++a) {
                word[k] = a;
                if (excluded(coarse(group, a), q) || separated(separate, a, q) || active.count(word) ||
                    excluded(bound_dot(norms, terminal_sample_norms_.data + a * n, n), q) ||
                    individual(norms, terminal_sample_norms_.data + a * n, q)) continue;
                const auto dx = channel(a);
                std::fill_n(values, n, 0.);
                for (size_t j = 0; j < l; ++j) {
                    const double* step = dx.values.data + j * dx.dimension * n;
                    for (size_t i = 0; i < n; ++i) values[i] += step[i] * factors[j * n + i];
                }
                ++stats.terminal_correlations; ++stats.evaluated;
                if (visit_column) { visit_column(word, values); continue; }
                const double score = siglasso::finite(std::abs(dot(values, residual.data, n)) / n * weights[k + q]);
                result.omitted = std::max(result.omitted, siglasso::guard(score));
                if (score > alpha && siglasso::shortlist(result.candidates, capacity, word, score, [&] {
                    auto column = alloc(n); copy(values, column.data, n); return column;
                })) ++stats.constructed;
            }
        };
        if (!highest_degree_only || !visit_column || k + 1 == options.degree) visit(1, node.integrals.data);
        if (levels == 2) {
            word.push_back(0);
            const auto group = grouped_norm(operators);
            const auto separate = response_groups(operators);
            for (size_t b = 0; b < d; ++b) {
                if (excluded(coarse(group, b), 2) || separated(separate, b, 2)) continue;
                if (excluded(bound_dot(operators, terminal_sample_norms_.data + b * n, n), 2) ||
                    individual(operators, terminal_sample_norms_.data + b * n, 2)) continue;
                word[k + 1] = b;
                const auto dx = channel(b);
                std::fill_n(tail, n, 0.); std::fill_n(norms, n, 0.);
                for (size_t j = l; j-- > 0;) {
                    const double* step = dx.values.data + j * dx.dimension * n;
                    const double* integral = node.integrals.data + j * n;
                    const double* linear = node.linear_integrals.data + j * n;
                    for (size_t i = 0; i < n; ++i) {
                        tail[i] += step[i];
                        const double h = integral[i] * tail[i] - linear[i] * step[i];
                        products[j * n + i] = h; norms[i] += h * h;
                    }
                }
                for (size_t i = 0; i < n; ++i) norms[i] = std::abs(residual.data[i]) * std::sqrt(norms[i] + tiny);
                visit(2, products);
            }
        }
        if (correlations_before == stats.terminal_correlations) ++stats.terminal_pruned;
        return true;
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
                const size_t start_q = highest_degree_only && visit_column ? levels - 1 : 0;
                for (size_t q = proposal_degree ? proposal_degree - task.node->word.size() : levels; q-- > start_q;) {
                    const auto envelope = [&](const siglasso::Array& moments) {
                    if (compact_panels_)
                        return compact_bound(task.block, moments.data + q * panels() * n, residual, operator_weights.data);
                    double total = 0;
                    const hn::ScalableTag<double> tag;
                    const size_t lanes = hn::Lanes(tag);
                    auto accumulator = hn::Zero(tag);
                    for (size_t j = 0; j < panels(); ++j) {
                        size_t i = 0;
                        for (; i + lanes <= n; i += lanes) accumulator = hn::Add(accumulator, hn::Mul(
                            hn::Mul(hn::Abs(hn::LoadU(tag, residual.data + i)), hn::LoadU(tag, moments.data + (q * panels() + j) * n + i)),
                            hn::Abs(hn::LoadU(tag, step_envelope(task.block, task.low, task.high, j) + i))));
                        for (; i < n; ++i) total += std::abs(residual.data[i]) * moments.data[(q * panels() + j) * n + i]
                            * std::abs(step_envelope(task.block, task.low, task.high, j)[i]);
                    }
                    return siglasso::finite(total + hn::ReduceSum(tag, accumulator));
                    };
                    double total = envelope(task.node->moments);
                    if (bound_type_ == 1 || bound_type_ >= 3) total += tt_error;
                    if (bound_type_ == 2 && total >= threshold) total = std::min(total, envelope(task.node->tt_moments) + tt_error);
                    if (q == 0 && total >= threshold && task.high - task.low > 1 && increments.owner) {
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
                                    hn::LoadU(tag, terminal_sample_norms_.data + norm_index(task.block) * n + i)), tag, operator_weights.data + i);
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
                                if (operator_weights.owner) operator_weights.data[i] = weighted * terminal_sample_norms_.data[norm_index(task.block) * n + i];
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
                                try { total = bound_dot(operator_weights.data, terminal_sample_norms_.data + norm_index(block) * n, n); }
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
                                        ? bound_dot(operator_weights.data, terminal_sample_norms_.data + b * n, n)
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
                                        sample_norm = hn::MulAdd(weighted, hn::LoadU(tag, terminal_sample_norms_.data + norm_index(task.block) * n + i), sample_norm);
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
                                        sample_bound += weighted * terminal_sample_norms_.data[norm_index(task.block) * n + i];
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
                                    ? bound_dot(terminal_products.data + n * l, terminal_sample_norms_.data + a * n, n)
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
    // Only I=integral(P) and J=integral(sP) are consumed by a terminal query.
    // For the child Q(s)=carry+D*integral_0^s P(t)dt, a parent monomial
    // F_a*s^a/a! contributes D*F_a/(a+1)! times 1/(a+2) or 1/(a+3).
    bool terminal_children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        if (!power_prefix()) return false;
        const size_t n = options.samples, l = options.length - 1;
        out.resize(tasks.size());
        const auto construct = [&](uint64_t begin, uint64_t end) {
            for (size_t t = begin; t < end; ++t) {
                const auto& parent = *tasks[t].node;
                const size_t k = parent.word.size(), letter = tasks[t].low;
                auto child = std::make_shared<siglasso::Node>();
                child->word = parent.word; child->word.push_back(letter);
                child->metadata = account(512 + child->word.size() * 64);
                const bool two = (proposal_degree ? proposal_degree : options.degree) - k - 1 == 2;
                if (visit_column && tasks.size() == 1) {
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
                out[t] = std::move(child);
            }
        };
        // Cache eviction under an explicit memory limit belongs to the search thread.
        if (options.memory_limit) construct(0, tasks.size());
        else batch(tasks.size(), construct);
        return true;
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
    void children(const std::vector<siglasso::Probe>& tasks, std::vector<siglasso::NodePtr>& out) override {
        const size_t n = options.samples, l = options.length - 1;
        out.resize(tasks.size());
        for (size_t t = 0; t < tasks.size(); ++t) {
            auto node = std::make_shared<siglasso::Node>(); node->word = tasks[t].node->word; node->word.push_back(tasks[t].low);
            node->metadata = account(512 + node->word.size() * 64);
            node->values = alloc(n);
            if (node->word.size() < options.degree) node->coefficients = alloc((power_prefix() ? 1 : node->word.size() + 1) * n * l);
            if (bound_type_ && node->word.size() < options.degree) node->tt_state = alloc(node->word.size() * n * l);
            out[t] = std::move(node);
        }
        if (power_prefix()) {
            batch(tasks.size(), [&](uint64_t begin, uint64_t end) {
                for (size_t t = begin; t < end; ++t) {
                    auto& child = *out[t]; const auto& task = tasks[t];
                    fill(child.values, 0);
                    const auto steps = channel(task.low);
                    for (size_t j = 0; j < l; ++j) {
                        if (child.coefficients.owner) copy(child.values.data, child.coefficients.data + j * n, n);
                        const double* step = steps.values.data + j * steps.dimension * n;
                        const double* integral = task.node->integrals.data + j * n;
                        for (size_t i = 0; i < n; ++i) child.values.data[i] += step[i] * integral[i];
                    }
                }
            });
        } else batch(tasks.size() * n, [&](uint64_t begin, uint64_t end) {
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
                    const auto steps = channel(tasks[t].low);
                    for (size_t a = 0; a <= k; ++a) for (size_t j = 0; j < l; ++j) {
                        const double* source = a ? parent.tt_state.data + ((a - 1) * l + j) * n : prefix_start(parent, j);
                        const double* increment = steps.values.data + j * steps.dimension * n;
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
    void release_fit_workspace() {
        searches_.clear(); search_workers_.clear(); shared_factors_.clear(); channels_.clear();
        for (auto& shard : cold_) shard.values = {};
        increments = {}; variation = {}; after = {}; alphabet_ratios = {}; compact_panels_.reset(); tt_suffix_ = {};
        responses_ = {}; normalization_ = {}; endpoint_variation_ = {}; terminal_workspace_ = {}; terminal_child_storage_ = {}; terminal_sample_norms_ = {}; channel_group_norms_ = {};
        terminal_channel_norms_ = {}; trim_caches();
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
        auto single = options;
        single.n_jobs = 1; single.summary_bytes /= parts;
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
    std::unique_ptr<siglasso::Backend> terminal_worker() override {
        auto single = options;
        single.n_jobs = 1;
        auto worker = std::make_unique<LassoCPU>(*this, single);
        worker->factor_parent_ = this;
        return worker;
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
    bool native_search(const siglasso::NodePtr& root, const siglasso::Array& residual, double alpha,
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
            auto single = options;
            single.n_jobs = 1; single.cache_bytes /= frontier_.size(); single.summary_bytes /= frontier_.size();
            const bool reusable = !options.experiment && !options.cache_bytes && !options.summary_flags;
            const size_t count = reusable ? std::min(frontier_.size(), threads) : frontier_.size();
            for (size_t j = 0; j < count; ++j) {
                auto worker = std::make_unique<LassoCPU>(*this, single);
                worker->increments = increments; worker->variation = variation; worker->after = after;
                worker->alphabet = alphabet; worker->alphabet_ratios = alphabet_ratios;
                worker->factor_parent_ = this;
                searches_.push_back(std::make_unique<siglasso::Search>(*worker, root));
                search_workers_.push_back(std::move(worker));
            }
        }
        std::vector<siglasso::SearchResult> found(frontier_.size());
        std::shared_ptr<siglasso::CandidateCutoff> cutoff;
        if ((options.traversal_order & 2) && capacity && searches_.size() > 1) {
            const size_t rank = options.pool_rank ? std::min<size_t>(capacity, options.pool_rank) : capacity;
            cutoff = std::make_shared<siglasso::CandidateCutoff>(rank, account(256 + siglasso::multiply(rank, sizeof(double))));
        }
        std::atomic<size_t> next{0};
        const bool reusable = searches_.size() < frontier_.size();
        spawn_batch_threads(searches_.size(), options.n_jobs, [&](size_t begin, size_t end) {
            for (size_t worker_id = begin; worker_id < end; ++worker_id) {
                auto& worker = *search_workers_[worker_id];
                worker.stats = {}; worker.experiment_stats = {}; worker.optimization_stats = {};
                worker.active_corr_cap = active_corr_cap; worker.verify_search = verify_search;
                worker.proposal_degree = proposal_degree;
                worker.highest_degree_only = highest_degree_only;
                worker.visit_column = visit_column; worker.batch_absolute = batch_absolute; worker.batch_ratios = batch_ratios; worker.batch_signed = batch_signed; worker.shared_cutoff = cutoff;
                for (size_t j = reusable ? next.fetch_add(1) : worker_id; j < frontier_.size();) {
                    const size_t low = options.search_method & 1 ? alphabet[frontier_[j]].first : frontier_[j];
                    const size_t high = options.search_method & 1 ? alphabet[frontier_[j]].second
                        : (j + 1 < frontier_.size() ? frontier_[j + 1] : options.dimension);
                    found[j] = searches_[worker_id]->query(residual, alpha, active, capacity, low, high, frontier_[j]);
                    if (!reusable) break;
                    j = next.fetch_add(1);
                }
                worker.shared_cutoff.reset(); worker.visit_column = {}; worker.batch_absolute.clear(); worker.batch_ratios.clear(); worker.batch_signed.clear();
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
    backend.release_fit_workspace();
    siglasso::experimental_last_stats = backend.experiment_stats;
    siglasso::experimental_optimization_stats = backend.optimization_stats;
    siglasso::export_result(backend, result, allocate, context);
}
template<typename T>
void sig_working_set_predict_(const T* path, const T* coefficients, const T* intercept, T* output,
    const uint64_t* letters, const uint64_t* offsets, uint64_t words, uint32_t feature_type, const SigLassoOptions* options) {
    if (feature_type > 2) throw std::invalid_argument("Invalid Lasso feature type");
    const size_t n = options->samples, targets = options->targets, stride = siglasso::multiply(options->length, options->dimension);
    for (size_t t = 0; t < targets; ++t) siglasso::finite(intercept[t]);
    for (size_t j = 0; j < siglasso::multiply(words, targets); ++j) siglasso::finite(coefficients[j]);
    auto budget = std::make_shared<siglasso::Budget>(options->memory_limit);
    std::vector<size_t> order;
    std::shared_ptr<void> indices;
    if (feature_type < 2 && options->degree > 1 && words * sizeof(size_t) <= budget->available() / 4) {
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
            auto opt = siglasso::SearchOptions(*options, feature_type, true); opt.samples = count;
            LassoCPU backend(opt, path + start * stride, static_cast<const T*>(nullptr), feature_type, budget);
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
            for (size_t i = 0; i < prediction.size; ++i) output[start * targets + i] = static_cast<T>(siglasso::finite(prediction.data[i]));
        } catch (const std::bad_alloc&) {
            if (!options->memory_limit || count == 1) throw;
            tile = (count + 1) / 2;
            continue;
        }
        start += count;
    }
}
template void sig_working_set_<float>(const float*, const float*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_tt_envelopes_<double>(const double*, const SigLassoOptions*, double*);
template void sig_working_set_query_benchmark_<double>(const double*, const double*, const double*, uint32_t, const SigLassoOptions*, double*, double*);
template void sig_working_set_<double>(const double*, const double*, const double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*, SigLassoAllocator, void*);
template void sig_working_set_predict_<float>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);
template void sig_working_set_predict_<double>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, uint64_t, uint32_t, const SigLassoOptions*);

} } // namespace cpsig::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
