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
#include "cp_lasso_features.h"
#include "../shared/preparation/log_sig/lyndon_words.h"
#include "sig_lasso_math.h"
#include "cp_dispatch.h"
#include "multithreading.h"
namespace cpsig {
HWY_EXPORT(log_polynomial_);
HWY_EXPORT(log_terminal_integrals_);
HWY_EXPORT(log_child_integrals_);
HWY_EXPORT(log_terminal_products_);
HWY_EXPORT(log_terminal_values_);
void log_polynomial(const double* parent, const double* increments, double* out, double* values,
    uint64_t n, uint64_t l, uint64_t d, uint64_t degree, uint64_t levels, uint64_t letter, uint64_t begin, uint64_t end) {
    HWY_DYNAMIC_DISPATCH(log_polynomial_)(parent, increments, out, values, n, l, d, degree, levels, letter, begin, end);
}
void log_child_integrals(const double* parent, const double* dx, double* out, double* values,
                         size_t n, size_t l, size_t stride, size_t degree, size_t begin, size_t end) {
    HWY_DYNAMIC_DISPATCH(log_child_integrals_)(parent, dx, out, values, n, l, stride, degree, begin, end);
}
void log_terminal_integrals(const double* polynomial, double* out, uint64_t n, uint64_t l, uint64_t degree, uint64_t levels, uint64_t begin, uint64_t end) {
    HWY_DYNAMIC_DISPATCH(log_terminal_integrals_)(polynomial, out, n, l, degree, levels, begin, end);
}
void log_terminal_products(const double* integrals, const double* increments, const double* residual, double* out,
    uint64_t n, uint64_t l, uint64_t d, uint64_t letter, uint64_t suffix, uint64_t begin, uint64_t end) {
    HWY_DYNAMIC_DISPATCH(log_terminal_products_)(integrals, increments, residual, out, n, l, d, letter, suffix, begin, end);
}
void log_terminal_values(const double* products, const double* increments, double* out,
    uint64_t n, uint64_t l, uint64_t d, uint64_t letter, uint64_t begin, uint64_t end) {
    HWY_DYNAMIC_DISPATCH(log_terminal_values_)(products, increments, out, n, l, d, letter, begin, end);
}
}

namespace siglasso {
namespace {
struct State {
    Word word;
    Array polynomial, values, terminal_integrals, variation_product;
    std::shared_ptr<void> metadata;
};
using StatePtr = std::shared_ptr<State>;

class LogSearch final : public FeatureSearch {
    Backend& b_;
    const size_t n_, l_, d_, m_;
    std::vector<double> caps_;
    StatePtr root_;
    std::vector<StatePtr> prediction_path_;
    std::shared_ptr<void> tables_metadata_;
    std::shared_ptr<float[]> log_channel_norms_;
    std::shared_ptr<void> log_norm_storage_;
    Array log_fourth_norms_, log_group_norms_;
    static constexpr size_t sample_groups = 8;
    Array channel_variations_, suffix_maxima_, suffix_factors_;
    size_t suffix_first_ = SIZE_MAX;
    // At each segment, Hölder in the sample index separates the letters of
    // any tensor word. The signed permutation formula for log contributes
    // only its degree-dependent Eulerian absolute coefficient sum.
    Array letter_factors_;
    std::array<double, sample_groups> residual_groups_{};
    void prepare_letter_factors() {
        if (d_ < 24 || m_ < 3 || m_ > 8 || !l_ || n_ < sample_groups) return;
        const size_t bytes = multiply(m_ * sample_groups * 2, d_) * sizeof(double);
        if (bytes > b_.budget->available() / 16) return;
        letter_factors_ = b_.alloc(bytes / sizeof(double));
        b_.fill(letter_factors_, 0.);
        auto run = [&](uint64_t low, uint64_t high) {
            for (size_t a = low; a < high; ++a) {
                const auto steps = b_.channel(a);
                for (size_t g = 0; g < sample_groups; ++g) {
                    const size_t begin = n_ * g / sample_groups, end = n_ * (g + 1) / sample_groups;
                    std::array<double, 8> maxima{};
                    for (size_t j = 0; j < l_; ++j) {
                        std::array<double, 8> sums{};
                        for (size_t i = begin; i < end; ++i) {
                            const double dx = steps.values.data[j * steps.dimension * n_ + i], square = dx * dx;
                            double power = square;
                            for (size_t k = 0; k < m_; ++k) { sums[k] += power; power *= square; }
                        }
                        for (size_t k = 0; k < m_; ++k) maxima[k] = std::max(maxima[k], sums[k]);
                    }
                    for (size_t k = 0; k < m_; ++k) {
                        const double safe = maxima[k] / (end - begin) + 32. * std::numeric_limits<double>::min();
                        letter_factors_.data[(k * 2 * d_ + a) * sample_groups + g] =
                            l_ * std::pow(safe, .5 / (k + 1)) * (1. + 1e-12 * (n_ + l_));
                    }
                }
            }
        };
        if (b_.options.n_jobs == 1) run(0, d_); else spawn_batch_threads(d_, b_.options.n_jobs, run);
        for (size_t k = 0; k < m_; ++k) for (size_t a = d_; a-- > 0;) for (size_t g = 0; g < sample_groups; ++g) {
            const size_t id = (k * 2 * d_ + a) * sample_groups + g;
            letter_factors_.data[id + d_ * sample_groups] = std::max(letter_factors_.data[id],
                a + 1 == d_ ? 0. : letter_factors_.data[id + (d_ + 1) * sample_groups]);
        }
    }
    double letter_bound(const Word& prefix, size_t next) const {
        const size_t h = prefix.size() + 1, first = prefix.empty() ? next : prefix.front();
        double best = 0;
        for (size_t k = b_.highest_degree_only && b_.visit_column ? m_ : h; k <= m_; ++k) {
            const double* factors = letter_factors_.data + (k - 1) * 2 * d_ * sample_groups;
            double sum = 0;
            for (size_t g = 0; g < sample_groups; ++g) {
                double value = factors[next * sample_groups + g];
                for (auto a : prefix) value *= factors[a * sample_groups + g];
                const double suffix = factors[(d_ + first) * sample_groups + g];
                for (size_t j = h; j < k; ++j) value *= suffix;
                sum += value * residual_groups_[g];
            }
            if (!std::isfinite(sum)) return std::numeric_limits<double>::infinity();
            best = std::max(best, sum * caps_[k]);
        }
        return padded(best + terminal_roundoff(m_, n_, l_) * feature_cap_ * residual_mass_, 1e-10 * (l_ + n_ + m_));
    }
    double residual_mass_ = 0, residual_mean_ = 0, residual_rms_ = 0;
    void prepare_residual(const Array& residual) {
        residual_mass_ = b_.absolute_sum(residual) / n_;
        residual_mean_ = std::abs(std::accumulate(residual.data, residual.data + n_, 0.)) / n_;
        const auto [low, high] = std::minmax_element(residual.data, residual.data + n_);
        const double maximum = std::max(std::abs(*low), std::abs(*high));
        // Scale the RMS before squaring to handle very large and very small gradients.
        double squared = 0;
        if (maximum) for (size_t i = 0; i < n_; ++i) {
            const double x = residual.data[i] / maximum; squared += x * x;
        }
        residual_rms_ = padded(maximum * std::sqrt(squared / n_), 1e-11 * n_);
        residual_mean_ += residual_rms_ * 1e-11 * n_;
        if (letter_factors_.owner) for (size_t g = 0; g < sample_groups; ++g) {
            const size_t begin = n_ * g / sample_groups, end = n_ * (g + 1) / sample_groups;
            double squared = 0;
            for (size_t i = begin; i < end; ++i) squared += residual.data[i] * residual.data[i];
            residual_groups_[g] = std::sqrt((squared + 32. * (n_ + 1) * std::numeric_limits<double>::min()) * (end - begin)) / n_;
        }
    }
    struct TerminalNorms {
        Array columns;
        std::shared_ptr<void> metadata;
        double max_radius = 0, prefix_radius = 0;
    };
    std::unordered_map<Word, TerminalNorms, WordHash> terminal_norms_;
    size_t terminal_bytes_ = 0;
    double norm_score(double radius) const {
        return padded(residual_mean_ * feature_cap_ + residual_rms_ * radius, 1e-10);
    }
    double column_radius(const double* values, size_t degree, double rounding) const {
        const double mean = std::accumulate(values, values + n_, 0.) / n_;
        double squared = 0;
        for (size_t i = 0; i < n_; ++i) {
            const double delta = values[i] - mean; squared += delta * delta;
        }
        const double tiny = 32. * (n_ + 1) * (l_ + 1) * std::numeric_limits<double>::min();
        return padded(b_.weights[degree] * std::sqrt((squared + tiny) / n_) + rounding * caps_[degree], rounding);
    }

    double padded(double value, double allowance) const {
        value *= 1 + allowance;
        return std::isfinite(value) ? value * (1 + 64 * std::numeric_limits<double>::epsilon()) + 1e-15 : std::numeric_limits<double>::infinity();
    }
    double feature_cap_ = std::numeric_limits<double>::infinity();
    bool zero_path_ = false;

    template<typename Function> void samples(Function action) {
        if (b_.options.n_jobs == 1 || n_ < 32) action(0, n_);
        else spawn_batch_threads(n_, b_.options.n_jobs, action);
    }
    StatePtr state(const Word& word, size_t count) {
        auto s = std::make_shared<State>(); s->word = word;
        s->metadata = b_.account(512 + word.size() * 32);
        s->polynomial = b_.alloc(multiply(multiply(count, l_), n_));
        s->values = b_.alloc(n_); b_.fill(s->values, 0);
        ++b_.stats.evaluated;
        return s;
    }
    StatePtr log_child(const StatePtr& parent, size_t letter, bool extendible = true) {
        Word w = parent->word; w.push_back(letter);
        ++b_.stats.cache_misses;
        const size_t h = w.size(), levels = extendible ? m_ - h : 0;
        auto out = state(w, levels * (h + 1));
        const auto steps = b_.channel(letter);
        samples([&](uint64_t begin, uint64_t end) {
            cpsig::log_polynomial(parent->polynomial.size ? parent->polynomial.data : nullptr, steps.values.data, out->polynomial.data, out->values.data,
                n_, l_, steps.dimension, h, levels, 0, begin, end);
        });
        if (is_lyndon(w)) ++b_.stats.constructed;
        return out;
    }
    void prepare_channel_variations() {
        const size_t count = multiply(d_, n_);
        channel_variations_ = b_.alloc(count);
        suffix_maxima_ = b_.alloc(count);
        suffix_factors_ = b_.alloc(multiply(m_ - 2, n_));
        for (size_t a = d_; a-- > 0;) {
            double* tv = channel_variations_.data + a * n_;
            const auto steps = b_.channel(a);
            std::fill_n(tv, n_, 0.);
            for (size_t j = 0; j < l_; ++j) {
                const double* dx = steps.values.data + j * steps.dimension * n_;
                for (size_t i = 0; i < n_; ++i) tv[i] += std::abs(dx[i]);
            }
            double* maxima = suffix_maxima_.data + a * n_;
            for (size_t i = 0; i < n_; ++i)
                maxima[i] = a + 1 == d_ ? tv[i] : std::max(tv[i], maxima[n_ + i]);
        }
    }
    double variation_bound(const StatePtr& u, size_t letter, const Array& residual) {
        const size_t h = u->word.size() + 1, first = h == 1 ? letter : u->word.front();
        const double scale = b_.weights[m_ - 1];
        if (suffix_first_ != first) {
            // All Lyndon completions use letters >= first. DFS reuses this table down the subtree.
            const double* maxima = suffix_maxima_.data + first * n_;
            for (size_t depth = m_ - 2; depth > 0; --depth) {
                double* factors = suffix_factors_.data + (depth - 1) * n_;
                for (size_t i = 0; i < n_; ++i)
                    factors[i] = depth == m_ - 2 ? maxima[i] * maxima[i] : maxima[i] * factors[n_ + i];
            }
            for (size_t depth = 1; depth <= m_ - 2; ++depth) {
                double* factors = suffix_factors_.data + (depth - 1) * n_;
                const double lower = b_.weights[depth - 1] / scale;
                for (size_t i = 0; i < n_; ++i) factors[i] = std::max(lower, factors[i]);
            }
            suffix_first_ = first;
        }
        if (!u->variation_product.owner) {
            // Scaling before multiplying small variations avoids premature underflow.
            u->variation_product = b_.alloc(n_);
            b_.fill(u->variation_product, scale);
            for (auto a : u->word) {
                const double* tv = channel_variations_.data + a * n_;
                for (size_t i = 0; i < n_; ++i) u->variation_product.data[i] *= tv[i];
            }
        }
        const double* factors = suffix_factors_.data + (h - 1) * n_;
        const double* next = channel_variations_.data + letter * n_;
        double sum = 0.;
        for (size_t i = 0; i < n_; ++i)
            sum += std::abs(residual.data[i]) * ((u->variation_product.data[i] * next[i]) * factors[i]);
        const double slack = terminal_roundoff(m_, n_, l_) * feature_cap_ * residual_mass_;
        return padded(sum / n_ + slack, 1e-11 * m_ * m_ * (l_ + n_ + 1));
    }
    StatePtr search_child(const StatePtr& parent, size_t letter) {
        const size_t h = parent->word.size() + 1;
        if (m_ - h != 2) return log_child(parent, letter);
        Word word = parent->word; word.push_back(letter);
        auto child = state(word, 0);
        child->terminal_integrals = b_.alloc(multiply(3 * l_, n_));
        const auto steps = b_.channel(letter);
        samples([&](size_t begin, size_t end) {
            cpsig::log_child_integrals(parent->polynomial.size ? parent->polynomial.data : nullptr, steps.values.data,
                child->terminal_integrals.data, child->values.data, n_, l_, steps.dimension, h, begin, end);
        });
        ++b_.stats.cache_misses; ++b_.stats.constructed;
        return child;
    }
    double offer(const StatePtr& u, const Array& residual, double alpha, size_t capacity,
        const ActiveWords& active, SearchResult& result, bool copy_values = false) {
        if (active.count(u->word)) return 0;
        if (b_.visit_column) return b_.visit_column(u->word, u->values.data)
            ? std::nextafter(alpha, std::numeric_limits<double>::infinity()) : 0.;
        const double score = std::abs(b_.dot(u->values.data, residual.data, n_)) * b_.weights[u->word.size()] / n_;
        result.omitted = std::max(result.omitted, guard(score));
        if (!capacity || score <= alpha) return score;
        shortlist(result.candidates, capacity, u->word, score, [&] { return copy_values ? b_.clone(u->values) : u->values; });
        return score;
    }
    struct TerminalWorker {
        std::unique_ptr<Backend> backend;
        std::unique_ptr<LogSearch> search;
    };
    std::vector<TerminalWorker> terminal_workers_;
    LogSearch(Backend& b, const LogSearch& other)
        : b_(b), n_(other.n_), l_(other.l_), d_(other.d_), m_(other.m_),
          caps_(other.caps_), log_channel_norms_(other.log_channel_norms_), log_norm_storage_(other.log_norm_storage_),
          log_fourth_norms_(other.log_fourth_norms_), log_group_norms_(other.log_group_norms_),
          feature_cap_(other.feature_cap_) {}
    void terminal_geometry(double underflow) {
        if (!log_channel_norms_) {
            const size_t count = multiply(n_, d_);
            log_norm_storage_ = b_.account(multiply(count, sizeof(float)));
            log_channel_norms_.reset(new float[count]);
            auto scratch = b_.alloc(n_);
            for (size_t a = 0; a < d_; ++a) {
                const auto steps = b_.channel(a);
                double* norms = scratch.data;
                std::fill_n(norms, n_, 0.);
                for (size_t j = 0; j < l_; ++j) {
                    const double* dx = steps.values.data + j * steps.dimension * n_;
                    for (size_t i = 0; i < n_; ++i) norms[i] += dx[i] * dx[i];
                }
                const double inflate = 1 + terminal_roundoff(0, multiply(n_, l_), l_);
                for (size_t i = 0; i < n_; ++i) {
                    const double upper = std::sqrt(norms[i] + underflow) * inflate;
                    const float value = static_cast<float>(upper);
                    log_channel_norms_[a * n_ + i] = value < upper ? std::nextafter(value, std::numeric_limits<float>::infinity()) : value;
                }
            }
        }
        const bool need_single = b_.visit_column && d_ >= 8 && !log_group_norms_.owner;
        if (!log_fourth_norms_.owner || need_single) {
            // Group norms over disjoint sample intervals; the second table
            // bounds all channels at or above a prenecklace's next letter.
            if (need_single) log_group_norms_ = b_.alloc(multiply(2 * d_, sample_groups));
            if (!log_fourth_norms_.owner) log_fourth_norms_ = b_.alloc(multiply(2 * d_, sample_groups));
            for (size_t a = d_; a-- > 0;) for (size_t g = 0; g < sample_groups; ++g) {
                double fourth = 0., sum = 0.;
                for (size_t i = n_ * g / sample_groups; i < n_ * (g + 1) / sample_groups; ++i) {
                    const double norm = log_channel_norms_[a * n_ + i];
                    const double square = norm * norm;
                    fourth += square * square; sum += square;
                }
                if (need_single) {
                    const double group = std::sqrt(sum + underflow);
                    log_group_norms_.data[a * sample_groups + g] = group;
                    log_group_norms_.data[(d_ + a) * sample_groups + g] = a + 1 == d_ ? group
                        : std::max(group, log_group_norms_.data[(d_ + a + 1) * sample_groups + g]);
                }
                const double norm = std::sqrt(std::sqrt(fourth + underflow));
                log_fourth_norms_.data[a * sample_groups + g] = norm;
                log_fourth_norms_.data[(d_ + a) * sample_groups + g] = a + 1 == d_ ? norm
                    : std::max(norm, log_fourth_norms_.data[(d_ + a + 1) * sample_groups + g]);
            }
        }
    }
    void visit_terminal_children(const StatePtr& u, size_t period, const std::vector<size_t>& letters,
                                 const Array& residual, double alpha, const ActiveWords& active, SearchResult& result) {
        // Each worker holds a segment operator for one child. Use a smaller
        // concurrency limit on narrow alphabets, where there are fewer siblings.
        const size_t limit = d_ >= 96 ? 4 : (d_ >= 24 ? 3 : 1);
        const size_t workers = std::min({limit, static_cast<size_t>(b_.options.n_jobs), letters.size()});
        terminal_geometry(32. * (n_ + 1) * (l_ + 1) * std::numeric_limits<double>::min());
        while (terminal_workers_.size() < workers) {
            auto backend = b_.terminal_worker();
            auto search = std::unique_ptr<LogSearch>(new LogSearch(*backend, *this));
            terminal_workers_.push_back({std::move(backend), std::move(search)});
        }
        const size_t h = u->word.size();
        std::atomic<size_t> next{0};
        std::vector<SearchResult> local(workers);
        spawn_batch_threads(workers, static_cast<int>(workers), [&](size_t begin, size_t end) {
            for (size_t t = begin; t < end; ++t) {
                auto& search = *terminal_workers_[t].search;
                auto& host = *terminal_workers_[t].backend;
                host.stats = {};
                host.highest_degree_only = b_.highest_degree_only;
                host.visit_column = b_.visit_column;
                host.batch_ratios = b_.batch_ratios;
                host.batch_signed = b_.batch_signed;
                search.residual_mass_ = residual_mass_;
                for (size_t slot = next.fetch_add(1); slot < letters.size(); slot = next.fetch_add(1)) {
                    const size_t a = letters[slot];
                    auto child = search.search_child(u, a);
                    const size_t child_period = h && a == u->word[h - period] ? period : h + 1;
                    if (child_period == h + 1) search.offer(child, residual, alpha, 0, active, local[t]);
                    search.log_terminal_search(child, child_period, residual, alpha, 0, active, local[t]);
                }
                host.visit_column = {};
                host.batch_ratios.clear();
                host.batch_signed.clear();
            }
        });
        for (size_t t = 0; t < workers; ++t) {
            const auto& stats = terminal_workers_[t].backend->stats;
            result.omitted = std::max(result.omitted, local[t].omitted);
            b_.stats.evaluated += stats.evaluated;
            b_.stats.constructed += stats.constructed;
            b_.stats.cache_misses += stats.cache_misses;
            b_.stats.block_queries += stats.block_queries;
            b_.stats.terminal_queries += stats.terminal_queries;
            b_.stats.pruned += stats.pruned;
            b_.stats.terminal_pruned += stats.terminal_pruned;
            b_.stats.terminal_correlations += stats.terminal_correlations;
            b_.stats.numerical_failures += stats.numerical_failures;
        }
    }
    void log_terminal_search(const StatePtr& u, size_t period, const Array& residual, double alpha, size_t capacity,
        const ActiveWords& active, SearchResult& result) {
        ++b_.stats.terminal_queries;
        const size_t h = u->word.size(), levels = m_ - h;
        const double rounding = terminal_roundoff(h, multiply(n_, l_), l_);
        const double underflow = 32. * (n_ + 1) * (l_ + 1) * std::numeric_limits<double>::min();
        const double mass = residual_mass_;
        TerminalNorms* norms = nullptr;
        bool build = false;
        if (!b_.visit_column && d_ <= 32 && n_ >= 32 && !b_.options.memory_limit) {
            auto found = terminal_norms_.find(u->word);
            if (found == terminal_norms_.end()) {
                const size_t entries = d_ + (levels == 2 ? d_ * d_ : 0);
                const size_t bytes = entries * sizeof(double) + 256 + h * sizeof(uint64_t);
                if (terminal_bytes_ + bytes <= (1u << 20)) {
                    auto storage = b_.alloc(entries); b_.fill(storage, 0);
                    found = terminal_norms_.emplace(u->word, TerminalNorms{std::move(storage), b_.account(bytes - entries * sizeof(double))}).first;
                    terminal_bytes_ += bytes; build = true;
                    if (h && h == period) found->second.prefix_radius = column_radius(u->values.data, h, rounding);
                }
            }
            if (found != terminal_norms_.end()) norms = &found->second;
            if (norms && !build) {
                const double bound = norm_score(norms->max_radius);
                const double cutoff = capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
                if (bound < cutoff) {
                    result.omitted = std::max(result.omitted, bound); ++b_.stats.terminal_pruned; return;
                }
            }
        }
        terminal_geometry(underflow);
        // At the root I=1, J=1/2 and K=-1/2. The kernels can apply these
        // constants directly without storing three copies across paths and time.
        const bool root_constants = h == 0;
        auto integrals = u->terminal_integrals;
        if (!root_constants && !integrals.owner) integrals = b_.alloc(multiply(3 * l_, n_));
        Array pair_products;
        auto single_stats = b_.alloc(2 * n_);
        auto probe = std::make_shared<State>(); probe->values = b_.alloc(n_);
        probe->metadata = b_.account(512 + m_ * 32);
        auto sample_bounds = b_.batch_ratios.empty() ? Array{} : b_.alloc(n_);
        if (!root_constants && !u->terminal_integrals.owner) samples([&](uint64_t begin, uint64_t end) {
            cpsig::log_terminal_integrals(u->polynomial.data, integrals.data, n_, l_, h, levels, begin, end);
        });
        // H^b = M D^b with M[j,t] = I[j] 1(t>=j) - J[j] 1(t==j) + K[j].
        // The Frobenius norm is independent of both terminal letters.
        const bool screen_pairs = !norms && levels == 2;
        auto pair_weights = screen_pairs ? b_.alloc(n_) : Array{};
        if (screen_pairs) {
            b_.fill(pair_weights, root_constants ? .25 * l_ * (l_ - 1.) : 0.);
            if (!root_constants) for (size_t j = 0; j < l_; ++j) {
                const double* I = integrals.data + j * n_;
                const double* J = I + l_ * n_;
                const double* K = J + l_ * n_;
                for (size_t i = 0; i < n_; ++i) {
                    const double upper = I[i] + K[i], diagonal = upper - J[i];
                    pair_weights.data[i] += j * K[i] * K[i] + diagonal * diagonal + (l_ - j - 1) * upper * upper;
                }
            }
            for (size_t i = 0; i < n_; ++i)
                pair_weights.data[i] = std::abs(residual.data[i]) * std::sqrt(pair_weights.data[i] + underflow);
        }
        const auto pair_scale = screen_pairs ? b_.alloc(n_) : Array{};
        std::array<double, sample_groups> operator_norms{};
        if (screen_pairs) for (size_t g = 0; g < sample_groups; ++g) {
            const size_t first = n_ * g / sample_groups, end = n_ * (g + 1) / sample_groups;
            const double squared = b_.bound_dot(pair_weights.data + first, pair_weights.data + first, end - first);
            operator_norms[g] = std::sqrt(squared + underflow);
        }
        std::array<std::array<double, sample_groups>, max_response_batch> group_bounds{};
        if (screen_pairs) for (size_t t = 0; t < b_.batch_ratios.size(); ++t) {
            for (size_t g = 0; g < sample_groups; ++g) {
                const size_t first = n_ * g / sample_groups, end = n_ * (g + 1) / sample_groups;
                const double squared = b_.weighted_square(pair_weights.data + first, b_.batch_ratios[t].data + first, end - first);
                group_bounds[t][g] = std::sqrt(squared + underflow);
            }
        }
        const auto cutoff_now = [&] {
            return capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
        };
        const auto pair_bound = [&](size_t a, size_t c) {
            const double* left = log_fourth_norms_.data + a * sample_groups;
            const double* right = log_fourth_norms_.data + c * sample_groups;
            double total = 0;
            for (size_t g = 0; g < sample_groups; ++g) total += operator_norms[g] * left[g] * right[g];
            const double factor = b_.weights[h + 2] / n_, slack = rounding * caps_[h + 2] * mass + underflow * factor;
            if (b_.batch_ratios.size() > 1 && (total * factor + slack) * (1 + rounding) + 1e-15 >= cutoff_now()) {
                total = 0.;
                for (size_t t = 0; t < b_.batch_ratios.size(); ++t) {
                    double sum = 0.;
                    for (size_t g = 0; g < sample_groups; ++g) sum += group_bounds[t][g] * left[g] * right[g];
                    total = std::max(total, sum);
                }
            }
            return padded(total * factor + slack, rounding);
        };
        const size_t first = h ? u->word[h % period] : 0;
        // All remaining Lyndon letters are at least the first letter of
        // the prefix. The suffix maxima bound both ends of a terminal pair.
        bool omit_pairs = false;
        if (screen_pairs) {
            const size_t second = h ? u->word.front() + 1 : 0;
            if (second == d_) omit_pairs = true;
            else {
                const double bound = pair_bound(d_ + first, d_ + second);
                omit_pairs = bound < cutoff_now();
                if (omit_pairs) result.omitted = std::max(result.omitted, bound);
            }
        }
        double maximum = 0;
        for (size_t q = b_.highest_degree_only && b_.visit_column ? levels : 1; q <= (omit_pairs ? 1 : levels); ++q) for (size_t c = 0; c < (q == 2 ? d_ : 1); ++c) {
            size_t low = first, high = d_;
            if (q == 1) low += (h != 0);
            else if (!h) high = c;
            else {
                if (c <= u->word.front()) continue;
                low += c <= u->word[(h + 1) % period];
            }
            if (low >= high) continue;
            if (q == 2 && screen_pairs) {
                const double bound = pair_bound(d_ + low, c);
                if (bound < cutoff_now()) { result.omitted = std::max(result.omitted, bound); continue; }
            }
            std::array<double, sample_groups> single_groups{};
            const bool screen_single = q == 1 && b_.visit_column && high - low >= 8;
            const auto single_bound = [&](size_t channel) {
                double sum = 0.;
                for (size_t g = 0; g < sample_groups; ++g)
                    sum += single_groups[g] * log_group_norms_.data[channel * sample_groups + g];
                return padded((sum + underflow) * b_.weights[h + 1] / n_ + rounding * caps_[h + 1] * mass, rounding);
            };
            bool prepared = false, scaled = false;
            double magnitude = 0.;
            const double* factors = integrals.data;
            double* weighted_norm = single_stats.data;
            const auto prepare_single = [&] {
                b_.fill(single_stats, root_constants ? double(l_) : 0.);
                if (!root_constants) for (size_t j = 0; j < l_; ++j) {
                    const double* row = integrals.data + j * n_;
                    for (size_t i = 0; i < n_; ++i) {
                        weighted_norm[i] += row[i] * row[i];
                        single_stats.data[n_ + i] += std::abs(row[i]);
                    }
                }
                for (size_t i = 0; i < n_; ++i) {
                    const double weight = std::abs(residual.data[i]);
                    weighted_norm[i] = weight * std::sqrt(weighted_norm[i] + underflow);
                    magnitude += weight * single_stats.data[n_ + i];
                }
                prepared = true;
            };
            if (screen_single) {
                prepare_single();
                for (size_t g = 0; g < sample_groups; ++g) {
                    const size_t start = n_ * g / sample_groups, size = n_ * (g + 1) / sample_groups - start;
                    single_groups[g] = std::sqrt(b_.bound_dot(weighted_norm + start, weighted_norm + start, size) + underflow);
                }
                const double bound = single_bound(d_ + low);
                if (bound < cutoff_now()) { result.omitted = std::max(result.omitted, bound); continue; }
            }
            Word word = u->word; word.push_back(low); if (q == 2) word.push_back(c);
            for (size_t a = low; a < high; ++a) {
                word[h] = a;
                if (!build && active.count(word)) continue;
                const double cutoff = cutoff_now();
                if (screen_single) {
                    const double bound = single_bound(a);
                    if (bound < cutoff) { result.omitted = std::max(result.omitted, bound); continue; }
                }
                const size_t entry = q == 1 ? a : d_ + c * d_ + a;
                if (norms && !build) {
                    const double upper = norm_score(norms->columns.data[entry]);
                    if (upper < cutoff) {
                        result.omitted = std::max(result.omitted, upper); maximum = std::max(maximum, upper);
                        continue;
                    }
                }
                if (q == 2 && screen_pairs) {
                    const double coarse = pair_bound(a, c);
                    if (coarse < cutoff) { result.omitted = std::max(result.omitted, coarse); continue; }
                    if (!scaled) {
                        const float* right = log_channel_norms_.get() + c * n_;
                        for (size_t i = 0; i < n_; ++i) pair_scale.data[i] = right[i] * pair_weights.data[i];
                        scaled = true;
                    }
                    const double sum = b_.bound_dot(log_channel_norms_.get() + a * n_, pair_scale.data, n_);
                    const double upper = padded((sum + underflow) * b_.weights[h + q] / n_ + rounding * caps_[h + q] * mass, rounding);
                    if (upper < cutoff) {
                        result.omitted = std::max(result.omitted, upper); maximum = std::max(maximum, upper);
                        continue;
                    }
                }
                if (!prepared) {
                    if (q == 1) prepare_single();
                    else {
                        if (!pair_products.owner) pair_products = b_.alloc(multiply(l_ + 2, n_));
                        const auto other = b_.channel(c);
                        samples([&](uint64_t begin, uint64_t end) {
                            cpsig::log_terminal_products(integrals.data, other.values.data, residual.data, pair_products.data,
                                n_, l_, other.dimension, 0, q, begin, end);
                        });
                        factors = pair_products.data;
                        weighted_norm = pair_products.data + l_ * n_;
                        magnitude = std::accumulate(weighted_norm + n_, weighted_norm + 2 * n_, 0.);
                    }
                    prepared = true;
                }
                if (build) {
                    const auto steps = b_.channel(a);
                    cpsig::log_terminal_values(factors, steps.values.data, probe->values.data, n_, l_, steps.dimension, 0, 0, n_);
                    const double radius = column_radius(probe->values.data, h + q, rounding);
                    norms->columns.data[entry] = radius;
                    norms->max_radius = std::max(norms->max_radius, radius);
                    probe->word = word;
                    ++b_.stats.terminal_correlations;
                    const size_t previous = result.candidates.size();
                    maximum = std::max(maximum, offer(probe, residual, alpha, capacity, active, result, true));
                    if (result.candidates.size() > previous) ++b_.stats.constructed;
                    continue;
                }
                double norm = std::numeric_limits<double>::infinity();
                try { norm = b_.bound_dot(log_channel_norms_.get() + a * n_, weighted_norm, n_); }
                catch (const NonfiniteError&) {}
                const double bound = padded((norm + rounding * magnitude + underflow) * b_.weights[h + q] / n_ + rounding * caps_[h + q] * mass, rounding);
                if (std::isfinite(bound) && bound < cutoff) {
                    result.omitted = std::max(result.omitted, bound); maximum = std::max(maximum, bound);
                    continue;
                }
                std::array<double, max_response_batch> remaining{}, partial{};
                const size_t count = b_.batch_ratios.size();
                const double factor = b_.weights[h + q] / n_;
                const auto bounded = [&](double sum) {
                    return padded((sum + rounding * (magnitude + 4 * norm) + underflow) * factor + rounding * caps_[h + q] * mass, rounding);
                };
                if (count && std::isfinite(norm)) {
                    for (size_t i = 0; i < n_; ++i) sample_bounds.data[i] = weighted_norm[i] * log_channel_norms_[a * n_ + i];
                    for (size_t t = 0; t < count; ++t) {
                        const auto* ratio = b_.batch_ratios[t].data;
                        remaining[t] = b_.bound_dot(ratio, sample_bounds.data, n_);
                    }
                    const double upper = bounded(*std::max_element(remaining.begin(), remaining.begin() + count));
                    if (upper < cutoff) { result.omitted = std::max(result.omitted, upper); continue; }
                }
                const auto steps = b_.channel(a);
                bool pruned = false;
                if (count && std::isfinite(norm)) {
                    for (size_t first = 0; first < n_; first += 64) {
                        const size_t last = std::min(n_, first + 64);
                        cpsig::log_terminal_values(factors, steps.values.data, probe->values.data, n_, l_, steps.dimension, 0, first, last);
                        double top = 0.;
                        for (size_t t = 0; t < count; ++t) {
                            const auto* ratio = b_.batch_ratios[t].data;
                            const auto* weight = b_.batch_signed[t].data;
                            const double signed_sum = b_.bound_dot(probe->values.data + first, weight + first, last - first);
                            const double used = b_.bound_dot(sample_bounds.data + first, ratio + first, last - first);
                            partial[t] += signed_sum; remaining[t] -= used;
                            top = std::max(top, std::abs(partial[t]) + std::max(0., remaining[t]));
                        }
                        const double upper = bounded(top);
                        if (upper < cutoff) { result.omitted = std::max(result.omitted, upper); pruned = true; break; }
                    }
                } else if (!b_.visit_column && std::isfinite(norm)) {
                    double partial = 0, correction = 0, used = 0;
                    for (size_t begin = 0; begin < n_; begin += 32) {
                        const size_t end = std::min(n_, begin + 32);
                        cpsig::log_terminal_values(factors, steps.values.data, probe->values.data, n_, l_, steps.dimension, 0, begin, end);
                        const double term = b_.dot(probe->values.data + begin, residual.data + begin, end - begin) - correction;
                        const double next = partial + term; correction = (next - partial) - term; partial = next;
                        used += b_.bound_dot(log_channel_norms_.get() + a * n_ + begin, weighted_norm + begin, end - begin);
                        const double remaining = std::max(0., norm - used) + rounding * (std::abs(norm) + std::abs(used));
                        const double refined = padded((std::abs(partial) + remaining + rounding * magnitude + underflow) * b_.weights[h + q] / n_
                            + rounding * caps_[h + q] * mass, rounding);
                        if (std::isfinite(refined) && refined < cutoff) {
                            result.omitted = std::max(result.omitted, refined); maximum = std::max(maximum, refined); pruned = true; break;
                        }
                    }
                } else samples([&](uint64_t begin, uint64_t end) {
                    cpsig::log_terminal_values(factors, steps.values.data, probe->values.data, n_, l_, steps.dimension, 0, begin, end);
                });
                if (pruned) continue;
                probe->word = word;
                ++b_.stats.terminal_correlations;
                const size_t previous = result.candidates.size();
                maximum = std::max(maximum, offer(probe, residual, alpha, capacity, active, result, true));
                if (result.candidates.size() > previous) ++b_.stats.constructed;
            }
        }
        if (maximum <= alpha) ++b_.stats.terminal_pruned;
    }
public:
    LogSearch(Backend& b, size_t degree)
        : b_(b), n_(b.options.samples), l_(b.options.length - 1), d_(b.options.dimension),
          m_(b.options.dimension == 1 ? 1 : degree) {
        root_ = state({}, 0);
        if (b.options.selected_only) return;
        tables_metadata_ = b_.account(multiply(m_ + 1, 3 * sizeof(double)));
        caps_.assign(m_ + 1, 1.);
        std::vector<double> euler(m_, 0.), binomial(m_, 1.);
        euler[0] = 1.;
        for (size_t h = 1; h <= m_; ++h) {
            for (size_t k = h - 1; k > 0; --k) {
                euler[k] = (k + 1) * euler[k] + (h - k) * euler[k - 1];
                if (k < h - 1) binomial[k] += binomial[k - 1];
            }
            caps_[h] = 0.;
            for (size_t k = 0; k < h; ++k) caps_[h] += euler[k] / binomial[k] / h;
        }
        zero_path_ = !l_ || b_.absolute_sum(b_.variation) == 0;
        if (!zero_path_ && m_ > 2) prepare_channel_variations();
        prepare_letter_factors();
        double variation = 1.;
        if (l_) for (size_t i = 0; i < n_; ++i)
            variation = std::max(variation, b_.variation.data[i] + b_.after.data[i]);
        feature_cap_ = 1.;
        for (size_t h = 1; h <= m_; ++h)
            feature_cap_ = std::max(feature_cap_, padded(caps_[h] * std::pow(variation, int(h)), 1e-11 * (h + 1) * (l_ + 1)));
    }
    double feature_cap() const override { return feature_cap_; }
    NodePtr selected(const Word& word, bool extendible = true) override {
        if (word.empty() || word.size() > m_ || std::any_of(word.begin(), word.end(), [&](uint64_t a) { return a >= d_; })) throw std::invalid_argument("Invalid feature word");
        StatePtr s;
        if (!is_lyndon(word)) throw std::invalid_argument("Log feature words must be Lyndon words");
        if (b_.options.selected_only || !extendible) {
            s = selected_prefix(prediction_path_, root_, word, extendible,
                [&](const StatePtr& parent, size_t a, bool keep) { return log_child(parent, a, keep); });
        } else {
            s = root_; for (auto a : word) s = log_child(s, a);
        }
        auto node = std::make_shared<Node>(); node->word = word; node->values = s->values; return node;
    }
    SearchResult query(const Array& residual, double alpha, const ActiveWords& active,
        size_t capacity) override {
        SearchResult result;
        prediction_path_.clear();
        if (zero_path_ || !l_) return result;
        prepare_residual(residual);
        const auto child_bound = [&](const StatePtr& stem, size_t a, double cutoff) {
            if (!b_.visit_column && !terminal_norms_.empty() && m_ - stem->word.size() == 3) {
                Word child = stem->word; child.push_back(a);
                if (auto it = terminal_norms_.find(child); it != terminal_norms_.end()) {
                    const auto& saved = it->second;
                    const double radius = active.count(child) ? saved.max_radius : std::max(saved.max_radius, saved.prefix_radius);
                    const double bound = norm_score(radius);
                    if (bound < cutoff) return bound;
                }
            }
            if (letter_factors_.owner) {
                const double bound = letter_bound(stem->word, a);
                if (bound < cutoff) return bound;
            }
            return variation_bound(stem, a, residual);
        };
        std::function<void(const StatePtr&)> visit_log;
        visit_log = [&](const StatePtr& u) {
            const size_t h = u->word.size();
            if (h) offer(u, residual, alpha, capacity, active, result);
            if (h == m_ || (h && u->word.front() == d_ - 1)) return;
            if (m_ - h <= 2) {
                log_terminal_search(u, h, residual, alpha, capacity, active, result); return;
            }
            // A direct child follows the periodic repetition of u, then increases one letter.
            StatePtr stem = u;
            const size_t last = h ? m_ - h - 3 : 0;
            for (size_t q = 0; q <= last; ++q) {
                const size_t next = h ? u->word[q % h] : 0;
                const bool terminal = m_ - h - q == 3;
                const bool parallel = terminal && b_.visit_column && b_.options.n_jobs != 1 && d_ >= 12 && !capacity;
                std::vector<size_t> letters;
                const auto descend = [&](size_t a) {
                    auto child = search_child(stem, a);
                    // Equality supplies only the shared state for the remaining terminal endings.
                    if (h && a == next) log_terminal_search(child, h, residual, alpha, capacity, active, result);
                    else visit_log(child);
                };
                for (size_t a = h && !terminal ? next + 1 : next; a < d_; ++a) {
                    if (!h || a > next) {
                        const double cutoff = capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
                        ++b_.stats.block_queries;
                        const double bound = child_bound(stem, a, cutoff);
                        if (bound < cutoff) { result.omitted = std::max(result.omitted, bound); ++b_.stats.pruned; continue; }
                    }
                    if (parallel) letters.push_back(a);
                    else descend(a);
                }
                if (letters.size() >= 4) visit_terminal_children(stem, h, letters, residual, alpha, active, result);
                else for (const size_t a : letters) descend(a);
                if (q < last) stem = log_child(stem, next);
            }
        };
        visit_log(root_);
        std::sort(result.candidates.begin(), result.candidates.end(), better); ++b_.stats.search_passes; return result;
    }
};
}
std::unique_ptr<FeatureSearch> make_log_search(Backend& b, size_t degree) {
    return std::make_unique<LogSearch>(b, degree ? degree : b.options.degree);
}
}
