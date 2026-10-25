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

#pragma once
// Celer working-set and extrapolation rules: eef3bb5f239cf1a0b3a6d0fc8997a20b5e73e21a.
// Copyright (c) 2018-2022, celer.
#include "sig_lasso_types.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace siglasso {

using Word = std::vector<uint64_t>;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
inline size_t multiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) throw std::out_of_range("sig_lasso allocation size overflow");
    return a * b;
}
class NonfiniteError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
inline double finite(double x) {
    if (!std::isfinite(x)) throw NonfiniteError("Nonfinite value in sig_lasso; reduce path scale or degree");
    return x;
}
inline double soft(double x, double t) { return std::copysign(std::max(0.0, std::abs(x) - t), x); }
inline double guard(double x) { return finite(x) * (1.0 + 64 * std::numeric_limits<double>::epsilon()) + 1e-15; }

struct WordHash {
    size_t operator()(const Word& word) const noexcept {
        size_t h = 0xcbf29ce484222325ULL;
        for (auto a : word) h = (h ^ a) * 0x100000001b3ULL;
        return h;
    }
};
inline bool word_less(const Word& a, const Word& b) { return a.size() == b.size() ? a < b : a.size() < b.size(); }

class Budget {
    size_t used_ = 0, peak_ = 0, limit_;
    mutable std::mutex mutex_;
public:
    explicit Budget(size_t limit) : limit_(limit) {}
    void acquire(size_t bytes) {
        std::lock_guard lock(mutex_);
        if (bytes > std::numeric_limits<size_t>::max() - used_ || (limit_ && bytes > limit_ - used_))
            throw std::bad_alloc();
        used_ += bytes;
        peak_ = std::max(peak_, used_);
    }
    void release(size_t bytes) { std::lock_guard lock(mutex_); used_ -= bytes; }
    size_t peak() const { std::lock_guard lock(mutex_); return peak_; }
    size_t used() const { std::lock_guard lock(mutex_); return used_; }
    size_t available() const { std::lock_guard lock(mutex_); return limit_ ? limit_ - used_ : std::numeric_limits<size_t>::max() - used_; }
    std::shared_ptr<void> reserve(size_t bytes, const std::shared_ptr<Budget>& owner) {
        acquire(bytes);
        return std::shared_ptr<void>(this, [owner, bytes](void*) { owner->release(bytes); });
    }
};

struct Array {
    double* data = nullptr;
    size_t size = 0;
    std::shared_ptr<void> owner;
    double* at(size_t offset) const { return data + offset; }
    size_t bytes() const { return multiply(size, sizeof(double)); }
};

struct Column {
    Word word;
    Array values;
    double mean = 0, norm2 = 0, inverse_norm2 = 0;
    std::shared_ptr<void> metadata;
};

struct Node {
    Word word;
    Array coefficients, moments, integrals, values;
    std::shared_ptr<void> metadata;
    size_t bytes() const { return coefficients.bytes() + moments.bytes() + integrals.bytes() + values.bytes() + 512 + word.size() * 64; }
};
using NodePtr = std::shared_ptr<Node>;
struct Probe { NodePtr node; uint64_t low, high; size_t block = 0; Array gram; };
struct Candidate { Word word; Array values; double score; };
inline bool better(const Candidate& a, const Candidate& b) {
    return a.score == b.score ? word_less(a.word, b.word) : a.score > b.score;
}
struct SearchResult { std::vector<Candidate> candidates; double omitted = 0; };

class Backend {
public:
    const SigLassoOptions options;
    const Clock::time_point started = Clock::now();
    std::shared_ptr<Budget> budget;
    SigLassoStats stats{};
    Array increments, variation, after, envelopes, alphabet_ratios;
    std::vector<std::pair<size_t, size_t>> alphabet;
    std::unordered_set<Word, WordHash> constructed;
    std::vector<std::shared_ptr<void>> record_storage;
    std::function<void()> trim_search;
    std::shared_ptr<void> metadata;
    Array column_slab;
    size_t slab_used = 0;
    std::vector<double> weights;
    explicit Backend(const SigLassoOptions& o, std::shared_ptr<Budget> shared_budget = {}) : options(o), budget(shared_budget ? std::move(shared_budget) : std::make_shared<Budget>(o.memory_limit)) {
        metadata = budget->reserve(multiply(o.samples, 64) + multiply(o.degree + o.dimension + o.targets, 256) + 4096, budget);
    }
    virtual ~Backend() = default;
    virtual Array alloc(size_t size) = 0;
    virtual void fill(const Array&, double) = 0;
    virtual void copy(const double*, double*, size_t) = 0;
    virtual void upload(const double*, double*, size_t) = 0;
    virtual void download(const double*, double*, size_t) = 0;
    virtual double dot(const double*, const double*, size_t) = 0;
    virtual double absolute_sum(const Array&) = 0;
    virtual void axpy(double, const double*, double*, size_t) = 0;
    virtual void recompute_residual(const Array& y, const std::vector<Column>& columns, const std::vector<double>& beta, Array& residual) {
        copy(y.data, residual.data, y.size);
        for (size_t j = 0; j < beta.size(); ++j) if (beta[j]) axpy(-beta[j], columns[j].values.data, residual.data, y.size);
    }
    virtual void scale(double, double*, size_t) = 0;
    virtual double center(double*, size_t, bool) = 0;
    virtual void round(double*, size_t) = 0;
    virtual void correlations(const std::vector<Column>&, const Array&, std::vector<double>&, const std::vector<size_t>* = nullptr) = 0;
    virtual void sweeps(const std::vector<Column>&, const std::vector<size_t>&, Array&, Array&, double, size_t, std::array<Array, 6>*, size_t&) = 0;
    virtual void small_gram(const std::array<Array, 5>& vectors, std::array<double, 25>& gram) {
        for (size_t j = 0; j < 5; ++j) for (size_t k = 0; k <= j; ++k)
            gram[j * 5 + k] = gram[k * 5 + j] = dot(vectors[j].data, vectors[k].data, vectors[j].size);
    }
    virtual void reset_solver() {}
    virtual void prepare() = 0;
    virtual Array factors(size_t k) = 0;
    virtual void moments(Node&, const Array&) = 0;
    virtual void bounds(const std::vector<Probe>&, const Array&, double, std::vector<double>&) = 0;
    virtual Array energy_gram(size_t, size_t) = 0;
    virtual void children(const std::vector<Probe>&, std::vector<NodePtr>&) = 0;
    virtual void full_columns(std::vector<Column>&) = 0;
    virtual Array response(size_t target) = 0;
    virtual bool native_search(const Array&, double, const std::unordered_map<Word, size_t, WordHash>&, size_t, SearchResult&) { return false; }
    virtual bool has_native_search() const { return false; }
    virtual void finish() {}
    virtual void for_targets(const std::function<void(Backend&, size_t)>& action) { for (size_t j = 0; j < options.targets; ++j) action(*this, j); }
    virtual void trim_caches() { if (trim_search) trim_search(); }
    std::shared_ptr<void> account(size_t bytes) {
        if (bytes > budget->available()) trim_caches();
        return budget->reserve(bytes, budget);
    }
    void record_word(const Word& word) {
        if (constructed.count(word)) return;
        record_storage.push_back(account(192 + word.size() * 16));
        constructed.insert(word); stats.constructed = constructed.size();
    }
    Array clone(const Array& source) { auto out = alloc(source.size); copy(source.data, out.data, source.size); return out; }
    Array column_storage() {
        const size_t n = options.samples;
        if (slab_used + n > column_slab.size) {
            size_t columns = std::min<size_t>(64, std::max<size_t>(1, 1024 * 1024 / (n * sizeof(double))));
            columns = std::min(columns, std::max<size_t>(1, budget->available() / (n * sizeof(double))));
            column_slab = alloc(multiply(columns, n)); slab_used = 0;
        }
        Array result{column_slab.data + slab_used, n, column_slab.owner}; slab_used += n;
        return result;
    }
    Array from_host(const std::vector<double>& source) { auto out = alloc(source.size()); upload(source.data(), out.data, source.size()); return out; }
    std::vector<double> host(const Array& source) { std::vector<double> out(source.size); download(source.data, out.data(), out.size()); return out; }
    void prepare_alphabet() {
        std::vector<std::pair<size_t, size_t>> pending{{0, options.dimension}};
        while (!pending.empty()) {
            auto interval = pending.back(); pending.pop_back(); alphabet.push_back(interval);
            if (interval.second - interval.first > 1) {
                const auto mid = (interval.first + interval.second) / 2;
                pending.emplace_back(mid, interval.second); pending.emplace_back(interval.first, mid);
            }
        }
        alphabet_ratios = alloc(multiply(multiply(options.samples, options.length - 1), alphabet.size()));
    }
};

struct Result {
    std::vector<Word> words;
    std::vector<double> coefficients, intercepts, objectives, gaps;
    std::vector<uint64_t> converged, iterations;
    SigLassoStats stats{};
    size_t targets = 0;
    void* stream = nullptr;
    int device = 0;
};

class Results {
    std::mutex mutex_;
    uint64_t next_ = 1;
    std::unordered_map<uint64_t, std::shared_ptr<Result>> values_;
public:
    uint64_t insert(Result result) {
        std::lock_guard lock(mutex_);
        const auto id = next_++;
        values_.emplace(id, std::make_shared<Result>(std::move(result)));
        return id;
    }
    std::shared_ptr<Result> get(uint64_t id) {
        std::lock_guard lock(mutex_);
        auto it = values_.find(id);
        if (it == values_.end()) throw std::invalid_argument("Invalid sig_lasso result handle");
        return it->second;
    }
    void erase(uint64_t id) { std::lock_guard lock(mutex_); values_.erase(id); }
};

class PrefixCache {
    size_t capacity_, used_ = 0;
    std::list<std::pair<Word, NodePtr>> entries_;
    std::unordered_map<Word, decltype(entries_)::iterator, WordHash> positions_;
public:
    explicit PrefixCache(size_t capacity) : capacity_(capacity) {}
    NodePtr get(const Word& word) {
        auto it = positions_.find(word);
        if (it == positions_.end()) return {};
        entries_.splice(entries_.begin(), entries_, it->second);
        return it->second->second;
    }
    void insert(const NodePtr& node) {
        const size_t bytes = node->bytes();
        if (!capacity_ || bytes > capacity_ || positions_.count(node->word)) return;
        while (used_ + bytes > capacity_ && !entries_.empty()) {
            used_ -= entries_.back().second->bytes();
            positions_.erase(entries_.back().first);
            entries_.pop_back();
        }
        entries_.emplace_front(node->word, node);
        positions_.emplace(node->word, entries_.begin());
        used_ += bytes;
    }
    void clear() { positions_.clear(); entries_.clear(); used_ = 0; }
};

class Search {
    Backend& b_;
    PrefixCache cache_;
    std::unordered_map<size_t, Array> factors_;
    std::unordered_map<size_t, Array> grams_;
    size_t factor_bytes_ = 0;
    size_t gram_bytes_ = 0;
    NodePtr root_;
    Array get_factors(size_t k) {
        auto it = factors_.find(k);
        if (it != factors_.end()) return it->second;
        auto value = b_.factors(k);
        const size_t limit = b_.options.cache_bytes / 4;
        if (value.bytes() <= limit) {
            if (factor_bytes_ + value.bytes() > limit) { factors_.clear(); factor_bytes_ = 0; }
            factors_.emplace(k, value);
            factor_bytes_ += value.bytes();
        }
        return value;
    }
public:
    explicit Search(Backend& b) : b_(b), cache_(b.options.cache_bytes / 8 * 3) {
        b_.trim_search = [&] { cache_.clear(); factors_.clear(); grams_.clear(); factor_bytes_ = gram_bytes_ = 0; };
        root_ = std::make_shared<Node>();
        root_->coefficients = b_.alloc(multiply(b.options.samples, b.options.length - 1));
        b_.fill(root_->coefficients, 1);
        b_.moments(*root_, get_factors(0));
    }
    ~Search() { b_.trim_search = {}; }
    NodePtr selected(const Word& word) {
        auto node = root_;
        for (auto letter : word) {
            Word child_word = node->word; child_word.push_back(letter);
            auto child = cache_.get(child_word);
            if (child) ++b_.stats.cache_hits;
            else {
                ++b_.stats.cache_misses;
                std::vector<NodePtr> children;
                b_.children({Probe{node, letter, letter + 1}}, children);
                child = children.front();
                b_.record_word(child->word);
                if (child->word.size() < b_.options.degree) b_.moments(*child, get_factors(child->word.size()));
                cache_.insert(child);
                ++b_.stats.evaluated;
            }
            node = child;
        }
        return node;
    }
    SearchResult query(const Array& residual, double alpha, const std::unordered_map<Word, size_t, WordHash>& active, size_t capacity) {
        SearchResult result;
        if (b_.has_native_search()) {
            cache_.clear(); factors_.clear(); grams_.clear(); root_.reset(); factor_bytes_ = gram_bytes_ = 0;
            b_.native_search(residual, alpha, active, capacity, result); ++b_.stats.search_passes; return result;
        }
        std::vector<Probe> stack{{root_, 0, b_.options.dimension}};
        while (!stack.empty()) {
            std::vector<Probe> batch;
            const auto k = stack.back().node->word.size();
            while (!stack.empty() && batch.size() < 32 && stack.back().node->word.size() == k) {
                batch.push_back(std::move(stack.back())); stack.pop_back();
            }
            double threshold = alpha;
            if (capacity && result.candidates.size() == capacity) threshold = std::max(threshold, result.candidates.front().score);
            std::vector<double> bounds;
            const size_t gram_size = multiply(multiply(b_.options.samples, b_.options.length - 1), (b_.options.length - 1) * sizeof(double));
            for (auto& task : batch) if (task.high - task.low > b_.options.length - 1 && gram_size && gram_size <= b_.options.cache_bytes / 8) {
                auto found = grams_.find(task.block);
                if (found == grams_.end()) {
                    if (gram_bytes_ + gram_size > b_.options.cache_bytes / 8) { grams_.clear(); gram_bytes_ = 0; }
                    auto gram = b_.energy_gram(task.low, task.high);
                    found = grams_.emplace(task.block, std::move(gram)).first; gram_bytes_ += gram_size;
                }
                task.gram = found->second;
            }
            b_.bounds(batch, residual, threshold, bounds);
            b_.stats.block_queries += batch.size();
            std::vector<Probe> leaves;
            for (size_t j = 0; j < batch.size(); ++j) {
                auto& task = batch[j];
                const double bound = guard(bounds[j] * (1.0 + 1e-10));
                if (bound < threshold) {
                    result.omitted = std::max(result.omitted, bound);
                    ++b_.stats.pruned;
                } else if (task.high - task.low > 1) {
                    const auto middle = task.low + (task.high - task.low) / 2;
                    stack.push_back({task.node, middle, task.high, task.block + 2 * (middle - task.low)});
                    stack.push_back({task.node, task.low, middle, task.block + 1});
                } else leaves.push_back(std::move(task));
            }
            std::vector<NodePtr> nodes(leaves.size());
            std::vector<Probe> missing;
            std::vector<size_t> missing_positions;
            for (size_t j = 0; j < leaves.size(); ++j) {
                Word word = leaves[j].node->word; word.push_back(leaves[j].low);
                nodes[j] = cache_.get(word);
                if (nodes[j]) ++b_.stats.cache_hits;
                else { missing.push_back(leaves[j]); missing_positions.push_back(j); ++b_.stats.cache_misses; }
            }
            if (!missing.empty()) {
                std::vector<NodePtr> made;
                b_.children(missing, made);
                auto factor = k + 1 < b_.options.degree ? get_factors(k + 1) : Array{};
                for (size_t j = 0; j < made.size(); ++j) {
                    b_.record_word(made[j]->word);
                    if (k + 1 < b_.options.degree) b_.moments(*made[j], factor);
                    cache_.insert(made[j]);
                    nodes[missing_positions[j]] = std::move(made[j]);
                }
                b_.stats.evaluated += made.size();
            }
            std::vector<Column> probe_columns;
            for (const auto& node : nodes) probe_columns.push_back({{}, node->values, 0, 0});
            std::vector<double> scores;
            b_.correlations(probe_columns, residual, scores);
            for (size_t j = 0; j < nodes.size(); ++j) {
                auto& node = nodes[j];
                const double score = finite(std::abs(scores[j]) * b_.weights[node->word.size()]);
                if (!active.count(node->word)) {
                    result.omitted = std::max(result.omitted, guard(score));
                    if (capacity && score > alpha) {
                        Candidate candidate{node->word, node->values, score};
                        if (result.candidates.size() < capacity) {
                            result.candidates.push_back(std::move(candidate));
                            std::push_heap(result.candidates.begin(), result.candidates.end(), better);
                        } else if (better(candidate, result.candidates.front())) {
                            std::pop_heap(result.candidates.begin(), result.candidates.end(), better);
                            result.candidates.back() = std::move(candidate);
                            std::push_heap(result.candidates.begin(), result.candidates.end(), better);
                        }
                    }
                }
                if (node->word.size() < b_.options.degree) stack.push_back({node, 0, b_.options.dimension});
            }
        }
        std::sort(result.candidates.begin(), result.candidates.end(), better);
        ++b_.stats.search_passes;
        return result;
    }
};

struct SolverState {
    Array y, residual, beta, theta, trial, inner_theta, anchor, difference;
    std::array<Array, 6> history;
    std::array<Array, 5> differences;
    std::vector<size_t> initial;
    std::vector<double> anchor_correlations;
    std::vector<uint8_t> screened;
    size_t history_count = 0, history_next = 0;
    double best_dual = -std::numeric_limits<double>::infinity();
    double inner_best_dual = -std::numeric_limits<double>::infinity();
    double mean = 0, scale = 1, objective = 0, gap = 0;
    uint64_t epochs = 0;
};

inline bool solve_five(std::array<double, 25>& a, std::array<double, 5>& rhs) {
    double largest = 0;
    for (size_t j = 0; j < 5; ++j) largest = std::max(largest, std::abs(a[j * 5 + j]));
    if (!(largest > 0)) return false;
    for (size_t j = 0; j < 5; ++j) {
        for (size_t k = 0; k <= j; ++k) {
            double v = a[j * 5 + k];
            for (size_t h = 0; h < k; ++h) v -= a[j * 5 + h] * a[k * 5 + h];
            if (j == k) {
                if (!(v > largest * 1e-12)) return false;
                a[j * 5 + k] = std::sqrt(v);
            } else a[j * 5 + k] = v / a[k * 5 + k];
        }
    }
    for (size_t j = 0; j < 5; ++j) {
        for (size_t k = 0; k < j; ++k) rhs[j] -= a[j * 5 + k] * rhs[k];
        rhs[j] /= a[j * 5 + j];
    }
    for (size_t j = 5; j-- > 0;) {
        for (size_t k = j + 1; k < 5; ++k) rhs[j] -= a[k * 5 + j] * rhs[k];
        rhs[j] /= a[j * 5 + j];
    }
    return std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); });
}

inline void refresh_residual(Backend& b, const std::vector<Column>& columns, SolverState& state, const std::vector<double>& beta) {
    b.recompute_residual(state.y, columns, beta, state.residual);
}

inline double dual_value(Backend& b, const SolverState& state, const Array& theta) {
    return finite(b.dot(state.y.data, theta.data, theta.size) - 0.5 * theta.size * b.dot(theta.data, theta.data, theta.size));
}

inline void consider_dual(Backend& b, const std::vector<Column>& columns, SolverState& state, const Array& candidate, double alpha, bool inner = false) {
    if (b.options.intercept) b.center(candidate.data, candidate.size, true);
    std::vector<double> correlations;
    std::vector<size_t> checked;
    bool refresh_anchor = !inner && ((b.options.solver_flags & 16) || state.anchor_correlations.size() != columns.size());
    if (!inner && !refresh_anchor && state.anchor.owner) {
        if (!state.difference.owner) state.difference = b.alloc(candidate.size);
        b.copy(candidate.data, state.difference.data, candidate.size);
        b.axpy(-1, state.anchor.data, state.difference.data, candidate.size);
        const double distance = guard(std::sqrt(b.dot(state.difference.data, state.difference.data, candidate.size)));
        for (size_t j = 0; j < columns.size(); ++j)
            if (guard(std::abs(state.anchor_correlations[j]) + std::sqrt(columns[j].norm2) * distance) >= alpha) checked.push_back(j);
        refresh_anchor = checked.size() > columns.size() / 2;
    } else refresh_anchor = !inner;
    if (inner || refresh_anchor) b.correlations(columns, candidate, correlations);
    else {
        b.correlations(columns, candidate, correlations, &checked);
    }
    double largest = alpha;
    for (double q : correlations) largest = std::max(largest, std::abs(q) * candidate.size);
    const double scaling = alpha / guard(largest);
    b.scale(scaling, candidate.data, candidate.size);
    if (refresh_anchor) {
        if (!state.anchor.owner) state.anchor = b.alloc(candidate.size);
        b.copy(candidate.data, state.anchor.data, candidate.size);
        state.anchor_correlations.resize(columns.size());
        for (size_t j = 0; j < columns.size(); ++j) state.anchor_correlations[j] = correlations[j] * candidate.size * scaling;
    }
    const double value = dual_value(b, state, candidate);
    auto& best_dual = inner ? state.inner_best_dual : state.best_dual;
    auto& best_theta = inner ? state.inner_theta : state.theta;
    if (value > best_dual) {
        best_dual = value;
        b.copy(candidate.data, best_theta.data, candidate.size);
    }
}

inline void extrapolate(Backend& b, const std::vector<Column>& columns, SolverState& state, double alpha, bool inner = false) try {
    const size_t n = state.y.size;
    if (state.history_count < 6) return;
    for (size_t j = 0; j < 5; ++j) {
        if (!state.differences[j].owner || (b.options.solver_flags & 8)) state.differences[j] = b.alloc(n);
        b.copy(state.history[(state.history_next + j + 1) % 6].data, state.differences[j].data, n);
        b.axpy(-1, state.history[(state.history_next + j) % 6].data, state.differences[j].data, n);
    }
    std::array<double, 25> gram{};
    b.small_gram(state.differences, gram);
    std::array<double, 5> z{1, 1, 1, 1, 1};
    if (!solve_five(gram, z)) return;
    const double total = std::accumulate(z.begin(), z.end(), 0.0);
    double magnitude = 0; for (auto v : z) magnitude += std::abs(v);
    if (!std::isfinite(total) || std::abs(total) < 1e-12 * magnitude) return;
    auto candidate = state.trial; b.fill(candidate, 0);
    for (size_t j = 0; j < 5; ++j) b.axpy(z[j] / (total * n), state.history[(state.history_next + j) % 6].data, candidate.data, n);
    consider_dual(b, columns, state, candidate, alpha, inner);
} catch (const NonfiniteError&) {
    // A failed extrapolation does not invalidate coordinate descent.
}

inline void restricted_solve(Backend& b, const std::vector<Column>& columns, SolverState& state, double alpha, double tolerance) {
    const size_t p = columns.size(), n = state.y.size;
    if (!p) return;
    auto beta = b.host(state.beta);
    if (state.screened.size() != p) state.screened.assign(p, 0);
    auto& screened = state.screened;
    std::vector<size_t> working;
    std::vector<double> priority(p), q;
    size_t requested = 10;
    double previous_gap = std::numeric_limits<double>::infinity();
    for (size_t pass = 0; state.epochs < b.options.max_epochs; ++pass) {
        beta = b.host(state.beta);
        refresh_residual(b, columns, state, beta);
        if (!state.trial.owner || (b.options.solver_flags & 8)) state.trial = b.alloc(n);
        auto candidate = state.trial; b.copy(state.residual.data, candidate.data, n); b.scale(1.0 / n, candidate.data, n);
        consider_dual(b, columns, state, std::move(candidate), alpha);
        const double penalty = std::accumulate(beta.begin(), beta.end(), 0.0, [](double s, double v) { return s + std::abs(v); });
        state.objective = finite(b.dot(state.residual.data, state.residual.data, n) / (2 * n) + alpha * penalty);
        state.gap = std::max(0.0, state.objective - state.best_dual);
        if (state.gap <= tolerance) break;
        std::vector<size_t> remaining;
        for (size_t j = 0; j < p; ++j) if (!screened[j] || beta[j]) remaining.push_back(j);
        b.correlations(columns, state.theta, q, &remaining);
        const double radius = std::sqrt(2 * state.gap / n);
        std::vector<size_t> eligible;
        size_t support = 0;
        for (size_t position = 0; position < remaining.size(); ++position) {
            const size_t j = remaining[position];
            if (!columns[j].norm2) { screened[j] = 1; continue; }
            const double norm = std::sqrt(columns[j].norm2);
            priority[j] = (alpha - std::abs(q[position]) * n) / norm;
            if (!(b.options.solver_flags & 2) && !beta[j] && priority[j] > radius + 1e-12) {
                if (!screened[j]) ++b.stats.screening_count;
                screened[j] = 1;
            }
            if (beta[j]) { screened[j] = 0; priority[j] = -std::numeric_limits<double>::infinity(); ++support; }
            if (!screened[j]) eligible.push_back(j);
        }
        requested = pass == 0 ? std::max<size_t>(10, support) : std::max<size_t>(10, 2 * support);
        if (state.gap > previous_gap * 0.9) requested = std::max(requested, 2 * working.size());
        previous_gap = state.gap;
        requested = std::min(requested, eligible.size());
        if (b.options.solver_flags & 1) requested = eligible.size();
        auto compare = [&](size_t a, size_t c) { return priority[a] == priority[c] ? a < c : priority[a] < priority[c]; };
        if (requested < eligible.size()) std::nth_element(eligible.begin(), eligible.begin() + requested, eligible.end(), compare);
        eligible.resize(requested); std::sort(eligible.begin(), eligible.end());
        if (!state.initial.empty()) {
            if (!support) {
                const size_t count = std::min<size_t>(10, state.initial.size());
                if (count < state.initial.size()) std::nth_element(state.initial.begin(), state.initial.begin() + count, state.initial.end(), compare);
                eligible.assign(state.initial.begin(), state.initial.begin() + count); std::sort(eligible.begin(), eligible.end());
            }
            state.initial.clear();
        }
        working = std::move(eligible);
        if (working.empty()) break;
        b.reset_solver();
        std::vector<Column> inner_columns; inner_columns.reserve(working.size());
        for (auto j : working) inner_columns.push_back(columns[j]);
        if (!state.inner_theta.owner) state.inner_theta = b.alloc(n);
        state.inner_best_dual = -std::numeric_limits<double>::infinity();
        state.history_count = state.history_next = 0;
        const double target = std::max(tolerance, 0.3 * state.gap);
        while (state.epochs < b.options.max_epochs) {
            const size_t sweeps = std::min<uint64_t>(10, b.options.max_epochs - state.epochs);
            if (!(b.options.solver_flags & 4)) for (auto& snapshot : state.history)
                if (!snapshot.owner || (b.options.solver_flags & 8)) snapshot = b.alloc(n);
            b.sweeps(columns, working, state.beta, state.residual, alpha, sweeps, (b.options.solver_flags & 4) ? nullptr : &state.history, state.history_next);
            state.history_count = std::min<size_t>(6, state.history_count + sweeps);
            state.epochs += sweeps; b.stats.coordinate_sweeps += sweeps;
            if (!(b.options.solver_flags & 4)) extrapolate(b, inner_columns, state, alpha, true);
            if (b.options.solver_flags & 8) state.trial = b.alloc(n);
            auto trial = state.trial; b.copy(state.residual.data, trial.data, n); b.scale(1.0 / n, trial.data, n);
            consider_dual(b, inner_columns, state, trial, alpha, true);
            const double l1 = b.absolute_sum(state.beta);
            state.objective = finite(b.dot(state.residual.data, state.residual.data, n) / (2 * n) + alpha * l1);
            const double inner_gap = std::max(0.0, state.objective - state.inner_best_dual);
            if (inner_gap <= target) break;
        }
        b.copy(state.inner_theta.data, state.trial.data, n);
        consider_dual(b, columns, state, state.trial, alpha);
    }
}

inline void append_column(Backend& b, std::vector<Column>& columns, const Word& word, const Array& raw) {
    auto metadata = b.account(1024 + word.size() * 64 + b.options.targets * 32);
    Column column{word, b.column_storage()}; column.metadata = std::move(metadata);
    b.copy(raw.data, column.values.data, raw.size);
    b.scale(b.weights[word.size()], column.values.data, column.values.size);
    column.mean = b.center(column.values.data, column.values.size, b.options.intercept != 0);
    column.norm2 = finite(b.dot(column.values.data, column.values.data, column.values.size));
    column.inverse_norm2 = column.norm2 ? 1.0 / column.norm2 : 0;
    columns.push_back(std::move(column));
}

inline Result fit(Backend& b, const std::vector<double>& alphas, const std::vector<Word>& initial) {
    auto started = b.started;
    std::vector<Column> columns;
    std::unordered_map<Word, size_t, WordHash> active;
    std::unique_ptr<Search> search;
    if (b.options.full) b.full_columns(columns);
    else { b.prepare(); search = std::make_unique<Search>(b); }
    for (size_t j = 0; j < columns.size(); ++j) active.emplace(columns[j].word, j);
    if (search) for (const auto& word : initial) if (!active.count(word)) {
        auto node = search->selected(word);
        active.emplace(word, columns.size()); append_column(b, columns, word, node->values);
    }
    b.finish(); b.stats.preparation_seconds = elapsed(started);
    std::vector<std::vector<double>> coefficients(b.options.targets);
    Result result; result.targets = b.options.targets;
    result.intercepts.resize(result.targets); result.objectives.resize(result.targets); result.gaps.resize(result.targets);
    result.converged.resize(result.targets); result.iterations.resize(result.targets);
    b.for_targets([&](Backend& b, size_t target_index) {
        SolverState state;
        state.y = b.response(target_index);
        state.mean = b.center(state.y.data, state.y.size, b.options.intercept != 0);
        state.residual = b.clone(state.y); state.theta = b.alloc(state.y.size); b.fill(state.theta, 0);
        state.beta = b.alloc(columns.size()); b.fill(state.beta, 0);
        for (const auto& word : initial) state.initial.push_back(active.at(word));
        state.scale = std::max(1.0, b.dot(state.y.data, state.y.data, state.y.size) / state.y.size);
        const double alpha = alphas[target_index], tolerance = b.options.tolerance * state.scale;
        double restricted_target = b.options.full ? tolerance * 0.1 : std::max(tolerance * 0.1, std::min(state.scale * 1e-3, alpha * alpha / 2));
        double final_gap = std::numeric_limits<double>::infinity(), final_objective = 0;
        bool converged = false;
        uint64_t iteration = 0;
        for (; iteration < b.options.max_iter; ++iteration) {
            if (state.beta.size != columns.size()) {
                auto enlarged = b.alloc(columns.size()); b.fill(enlarged, 0); b.copy(state.beta.data, enlarged.data, state.beta.size);
                state.beta = std::move(enlarged); state.best_dual = -std::numeric_limits<double>::infinity();
                state.history_count = state.history_next = 0; b.reset_solver();
                consider_dual(b, columns, state, b.clone(state.theta), alpha);
            }
            auto solve_started = Clock::now();
            restricted_solve(b, columns, state, alpha, restricted_target);
            b.stats.solve_seconds += elapsed(solve_started);
            auto rounded = b.clone(state.beta); if (b.options.single_precision) b.round(rounded.data, rounded.size);
            const auto beta = b.host(rounded);
            refresh_residual(b, columns, state, beta);
            const double penalty = std::accumulate(beta.begin(), beta.end(), 0.0, [](double s, double v) { return s + std::abs(v); });
            final_objective = finite(b.dot(state.residual.data, state.residual.data, state.y.size) / (2 * state.y.size) + alpha * penalty);
            SearchResult found;
            if (search) {
                auto t = Clock::now(); found = search->query(state.residual, alpha, active, b.options.batch_size);
                b.stats.search_seconds += elapsed(t);
            }
            std::vector<double> scores; b.correlations(columns, state.residual, scores);
            double maximum = std::max(alpha, found.omitted);
            for (auto q : scores) maximum = std::max(maximum, guard(std::abs(q)));
            auto theta = b.clone(state.residual);
            if (b.options.intercept) b.center(theta.data, theta.size, true);
            b.scale(alpha / (maximum * theta.size), theta.data, theta.size);
            double dual = dual_value(b, state, theta);
            if (std::isfinite(state.best_dual) && state.best_dual > dual) {
                auto accelerated = b.clone(state.theta);
                double dual_max = alpha;
                if (search) {
                    auto t = Clock::now();
                    auto scaled = b.clone(accelerated); b.scale(state.y.size, scaled.data, scaled.size);
                    dual_max = std::max(dual_max, search->query(scaled, alpha, active, 0).omitted);
                    b.stats.search_seconds += elapsed(t);
                }
                b.correlations(columns, accelerated, scores);
                for (auto q : scores) dual_max = std::max(dual_max, guard(std::abs(q) * state.y.size));
                b.scale(alpha / dual_max, accelerated.data, accelerated.size);
                dual = std::max(dual, dual_value(b, state, accelerated));
            }
            final_gap = std::max(0.0, final_objective - dual);
            if (final_gap <= tolerance) { state.beta = std::move(rounded); converged = true; ++iteration; break; }
            if (state.epochs >= b.options.max_epochs) { state.beta = std::move(rounded); ++iteration; break; }
            if (!found.candidates.empty()) {
                auto t = Clock::now();
                for (const auto& candidate : found.candidates) if (!active.count(candidate.word)) {
                    active.emplace(candidate.word, columns.size()); append_column(b, columns, candidate.word, candidate.values);
                }
                b.stats.insertion_seconds += elapsed(t);
            } else restricted_target *= 0.1;
        }
        if (b.options.single_precision) b.round(state.beta.data, state.beta.size);
        coefficients[target_index] = b.host(state.beta);
        double intercept = state.mean;
        for (size_t j = 0; j < coefficients[target_index].size(); ++j) intercept -= columns[j].mean * coefficients[target_index][j];
        if (b.options.single_precision) {
            const double rounded_intercept = static_cast<float>(intercept);
            const double loss = 0.5 * (rounded_intercept - intercept) * (rounded_intercept - intercept);
            final_objective += loss; final_gap += loss; intercept = rounded_intercept;
            converged = converged && final_gap <= tolerance;
        }
        result.intercepts[target_index] = intercept; result.objectives[target_index] = final_objective; result.gaps[target_index] = final_gap;
        result.converged[target_index] = converged; result.iterations[target_index] = iteration;
    });
    std::vector<size_t> selected;
    for (size_t j = 0; j < columns.size(); ++j) {
        bool nonzero = false;
        for (const auto& values : coefficients) nonzero = nonzero || (j < values.size() && values[j] != 0);
        if (nonzero) selected.push_back(j);
    }
    std::sort(selected.begin(), selected.end(), [&](size_t a, size_t c) { return word_less(columns[a].word, columns[c].word); });
    for (auto j : selected) {
        result.words.push_back(columns[j].word);
        for (const auto& values : coefficients) result.coefficients.push_back(j < values.size() ? values[j] : 0);
    }
    b.finish(); b.stats.columns = columns.size(); b.stats.peak_bytes = b.budget->peak(); b.stats.total_seconds = elapsed(started);
    result.stats = b.stats;
    return result;
}

} // namespace siglasso
