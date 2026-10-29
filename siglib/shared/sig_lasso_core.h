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
#include "sig_lasso_types.h"
#include "sig_lasso_active.h"
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
    double mean = 0, norm2 = 0;
    std::shared_ptr<void> metadata;
};

struct Node {
    Word word;
    Array coefficients, moments, integrals, values, time_values, linear_integrals;
    bool uncertain_moments = false;
    std::shared_ptr<void> metadata;
    size_t bytes() const { return coefficients.bytes() + moments.bytes() + integrals.bytes() + values.bytes() + time_values.bytes() + linear_integrals.bytes() + 512 + word.size() * 64; }
};
using NodePtr = std::shared_ptr<Node>;
struct Ancestors { std::vector<NodePtr> nodes; std::shared_ptr<void> metadata; };
using AncestorPtr = std::shared_ptr<Ancestors>;
struct Probe { NodePtr node; uint64_t low, high; size_t block = 0; Array gram; AncestorPtr ancestors; };
struct Candidate { Word word; Array values; double score; };
inline bool better(const Candidate& a, const Candidate& b) {
    return a.score == b.score ? word_less(a.word, b.word) : a.score > b.score;
}
struct SearchResult { std::vector<Candidate> candidates; double omitted = 0; };

class HostSearch {
public:
    virtual ~HostSearch() = default;
    virtual void begin_query(const Array&, const std::unordered_map<Word, size_t, WordHash>&) {}
    virtual void bounds(const std::vector<Probe>&, const Array&, double, std::vector<double>&) = 0;
};

class Backend {
public:
    const SigLassoOptions options;
    const Clock::time_point started = Clock::now();
    std::shared_ptr<Budget> budget;
    SigLassoStats stats{};
    Array increments, variation, after, alphabet_ratios;
    std::vector<std::pair<size_t, size_t>> alphabet;
    std::unordered_set<Word, WordHash> constructed;
    std::vector<std::shared_ptr<void>> record_storage;
    std::function<void()> trim_search;
    std::shared_ptr<void> metadata;
    Array column_slab;
    size_t slab_used = 0;
    std::vector<double> weights, reciprocals;
    explicit Backend(const SigLassoOptions& o, std::shared_ptr<Budget> shared_budget = {}) : options(o), budget(shared_budget ? std::move(shared_budget) : std::make_shared<Budget>(o.memory_limit)) {
        if (o.search_method > 3) throw std::invalid_argument("Invalid sig_lasso search method");
        if (o.bound_method > 1) throw std::invalid_argument("Invalid sig_lasso bound method");
        if (o.degree > 170) throw std::out_of_range("sig_lasso degree exceeds the finite factorial range");
        weights.resize(o.degree + 1, 1.0);
        for (size_t k = 1; k <= o.degree; ++k) weights[k] = weights[k - 1] * static_cast<double>(k);
        reciprocals.resize(o.degree + 3);
        for (size_t k = 1; k < reciprocals.size(); ++k) reciprocals[k] = 1.0 / k;
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
    virtual void scale(double, double*, size_t) = 0;
    virtual double center(double*, size_t, bool) = 0;
    virtual void correlations(const std::vector<Column>&, const Array&, std::vector<double>&, const std::vector<size_t>* = nullptr) = 0;
    virtual std::unique_ptr<ActiveSet> active_set(const Array&, double alpha) = 0;
    virtual double correlation_max(const std::vector<Column>& columns, const Array& vector) {
        std::vector<double> scores; correlations(columns, vector, scores);
        double maximum = 0;
        for (auto value : scores) maximum = std::max(maximum, guard(std::abs(value)));
        return maximum;
    }
    virtual std::vector<size_t> dense_candidates(const std::vector<Column>& columns, const Array& vector,
        const std::unordered_set<size_t>& offered, double alpha, size_t capacity) {
        std::vector<double> scores; correlations(columns, vector, scores);
        std::vector<size_t> selected;
        for (size_t j = 0; j < columns.size(); ++j) if (columns[j].norm2 && !offered.count(j) && std::abs(scores[j]) > alpha) selected.push_back(j);
        auto compare = [&](size_t a, size_t b) { return std::abs(scores[a]) == std::abs(scores[b])
            ? word_less(columns[a].word, columns[b].word) : std::abs(scores[a]) > std::abs(scores[b]); };
        if (selected.size() > capacity) { std::nth_element(selected.begin(), selected.begin() + capacity, selected.end(), compare); selected.resize(capacity); }
        std::sort(selected.begin(), selected.end(), compare);
        return selected;
    }
    virtual void prepare() = 0;
    virtual Array factors(size_t k) = 0;
    virtual void moments(Node&, const Array&) = 0;
    virtual HostSearch* host_search() { return nullptr; }
    virtual Array energy_gram(size_t, size_t) = 0;
    virtual void children(const std::vector<Probe>&, std::vector<NodePtr>&) = 0;
    virtual void full_columns(std::vector<Column>&) = 0;
    virtual Array response(size_t target) = 0;
    virtual void export_result(const Result&, const SigLassoOutput&) = 0;
    virtual bool native_search(const Array&, double, const std::unordered_map<Word, size_t, WordHash>&, size_t, SearchResult&) { return false; }
    virtual bool has_native_search() const { return false; }
    virtual bool reuse_search_bound() const { return false; }
    virtual void finish() {}
    virtual std::unique_ptr<Backend> solver_worker() { return {}; }
    virtual void for_targets(const std::function<void(size_t)>& action) { for (size_t j = 0; j < options.targets; ++j) action(j); }
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
        if (!(options.search_method & 1)) {
            for (size_t a = 0; a < options.dimension; ++a) alphabet.emplace_back(a, a + 1);
            alphabet_ratios = alloc(multiply(multiply(options.samples, options.length - 1), alphabet.size()));
            return;
        }
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
    std::vector<Column> selected_columns;
    SigLassoStats stats{};
    size_t targets = 0;
};

inline SigLassoOutput allocate_output(const Result& result, SigLassoAllocator allocate, void* context) {
    SigLassoShape shape{result.words.size(), 0, result.targets, result.stats};
    for (const auto& word : result.words) shape.letters += word.size();
    SigLassoOutput output{};
    if (!allocate) throw std::invalid_argument("sig_lasso requires an output allocator");
    if (allocate(context, &shape, &output)) throw std::runtime_error("sig_lasso output allocation failed");
    if ((shape.letters && !output.letters) || !output.offsets || !output.converged || !output.iterations ||
        (shape.words && (!output.coefficients || !output.signatures)) || !output.intercepts || !output.objectives || !output.gaps || !output.stats)
        throw std::invalid_argument("sig_lasso output allocator returned a null buffer");
    size_t count = 0; output.offsets[0] = 0;
    for (size_t j = 0; j < result.words.size(); ++j) {
        for (auto a : result.words[j]) output.letters[count++] = a;
        output.offsets[j + 1] = count;
    }
    std::copy(result.converged.begin(), result.converged.end(), output.converged);
    std::copy(result.iterations.begin(), result.iterations.end(), output.iterations);
    return output;
}

inline void export_result(Backend& backend, Result& result, SigLassoAllocator allocate, void* context) {
    const auto started = Clock::now();
    size_t letters = 0;
    for (const auto& word : result.words) letters += word.size();
    const size_t scalars = multiply(result.words.size(), backend.options.samples + result.targets) + 3 * result.targets;
    backend.stats.output_bytes = multiply(scalars, backend.options.single_precision ? sizeof(float) : sizeof(double))
        + multiply(letters + result.words.size() + 1 + 2 * result.targets, sizeof(uint64_t));
    auto output_storage = backend.account(backend.stats.output_bytes);
    result.stats = backend.stats;
    const auto output = allocate_output(result, allocate, context);
    backend.export_result(result, output);
    backend.finish();
    backend.stats.output_seconds = elapsed(started);
    backend.stats.peak_bytes = backend.budget->peak();
    backend.stats.total_seconds = elapsed(backend.started);
    *output.stats = backend.stats;
}

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
    AncestorPtr extend(const AncestorPtr& parent, const NodePtr& node) {
        if (b_.options.bound_method) return {};
        const size_t count = parent ? parent->nodes.size() + 1 : 1;
        auto metadata = b_.account(128 + multiply(count, sizeof(NodePtr)));
        auto result = std::make_shared<Ancestors>(); result->metadata = std::move(metadata);
        result->nodes.reserve(count);
        if (parent) result->nodes.insert(result->nodes.end(), parent->nodes.begin(), parent->nodes.end());
        result->nodes.push_back(node);
        return result;
    }
    Array get_factors(size_t k) {
        if (!b_.options.bound_method) return {};
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
    explicit Search(Backend& b, NodePtr root = {}) : b_(b),
        cache_(b.options.cache_bytes / (!b.options.bound_method && b.options.search_method == 3 ? 16 : 8) * 3), root_(std::move(root)) {
        b_.trim_search = [&] { cache_.clear(); factors_.clear(); grams_.clear(); factor_bytes_ = gram_bytes_ = 0; };
        if (root_) return;
        root_ = std::make_shared<Node>();
        if (b.options.bound_method) {
            root_->coefficients = b_.alloc(multiply(b.options.samples, b.options.length - 1));
            b_.fill(root_->coefficients, 1);
        } else {
            root_->time_values = b_.alloc(multiply(b.options.samples, b.options.length));
            b_.fill(root_->time_values, 1);
        }
        b_.moments(*root_, get_factors(0));
    }
    ~Search() { b_.trim_search = {}; }
    const NodePtr& root() const { return root_; }
    NodePtr selected(const Word& word) {
        auto node = root_;
        auto ancestors = extend({}, node);
        for (auto letter : word) {
            Word child_word = node->word; child_word.push_back(letter);
            auto child = cache_.get(child_word);
            if (child) ++b_.stats.cache_hits;
            else {
                ++b_.stats.cache_misses;
                std::vector<NodePtr> children;
                b_.children({Probe{node, letter, letter + 1, 0, {}, ancestors}}, children);
                child = children.front();
                b_.record_word(child->word);
                if (child->word.size() < b_.options.degree) b_.moments(*child, get_factors(child->word.size()));
                cache_.insert(child);
                ++b_.stats.evaluated;
            }
            node = child;
            if (node->word.size() < b_.options.degree) ancestors = extend(ancestors, node);
        }
        return node;
    }
    SearchResult query(const Array& residual, double alpha, const std::unordered_map<Word, size_t, WordHash>& active, size_t capacity,
        size_t low = 0, size_t high = 0, size_t block = 0) {
        SearchResult result;
        if (b_.has_native_search()) {
            cache_.clear(); factors_.clear(); grams_.clear(); root_.reset(); factor_bytes_ = gram_bytes_ = 0;
            b_.native_search(residual, alpha, active, capacity, result); ++b_.stats.search_passes; return result;
        }
        b_.host_search()->begin_query(residual, active);
        std::vector<Probe> stack;
        const auto push = [&](const NodePtr& node, size_t first, size_t last, size_t index, const AncestorPtr& ancestors) {
            if (b_.options.search_method & 1) stack.push_back({node, first, last, index, {}, ancestors});
            else for (size_t a = last; a-- > first;) stack.push_back({node, a, a + 1, a, {}, ancestors});
        };
        push(root_, low, high ? high : b_.options.dimension, block, extend({}, root_));
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
            b_.host_search()->bounds(batch, residual, threshold, bounds);
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
                    stack.push_back({task.node, middle, task.high, task.block + 2 * (middle - task.low), {}, task.ancestors});
                    stack.push_back({task.node, task.low, middle, task.block + 1, {}, task.ancestors});
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
                if (node->word.size() < b_.options.degree) {
                    push(node, 0, b_.options.dimension, 0, extend(leaves[j].ancestors, node));
                }
            }
        }
        std::sort(result.candidates.begin(), result.candidates.end(), better);
        ++b_.stats.search_passes;
        return result;
    }
};

} // namespace siglasso

#include "sig_lasso_fit.h"
