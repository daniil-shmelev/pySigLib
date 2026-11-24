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

struct Result;

inline constexpr size_t max_response_batch = 10;

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
};

struct Channel {
    Array values;
    size_t dimension; // Stride between successive time slices, in sample vectors.
};
struct Column {
    Word word;
    Array values;
    double mean = 0, norm2 = 0;
    std::shared_ptr<void> metadata;
    uint64_t id = 0;
};

struct Node {
    Word word;
    Array coefficients, moments, integrals, values, linear_integrals, tt_state;
    std::shared_ptr<void> metadata;
};
using NodePtr = std::shared_ptr<Node>;
struct Probe { NodePtr node; uint64_t low, high; size_t block = 0; };
// Depth-first evaluation of selected words retains just the common prefix.
// A leaf needs its endpoint only; its segment polynomial can be discarded.
template<class Ptr, class Child>
Ptr selected_prefix(std::vector<Ptr>& path, const Ptr& root, const Word& word, bool extendible, Child child) {
    size_t common = 0;
    while (common < path.size() && common < word.size() && path[common]->word.back() == word[common]) ++common;
    path.resize(common);
    Ptr node = common ? path.back() : root;
    for (size_t j = common; j < word.size(); ++j) {
        const bool keep = j + 1 < word.size() || extendible;
        node = child(node, word[j], keep);
        if (keep) path.push_back(node);
    }
    return node;
}
struct QuantizedColumn {
    std::vector<int16_t> samples;
    std::shared_ptr<void> storage;
    double offset = 0., unit = 0., error = 0.;
};
struct Candidate {
    Word word;
    Array values;
    double score;
    std::shared_ptr<const QuantizedColumn> proposal;
};
inline bool better(const Candidate& a, const Candidate& b) {
    return a.score == b.score ? word_less(a.word, b.word) : a.score > b.score;
}
// Keep a bounded heap. A borrowed scratch column only needs copying if it qualifies.
template<class Store>
inline bool retain_candidate(std::vector<Candidate>& heap, size_t capacity, const Word& word, double score, Store store) {
    if (!capacity) return false;
    if (heap.size() == capacity) {
        const auto& last = heap.front();
        if (score < last.score || (score == last.score && !word_less(word, last.word))) return false;
        std::pop_heap(heap.begin(), heap.end(), better);
        heap.pop_back();
    }
    heap.push_back({word, {}, score, {}});
    store(heap.back());
    std::push_heap(heap.begin(), heap.end(), better);
    return true;
}
template<class Values>
inline bool shortlist(std::vector<Candidate>& heap, size_t capacity, const Word& word, double score, Values values) {
    return retain_candidate(heap, capacity, word, score, [&](Candidate& candidate) { candidate.values = values(); });
}
struct SearchResult { std::vector<Candidate> candidates; double omitted = 0; };

using ActiveWords = std::unordered_map<Word, size_t, WordHash>;
// The restricted solvers suspend only at an inactive-coordinate search. The
// referenced residual and active set stay alive in the suspended target.
struct Separation {
    const Array& residual;
    double penalty, threshold;
    const ActiveWords& active;
    size_t capacity;
    SearchResult found;
    double seconds = 0.;
    // A nonempty shortlist can advance this fit. A final or iteration-limited
    // request instead needs a bound over the entire requested dictionary.
    bool allow_lower_degree = false;
};

class FeatureSearch {
public:
    virtual ~FeatureSearch() = default;
    virtual double feature_cap() const { return std::numeric_limits<double>::infinity(); }
    virtual size_t response_batch() const { return 1; }
    virtual void query_batch(const std::vector<Separation*>& queries) {
        for (auto* q : queries) q->found = query(q->residual, q->threshold, q->active, q->capacity);
    }
    virtual NodePtr selected(const Word&, bool extendible = true) = 0;
    virtual SearchResult query(const Array&, double threshold, const ActiveWords&, size_t capacity) = 0;
};

struct SearchOptions : SigLassoOptions {
    bool selected_only;
    bool accelerate_solver;

    explicit SearchOptions(const SigLassoOptions& o, bool predict = false)
        : SigLassoOptions(o), selected_only(predict),
          accelerate_solver(!predict && !(o.degree == 1 && o.memory_limit)) {}
};

class Backend {
public:
    const SearchOptions options;
    const Clock::time_point started = Clock::now();
    std::shared_ptr<Budget> budget;
    SigLassoStats stats{};
    bool highest_degree_only = false;
    std::function<bool(const Word&, const double*)> visit_column;
    std::vector<Array> batch_absolute, batch_ratios, batch_signed;
    Array increments, variation, after, alphabet_ratios;
    std::vector<std::pair<size_t, size_t>> alphabet;
    std::shared_ptr<void> metadata;
    std::vector<double> weights;
    explicit Backend(const SearchOptions& o, std::shared_ptr<Budget> shared_budget = {}) : options(o), budget(shared_budget ? std::move(shared_budget) : std::make_shared<Budget>(o.memory_limit)) {
        if (o.degree > 170) throw std::out_of_range("sig_lasso degree exceeds the finite factorial range");
        weights.resize(o.degree + 1, 1.0);
        for (size_t k = 1; k <= o.degree; ++k) weights[k] = weights[k - 1] * static_cast<double>(k);
        metadata = budget->reserve(multiply(o.samples, 64) + multiply(o.degree + o.dimension + o.targets, 256) + 4096, budget);
    }
    virtual ~Backend() = default;
    virtual Array alloc(size_t size) = 0;
    virtual void fill(const Array&, double) = 0;
    virtual void copy(const double*, double*, size_t) = 0;
    virtual double dot(const double*, const double*, size_t) = 0;
    virtual double gram_dot(const double*, const double*, size_t) = 0;
    virtual double absolute_sum(const Array&) = 0;
    virtual void axpy(double, const double*, double*, size_t) = 0;
    virtual void scale(double, double*, size_t) = 0;
    virtual double center(double*, size_t, bool) = 0;
    virtual void correlations(const std::vector<Column>&, const Array&, std::vector<double>&) = 0;
    virtual Channel channel(size_t a) {
        return {{increments.size ? increments.data + a * options.samples : increments.data, 0, increments.owner}, options.dimension};
    }
    virtual std::unique_ptr<Backend> terminal_worker() = 0;
    virtual double bound_dot(const double*, const double*, size_t) = 0;
    virtual double bound_dot(const float*, const double*, size_t) = 0;
    virtual double weighted_square(const double*, const double*, size_t) = 0;
    virtual void prepare() = 0;
    virtual std::unique_ptr<FeatureSearch> make_search();
    virtual void moments(Node&) = 0;
    virtual void begin_query(const Array&) = 0;
    virtual double bound(const Probe&, const Array&, double threshold) = 0;
    virtual void terminal_query(const Probe&, const Array&, double, const ActiveWords&, size_t, SearchResult&) = 0;
    virtual NodePtr terminal_child(const Probe&) = 0;
    virtual NodePtr child(const NodePtr&, size_t letter) = 0;
    virtual NodePtr prediction_child(const NodePtr&, size_t letter, bool extendible) = 0;
    virtual Array response(size_t target) = 0;
    virtual void export_result(const Result&, const SigLassoOutput&) = 0;
    virtual bool parallel_search(const NodePtr&, const Array&, double, const ActiveWords&, size_t, SearchResult&) = 0;
    std::shared_ptr<void> account(size_t bytes) {
        return budget->reserve(bytes, budget);
    }
    Array clone(const Array& source) { auto out = alloc(source.size); copy(source.data, out.data, source.size); return out; }
    void prepare_alphabet() {
        std::vector<std::pair<size_t, size_t>> pending{{0, options.dimension}};
        while (!pending.empty()) {
            auto interval = pending.back(); pending.pop_back(); alphabet.push_back(interval);
            if (interval.second - interval.first > 1) {
                const auto mid = (interval.first + interval.second) / 2;
                pending.emplace_back(mid, interval.second); pending.emplace_back(interval.first, mid);
            }
        }
    }
};

struct Result {
    std::vector<Word> words;
    std::vector<double> coefficients, intercepts, objectives, gaps, kkt;
    std::vector<uint64_t> converged, iterations;
    std::vector<Column> selected_columns;
    std::vector<std::shared_ptr<void>> storage;
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
        (shape.words && (!output.coefficients || !output.signatures)) || !output.intercepts || !output.objectives || !output.gaps || !output.kkt || !output.stats)
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
    const size_t scalars = multiply(result.words.size(), backend.options.samples + result.targets) + 4 * result.targets;
    backend.stats.output_bytes = multiply(scalars, backend.options.single_precision ? sizeof(float) : sizeof(double))
        + multiply(letters + result.words.size() + 1 + 2 * result.targets, sizeof(uint64_t));
    auto output_storage = backend.account(backend.stats.output_bytes);
    result.stats = backend.stats;
    const auto output = allocate_output(result, allocate, context);
    backend.export_result(result, output);
    backend.stats.output_seconds = elapsed(started);
    backend.stats.peak_bytes = backend.budget->peak();
    backend.stats.total_seconds = elapsed(backend.started);
    *output.stats = backend.stats;
}

class Search final : public FeatureSearch {
    Backend& b_;
    NodePtr root_;
    std::vector<NodePtr> prediction_path_;
public:
    explicit Search(Backend& b, NodePtr root = {}) : b_(b), root_(std::move(root)) {
        if (root_) return;
        root_ = std::make_shared<Node>();
        root_->coefficients = b_.alloc(multiply(b.options.samples, b.options.length - 1));
        b_.fill(root_->coefficients, 1);
        if (!b.options.selected_only) b_.moments(*root_);
    }

    NodePtr selected(const Word& word, bool extendible = true) override {
        if (b_.options.selected_only || !extendible)
            return selected_prefix(prediction_path_, root_, word, extendible,
                [&](const NodePtr& parent, size_t a, bool keep) { return b_.prediction_child(parent, a, keep); });
        auto node = root_;
        for (auto letter : word) {
            node = b_.child(node, letter);
            if (node->word.size() < b_.options.degree) b_.moments(*node);
            ++b_.stats.cache_misses;
            ++b_.stats.constructed;
            ++b_.stats.evaluated;
        }
        return node;
    }

    SearchResult query(const Array& residual, double alpha, const ActiveWords& active, size_t capacity) override {
        prediction_path_.clear();
        SearchResult result;
        if (!b_.parallel_search(root_, residual, alpha, active, capacity, result))
            result = query_block(residual, alpha, active, capacity, 0);
        ++b_.stats.search_passes;
        return result;
    }

    SearchResult query_block(const Array& residual, double alpha, const ActiveWords& active, size_t capacity, size_t block) {
        const auto [low, high] = b_.alphabet[block];
        SearchResult result;
        b_.begin_query(residual);
        std::vector<Probe> stack{{root_, low, high, block}};
        while (!stack.empty()) {
            Probe task = std::move(stack.back());
            stack.pop_back();
            const size_t levels = b_.options.degree - task.node->word.size();
            if (levels <= 2) {
                b_.terminal_query(task, residual, alpha, active, capacity, result);
                continue;
            }

            double threshold = alpha;
            if (capacity && result.candidates.size() == capacity)
                threshold = std::max(alpha, result.candidates.front().score);
            const double estimate = b_.bound(task, residual, threshold) * (1. + 1e-10);
            const double bound = estimate >= 0 && std::isfinite(estimate) ? guard(estimate) : std::numeric_limits<double>::infinity();
            ++b_.stats.block_queries;
            if (bound < threshold) {
                result.omitted = std::max(result.omitted, bound);
                ++b_.stats.pruned;
                continue;
            }
            if (task.high - task.low > 1) {
                const size_t mid = task.low + (task.high - task.low) / 2;
                stack.push_back({task.node, mid, task.high, task.block + 2 * (mid - task.low)});
                stack.push_back({task.node, task.low, mid, task.block + 1});
                continue;
            }
            NodePtr node;
            if (levels == 3) node = b_.terminal_child(task);
            else {
                node = b_.child(task.node, task.low);
                b_.moments(*node);
            }
            ++b_.stats.cache_misses;
            ++b_.stats.constructed;
            ++b_.stats.evaluated;
            if (!active.count(node->word)) {
                if (b_.visit_column) b_.visit_column(node->word, node->values.data);
                else {
                    const double score = finite(std::abs(b_.dot(node->values.data, residual.data, residual.size) / residual.size) * b_.weights[node->word.size()]);
                    result.omitted = std::max(result.omitted, guard(score));
                    if (score > alpha)
                        shortlist(result.candidates, capacity, node->word, score, [&] { return node->values; });
                }
            }
            stack.push_back({node, 0, b_.options.dimension, 0});
        }
        std::sort(result.candidates.begin(), result.candidates.end(), better);
        return result;
    }
};

inline std::unique_ptr<FeatureSearch> Backend::make_search() { return std::make_unique<Search>(*this); }

Result fit(Backend&, bool log_features, const std::vector<double>& alphas, const std::vector<Word>& initial);

} // namespace siglasso
