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

inline std::atomic<unsigned> experimental_signature_bound{1};
inline std::atomic<unsigned> experimental_candidate_pool{256};
inline std::atomic<unsigned> experimental_search_strategy{0};
inline std::atomic<unsigned> experimental_search_batch{4}, experimental_pool_rank{16}, experimental_traversal_order{4};
inline std::atomic<unsigned> experimental_search_partitions{0}, experimental_search_method{3};
inline std::atomic<size_t> experimental_cache_bytes{0};
inline std::atomic<double> experimental_pool_refresh{0};
inline std::atomic<unsigned> experimental_terminal_kernel{14}, experimental_restricted_solver{7};
inline std::atomic<unsigned> experimental_feature_kernel{2039};
inline std::atomic<unsigned> experimental_summary_flags{0};
inline std::atomic<size_t> experimental_summary_bytes{0};
enum OptimizationCounter { AndersonAttempts, AndersonAccepted, GramBuilds, GramReusedEntries, GramSeconds,
    MatrixBlocks, MatrixCorrelations, MatrixSeconds, SummaryHits, SummaryMisses, SummaryPruned, SegmentRefinements, OptimizationCounterCount };
inline thread_local std::array<double, OptimizationCounterCount> experimental_optimization_stats{};
enum ExperimentCounter { RetainedChecks, RetainedHits, NormCalls, PermanentHits, ScreenedEntries, ScreenPasses,
    InnerSweeps, AdaptiveRefinements, FinalAudits, AuditRepairs, ScreenSeconds, ExperimentCounterCount };
inline thread_local std::array<double, ExperimentCounterCount> experimental_last_stats{};

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
    Array coefficients, moments, integrals, values, linear_integrals, tt_state, tt_moments;
    bool uncertain_moments = false;
    std::shared_ptr<void> metadata;
    size_t bytes() const { return coefficients.bytes() + moments.bytes() + integrals.bytes() + values.bytes() + linear_integrals.bytes() + tt_state.bytes() + tt_moments.bytes() + 512 + word.size() * 64; }
};
using NodePtr = std::shared_ptr<Node>;
struct Probe { NodePtr node; uint64_t low, high; size_t block = 0; Array gram; double norm = -1; };
struct Candidate { Word word; Array values; double score; };
inline bool better(const Candidate& a, const Candidate& b) {
    return a.score == b.score ? word_less(a.word, b.word) : a.score > b.score;
}
struct SearchResult { std::vector<Candidate> candidates; double omitted = 0; };

class CandidateCutoff {
    size_t capacity_;
    std::vector<double> scores_;
    std::mutex mutex_;
    std::shared_ptr<void> metadata_;
public:
    std::atomic<double> threshold{0};
    CandidateCutoff(size_t capacity, std::shared_ptr<void> metadata) : capacity_(capacity), metadata_(std::move(metadata)) { scores_.reserve(capacity); }
    void offer(double score) {
        if (score <= threshold.load(std::memory_order_relaxed)) return;
        std::lock_guard lock(mutex_);
        if (scores_.size() < capacity_) { scores_.push_back(score); std::push_heap(scores_.begin(), scores_.end(), std::greater<double>{}); }
        else if (score > scores_.front()) {
            std::pop_heap(scores_.begin(), scores_.end(), std::greater<double>{});
            scores_.back() = score; std::push_heap(scores_.begin(), scores_.end(), std::greater<double>{});
        }
        if (scores_.size() == capacity_) threshold.store(scores_.front(), std::memory_order_relaxed);
    }
};

class FeatureSearch {
public:
    virtual ~FeatureSearch() = default;
    virtual void new_target() {}
    virtual double stamp() const { return 0; }
    virtual void screen(double, double, double, double) {}
    virtual NodePtr selected(const Word&) = 0;
    virtual SearchResult query(const Array&, double, const std::unordered_map<Word, size_t, WordHash>&, size_t,
        size_t = 0, size_t = 0, size_t = 0) = 0;
    virtual SearchResult verify(const Array& r, double alpha, const std::unordered_map<Word, size_t, WordHash>& active, size_t capacity) {
        return query(r, alpha, active, capacity);
    }
};

class HostSearch {
public:
    virtual ~HostSearch() = default;
    virtual void begin_query(const Array&, const std::unordered_map<Word, size_t, WordHash>&) {}
    virtual double complete_bound(const Probe&, double value) { return value; }
    virtual double subtree_norm(const Probe&) { return std::numeric_limits<double>::infinity(); }
    virtual void bounds(const std::vector<Probe>&, const Array&, double, std::vector<double>&) = 0;
};

struct SearchOptions : SigLassoOptions {
    size_t cache_bytes;
    unsigned search_method;
    unsigned experiment;
    unsigned batch_width, pool_rank, traversal_order, partitions;
    unsigned terminal_kernel, restricted_solver, feature_kernel;
    unsigned summary_flags;
    size_t summary_bytes;
    bool selected_only;
    SearchOptions(const SigLassoOptions& o, uint32_t mode, bool predict = false) : SigLassoOptions(o), selected_only(predict) {
        const size_t limit = predict ? 64u << 20 : experimental_cache_bytes.load();
        cache_bytes = o.memory_limit ? std::min<size_t>(limit, o.memory_limit / 4) : limit;
        search_method = predict || mode == 2 ? 0 : (!mode && o.degree >= 4 ? 3 : 2);
        if (!predict && !mode && experimental_search_method.load() < 4) search_method = experimental_search_method.load();
        experiment = predict ? 0 : experimental_search_strategy.load();
        batch_width = predict ? 32 : experimental_search_batch.load();
        pool_rank = predict ? 0 : experimental_pool_rank.load();
        traversal_order = predict ? 0 : experimental_traversal_order.load();
        partitions = predict ? 0 : experimental_search_partitions.load();
        terminal_kernel = predict || mode || o.degree < 3 ? 0 : experimental_terminal_kernel.load();
        restricted_solver = predict || o.degree == 1 ? 0 : experimental_restricted_solver.load();
        feature_kernel = experimental_feature_kernel.load();
        summary_flags = predict || mode ? 0 : experimental_summary_flags.load();
        summary_bytes = predict || mode ? 0 : experimental_summary_bytes.load();
        if (o.memory_limit) summary_bytes = std::min<size_t>(summary_bytes, o.memory_limit / 8);
    }
};

class Backend {
public:
    const SearchOptions options;
    const Clock::time_point started = Clock::now();
    std::shared_ptr<Budget> budget;
    SigLassoStats stats{};
    bool uncertain_features = false;
    bool verify_search = false;
    double active_corr_cap = 0;
    std::shared_ptr<CandidateCutoff> shared_cutoff;
    std::array<double, ExperimentCounterCount> experiment_stats{};
    std::array<double, OptimizationCounterCount> optimization_stats{};
    Array increments, variation, after, alphabet_ratios;
    std::vector<std::pair<size_t, size_t>> alphabet;
    std::function<void()> trim_search;
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
    virtual double absolute_sum(const Array&) = 0;
    virtual void axpy(double, const double*, double*, size_t) = 0;
    virtual void scale(double, double*, size_t) = 0;
    virtual double center(double*, size_t, bool) = 0;
    virtual void correlations(const std::vector<Column>&, const Array&, std::vector<double>&, const std::vector<size_t>* = nullptr) = 0;
    virtual void prepare() = 0;
    virtual std::unique_ptr<FeatureSearch> make_search();
    virtual Array factors(size_t k) = 0;
    virtual void moments(Node&, const Array&) = 0;
    virtual HostSearch* host_search() { return nullptr; }
    virtual Array energy_gram(size_t, size_t) = 0;
    virtual void children(const std::vector<Probe>&, std::vector<NodePtr>&) = 0;
    virtual Array response(size_t target) = 0;
    virtual void export_result(const Result&, const SigLassoOutput&) = 0;
    virtual bool native_search(const Array&, double, const std::unordered_map<Word, size_t, WordHash>&, size_t, SearchResult&) { return false; }
    virtual bool has_native_search() const { return false; }
    virtual void native_new_target() {}
    virtual void native_screen(double, double, double, double) {}
    virtual void finish() {}
    virtual void trim_caches() { if (trim_search) trim_search(); }
    std::shared_ptr<void> account(size_t bytes) {
        if (bytes > budget->available()) trim_caches();
        return budget->reserve(bytes, budget);
    }
    Array clone(const Array& source) { auto out = alloc(source.size); copy(source.data, out.data, source.size); return out; }
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

class Search : public FeatureSearch {
    Backend& b_;
    PrefixCache cache_;
    std::unordered_map<size_t, Array> factors_;
    std::unordered_map<size_t, Array> grams_;
    size_t factor_bytes_ = 0;
    size_t gram_bytes_ = 0;
    NodePtr root_;
    struct Certificate {
        double bound = 0, norm = 0, stamp = 0;
        bool permanent = false;
        std::shared_ptr<void> metadata;
    };
    std::unordered_map<Word, Certificate, WordHash> certificates_;
    Array previous_residual_;
    double travel_ = 0;
    void advance(const Array& residual) {
        if (!previous_residual_.owner) previous_residual_ = b_.clone(residual);
        else {
            double squared = 0;
            for (size_t i = 0; i < residual.size; ++i) {
                const double difference = residual.data[i] - previous_residual_.data[i];
                squared += difference * difference;
            }
            travel_ = guard(travel_ + guard(std::sqrt(squared)) / residual.size);
            b_.copy(residual.data, previous_residual_.data, residual.size);
        }
    }
    void retain(Word key, double bound, double norm) {
        auto found = certificates_.find(key);
        if (found == certificates_.end()) {
            auto memory = b_.account(sizeof(Certificate) + key.size() * 16 + 192);
            found = certificates_.emplace(std::move(key), Certificate{0, guard(norm), 0, false, std::move(memory)}).first;
        }
        found->second.bound = guard(bound); found->second.stamp = travel_;
    }
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
    explicit Search(Backend& b, NodePtr root = {}) : b_(b),
        cache_(b.options.cache_bytes / 8 * 3), root_(std::move(root)) {
        b_.trim_search = [&] { cache_.clear(); factors_.clear(); grams_.clear(); factor_bytes_ = gram_bytes_ = 0; };
        if (root_) return;
        root_ = std::make_shared<Node>();
        root_->coefficients = b_.alloc(multiply(b.options.samples, b.options.length - 1));
        b_.fill(root_->coefficients, 1);
        if (!b.options.selected_only) b_.moments(*root_, get_factors(0));
    }
    ~Search() { b_.trim_search = {}; }
    const NodePtr& root() const { return root_; }
    void new_target() override {
        certificates_ = decltype(certificates_){};
        previous_residual_ = {}; travel_ = 0;
        b_.native_new_target();
    }
    double stamp() const override { return travel_; }
    void screen(double scale, double radius, double stamp, double alpha) override {
        if (b_.has_native_search()) { b_.native_screen(scale, radius, stamp, alpha); return; }
        const auto start = Clock::now();
        for (auto& [key, entry] : certificates_) if (!entry.permanent) {
            const double correlation = guard(entry.bound + entry.norm * std::abs(entry.stamp - stamp));
            if (guard(scale * correlation + entry.norm * radius) < alpha) {
                entry.permanent = true; ++b_.experiment_stats[ScreenedEntries];
            }
        }
        ++b_.experiment_stats[ScreenPasses]; b_.experiment_stats[ScreenSeconds] += elapsed(start);
    }
    SearchResult verify(const Array& residual, double alpha, const std::unordered_map<Word, size_t, WordHash>& active, size_t capacity) override {
        b_.verify_search = true;
        try { auto result = query(residual, alpha, active, capacity); b_.verify_search = false; return result; }
        catch (...) { b_.verify_search = false; throw; }
    }
    NodePtr selected(const Word& word) {
        auto node = root_;
        for (auto letter : word) {
            Word child_word = node->word; child_word.push_back(letter);
            auto child = cache_.get(child_word);
            if (child) ++b_.stats.cache_hits;
            else {
                ++b_.stats.cache_misses;
                std::vector<NodePtr> children;
                b_.children({Probe{node, letter, letter + 1, 0, {}}}, children);
                child = children.front();
                ++b_.stats.constructed;
                if (!b_.options.selected_only && child->word.size() < b_.options.degree) b_.moments(*child, get_factors(child->word.size()));
                cache_.insert(child);
                ++b_.stats.evaluated;
            }
            node = child;
        }
        return node;
    }
    SearchResult query(const Array& residual, double alpha, const std::unordered_map<Word, size_t, WordHash>& active, size_t capacity,
        size_t low = 0, size_t high = 0, size_t block = 0) {
        SearchResult result;
        if (b_.options.experiment & 5) advance(residual);
        if (b_.has_native_search()) {
            cache_.clear(); factors_.clear(); grams_.clear(); factor_bytes_ = gram_bytes_ = 0;
            b_.native_search(residual, alpha, active, capacity, result); ++b_.stats.search_passes; return result;
        }
        const bool retain_bounds = (b_.options.experiment & 5) && !b_.verify_search;
        const size_t rank = b_.options.pool_rank ? std::min<size_t>(capacity, b_.options.pool_rank) : capacity;
        auto leader_storage = rank < capacity ? b_.account(multiply(rank, sizeof(double))) : std::shared_ptr<void>{};
        std::vector<double> leaders;
        if (rank < capacity) leaders.reserve(rank);
        b_.host_search()->begin_query(residual, active);
        std::vector<Probe> stack;
        const auto push = [&](const NodePtr& node, size_t first, size_t last, size_t index) {
            if (b_.options.search_method & 1) stack.push_back({node, first, last, index, {}});
            else for (size_t a = last; a-- > first;) stack.push_back({node, a, a + 1, a, {}});
        };
        push(root_, low, high ? high : b_.options.dimension, block);
        while (!stack.empty()) {
            std::vector<Probe> batch;
            const auto k = stack.back().node->word.size();
            while (!stack.empty() && batch.size() < b_.options.batch_width && stack.back().node->word.size() == k) {
                batch.push_back(std::move(stack.back())); stack.pop_back();
            }
            double threshold = alpha;
            if (rank < capacity) {
                if (leaders.size() == rank) threshold = std::max(threshold, leaders.front());
            } else if (capacity && result.candidates.size() == capacity) threshold = std::max(threshold, result.candidates.front().score);
            if (b_.shared_cutoff) threshold = std::max(threshold, b_.shared_cutoff->threshold.load(std::memory_order_relaxed));
            if (retain_bounds) {
                size_t kept = 0;
                for (size_t j = 0; j < batch.size(); ++j) {
                    auto& task = batch[j];
                    Word key = task.node->word; key.push_back(b_.options.dimension + task.block);
                    const auto found = certificates_.find(key);
                    if (found != certificates_.end()) {
                        ++b_.experiment_stats[RetainedChecks];
                        const auto& entry = found->second;
                        task.norm = entry.norm;
                        const double bound = guard(entry.bound + entry.norm * std::max(0., travel_ - entry.stamp));
                        if (entry.permanent || ((b_.options.experiment & 1) && bound < threshold)) {
                            ++b_.experiment_stats[entry.permanent ? PermanentHits : RetainedHits];
                            result.omitted = std::max(result.omitted, bound);
                            ++b_.stats.pruned; continue;
                        }
                    }
                    if (kept != j) batch[kept] = std::move(task);
                    ++kept;
                }
                batch.resize(kept);
            }
            if (batch.empty()) continue;
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
                const double estimate = bounds[j] * (1.0 + 1e-10);
                const double bound = estimate >= 0 && std::isfinite(estimate) ? guard(estimate) : std::numeric_limits<double>::infinity();
                if (bound < threshold) {
                    result.omitted = std::max(result.omitted, bound);
                    ++b_.stats.pruned;
                    if (retain_bounds) {
                        if (task.norm < 0) { ++b_.experiment_stats[NormCalls]; task.norm = b_.host_search()->subtree_norm(task); }
                        Word key = task.node->word; key.push_back(b_.options.dimension + task.block);
                        retain(std::move(key), b_.host_search()->complete_bound(task, bound), task.norm);
                    }
                } else if (task.high - task.low > 1) {
                    const auto middle = task.low + (task.high - task.low) / 2;
                    stack.push_back({task.node, middle, task.high, task.block + 2 * (middle - task.low), {}});
                    stack.push_back({task.node, task.low, middle, task.block + 1, {}});
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
                    ++b_.stats.constructed;
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
                bool screened = false;
                if (retain_bounds && (b_.options.experiment & 4)) {
                    const auto found = certificates_.find(node->word);
                    const double norm = found == certificates_.end()
                        ? b_.weights[node->word.size()] * std::sqrt(b_.dot(node->values.data, node->values.data, residual.size))
                            + 1e-11 * std::sqrt(static_cast<double>(residual.size))
                        : found->second.norm;
                    screened = found != certificates_.end() && found->second.permanent;
                    retain(node->word, score, norm);
                    if (screened) ++b_.experiment_stats[PermanentHits];
                }
                if (!active.count(node->word)) {
                    result.omitted = std::max(result.omitted, guard(score));
                    if (!screened && b_.shared_cutoff && score > alpha) b_.shared_cutoff->offer(score);
                    if (!screened && rank < capacity && score > alpha) {
                        if (leaders.size() < rank) { leaders.push_back(score); std::push_heap(leaders.begin(), leaders.end(), std::greater<double>{}); }
                        else if (score > leaders.front()) {
                            std::pop_heap(leaders.begin(), leaders.end(), std::greater<double>{});
                            leaders.back() = score; std::push_heap(leaders.begin(), leaders.end(), std::greater<double>{});
                        }
                    }
                    if (!screened && capacity && score > alpha) {
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
            }
            if ((b_.options.traversal_order & 1) && nodes.size() > 1) {
                std::vector<size_t> order(nodes.size()); std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return std::abs(scores[a]) < std::abs(scores[c]); });
                for (auto j : order) if (nodes[j]->word.size() < b_.options.degree) push(nodes[j], 0, b_.options.dimension, 0);
            } else for (const auto& node : nodes) if (node->word.size() < b_.options.degree) push(node, 0, b_.options.dimension, 0);
        }
        std::sort(result.candidates.begin(), result.candidates.end(), better);
        ++b_.stats.search_passes;
        return result;
    }
};

inline std::unique_ptr<FeatureSearch> Backend::make_search() { return std::make_unique<Search>(*this); }

} // namespace siglasso

namespace siglasso {
Result fit(Backend&, uint32_t mode, const std::vector<double>& alphas, const std::vector<Word>& initial);
}
