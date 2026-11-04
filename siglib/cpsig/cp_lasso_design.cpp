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

#include "cpsig.h"
#include "cp_lasso_hall.h"
#include "multithreading.h"
#include <map>

namespace siglasso {
namespace {
struct Projection {
    Word word;
    size_t index, end;
};

size_t tensor_index(const Word& word, size_t dimension) {
    size_t code = 0;
    for (auto a : word)
        code = code * dimension + a + 1;
    return code - 1;
}

using Expansion = std::map<Word, double>;
void shuffle_terms(const Word& a, size_t i, const Word& b, size_t j, Word& prefix, double weight, Expansion& out) {
    if (i == a.size() && j == b.size()) {
        out[prefix] += weight;
        return;
    }
    if (i < a.size()) {
        prefix.push_back(a[i]);
        shuffle_terms(a, i + 1, b, j, prefix, weight, out);
        prefix.pop_back();
    }
    if (j < b.size()) {
        prefix.push_back(b[j]);
        shuffle_terms(a, i, b, j + 1, prefix, weight, out);
        prefix.pop_back();
    }
}

class TensorDesign final : public FeatureSearch {
    Backend& b_;
    Array paths_, packed_, cached_, anchor_, scores_, norms_;
    size_t degree_, mode_, width_, n_, d_, tile_, cache_count_;
    std::vector<Projection> features_;
    std::vector<std::pair<size_t, double>> terms_;
    std::vector<size_t> cache_ids_;
    std::shared_ptr<void> metadata_;
    bool anchored_ = false;
    double max_norm_ = 0;

    void hall_projection() {
        std::vector<std::vector<HallPtr>> levels(degree_ + 1);
        std::vector<Expansion> expansions;
        size_t first = 0;
        for (size_t k = 1; k <= degree_; ++k) {
            for (size_t a = 0; k == 1 && a < d_; ++a) {
                auto h = std::make_shared<Hall>();
                h->word = {a};
                levels[1].push_back(h);
            }
            for (size_t l = 1; l < k; ++l)
                for (const auto& u : levels[l])
                    for (const auto& v : levels[k - l])
                        if (admissible(u, v))
                            levels[k].push_back(bracket(u, v));
            std::sort(levels[k].begin(), levels[k].end(), [](const HallPtr& a, const HallPtr& b) { return a->word < b->word; });
            for (const auto& h : levels[k]) {
                Expansion row;
                if (!h->left)
                    row[h->word] = 1;
                else
                    for (const auto& [a, ca] : expansions[h->left->slot])
                        for (const auto& [b, cb] : expansions[h->right->slot]) {
                            Word prefix{a.front()};
                            shuffle_terms(a, 1, b, 0, prefix, ca * cb / h->run, row);
                        }
                for (auto& [word, value] : row)
                    terms_.emplace_back(tensor_index(word, d_), value);
                features_.push_back({h->word, first, terms_.size()});
                first = terms_.size();
                h->slot = expansions.size();
                // Only lower degrees are needed to form the next brackets.
                if (k < degree_)
                    expansions.push_back(std::move(row));
            }
        }
    }
    template <class F> void tiles(F process) {
        const size_t length = b_.options.length;
        auto tensor = b_.alloc(tile_ * width_);
        auto logarithm = mode_ && degree_ > 1 ? b_.alloc(tensor.size) : Array{};
        const size_t workers = std::min(
            tile_, static_cast<size_t>(b_.options.n_jobs > 0 ? b_.options.n_jobs : get_max_threads() + 1 + b_.options.n_jobs));
        auto workspace = b_.account(
            degree_ == 1 || d_ == 1 || length == 1 ? 0 : workers * (12 * (width_ + 1) + 8 * d_ * degree_ + 256) * sizeof(double));
        for (size_t start = 0; start < n_; start += tile_) {
            const size_t rows = std::min(tile_, n_ - start);
            int status = 0;
            if (degree_ == 1 || length == 1 || d_ == 1) {
                for (size_t i = 0; i < rows; ++i) {
                    const double* end = paths_.data + ((start + i + 1) * length - 1) * d_;
                    if (length == 1)
                        std::fill_n(tensor.data + i * width_, width_, 0.);
                    else if (degree_ == 1)
                        std::copy_n(end, d_, tensor.data + i * d_);
                    else {
                        double value = 1;
                        for (size_t k = 1; k <= degree_; ++k) {
                            value *= end[0] / k;
                            tensor.data[i * width_ + k - 1] = mode_ && k > 1 ? 0 : value;
                        }
                    }
                }
            } else {
                ++b_.stats.full_transform_calls;
                status = signature_d(paths_.data + start * length * d_, tensor.data, rows, d_, length, degree_, false, false, 1.,
                                     true, false, b_.options.n_jobs);
                if (!status && mode_)
                    status = sig_to_log_sig_d(tensor.data, logarithm.data, rows, d_, degree_, false, false, 0, false,
                                              b_.options.n_jobs);
            }
            if (status == 1)
                throw std::bad_alloc();
            if (status)
                throw std::runtime_error("Working-set tensor transform failed");
            const double* block = logarithm.owner && length > 1 && d_ > 1 ? logarithm.data : tensor.data;
            process(start, rows, block);
        }
    }
    double coordinate(const double* row, size_t j) const {
        const auto& f = features_[j];
        if (mode_ != 2)
            return row[f.index];
        double x = 0;
        for (size_t k = f.index; k < f.end; ++k)
            x += row[terms_[k].first] * terms_[k].second;
        return x;
    }
    Array gather(const std::vector<size_t>& ids) {
        auto matrix = b_.alloc(multiply(ids.size(), n_));
        tiles([&](size_t start, size_t rows, const double* block) {
            spawn_batch_threads(ids.size(), b_.options.n_jobs, [&](size_t first, size_t last) {
                for (size_t j = first; j < last; ++j)
                    for (size_t i = 0; i < rows; ++i)
                        matrix.data[j * n_ + start + i] = finite(coordinate(block + i * width_, ids[j]));
            });
        });
        return matrix;
    }
    size_t locate(const Word& w) const {
        auto it = std::lower_bound(features_.begin(), features_.end(), w,
                                   [](const Projection& f, const Word& word) { return word_less(f.word, word); });
        if (it == features_.end() || it->word != w)
            throw std::invalid_argument("Word outside the selected dictionary");
        return it - features_.begin();
    }

  public:
    TensorDesign(Backend& b, const Array& paths, size_t degree, uint32_t mode, size_t width, size_t metadata, bool& packed,
                 size_t packed_bytes)
        : b_(b), paths_(paths), degree_(degree), mode_(mode), width_(width), n_(b.options.samples), d_(b.options.dimension) {
        metadata_ = b.account(metadata);
        if (mode == 2)
            hall_projection();
        else
            for (size_t k = 1, code = 0; k <= degree; ++k) {
                Word w(k, 0);
                for (;;) {
                    if (!mode || lyndon(w))
                        features_.push_back({w, code, 0});
                    ++code;
                    size_t j = k;
                    while (j && ++w[j - 1] == d_) {
                        w[j - 1] = 0;
                        --j;
                    }
                    if (!j)
                        break;
                }
            }
        size_t p = features_.size();
        packed = n_ <= std::min(packed_bytes, b.budget->available() / 2) / (sizeof(double) * p);
        const size_t row_cost = (36 * (width + 1) + 8 * d_ * degree) * sizeof(double);
        tile_ = std::min(n_, std::max<size_t>(1, std::min<size_t>((1 << 20) / (8 * (4 * width + 3 * d_ * b.options.length)),
                                                                  b.budget->available() / 4 / row_cost)));
        cache_count_ = std::min(p, std::max<size_t>(64, std::min<size_t>(4 * static_cast<size_t>(std::ceil(std::sqrt(double(p)))),
                                                                         (2 << 20) / (8 * n_))));
        cache_count_ = std::max<size_t>(1, std::min(cache_count_, b.budget->available() / 4 / (8 * n_)));
        norms_ = b.alloc(p);
        if (packed) {
            std::vector<size_t> ids(p);
            std::iota(ids.begin(), ids.end(), 0);
            packed_ = gather(ids);
            paths_ = {};
            for (size_t j = 0; j < p; ++j) {
                const double* x = packed_.at(j * n_);
                norms_.data[j] = std::sqrt(b_.dot(x, x, n_)) * b_.weights[features_[j].word.size()] *
                                 (1 + 32 * n_ * std::numeric_limits<double>::epsilon());
                max_norm_ = std::max(max_norm_, norms_.data[j]);
            }
        } else {
            scores_ = b.alloc(p);
            anchor_ = b.alloc(n_);
        }
        b.stats.engine = packed ? 1 : 2;
    }
    double mean_bound() const override {
        return max_norm_ / std::sqrt(double(n_));
    }
    NodePtr selected(const Word& w) override {
        size_t j = locate(w);
        auto node = std::make_shared<Node>();
        node->word = w;
        node->values = packed_.owner ? Array{packed_.at(j * n_), n_, packed_.owner} : gather({j});
        return node;
    }
    SearchResult query(const Array& r, double threshold, const std::unordered_map<Word, size_t, WordHash>& active, size_t cap,
                       size_t, size_t, size_t) override {
        ++b_.stats.search_passes;
        SearchResult result;
        const size_t p = features_.size();
        const double eps = 32 * n_ * std::numeric_limits<double>::epsilon();
        const double dot_error = eps * std::sqrt(b_.dot(r.data, r.data, n_)) / n_;
        auto consider = [&](size_t j, const Array& raw) {
            if (active.count(features_[j].word))
                return;
            double score = finite(std::abs(b_.dot(raw.data, r.data, n_) / n_) * b_.weights[features_[j].word.size()]);
            result.omitted = std::max(result.omitted, guard(score + dot_error * norms_.data[j]));
            if (score <= threshold || !cap)
                return;
            Candidate c{features_[j].word, raw, score};
            if (result.candidates.size() < cap) {
                result.candidates.push_back(std::move(c));
                std::push_heap(result.candidates.begin(), result.candidates.end(), better);
            } else if (better(c, result.candidates.front())) {
                std::pop_heap(result.candidates.begin(), result.candidates.end(), better);
                result.candidates.back() = std::move(c);
                std::push_heap(result.candidates.begin(), result.candidates.end(), better);
            }
        };
        auto check_cache = [&] {
            for (size_t i = 0; i < cache_ids_.size(); ++i)
                consider(cache_ids_[i], {cached_.at(i * n_), n_, cached_.owner});
        };
        if (packed_.owner) {
            for (size_t j = 0; j < p; ++j)
                consider(j, {packed_.at(j * n_), n_, packed_.owner});
        } else {
            check_cache();
            if (anchored_ && result.candidates.empty()) {
                const double norm2 = b_.dot(anchor_.data, anchor_.data, n_);
                const double t = norm2 ? b_.dot(r.data, anchor_.data, n_) / norm2 : 0;
                double delta = 0;
                for (size_t i = 0; i < n_; ++i) {
                    const double e = r.data[i] - t * anchor_.data[i];
                    delta += e * e;
                }
                delta =
                    (std::sqrt(delta) + eps * (1 + std::sqrt(b_.dot(r.data, r.data, n_)) + std::abs(t) * std::sqrt(norm2))) / n_;
                double bound = result.omitted;
                for (size_t j = 0; j < p; ++j) {
                    if (active.count(features_[j].word) || std::binary_search(cache_ids_.begin(), cache_ids_.end(), j))
                        continue;
                    bound = std::max(bound, guard(std::abs(t * scores_.data[j]) + norms_.data[j] * delta));
                }
                if (bound <= threshold) {
                    result.omitted = bound;
                    ++b_.stats.cache_hits;
                    return result;
                }
            }
            if (result.candidates.empty()) {
                ++b_.stats.cache_misses;
                b_.fill(scores_, 0);
                if (!anchored_)
                    b_.fill(norms_, 0);
                tiles([&](size_t start, size_t rows, const double* block) {
                    spawn_batch_threads(p, b_.options.n_jobs, [&](size_t first, size_t last) {
                        for (size_t j = first; j < last; ++j) {
                            double sum = 0, sq = 0;
                            for (size_t i = 0; i < rows; ++i) {
                                const double x = coordinate(block + i * width_, j);
                                sum += x * r.data[start + i];
                                if (!anchored_)
                                    sq += x * x;
                            }
                            scores_.data[j] += sum;
                            if (!anchored_)
                                norms_.data[j] += sq;
                        }
                    });
                });
                std::vector<size_t> order;
                order.reserve(p);
                result.omitted = 0;
                bool violating = false;
                for (size_t j = 0; j < p; ++j) {
                    double weight = b_.weights[features_[j].word.size()];
                    scores_.data[j] = finite(scores_.data[j] / n_ * weight);
                    if (!anchored_)
                        norms_.data[j] = finite(std::sqrt(norms_.data[j]) * weight * (1 + eps));
                    if (!anchored_)
                        max_norm_ = std::max(max_norm_, norms_.data[j]);
                    if (active.count(features_[j].word))
                        continue;
                    result.omitted = std::max(result.omitted, guard(std::abs(scores_.data[j]) + dot_error * norms_.data[j]));
                    violating |= std::abs(scores_.data[j]) > threshold;
                    order.push_back(j);
                }
                b_.copy(r.data, anchor_.data, n_);
                anchored_ = true;
                if (!violating || !cap)
                    return result;
                const auto distance = [&](size_t j) {
                    return norms_.data[j] ? (threshold - std::abs(scores_.data[j])) / norms_.data[j]
                                          : std::numeric_limits<double>::infinity();
                };
                size_t keep = std::min(cache_count_, order.size());
                std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](size_t a, size_t b) {
                    double x = distance(a), y = distance(b);
                    return x == y ? a < b : x < y;
                });
                order.resize(keep);
                std::sort(order.begin(), order.end());
                cached_ = {};
                cache_ids_ = std::move(order);
                cached_ = gather(cache_ids_);
                check_cache();
                if (result.candidates.empty() && cap)
                    throw NonfiniteError("Cached scores disagree with streamed correlations");
            } else
                result.omitted = std::numeric_limits<double>::infinity();
        }
        std::sort(result.candidates.begin(), result.candidates.end(), better);
        return result;
    }
};
} // namespace

std::unique_ptr<FeatureSearch> make_design(Backend& b, const Array& paths, size_t degree, uint32_t mode, bool& packed,
                                           size_t packed_bytes) {
    if (mode == 2 && degree > 5)
        return {};
    const size_t d = b.options.dimension;
    size_t width = 0, level = 1;
    for (size_t k = 1; k <= degree; ++k) {
        if (level > ((1 << 17) - width) / d)
            return {};
        level *= d;
        width += level;
    }
    size_t metadata = 0, power = 1;
    std::vector<size_t> primitive(degree + 1);
    for (size_t k = 1; k <= degree; ++k) {
        power *= d;
        primitive[k] = power;
        for (size_t i = 1; i < k; ++i)
            if (!(k % i))
                primitive[k] -= primitive[i];
        size_t count = mode ? primitive[k] / k : power;
        size_t bytes = 144 + 16 * k;
        if (mode == 2)
            bytes += 192 + 16 * k + static_cast<size_t>(b.weights[k - 1]) * (k < degree ? 144 + 16 * k : 32);
        metadata += multiply(count, bytes);
    }
    const size_t available = b.budget->available();
    if (metadata > available / 2 || (36 * (width + 1) + 8 * d * degree) * 8 > available / 3)
        return {};
    return std::make_unique<TensorDesign>(b, paths, degree, mode, width, metadata, packed, packed_bytes);
}
} // namespace siglasso
