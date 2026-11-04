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
#include "cp_dispatch.h"
#include "multithreading.h"
#include <Eigen/Eigenvalues>
#include <complex>
#include <map>

namespace cpsig {
HWY_EXPORT(hall_polynomial_);
HWY_EXPORT(hall_group_bound_);
void hall_polynomial(const siglasso::HallKernelPlan& plan, const double* left, const double* right, double* out, double* values,
    uint64_t n, uint64_t l, uint64_t begin, uint64_t end, double run, double* scratch) {
    HWY_DYNAMIC_DISPATCH(hall_polynomial_)(plan, left, right, out, values, n, l, begin, end, run, scratch);
}
double hall_group_bound(const double* global, const double* side, const double* residual, const siglasso::HallBoundFactor* factors,
    uint64_t count, const double* weights, uint64_t degree, uint64_t levels, uint64_t n) {
    return HWY_DYNAMIC_DISPATCH(hall_group_bound_)(global, side, residual, factors, count, weights, degree, levels, n);
}
}

namespace siglasso {
namespace {
using Complex = std::complex<double>;

bool lyndon(const Word& w) {
    for (size_t k = 1; k < w.size(); ++k)
        if (!std::lexicographical_compare(w.begin(), w.end(), w.begin() + k, w.end())) return false;
    return !w.empty();
}

struct Hall {
    Word word;
    std::shared_ptr<Hall> left, right;
    size_t run = 0;
    size_t slot = 0;
};
using HallPtr = std::shared_ptr<Hall>;
int compare(const HallPtr& a, const HallPtr& b) {
    if (a == b) return 0;
    if (a->word.size() != b->word.size()) return a->word.size() > b->word.size() ? -1 : 1;
    if (!a->left) return a->word < b->word ? -1 : a->word == b->word ? 0 : 1;
    const int first = compare(a->left, b->left);
    return first ? first : compare(a->right, b->right);
}
bool admissible(const HallPtr& u, const HallPtr& v) {
    return compare(u, v) < 0 && (!u->right || compare(v, u->right) <= 0);
}
HallPtr bracket(const HallPtr& u, const HallPtr& v) {
    auto h = std::make_shared<Hall>(); h->left = u; h->right = v;
    h->word = u->word; h->word.insert(h->word.end(), v->word.begin(), v->word.end());
    h->run = u->right && compare(u->right, v) == 0 ? u->run + 1 : 1;
    return h;
}

struct State {
    Word word;
    Array polynomial, values, moments, envelope, global;
    HallPtr hall;
    std::shared_ptr<void> metadata;
};
using StatePtr = std::shared_ptr<State>;

class PolynomialSearch final : public FeatureSearch {
    Backend& b_;
    const size_t n_, l_, d_, m_;
    const uint32_t mode_;
    std::vector<std::vector<double>> choose_, stirling_, time_basis_;
    std::vector<double> caps_, time_nodes_, time_weights_, gaussian_weights_;
    std::vector<Complex> gaussian_nodes_;
    std::vector<HallPtr> sides_;
    std::unordered_map<Word, HallPtr, WordHash> parsed_;
    std::unordered_set<Word, WordHash> invalid_words_;
    std::unordered_map<Word, StatePtr, WordHash> side_states_, cache_, summaries_;
    std::list<Word> cache_order_, summary_order_;
    size_t cache_size_ = 0, summary_bytes_ = 0;
    StatePtr root_;
    std::shared_ptr<void> tables_metadata_, time_basis_metadata_;
    size_t auxiliary_bytes_ = 0;
    Array side_maxima_;
    struct ProductTable { Array values; std::shared_ptr<void> metadata; };
    std::unordered_map<Word, ProductTable, WordHash> products_;
    size_t product_bytes_ = 0;
    std::map<std::pair<size_t, size_t>, std::shared_ptr<HallKernelPlan>> hall_plans_;
    Array hall_scratch_;
    std::vector<HallBoundFactor> group_factors_;

    double padded(double value, double allowance) const {
        value *= 1 + allowance;
        return std::isfinite(value) ? value * (1 + 64 * std::numeric_limits<double>::epsilon()) + 1e-15 : std::numeric_limits<double>::infinity();
    }
    bool zero_path_ = false;
    bool gaussian_reliable_ = true;

    template<typename Function> void samples(Function action) {
        if (b_.options.n_jobs == 1 || n_ < 32) action(0, n_);
        else spawn_batch_threads(n_, b_.options.n_jobs, action);
    }
    size_t triangle(size_t h) const { return (h + 1) * (h + 2) / 2; }
    size_t offset(size_t h, size_t p) const { return p * (h + 1) - p * (p - 1) / 2; }
    double& at(const Array& a, size_t c, size_t j, size_t i) const { return a.data[(c * l_ + j) * n_ + i]; }
    double step(size_t j, size_t a, size_t i) const { return b_.increments.data[(j * d_ + a) * n_ + i]; }
    StatePtr state(const Word& word, size_t count) {
        auto s = std::make_shared<State>(); s->word = word;
        s->metadata = b_.account(512 + word.size() * 32);
        s->polynomial = b_.alloc(multiply(multiply(count, l_), n_));
        s->values = b_.alloc(n_); b_.fill(s->values, 0);
        ++b_.stats.evaluated;
        return s;
    }
    void remember(const StatePtr& s) {
        const size_t bytes = s->values.bytes() + s->moments.bytes() + s->global.bytes() + 1024;
        const size_t capacity = b_.options.cache_bytes / 8 * 3;
        if (mode_ != 2 || bytes > capacity) return;
        if (auto it = summaries_.find(s->word); it != summaries_.end()) {
            const auto& old = it->second;
            if (s->moments.size <= old->moments.size && s->global.size <= old->global.size) return;
            summary_bytes_ -= old->values.bytes() + old->moments.bytes() + old->global.bytes() + 1024;
            summaries_.erase(it); summary_order_.remove(s->word);
        }
        while (!summary_order_.empty() && summary_bytes_ + bytes > capacity) {
            auto it = summaries_.find(summary_order_.front());
            summary_bytes_ -= it->second->values.bytes() + it->second->moments.bytes() + it->second->global.bytes() + 1024;
            summaries_.erase(it); summary_order_.pop_front();
        }
        const size_t metadata_bytes = 512 + s->word.size() * 32;
        if (metadata_bytes > b_.budget->available()) return;
        auto summary = std::make_shared<State>();
        summary->metadata = b_.account(metadata_bytes);
        summary->word = s->word; summary->hall = s->hall; summary->values = s->values;
        summary->moments = s->moments; summary->global = s->global;
        summaries_.emplace(s->word, summary); summary_order_.push_back(s->word); summary_bytes_ += bytes;
    }
    void cache(const StatePtr& s) {
        const size_t bytes = s->polynomial.bytes() + s->values.bytes() + s->moments.bytes() + s->global.bytes() + 1024;
        const size_t capacity = b_.options.cache_bytes / (mode_ == 2 ? 4 : 2);
        if (cache_.count(s->word)) return;
        if (bytes > capacity) { remember(s); return; }
        while (!cache_order_.empty() && cache_size_ + bytes > capacity) {
            auto it = cache_.find(cache_order_.front());
            cache_size_ -= it->second->polynomial.bytes() + it->second->values.bytes() + it->second->moments.bytes() + it->second->global.bytes() + 1024;
            auto evicted = it->second;
            cache_.erase(it); cache_order_.pop_front();
            remember(evicted);
        }
        cache_.emplace(s->word, s); cache_order_.push_back(s->word); cache_size_ += bytes;
    }
    double bernstein(const Array& a, size_t base, size_t degree, size_t j, size_t i, double t, bool absolute) const {
        double value = 0;
        for (size_t k = 0; k <= degree; ++k) {
            const double c = at(a, base + k, j, i);
            value += (absolute ? std::abs(c) : c) * choose_[degree][k] * std::pow(t, static_cast<int>(k)) * std::pow(1 - t, static_cast<int>(degree - k));
        }
        return value;
    }
    StatePtr log_child(const StatePtr& parent, size_t letter) {
        Word w = parent->word; w.push_back(letter);
        if (auto it = cache_.find(w); it != cache_.end()) { ++b_.stats.cache_hits; return it->second; }
        ++b_.stats.cache_misses;
        const size_t h = w.size(), levels = m_ - h;
        auto out = state(w, levels * (h + 1));
        samples([&](uint64_t begin, uint64_t end) {
            std::vector<double> totals(levels + 1);
            for (size_t i = begin; i < end; ++i) {
                std::fill(totals.begin(), totals.end(), 0);
                for (size_t p = 0; p <= levels; ++p) for (size_t j = 0; j < l_; ++j) {
                    double value = totals[p];
                    if (p < levels) at(out->polynomial, p * (h + 1), j, i) = value;
                    const double dx = step(j, letter, i) / h;
                    for (size_t k = 0; k < h; ++k) {
                        value += dx * at(parent->polynomial, p * h + k, j, i);
                        if (p < levels) at(out->polynomial, p * (h + 1) + k + 1, j, i) = value;
                    }
                    totals[p] = value;
                }
                out->values.data[i] = finite(totals[0]);
                for (size_t p = 0; p < levels; ++p) for (size_t j = 0; j < l_; ++j)
                    for (size_t k = 0; k <= h; ++k) at(out->polynomial, p * (h + 1) + k, j, i) += totals[p + 1];
            }
        });
        if (lyndon(w)) b_.record_word(w);
        cache(out); return out;
    }
    HallPtr parse(const Word& w) {
        if (auto it = parsed_.find(w); it != parsed_.end()) return it->second;
        if (invalid_words_.count(w)) return {};
        if (w.empty()) return {};
        if (w.size() == 1) {
            if (w[0] >= d_) return {};
            auto h = std::make_shared<Hall>(); h->word = w; parsed_.emplace(w, h);
            metadata_.push_back(b_.account(512 + w.size() * 32)); return h;
        }
        for (size_t cut = (w.size() + 1) / 2; cut < w.size(); ++cut) {
            auto v = parse(Word(w.begin() + cut, w.end()));
            if (!v) continue;
            auto u = parse(Word(w.begin(), w.begin() + cut));
            if (u && admissible(u, v)) {
                auto h = bracket(u, v); parsed_.emplace(w, h);
                metadata_.push_back(b_.account(512 + w.size() * 32)); return h;
            }
        }
        invalid_words_.insert(w); metadata_.push_back(b_.account(256 + w.size() * 16)); return {};
    }
    const HallKernelPlan& hall_plan(size_t a, size_t c) {
        const auto key = std::make_pair(a, c);
        if (auto it = hall_plans_.find(key); it != hall_plans_.end()) return *it->second;
        const size_t degree = a + c;
        size_t count = 0;
        for (size_t p = 0; p < degree; ++p) for (size_t pu = 0; pu < a && pu <= p; ++pu)
            if (p - pu <= c) count += multiply(a - pu, c - (p - pu) + 1);
        auto plan = std::make_shared<HallKernelPlan>(); plan->degree = degree;
        plan->metadata = b_.account(multiply(count, sizeof(HallProductTerm)) + multiply((degree + 1) * (degree + 1), sizeof(double)) + 256);
        plan->terms.reserve(count); plan->choose.resize((degree + 1) * (degree + 1));
        for (size_t k = 0; k <= degree; ++k) for (size_t p = 0; p <= k; ++p) plan->choose[k * (degree + 1) + p] = choose_[k][p];
        for (size_t p = 0; p < degree; ++p) for (size_t pu = 0; pu < a && pu <= p; ++pu) {
            const size_t pv = p - pu; if (pv > c) continue;
            const size_t du = a - pu - 1, dv = c - pv;
            for (size_t k = 0; k <= du; ++k) for (size_t t = 0; t <= dv; ++t)
                plan->terms.push_back({offset(a, pu) + k, offset(c, pv) + t, offset(degree - 1, p) + k + t,
                    (a - pu) * choose_[du][k] * choose_[dv][t] / choose_[du + dv][k + t]});
        }
        hall_plans_.emplace(key, plan); return *plan;
    }
    StatePtr hall_state(const HallPtr& h, bool required = false, bool polynomial = true) {
        if (auto it = side_states_.find(h->word); it != side_states_.end()) return it->second;
        if (auto it = cache_.find(h->word); it != cache_.end()) { ++b_.stats.cache_hits; return it->second; }
        StatePtr summary;
        if (auto it = summaries_.find(h->word); it != summaries_.end()) {
            summary = it->second;
            if (!summary->hall) summary->hall = h;
            if (!polynomial) { ++b_.stats.cache_hits; return summary; }
        }
        ++b_.stats.cache_misses;
        const size_t degree = h->word.size();
        const bool terminal = degree == m_ && h->left;
        auto out = state(h->word, terminal ? 0 : triangle(degree)); out->hall = h;
        if (summary) { out->moments = summary->moments; out->global = summary->global; }
        if (!h->left) {
            samples([&](uint64_t begin, uint64_t end) {
                for (size_t i = begin; i < end; ++i) {
                    double total = 0; for (size_t j = 0; j < l_; ++j) total += step(j, h->word[0], i);
                    double prefix = -0.5 * total;
                    for (size_t j = 0; j < l_; ++j) {
                        at(out->polynomial, 0, j, i) = prefix;
                        prefix += step(j, h->word[0], i);
                        at(out->polynomial, 1, j, i) = prefix;
                        at(out->polynomial, 2, j, i) = total;
                    }
                    out->values.data[i] = total;
                }
            });
        } else {
            auto u = hall_state(h->left), v = hall_state(h->right);
            const size_t a = h->left->word.size(), c = h->right->word.size();
            const auto& plan = hall_plan(a, c);
            const int64_t jobs = b_.options.n_jobs > 0 ? b_.options.n_jobs : static_cast<int64_t>(get_max_threads()) + 1 + b_.options.n_jobs;
            if (jobs < 1) throw std::invalid_argument("Invalid n_jobs");
            const size_t workers = n_ < 32 ? 1 : std::min(n_, static_cast<size_t>(jobs));
            const size_t stride = multiply(triangle(degree - 1) + 2 * (degree + 1), 4);
            if (hall_scratch_.size < multiply(workers, stride)) { hall_scratch_ = {}; hall_scratch_ = b_.alloc(multiply(workers, stride)); }
            auto update = [&](uint64_t begin, uint64_t end) {
                for (size_t worker = begin; worker < end; ++worker) {
                    cpsig::hall_polynomial(plan, u->polynomial.data, v->polynomial.data,
                        terminal ? nullptr : out->polynomial.data, out->values.data, n_, l_, n_ * worker / workers,
                        n_ * (worker + 1) / workers, static_cast<double>(h->run), hall_scratch_.data + worker * stride);
                }
            };
            if (workers == 1) update(0, 1); else spawn_batch_threads(workers, b_.options.n_jobs, update);
        }
        b_.record_word(h->word);
        if (required) side_states_.emplace(h->word, out); else cache(out);
        return out;
    }
    void rule(size_t count, bool logistic) {
        Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(count, count);
        for (size_t k = 1; k < count; ++k) {
            const double beta = logistic ? std::pow(static_cast<double>(k), 4) / (4 * (4.0 * k * k - 1)) : k * k / (4.0 * k * k - 1);
            jacobi(k - 1, k) = jacobi(k, k - 1) = std::sqrt(beta);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(jacobi);
        if (eig.info() != Eigen::Success) throw NonfiniteError("Gaussian rule failed");
        for (size_t k = 0; k < count; ++k) {
            const double weight = eig.eigenvectors()(0, k) * eig.eigenvectors()(0, k);
            if (logistic) {
                if (eig.eigenvalues()[k] < -1e-14) continue;
                gaussian_nodes_.emplace_back(0, std::abs(eig.eigenvalues()[k]) < 1e-14 ? 0 : eig.eigenvalues()[k]);
                gaussian_weights_.push_back(weight * (eig.eigenvalues()[k] > 1e-14 ? 2 : 1));
            }
            else { time_nodes_.push_back((1 + eig.eigenvalues()[k]) / 2); time_weights_.push_back(weight); }
        }
    }
    double weight(size_t q, size_t p, double r, double v) const {
        double sum = 0;
        for (size_t k = p - 1; k < q; ++k)
            sum += (stirling_[k][p - 1] / b_.weights[k]) * b_.weights[p - 1] * std::pow(v, static_cast<int>(k))
                * std::pow(r, static_cast<int>(q - 1 - k)) / b_.weights[q - 1 - k];
        return sum;
    }
    void log_moments(const StatePtr& u) {
        if (u->moments.owner) return;
        const size_t h = u->word.size(), levels = m_ - h;
        u->moments = b_.alloc(multiply(multiply(levels, l_), n_));
        if (cache_.count(u->word)) cache_size_ += u->moments.bytes();
        samples([&](uint64_t begin, uint64_t end) {
            for (size_t i = begin; i < end; ++i) {
                double v = l_ ? b_.variation.data[i] + b_.after.data[i] : 0;
                for (size_t j = 0; j < l_; ++j) for (size_t q = 1; q <= levels; ++q) {
                    double sum = 0;
                    const size_t count = b_.options.bound_method ? time_nodes_.size() : 2;
                    for (size_t k = 0; k < count; ++k) {
                        const double t = b_.options.bound_method ? time_nodes_[k] : static_cast<double>(k);
                        const double w = b_.options.bound_method ? time_weights_[k] : 0.5;
                        const double r = b_.after.data[j * n_ + i] + (1 - t) * b_.variation.data[j * n_ + i];
                        for (size_t p = 1; p <= q; ++p) sum += w * weight(q, p, r, v)
                            * std::abs(bernstein(u->polynomial, (p - 1) * (h + 1), h, j, i, t, b_.options.bound_method != 0));
                    }
                    const double value = std::abs(sum) * b_.weights[h + q];
                    at(u->moments, q - 1, j, i) = std::isfinite(value) ? value : std::numeric_limits<double>::infinity();
                }
            }
        });
    }
    double log_bound(const StatePtr& u, size_t low, size_t high, const Array& residual) {
        log_moments(u); double best = 0;
        const size_t h = u->word.size();
        for (size_t q = 1; q <= m_ - h; ++q) {
            double sum = 0;
            for (size_t i = 0; i < n_; ++i) {
                double bound = 0;
                for (size_t j = 0; j < l_; ++j) {
                    double dx = 0; for (size_t a = low; a < high; ++a) dx = std::max(dx, std::abs(step(j, a, i)));
                    bound += dx * at(u->moments, q - 1, j, i);
                }
                if (!std::isfinite(bound)) bound = caps_[h + q];
                sum += std::abs(residual.data[i]) * std::min(bound, caps_[h + q]);
            }
            if (!std::isfinite(sum)) return std::numeric_limits<double>::infinity();
            best = std::max(best, sum / n_);
        }
        return padded(best, 1e-11 * m_ * m_ * (l_ + 1));
    }
    Complex hall_value(const StatePtr& s, size_t j, size_t i, Complex z, size_t point, bool derivative) const {
        const size_t h = s->word.size(); Complex result = 0, power = 1;
        for (size_t p = 0; p <= h; ++p) {
            const size_t degree = h - p, base = offset(h, p);
            double value = 0;
            if (!derivative) for (size_t k = 0; k <= degree; ++k)
                value += at(s->polynomial, base + k, j, i) * time_basis_[degree][point * (degree + 1) + k];
            else for (size_t k = 0; k < degree; ++k)
                value += degree * (at(s->polynomial, base + k + 1, j, i) - at(s->polynomial, base + k, j, i))
                    * time_basis_[degree - 1][point * degree + k];
            result += value * power; power *= z;
        }
        return result;
    }
    double hall_envelope(const StatePtr& s, size_t j, size_t i, Complex z, bool derivative) const {
        const size_t h = s->word.size(), degree = h - (derivative ? 1 : 0);
        double envelope = 0;
        for (size_t k = 0; k <= degree; ++k) {
            Complex value = 0, power = 1;
            for (size_t p = 0; p <= h; ++p) {
                const size_t td = h - p; double elevated = 0;
                if (derivative && !td) { power *= z; continue; }
                const size_t source_degree = td - (derivative ? 1 : 0);
                for (size_t a = 0; a <= source_degree; ++a) if (k >= a && k - a <= degree - source_degree) {
                    double c = at(s->polynomial, offset(h, p) + a, j, i);
                    if (derivative) c = td * (at(s->polynomial, offset(h, p) + a + 1, j, i) - c);
                    elevated += c * choose_[source_degree][a] * choose_[degree - source_degree][k - a] / choose_[degree][k];
                }
                value += power * elevated; power *= z;
            }
            const double magnitude = std::abs(value);
            if (!std::isfinite(magnitude)) return std::numeric_limits<double>::infinity();
            if (derivative) envelope += magnitude / (degree + 1); else envelope = std::max(envelope, magnitude);
        }
        return envelope;
    }
    double hall_bound(const StatePtr& u, const Array& residual) {
        if (!gaussian_reliable_) return std::numeric_limits<double>::infinity();
        const size_t levels = m_ - u->word.size();
        if (!u->moments.owner) {
        u->moments = b_.alloc(multiply(levels, n_)); b_.fill(u->moments, 0);
        if (cache_.count(u->word)) cache_size_ += u->moments.bytes();
        for (const auto& v : sides_) {
            auto s = side_states_.at(v->word);
            const size_t points = b_.options.bound_method ? 1 : 2;
            const size_t bytes = multiply(multiply(multiply(gaussian_nodes_.size() * points, n_), l_), sizeof(double));
            if (s->envelope.owner || bytes > b_.options.cache_bytes / 4 - std::min(auxiliary_bytes_, static_cast<size_t>(b_.options.cache_bytes / 4))) continue;
            s->envelope = b_.alloc(bytes / sizeof(double)); auxiliary_bytes_ += bytes;
            samples([&](uint64_t begin, uint64_t end) {
                for (size_t i = begin; i < end; ++i) for (size_t j = 0; j < l_; ++j) for (size_t g = 0; g < gaussian_nodes_.size(); ++g)
                    for (size_t p = 0; p < points; ++p) at(s->envelope, g * points + p, j, i) = b_.options.bound_method
                        ? hall_envelope(s, j, i, gaussian_nodes_[g], false) : std::abs(hall_value(s, j, i, gaussian_nodes_[g], p, false));
            });
        }
        const size_t points = b_.options.bound_method ? 1 : 2;
        Word key{levels, u->hall->run, u->hall->right ? u->hall->right->word.size() : 0};
        if (u->hall->right) key.insert(key.end(), u->hall->right->word.begin(), u->hall->right->word.end());
        Array products;
        bool ready = false;
        if (auto found = products_.find(key); found != products_.end()) { products = found->second.values; ready = true; }
        else {
            const size_t count = multiply(multiply((levels + 1) * points * gaussian_nodes_.size(), l_), n_);
            const size_t bytes = multiply(count, sizeof(double));
            if (bytes <= b_.options.cache_bytes / 8) {
                if (product_bytes_ + bytes > b_.options.cache_bytes / 8) { products_.clear(); product_bytes_ = 0; }
                products = b_.alloc(count);
            }
        }
        samples([&](uint64_t begin, uint64_t end) {
        std::vector<double> dp(levels + 1), next(levels + 1);
        for (size_t i = begin; i < end; ++i) for (size_t j = 0; j < l_; ++j) for (size_t g = 0; g < gaussian_nodes_.size(); ++g) {
            const size_t points = b_.options.bound_method ? 1 : 2;
            for (size_t point = 0; point < points; ++point) {
                if (ready) for (size_t q = 0; q <= levels; ++q) dp[q] = at(products, (g * points + point) * (levels + 1) + q, j, i);
                else {
                std::fill(dp.begin(), dp.end(), 0); dp[0] = 1;
                for (const auto& v : sides_) {
                    if (u->hall->right && compare(v, u->hall->right) > 0) continue;
                    const size_t degree = v->word.size(); if (degree > levels) continue;
                    const auto& vs = side_states_.at(v->word);
                    const double a = vs->envelope.owner ? at(vs->envelope, g * points + point, j, i)
                        : b_.options.bound_method ? hall_envelope(vs, j, i, gaussian_nodes_[g], false) : std::abs(hall_value(vs, j, i, gaussian_nodes_[g], point, false));
                    const size_t inherited = u->hall->right && compare(v, u->hall->right) == 0 ? u->hall->run : 0;
                    next = dp;
                    double power = 1;
                    for (size_t count = 1; count * degree <= levels; ++count) {
                        power *= a / (inherited + count);
                        for (size_t q = count * degree; q <= levels; ++q) {
                            const double value = dp[q - count * degree] * power;
                            next[q] = std::isfinite(value) ? std::max(next[q], value) : std::numeric_limits<double>::infinity();
                        }
                    }
                    dp.swap(next);
                }
                if (products.owner) for (size_t q = 0; q <= levels; ++q) at(products, (g * points + point) * (levels + 1) + q, j, i) = dp[q];
                }
                const double derivative = b_.options.bound_method ? hall_envelope(u, j, i, gaussian_nodes_[g], true)
                    : std::abs(hall_value(u, j, i, gaussian_nodes_[g], point, true)) * 0.5;
                for (size_t q = 1; q <= levels; ++q) {
                    const double value = gaussian_weights_[g] * derivative * dp[q] * b_.weights[u->word.size() + q];
                    u->moments.data[(q - 1) * n_ + i] += std::isfinite(value) ? value : std::numeric_limits<double>::infinity();
                }
            }
        }
        });
        if (!ready && products.owner) {
            auto metadata = b_.account(512 + key.size() * 32);
            products_.emplace(key, ProductTable{products, std::move(metadata)}); product_bytes_ += products.bytes();
        }
        }
        double best = 0;
        for (size_t q = 0; q < levels; ++q) {
            double sum = 0; for (size_t i = 0; i < n_; ++i) sum += std::abs(residual.data[i]) * u->moments.data[q * n_ + i];
            if (!std::isfinite(sum)) return std::numeric_limits<double>::infinity();
            best = std::max(best, sum / n_);
        }
        return padded(best, 1e-10 * m_ * m_ * (l_ + 1));
    }
    double hall_group_bound(const StatePtr& u, const std::vector<HallPtr>& factors, size_t low, size_t high, const Array& residual) {
        const size_t levels = m_ - u->word.size();
        if (!side_maxima_.owner) {
            side_maxima_ = b_.alloc(multiply(sides_.size(), n_)); b_.fill(side_maxima_, 0);
            samples([&](uint64_t begin, uint64_t end) {
                for (size_t v = 0; v < sides_.size(); ++v) for (size_t i = begin; i < end; ++i)
                    for (size_t j = 0; j < l_; ++j) for (size_t g = 0; g < gaussian_nodes_.size(); ++g)
                        side_maxima_.data[v * n_ + i] = std::max(side_maxima_.data[v * n_ + i], hall_envelope(side_states_.at(sides_[v]->word), j, i, gaussian_nodes_[g], false));
            });
        }
        if (!u->global.owner) {
            u->global = b_.alloc(multiply(levels + 2, n_)); b_.fill(u->global, 0);
            if (cache_.count(u->word)) cache_size_ += u->global.bytes();
            samples([&](uint64_t begin, uint64_t end) {
                std::vector<double> dp(levels + 1), next(levels + 1);
                for (size_t i = begin; i < end; ++i) {
                    std::fill(dp.begin(), dp.end(), 0); dp[0] = 1;
                    double mass = 0;
                    for (size_t g = 0; g < gaussian_nodes_.size(); ++g) {
                        double value = 0; for (size_t j = 0; j < l_; ++j) value += hall_envelope(u, j, i, gaussian_nodes_[g], true);
                        mass += gaussian_weights_[g] * value;
                    }
                    u->global.data[i] = mass;
                    for (size_t v = 0; v < sides_.size(); ++v) {
                        if (u->hall->right && compare(sides_[v], u->hall->right) > 0) continue;
                        const size_t degree = sides_[v]->word.size();
                        const size_t inherited = u->hall->right && compare(sides_[v], u->hall->right) == 0 ? u->hall->run : 0;
                        next = dp; double power = 1;
                        for (size_t count = 1; count * degree <= levels; ++count) {
                            power *= side_maxima_.data[v * n_ + i] / (inherited + count);
                            for (size_t q = count * degree; q <= levels; ++q) {
                                const double value = dp[q - count * degree] * power;
                                next[q] = std::isfinite(value) ? std::max(next[q], value) : std::numeric_limits<double>::infinity();
                            }
                        }
                        dp.swap(next);
                    }
                    for (size_t q = 0; q <= levels; ++q) u->global.data[(q + 1) * n_ + i] = dp[q];
                }
            });
        }
        for (size_t index = low; index < high; ++index) {
            const auto& v = factors[index];
            const size_t run = u->hall->right && compare(v, u->hall->right) == 0 ? u->hall->run + 1 : 1;
            group_factors_[index - low] = {v->word.size(), v->slot, 1. / run};
        }
        const double best = cpsig::hall_group_bound(u->global.data, side_maxima_.data, residual.data, group_factors_.data(),
            high - low, b_.weights.data(), u->word.size(), levels, n_);
        return padded(best, 1e-10 * m_ * m_ * (l_ + 1));
    }
    double offer(const StatePtr& u, const Array& residual, double alpha, size_t capacity,
        const std::unordered_map<Word, size_t, WordHash>& active, SearchResult& result) {
        if (active.count(u->word)) return 0;
        const double score = std::abs(b_.dot(u->values.data, residual.data, n_)) * b_.weights[u->word.size()] / n_;
        result.omitted = std::max(result.omitted, guard(score));
        if (!capacity || score <= alpha) return score;
        Candidate c{u->word, u->values, score};
        if (result.candidates.size() < capacity) { result.candidates.push_back(std::move(c)); std::push_heap(result.candidates.begin(), result.candidates.end(), better); }
        else if (better(c, result.candidates.front())) {
            std::pop_heap(result.candidates.begin(), result.candidates.end(), better); result.candidates.back() = std::move(c);
            std::push_heap(result.candidates.begin(), result.candidates.end(), better);
        }
        return score;
    }
    void terminals(StatePtr u, const Array& residual, double alpha, size_t capacity,
        const std::unordered_map<Word, size_t, WordHash>& active, SearchResult& result) {
        ++b_.stats.terminal_queries;
        const size_t h = u->word.size(), levels = m_ - h;
        double maximum = 0;
        Array integrals;
        if (mode_ == 1) {
            integrals = b_.alloc(multiply(3 * l_, n_));
            samples([&](uint64_t begin, uint64_t end) {
                for (size_t j = 0; j < l_; ++j) for (size_t i = begin; i < end; ++i) {
                    double first = 0, time = 0, second = 0;
                    for (size_t k = 0; k <= h; ++k) {
                        const double value = at(u->polynomial, k, j, i) / (h + 1);
                        first += value; time += (k + 1) * value / (h + 2);
                        if (levels == 2) second += at(u->polynomial, h + 1 + k, j, i) / (h + 1);
                    }
                    at(integrals, 0, j, i) = first; at(integrals, 1, j, i) = time; at(integrals, 2, j, i) = second;
                }
            });
        }
        auto evaluate = [&](const Word& word, const HallPtr& v, const HallPtr& w, size_t a, size_t c) {
            if (active.count(word)) return;
            if (mode_ == 2) {
                StatePtr saved;
                if (auto it = cache_.find(word); it != cache_.end()) saved = it->second;
                else if (auto it = summaries_.find(word); it != summaries_.end()) saved = it->second;
                if (saved) {
                    ++b_.stats.cache_hits; ++b_.stats.terminal_correlations;
                    maximum = std::max(maximum, offer(saved, residual, alpha, capacity, active, result));
                    return;
                }
                if (!u->polynomial.owner) u = hall_state(u->hall);
            }
            auto probe = std::make_shared<State>(); probe->word = word; probe->values = b_.alloc(n_);
            const size_t q = word.size() - h;
            StatePtr vs = v ? side_states_.at(v->word) : StatePtr{};
            StatePtr ws = w ? side_states_.at(w->word) : StatePtr{};
            const size_t run = v && u->hall->right && compare(v, u->hall->right) == 0 ? u->hall->run + 1 : 1;
            const size_t second_run = w && compare(v, w) == 0 ? run + 1 : 1;
            samples([&](uint64_t begin, uint64_t end) {
                for (size_t i = begin; i < end; ++i) {
                    double total = 0, tail = 0, displacement = 0;
                    if (mode_ == 1 && q == 2) for (size_t j = 0; j < l_; ++j) displacement += step(j, c, i);
                    tail = displacement;
                    for (size_t j = 0; j < l_; ++j) {
                        if (mode_ == 1) {
                            double value = at(integrals, 0, j, i);
                            if (q == 2) value = value * tail - step(j, c, i) * at(integrals, 1, j, i) + displacement * at(integrals, 2, j, i);
                            total += step(j, a, i) * value;
                            if (q == 2) tail -= step(j, c, i);
                            continue;
                        }
                        for (size_t t = 0; t < time_nodes_.size(); ++t) {
                            for (size_t g = 0; g < gaussian_nodes_.size(); ++g) {
                                Complex value = hall_value(u, j, i, gaussian_nodes_[g], t + 2, true)
                                    * hall_value(vs, j, i, gaussian_nodes_[g], t + 2, false) / static_cast<double>(run);
                                if (ws) value *= hall_value(ws, j, i, gaussian_nodes_[g], t + 2, false) / static_cast<double>(second_run);
                                total += time_weights_[t] * gaussian_weights_[g] * value.real();
                            }
                        }
                    }
                    probe->values.data[i] = finite(mode_ == 2 && word.size() % 2 == 0 ? -total : total);
                }
            });
            ++b_.stats.terminal_correlations;
            if (mode_ == 2) remember(probe);
            const size_t previous = result.candidates.size();
            maximum = std::max(maximum, offer(probe, residual, alpha, capacity, active, result));
            if (result.candidates.size() > previous) b_.record_word(word);
        };
        if (mode_ == 1) {
            for (size_t a = 0; a < d_; ++a) {
                Word word = u->word; word.push_back(a);
                if (lyndon(word)) evaluate(word, {}, {}, a, 0);
                if (levels == 2) for (size_t c = 0; c < d_; ++c) {
                    word.push_back(c); if (lyndon(word)) evaluate(word, {}, {}, a, c); word.pop_back();
                }
            }
        } else for (const auto& v : sides_) if (v->word.size() <= levels && admissible(u->hall, v)) {
            auto child = bracket(u->hall, v); evaluate(child->word, v, {}, 0, 0);
            if (v->word.size() == 1 && levels == 2) for (const auto& w : sides_)
                if (w->word.size() == 1 && admissible(child, w)) evaluate(bracket(child, w)->word, v, w, 0, 0);
        }
        if (maximum <= alpha) ++b_.stats.terminal_pruned;
    }
public:
    PolynomialSearch(Backend& b, uint32_t mode, bool selected_only = false) : b_(b), n_(b.options.samples), l_(b.options.length - 1), d_(b.options.dimension), m_(b.options.dimension == 1 ? 1 : b.options.degree), mode_(mode) {
        b_.trim_search = [this] {
            cache_.clear(); cache_order_.clear(); cache_size_ = 0;
            summaries_.clear(); summary_order_.clear(); summary_bytes_ = 0;
            for (auto& item : side_states_) item.second->envelope = {};
            auxiliary_bytes_ = 0;
            products_.clear(); product_bytes_ = 0;
        };
        tables_metadata_ = b_.account(multiply((m_ + 1) * (m_ + 1), 32));
        choose_.resize(m_ + 1); stirling_.assign(m_ + 1, std::vector<double>(m_ + 1));
        caps_.assign(m_ + 1, 1); stirling_[0][0] = 1;
        std::vector<double> euler(1, 1);
        for (size_t h = 0; h <= m_; ++h) {
            choose_[h].resize(h + 1, 1);
            for (size_t k = 1; k < h; ++k) choose_[h][k] = choose_[h - 1][k - 1] + choose_[h - 1][k];
            if (!h) continue;
            for (size_t k = 1; k <= h; ++k) stirling_[h][k] = stirling_[h - 1][k - 1] + k * stirling_[h - 1][k];
            if (h > 1) {
                std::vector<double> next(h);
                for (size_t k = 0; k < h; ++k) next[k] = (k < euler.size() ? (k + 1) * euler[k] : 0) + (k ? (h - k) * euler[k - 1] : 0);
                euler.swap(next);
            }
            caps_[h] = 0; for (size_t k = 0; k < h; ++k) caps_[h] += euler[k] / choose_[h - 1][k] / h;
        }
        rule((m_ + 1) / 2, false);
        if (mode_ == 1) {
            root_ = state({}, m_);
            for (size_t p = 0; p < m_; ++p) for (size_t j = 0; j < l_; ++j) for (size_t i = 0; i < n_; ++i)
                at(root_->polynomial, p, j, i) = (p % 2 ? -1.0 : 1.0) / (p + 1);
        } else {
            rule((m_ + 1) / 2, true);
            const size_t points = time_nodes_.size() + 2;
            time_basis_metadata_ = b_.account(multiply(multiply(triangle(m_), points), sizeof(double)) + (m_ + 1) * 32);
            time_basis_.resize(m_ + 1);
            for (size_t h = 0; h <= m_; ++h) {
                time_basis_[h].resize(points * (h + 1));
                for (size_t point = 0; point < points; ++point) {
                    const double t = point < 2 ? static_cast<double>(point) : time_nodes_[point - 2];
                    for (size_t k = 0; k <= h; ++k) time_basis_[h][point * (h + 1) + k] = choose_[h][k]
                        * std::pow(t, static_cast<int>(k)) * std::pow(1 - t, static_cast<int>(h - k));
                }
            }
            std::vector<double> bernoulli(m_, 1);
            for (size_t k = 1; k < m_; ++k) {
                double sum = 0, magnitude = 0;
                for (size_t p = 0; p < k; ++p) { const double term = choose_[k + 1][p] * bernoulli[p]; sum += term; magnitude += std::abs(term); }
                bernoulli[k] = -sum / (k + 1);
                const double expected = k % 2 ? 0 : (std::pow(2., 1 - static_cast<int>(k)) - 1) * bernoulli[k];
                Complex moment = 0;
                for (size_t g = 0; g < gaussian_nodes_.size(); ++g) moment += gaussian_weights_[g] * std::pow(gaussian_nodes_[g], static_cast<int>(k));
                const double defect = std::abs(moment.real() - expected) + 32 * std::numeric_limits<double>::epsilon() * magnitude;
                if (!std::isfinite(defect) || defect > 1e-10 * std::max(1., std::abs(expected))) gaussian_reliable_ = false;
            }
            if (!gaussian_reliable_ && b_.absolute_sum(b_.increments) > 0) { b_.uncertain_features = true; ++b_.stats.numerical_failures; }
            if (selected_only) return;
            zero_path_ = !b_.options.full && (!l_ || b_.absolute_sum(b_.increments) == 0);
            if (zero_path_) return;
            long double power = 1, bytes = 0;
            std::vector<long double> primitive(m_ / 2 + 1);
            for (size_t degree = 1; degree <= m_ / 2; ++degree) {
                power *= d_; primitive[degree] = power;
                for (size_t k = 1; k < degree; ++k) if (degree % k == 0) primitive[degree] -= primitive[k];
                bytes += primitive[degree] / degree * (triangle(degree) * static_cast<long double>(n_) * l_ * sizeof(double) + 1024);
                if (bytes > b_.budget->available()) throw std::bad_alloc();
            }
            for (size_t a = 0; a < d_; ++a) sides_.push_back(parse(Word{a}));
            for (size_t degree = 2; degree <= m_ / 2; ++degree) {
                const size_t count = sides_.size(); std::vector<HallPtr> next;
                for (size_t a = 0; a < count; ++a) for (size_t c = 0; c < count; ++c)
                    if (sides_[a]->word.size() + sides_[c]->word.size() == degree && admissible(sides_[a], sides_[c])) {
                        auto h = bracket(sides_[a], sides_[c]);
                        auto metadata = b_.account(512 + degree * 32);
                        next.push_back(h); parsed_.emplace(h->word, h); metadata_.push_back(std::move(metadata));
                    }
                sides_.insert(sides_.end(), next.begin(), next.end());
            }
            for (size_t i = 0; i < sides_.size(); ++i) { sides_[i]->slot = i; hall_state(sides_[i], true); }
            metadata_.push_back(b_.account(multiply(sides_.size(), sizeof(HallBoundFactor))));
            group_factors_.resize(sides_.size());
        }
    }
    std::vector<std::shared_ptr<void>> metadata_;
    ~PolynomialSearch() override { b_.trim_search = {}; }
    NodePtr selected(const Word& word) override {
        if (word.empty() || word.size() > m_ || std::any_of(word.begin(), word.end(), [&](uint64_t a) { return a >= d_; })) throw std::invalid_argument("Invalid feature word");
        StatePtr s;
        if (mode_ == 1) {
            if (!lyndon(word)) throw std::invalid_argument("Log feature words must be Lyndon words");
            s = root_; for (auto a : word) s = log_child(s, a);
        } else {
            auto h = parse(word); if (!h) throw std::invalid_argument("Feature word is not in the reverse-degree Hall set");
            s = hall_state(h, false, false);
        }
        auto node = std::make_shared<Node>(); node->word = word; node->values = s->values; return node;
    }
    SearchResult query(const Array& input, double alpha, const std::unordered_map<Word, size_t, WordHash>& active,
        size_t capacity, size_t = 0, size_t = 0, size_t = 0) override {
        Array residual = input;
        if (b_.options.intercept) { residual = b_.alloc(n_); b_.copy(input.data, residual.data, n_); b_.center(residual.data, n_, true); }
        SearchResult result;
        if (zero_path_ || !l_ || b_.absolute_sum(b_.increments) == 0) return result;
        std::function<void(const StatePtr&, size_t)> visit_log;
        visit_log = [&](const StatePtr& u, size_t period) {
            const size_t h = u->word.size();
            if (h && period == h) offer(u, residual, alpha, capacity, active, result);
            if (h == m_) return;
            const size_t first = h ? u->word[h - period] : 0;
            const double threshold = capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
            if (b_.options.search_method & 1) {
                ++b_.stats.block_queries;
                const double bound = log_bound(u, first, d_, residual);
                if (bound < threshold) { result.omitted = std::max(result.omitted, bound); ++b_.stats.pruned; return; }
            }
            if ((b_.options.search_method & 2) && m_ - h <= 2) {
                terminals(u, residual, alpha, capacity, active, result); return;
            }
            for (size_t a = first; a < d_; ++a) {
                if (!(b_.options.search_method & 1)) {
                    ++b_.stats.block_queries; const double bound = log_bound(u, a, a + 1, residual);
                    if (bound < threshold) { result.omitted = std::max(result.omitted, bound); ++b_.stats.pruned; continue; }
                }
                visit_log(log_child(u, a), h && a == first ? period : h + 1);
            }
        };
        std::function<void(const StatePtr&)> visit_hall;
        visit_hall = [&](StatePtr u) {
            offer(u, residual, alpha, capacity, active, result);
            if (u->word.size() == m_) return;
            if (!u->polynomial.owner && !u->moments.owner) u = hall_state(u->hall);
            const double threshold = capacity && result.candidates.size() == capacity ? std::max(alpha, result.candidates.front().score) : alpha;
            ++b_.stats.block_queries; const double bound = hall_bound(u, residual);
            if (bound < threshold) {
                result.omitted = std::max(result.omitted, bound); ++b_.stats.pruned;
                if (!cache_.count(u->word)) remember(u);
                return;
            }
            if ((b_.options.search_method & 2) && gaussian_reliable_ && m_ - u->word.size() <= 2) {
                size_t count = 0;
                for (const auto& v : sides_) if (u->word.size() + v->word.size() <= m_ && admissible(u->hall, v)) {
                    ++count;
                    if (m_ - u->word.size() == 2 && v->word.size() == 1) {
                        auto child = bracket(u->hall, v);
                        for (const auto& w : sides_) if (w->word.size() == 1 && admissible(child, w)) ++count;
                    }
                }
                if (count <= 8) {
                    terminals(u, residual, alpha, capacity, active, result);
                    if (!cache_.count(u->word)) remember(u);
                    return;
                }
            }
            if (!u->polynomial.owner && !u->global.owner) u = hall_state(u->hall);
            std::vector<HallPtr> factors;
            for (const auto& v : sides_) if (u->word.size() + v->word.size() <= m_ && admissible(u->hall, v)) factors.push_back(v);
            std::function<void(size_t, size_t)> group = [&](size_t low, size_t high) {
                if (low == high) return;
                ++b_.stats.block_queries;
                const double child_bound = hall_group_bound(u, factors, low, high, residual);
                if (child_bound < threshold) { result.omitted = std::max(result.omitted, child_bound); ++b_.stats.pruned; return; }
                if (high - low == 1) visit_hall(hall_state(bracket(u->hall, factors[low]), false, false));
                else { const size_t middle = (low + high) / 2; group(low, middle); group(middle, high); }
            };
            if (b_.options.search_method & 1) group(0, factors.size());
            else for (size_t v = 0; v < factors.size(); ++v) group(v, v + 1);
            if (!cache_.count(u->word)) remember(u);
        };
        if (mode_ == 1) visit_log(root_, 0);
        else for (size_t a = 0; a < d_; ++a) visit_hall(hall_state(parse(Word{a})));
        std::sort(result.candidates.begin(), result.candidates.end(), better); ++b_.stats.search_passes; return result;
    }
    void full(std::vector<Column>& columns) {
        if (zero_path_) return;
        std::function<void(const StatePtr&, size_t)> log_visit;
        log_visit = [&](const StatePtr& u, size_t period) {
            const size_t h = u->word.size();
            if (h && h == period) append_column(b_, columns, u->word, u->values);
            if (h == m_) return;
            const size_t first = h ? u->word[h - period] : 0;
            for (size_t a = first; a < d_; ++a) log_visit(log_child(u, a), h && a == first ? period : h + 1);
        };
        std::function<void(const StatePtr&)> hall_visit;
        hall_visit = [&](const StatePtr& u) {
            append_column(b_, columns, u->word, u->values);
            for (const auto& v : sides_) if (u->word.size() + v->word.size() <= m_ && admissible(u->hall, v)) hall_visit(hall_state(bracket(u->hall, v)));
        };
        if (mode_ == 1) log_visit(root_, 0);
        else for (size_t a = 0; a < d_; ++a) hall_visit(hall_state(parse(Word{a})));
    }
    size_t audit() {
        std::vector<Column> columns; full(columns);
        auto residual = b_.alloc(n_);
        for (size_t i = 0; i < n_; ++i) residual.data[i] = std::sin(1.7 * i) - .3;
        b_.center(residual.data, n_, true);
        std::vector<StatePtr> states;
        if (root_) states.push_back(root_);
        for (const auto& entry : cache_) states.push_back(entry.second);
        for (const auto& entry : side_states_) if (!cache_.count(entry.first)) states.push_back(entry.second);
        size_t checks = 0;
        for (const auto& u : states) {
            const size_t h = u->word.size(); if (h == m_) continue;
            double actual = 0;
            for (const auto& column : columns) {
                bool descendant = column.word.size() > h;
                if (mode_ == 1) descendant = descendant && std::equal(u->word.begin(), u->word.end(), column.word.begin());
                else {
                    auto tree = parse(column.word);
                    while (tree && tree->word.size() > h) tree = tree->left;
                    descendant = descendant && tree && tree->word == u->word;
                }
                if (descendant) actual = std::max(actual, std::abs(b_.dot(column.values.data, residual.data, n_)) / n_);
            }
            const double bound = mode_ == 1 ? log_bound(u, 0, d_, residual) : hall_bound(u, residual);
            if (actual > bound + 1e-12) throw std::runtime_error("Descendant bound audit failed");
            ++checks;
            if (actual > 1e-8 && u->polynomial.size) {
                const double saved = u->polynomial.data[0];
                u->polynomial.data[0] = std::numeric_limits<double>::quiet_NaN();
                if (cache_.count(u->word)) cache_size_ -= u->moments.bytes();
                u->moments = {};
                const double uncertain = mode_ == 1 ? log_bound(u, 0, d_, residual) : hall_bound(u, residual);
                u->polynomial.data[0] = saved;
                if (std::isnan(uncertain) || uncertain < actual) throw std::runtime_error("Nonfinite envelope fallback failed");
                ++checks;
            }
            if (mode_ == 2) {
                std::vector<HallPtr> factors;
                for (const auto& v : sides_) if (h + v->word.size() <= m_ && admissible(u->hall, v)) factors.push_back(v);
                std::function<void(size_t, size_t)> audit_group = [&](size_t low, size_t high) {
                    if (low == high) return;
                    double expected = 0;
                    for (const auto& column : columns) {
                        auto tree = parse(column.word);
                        while (tree->left && tree->left->word.size() > h) tree = tree->left;
                        if (!tree->left || tree->left->word != u->word) continue;
                        if (std::none_of(factors.begin() + low, factors.begin() + high,
                            [&](const HallPtr& v) { return v->word == tree->right->word; })) continue;
                        expected = std::max(expected, std::abs(b_.dot(column.values.data, residual.data, n_)) / n_);
                    }
                    if (expected > hall_group_bound(u, factors, low, high, residual) + 1e-12)
                        throw std::runtime_error("Hall group bound audit failed");
                    ++checks;
                    if (high - low > 1) { const size_t middle = (low + high) / 2; audit_group(low, middle); audit_group(middle, high); }
                };
                audit_group(0, factors.size());
                if (!factors.empty() && actual > 1e-8) {
                    double& value = side_maxima_.data[factors.front()->slot * n_];
                    const double saved = value; value = std::numeric_limits<double>::quiet_NaN();
                    const double uncertain = hall_group_bound(u, factors, 0, factors.size(), residual);
                    value = saved;
                    if (std::isnan(uncertain) || uncertain < actual) throw std::runtime_error("Nonfinite group bound fallback failed");
                    ++checks;
                }
            }
        }
        return checks;
    }
};
}
std::unique_ptr<FeatureSearch> make_cpu_feature_search(Backend& b, uint32_t mode, bool selected_only) { return std::make_unique<PolynomialSearch>(b, mode, selected_only); }
void full_cpu_features(Backend& b, uint32_t mode, std::vector<Column>& columns) { PolynomialSearch search(b, mode); search.full(columns); }
size_t audit_cpu_feature_bounds(Backend& b, uint32_t mode) { PolynomialSearch search(b, mode); return search.audit(); }
}
