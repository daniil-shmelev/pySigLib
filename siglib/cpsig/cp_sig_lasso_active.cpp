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

#include "sig_lasso_core.h"
#include "sig_lasso_active.h"
#include <Eigen/QR>
#include <Eigen/SVD>

namespace siglasso {
namespace {
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;
constexpr double eps = std::numeric_limits<double>::epsilon();

class CPUActive final : public ActiveSet {
    Backend& b_;
    Array y_, residual_, q_, r_, projected_, temporary_, direction_, trial_;
    std::vector<Column> columns_;
    std::vector<size_t> slots_, selected_;
    std::vector<double> gamma_, signs_, scales_, scores_;
    std::shared_ptr<void> metadata_;
    size_t capacity_ = 0, rows_ = 0;
    double alpha_;
    bool rebuild_ = false;

    size_t n() const { return y_.size; }
    size_t stride() const { return std::min(n(), capacity_); }
    double& R(size_t i, size_t j) { return r_.data[j * stride() + i]; }
    double* Q(size_t j) { return q_.data + j * n(); }
    double penalty_at(size_t j) const { return n() * alpha_ * signs_[j] / scales_[j]; }

    void reserve(size_t count) {
        if (count <= capacity_) return;
        const size_t next = std::max<size_t>(16, std::max(count, 2 * capacity_));
        auto q = b_.alloc(multiply(n(), std::min(n(), next)));
        auto r = b_.alloc(multiply(std::min(n(), next), next));
        b_.fill(r, 0);
        for (size_t j = 0; j < rows_; ++j) b_.copy(Q(j), q.data + j * n(), n());
        for (size_t j = 0; j < size(); ++j) for (size_t i = 0; i < rows_; ++i)
            r.data[j * std::min(n(), next) + i] = R(i, j);
        auto projected = b_.alloc(next), temporary = b_.alloc(next), direction = b_.alloc(next);
        if (rows_) b_.copy(projected_.data, projected.data, rows_);
        auto metadata = b_.account(multiply(next, 256 + b_.options.degree * 8));
        columns_.reserve(next); slots_.reserve(next); gamma_.reserve(next); signs_.reserve(next); scales_.reserve(next);
        q_ = std::move(q); r_ = std::move(r); projected_ = std::move(projected);
        temporary_ = std::move(temporary); direction_ = std::move(direction); metadata_ = std::move(metadata);
        capacity_ = next;
    }

    void rebuild() {
        auto started = Clock::now();
        const size_t p = size();
        rows_ = std::min(n(), p);
        if (!p) { rebuild_ = false; return; }
        auto workspace = b_.account(multiply(3 * multiply(n(), p) + 16 * multiply(p, p) + 32 * p, sizeof(double)));
        Matrix normalized(n(), p);
        for (size_t j = 0; j < p; ++j) for (size_t i = 0; i < n(); ++i)
            normalized(i, j) = columns_[j].values.data[i] / scales_[j];
        Eigen::HouseholderQR<Matrix> qr(normalized);
        Eigen::Map<Matrix>(q_.data, n(), stride()).leftCols(rows_) = qr.householderQ() * Matrix::Identity(n(), rows_);
        b_.fill(r_, 0);
        for (size_t j = 0; j < p; ++j) for (size_t i = 0; i <= j && i < rows_; ++i) R(i, j) = qr.matrixQR()(i, j);
        for (size_t j = 0; j < rows_; ++j) projected_.data[j] = b_.dot(Q(j), y_.data, n());
        rebuild_ = false;
        ++b_.stats.qr_rebuilds;
        b_.stats.factorization_seconds += elapsed(started);
    }

    bool full_rank() {
        if (rows_ != size()) return false;
        double largest = 0, smallest = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < size(); ++j) { largest = std::max(largest, std::abs(R(j, j))); smallest = std::min(smallest, std::abs(R(j, j))); }
        return smallest > eps * std::max(n(), size()) * largest;
    }

    void lower(double* v) {
        for (size_t j = 0; j < size(); ++j) {
            for (size_t k = 0; k < j; ++k) v[j] -= R(k, j) * v[k];
            v[j] = finite(v[j] / R(j, j));
        }
    }

    void upper(double* v) {
        for (size_t j = size(); j-- > 0;) {
            for (size_t k = j + 1; k < size(); ++k) v[j] -= R(j, k) * v[k];
            v[j] = finite(v[j] / R(j, j));
        }
    }

    void residual_for(const double* gamma, Array& out) {
        b_.copy(y_.data, out.data, n());
        for (size_t j = 0; j < size(); ++j) if (gamma[j]) b_.axpy(-gamma[j] / scales_[j], columns_[j].values.data, out.data, n());
    }

    bool direction() {
        const size_t p = size();
        if (full_rank()) {
            for (size_t j = 0; j < p; ++j) temporary_.data[j] = penalty_at(j);
            lower(temporary_.data);
            for (size_t j = 0; j < p; ++j) direction_.data[j] = projected_.data[j] - temporary_.data[j];
            upper(direction_.data);
            for (size_t attempt = 0; attempt < 3; ++attempt) {
                residual_for(direction_.data, trial_);
                for (size_t j = 0; j < p; ++j)
                    temporary_.data[j] = b_.dot(columns_[j].values.data, trial_.data, n()) / scales_[j] - penalty_at(j);
                lower(temporary_.data); upper(temporary_.data);
                for (size_t j = 0; j < p; ++j) direction_.data[j] += temporary_.data[j];
            }
            for (size_t j = 0; j < p; ++j) direction_.data[j] -= gamma_[j];
            return false;
        }
        ++b_.stats.singular_steps;
        auto started = Clock::now();
        auto workspace = b_.account(multiply(24 * multiply(p, p) + 64 * p, sizeof(double)));
        Matrix factor(rows_, p);
        for (size_t j = 0; j < p; ++j) for (size_t i = 0; i < rows_; ++i) factor(i, j) = R(i, j);
        Eigen::JacobiSVD<Matrix> svd(factor, Eigen::ComputeThinU | Eigen::ComputeFullV);
        if (svd.info() != Eigen::Success) throw NonfiniteError("SVD failed in sig_lasso");
        const auto& singular = svd.singularValues();
        const double threshold = singular.size() ? eps * std::max(n(), p) * singular[0] : 0;
        size_t rank = 0;
        while (rank < static_cast<size_t>(singular.size()) && singular[rank] > threshold) ++rank;
        Vector penalty(p);
        for (size_t j = 0; j < p; ++j) penalty[j] = penalty_at(j);
        Vector projected = svd.matrixV().rightCols(p - rank).transpose() * penalty;
        const bool null_step = projected.norm() > 64 * eps * penalty.norm();
        if (null_step) {
            Eigen::Map<Vector>(direction_.data, p) = -svd.matrixV().rightCols(p - rank) * projected;
        } else {
            Vector value = svd.matrixU().leftCols(rank).transpose() * Eigen::Map<Vector>(projected_.data, rows_);
            Vector signed_penalty = svd.matrixV().leftCols(rank).transpose() * penalty;
            for (size_t j = 0; j < rank; ++j) value[j] = value[j] / singular[j] - signed_penalty[j] / (singular[j] * singular[j]);
            Eigen::Map<Vector>(direction_.data, p) = svd.matrixV().leftCols(rank) * value;
            for (size_t j = 0; j < p; ++j) direction_.data[j] -= gamma_[j];
        }
        b_.stats.factorization_seconds += elapsed(started);
        return null_step;
    }

    void erase(size_t position) {
        auto started = Clock::now();
        const size_t p = size();
        for (size_t j = position; j + 1 < p; ++j) for (size_t i = 0; i < rows_; ++i) R(i, j) = R(i, j + 1);
        for (size_t j = position; j + 1 < rows_ && j < p - 1; ++j) {
            const double radius = std::hypot(R(j, j), R(j + 1, j));
            const double c = radius ? R(j, j) / radius : 1, s = radius ? R(j + 1, j) / radius : 0;
            for (size_t k = j; k + 1 < p; ++k) {
                const double first = R(j, k), second = R(j + 1, k);
                R(j, k) = c * first + s * second; R(j + 1, k) = -s * first + c * second;
            }
            temporary_.data[j] = c; direction_.data[j] = s;
            const double first = projected_.data[j], second = projected_.data[j + 1];
            projected_.data[j] = c * first + s * second; projected_.data[j + 1] = -s * first + c * second;
        }
        for (size_t begin = 0; begin < n(); begin += 128)
            for (size_t j = position; j + 1 < rows_ && j < p - 1; ++j) {
                const double c = temporary_.data[j], s = direction_.data[j];
                for (size_t i = begin; i < std::min(n(), begin + 128); ++i) {
                    const double first = Q(j)[i], second = Q(j + 1)[i];
                    Q(j)[i] = c * first + s * second; Q(j + 1)[i] = -s * first + c * second;
                }
            }
        columns_.erase(columns_.begin() + position); slots_.erase(slots_.begin() + position);
        gamma_.erase(gamma_.begin() + position); signs_.erase(signs_.begin() + position); scales_.erase(scales_.begin() + position);
        rows_ = std::min(n(), size());
        ++b_.stats.drops; ++b_.stats.qr_deletions;
        b_.stats.factorization_seconds += elapsed(started);
    }

public:
    CPUActive(Backend& b, const Array& y, double alpha) : b_(b), y_(y), residual_(b.clone(y)), trial_(b.alloc(y.size)), alpha_(alpha) {}
    size_t size() const override { return slots_.size(); }
    const Array& residual() const override { return residual_; }

    void activate(const Column& column, size_t slot, double correlation) override {
        auto started = Clock::now();
        reserve(size() + 1);
        const size_t p = size();
        const double scale = std::sqrt(column.norm2);
        if (!(scale > 0)) throw std::logic_error("Cannot activate a zero signature column");
        b_.copy(column.values.data, trial_.data, n()); b_.scale(1 / scale, trial_.data, n());
        for (size_t j = 0; j < stride(); ++j) R(j, p) = 0;
        for (size_t pass = 0; pass < 2; ++pass) for (size_t j = 0; j < rows_; ++j) {
            const double projection = b_.dot(Q(j), trial_.data, n());
            R(j, p) += projection; b_.axpy(-projection, Q(j), trial_.data, n());
        }
        if (p < n()) {
            const double norm = std::sqrt(std::max(0.0, b_.dot(trial_.data, trial_.data, n())));
            R(p, p) = norm;
            if (norm > eps * std::max(n(), p + 1)) {
                b_.copy(trial_.data, Q(p), n()); b_.scale(1 / norm, Q(p), n());
                projected_.data[p] = b_.dot(Q(p), y_.data, n());
            } else { std::fill_n(Q(p), n(), 0); projected_.data[p] = 0; rebuild_ = true; }
            ++rows_;
        }
        columns_.push_back(column); slots_.push_back(slot); gamma_.push_back(0);
        signs_.push_back(std::copysign(1.0, correlation)); scales_.push_back(scale);
        ++b_.stats.activations; ++b_.stats.qr_appends;
        b_.stats.factorization_seconds += elapsed(started);
    }

    Correction correct(bool force_rebuild) override {
        if (!size()) return Correction::stationary;
        if (force_rebuild || rebuild_) rebuild();
        const double previous = b_.dot(residual_.data, residual_.data, n()) / (2 * n()) + alpha_ * penalty();
        for (size_t attempt = 0; attempt < 2; ++attempt) {
            const bool null_step = direction();
            double crossing = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < size(); ++j) if (signs_[j] * direction_.data[j] < 0)
                crossing = std::min(crossing, std::max(0.0, -gamma_[j] / direction_.data[j]));
            const double step = null_step ? crossing : std::min(1.0, crossing);
            if (!std::isfinite(step)) return Correction::failed;
            for (size_t j = 0; j < size(); ++j) {
                const double ratio = signs_[j] * direction_.data[j] < 0 ? -gamma_[j] / direction_.data[j] : std::numeric_limits<double>::infinity();
                temporary_.data[j] = ratio <= step + 16 * eps * std::max(1.0, std::abs(step)) ? 0 : finite(gamma_[j] + step * direction_.data[j]);
            }
            residual_for(temporary_.data, trial_);
            double l1 = 0;
            for (size_t j = 0; j < size(); ++j) l1 += std::abs(temporary_.data[j] / scales_[j]);
            const double squared = b_.dot(trial_.data, trial_.data, n());
            const double objective = finite(squared / (2 * n()) + alpha_ * l1);
            bool accurate = true;
            if (!null_step && step == 1) for (size_t j = 0; j < size(); ++j) if (temporary_.data[j]) {
                const double error = b_.dot(columns_[j].values.data, trial_.data, n()) / scales_[j] - penalty_at(j);
                accurate = accurate && std::abs(error) <= 64 * eps * (n() + size()) * (1 + std::sqrt(squared) + std::abs(penalty_at(j)));
            }
            if (accurate && objective <= previous + 256 * eps * std::max(1.0, previous)) {
                for (size_t j = 0; j < size(); ++j) gamma_[j] = temporary_.data[j];
                b_.copy(trial_.data, residual_.data, n());
                bool dropped = false;
                for (size_t j = size(); j-- > 0;) if (!gamma_[j]) { erase(j); dropped = true; }
                return dropped || step < 1 || null_step ? Correction::boundary : Correction::stationary;
            }
            if (attempt == 0) { rebuild(); refresh(false); }
        }
        return Correction::failed;
    }

    void refresh(bool rounded) override {
        if (rounded && b_.options.single_precision) for (size_t j = 0; j < size(); ++j)
            gamma_[j] = finite(static_cast<float>(gamma_[j] / scales_[j])) * scales_[j];
        residual_for(gamma_.data(), residual_);
    }

    double penalty() override {
        double result = 0;
        for (size_t j = 0; j < size(); ++j) result += std::abs(gamma_[j] / scales_[j]);
        return finite(result);
    }

    void dual_residual(Array& out) override {
        b_.copy(residual_.data, out.data, n());
        if (!size() || !full_rank()) return;
        for (size_t pass = 0; pass < 5; ++pass) {
            for (size_t j = 0; j < size(); ++j) temporary_.data[j] = penalty_at(j) - b_.dot(columns_[j].values.data, out.data, n()) / scales_[j];
            lower(temporary_.data);
            for (size_t j = 0; j < size(); ++j) b_.axpy(temporary_.data[j], Q(j), out.data, n());
        }
    }

    Selection select(const std::vector<Column>& columns, const std::vector<size_t>& pool) override {
        selected_.clear();
        for (auto slot : pool) if (columns[slot].norm2 && std::find(slots_.begin(), slots_.end(), slot) == slots_.end()) selected_.push_back(slot);
        b_.correlations(columns, residual_, scores_, &selected_);
        Selection best;
        for (size_t j = 0; j < selected_.size(); ++j) {
            const size_t slot = selected_[j];
            if (std::abs(scores_[j]) <= alpha_ * (1 + 32 * eps)) continue;
            if (best.slot == static_cast<size_t>(-1) || std::abs(scores_[j]) > std::abs(best.correlation)
                || (std::abs(scores_[j]) == std::abs(best.correlation) && word_less(columns[slot].word, columns[best.slot].word))) best = {slot, scores_[j]};
        }
        return best;
    }

    void output(std::vector<size_t>& slots, std::vector<double>& beta) override {
        slots = slots_; beta.resize(size());
        for (size_t j = 0; j < size(); ++j) beta[j] = gamma_[j] / scales_[j];
    }
};
} // namespace

std::unique_ptr<ActiveSet> make_cpu_active(Backend& b, const Array& y, double alpha) { return std::make_unique<CPUActive>(b, y, alpha); }
} // namespace siglasso
