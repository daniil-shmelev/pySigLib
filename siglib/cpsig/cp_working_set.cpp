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
#include <map>

namespace siglasso {
namespace {
struct Term {
    Column column;
    double beta, max_abs;
};

struct Restricted {
    Backend& b;
    Array y, eta, gradient, residual;
    std::vector<Term> terms;
    double alpha, intercept = 0, value = 0, intercept_error = 0, active_error = 0;
    bool has_intercept() const {
        return b.options.intercept != 0;
    }

    void add(const Word& word, const Array& raw) {
        auto values = b.alloc(y.size);
        b.copy(raw.data, values.data, y.size);
        b.scale(b.weights[word.size()], values.data, y.size);
        const double mean = b.center(values.data, y.size, has_intercept());
        const double norm2 = finite(b.dot(values.data, values.data, y.size));
        if (norm2 > 0) {
            double max_abs = 0;
            if (b.options.loss == 1 || b.options.loss == 2)
                for (size_t i = 0; i < y.size; ++i)
                    max_abs = std::max(max_abs, std::abs(values.data[i]));
            terms.push_back({{word, values, mean, norm2, b.account(192 + 16 * word.size())}, 0, max_abs});
        }
    }
    double update_gradient(const Array& curvature, bool objective) {
        double total = 0;
        const double inv = 1. / y.size;
        for (size_t i = 0; i < y.size; ++i) {
            const double z = eta.data[i], target = y.data[i], e = z - target;
            double g;
            switch (b.options.loss) {
            case 0:
                g = e;
                if (objective)
                    total += .5 * e * e;
                break;
            case 1: {
                const double t = std::exp(-std::abs(z));
                g = target == 0 ? (z >= 0 ? 1. : t) / (1 + t) : -(z >= 0 ? t : 1.) / (1 + t);
                if (curvature.data)
                    curvature.data[i] = t / ((1 + t) * (1 + t));
                if (objective)
                    total += std::max(z, 0.) - target * z + std::log1p(t);
                break;
            }
            case 2: {
                const double mu = std::exp(z);
                g = mu - target;
                if (curvature.data)
                    curvature.data[i] = mu;
                if (objective)
                    total += mu - target * z;
                break;
            }
            default:
                g = std::clamp(e, -1., 1.);
                if (objective)
                    total += g * (e - .5 * g);
                break;
            }
            gradient.data[i] = g * inv;
        }
        return total * inv;
    }
    void refresh(bool rounded, const Array& curvature = {}) {
        double raw_intercept = intercept;
        for (const auto& t : terms)
            raw_intercept -= t.beta * t.column.mean;
        if (rounded)
            raw_intercept = static_cast<float>(raw_intercept);
        b.fill(eta, raw_intercept);
        double shift = 0, penalty = 0;
        for (auto& t : terms) {
            if (rounded)
                t.beta = static_cast<float>(t.beta);
            shift += t.beta * t.column.mean;
            penalty += std::abs(t.beta);
            b.axpy(t.beta, t.column.values.data, eta.data, y.size);
        }
        for (size_t i = 0; i < y.size; ++i)
            eta.data[i] += shift;
        intercept = raw_intercept + shift;
        value = finite(update_gradient(curvature, true) + alpha * penalty);
        double sum = std::accumulate(gradient.data, gradient.data + y.size, 0.);
        intercept_error = has_intercept() ? std::abs(sum) : 0;
        active_error = 0;
        for (const auto& t : terms) {
            const double c = b.dot(t.column.values.data, gradient.data, y.size);
            active_error =
                std::max(active_error, t.beta ? std::abs(c + std::copysign(alpha, t.beta)) : std::max(0., std::abs(c) - alpha));
        }
        for (size_t i = 0; i < y.size; ++i)
            residual.data[i] = -(gradient.data[i] - (has_intercept() ? sum / y.size : 0)) * y.size;
    }
    void coordinate(double& coef, const Column* column, double max_abs, const Array& curvature) {
        const double* x = column ? column->values.data : nullptr;
        const double penalty = column ? alpha : 0.;
        const double g = x ? b.dot(x, gradient.data, y.size) : std::accumulate(gradient.data, gradient.data + y.size, 0.);
        if (coef == 0 && std::abs(g) <= penalty)
            return;
        double h = x ? column->norm2 / y.size : 1.;
        if (curvature.data) {
            double sum = 0;
            for (size_t i = 0; i < y.size; ++i)
                sum += curvature.data[i] * (x ? x[i] * x[i] : 1.);
            h = finite(sum / y.size);
        }
        const double limit = curvature.data ? .75 / max_abs : std::numeric_limits<double>::infinity();
        double next;
        if (h > 0) {
            const double u = h * coef - g;
            next = std::copysign(std::max(0., std::abs(u) - penalty), u) / h;
        } else {
            // In the tails the GLM curvature can underflow to zero.
            next = std::abs(g) <= penalty ? 0. : coef - std::copysign(limit, g);
        }
        double delta = next - coef;
        if (std::abs(delta) > limit) {
            next = coef + std::copysign(limit, delta);
            delta = next - coef;
        }
        finite(delta);
        if (!delta)
            return;
        coef = next;
        if (b.options.loss == 0) {
            // The gradient changes linearly until the next exact refresh.
            if (x)
                b.axpy(delta / y.size, x, gradient.data, y.size);
            else
                for (size_t i = 0; i < y.size; ++i)
                    gradient.data[i] += delta / y.size;
        } else {
            if (x)
                b.axpy(delta, x, eta.data, y.size);
            else
                for (size_t i = 0; i < y.size; ++i)
                    eta.data[i] += delta;
            update_gradient(curvature, false);
        }
    }
    bool covariance_sweeps(size_t k, double feature_tol, double intercept_tol) {
        auto gram = b.alloc(multiply(k, k));
        auto corr = b.alloc(k);
        const size_t offset = has_intercept();
        const double inv = 1. / y.size;
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l <= j; ++l) {
                double entry;
                if (l < offset)
                    entry = j < offset ? 1. : inv * std::accumulate(terms[j-offset].column.values.data,
                                                                  terms[j-offset].column.values.data+y.size, 0.);
                else
                    entry = inv * b.dot(terms[j-offset].column.values.data, terms[l-offset].column.values.data, y.size);
                gram.data[j*k+l] = gram.data[l*k+j] = entry;
            }
        }
        for (size_t sweep = 0; sweep < 250000;) {
            corr.data[0] = std::accumulate(gradient.data, gradient.data+y.size, 0.);
            for (size_t j = offset; j < k; ++j)
                corr.data[j] = b.dot(terms[j-offset].column.values.data, gradient.data, y.size);
            for (size_t inner = 0; inner < 256 && sweep < 250000; ++inner) {
                ++sweep;
                for (size_t j = 0; j < k; ++j) {
                    double& coef = j < offset ? intercept : terms[j-offset].beta;
                    const double penalty = j < offset ? 0. : alpha;
                    if (coef == 0 && std::abs(corr.data[j]) <= penalty)
                        continue;
                    const double h = gram.data[j*k+j];
                    const double u = h * coef - corr.data[j];
                    const double next = std::copysign(std::max(0., std::abs(u)-penalty), u)/h;
                    const double delta = finite(next-coef);
                    if (delta) {
                        coef = next;
                        b.axpy(delta, gram.at(j*k), corr.data, k);
                    }
                }
                bool done = !offset || std::abs(corr.data[0]) <= intercept_tol;
                for (size_t j = offset; done && j < k; ++j) {
                    const double coef = terms[j-offset].beta, g = corr.data[j];
                    done = (coef ? std::abs(g + std::copysign(alpha, coef)) : std::abs(g)-alpha) <= feature_tol;
                }
                if (done)
                    break;
            }
            refresh(false);
            if (intercept_error <= intercept_tol && active_error <= feature_tol)
                return true;
        }
        return false;
    }
    bool solve(double tau) {
        const auto curvature = b.options.loss == 1 || b.options.loss == 2 ? b.alloc(y.size) : Array{};
        const double fraction = b.options.single_precision ? .1 : .5;
        for (size_t sweeps = 0;;) {
            refresh(false, curvature);
            if (intercept_error <= b.options.tolerance * fraction && active_error <= tau * fraction)
                return true;
            const size_t k = terms.size() + has_intercept();
            if (sweeps == 65 && b.options.loss == 0 && k > 0 && k <= y.size &&
                k + 1 <= b.budget->available() / sizeof(double) / k)
                return covariance_sweeps(k, tau * fraction, b.options.tolerance * fraction);
            if (sweeps == 250000)
                return false;
            const size_t count = sweeps ? std::min<size_t>(4, 250000 - sweeps) : 1;
            for (size_t pass = 0; pass < count; ++pass) {
                if (has_intercept())
                    coordinate(intercept, nullptr, 1., curvature);
                for (auto& t : terms)
                    coordinate(t.beta, &t.column, t.max_abs, curvature);
            }
            sweeps += count;
        }
    }
    double initial_intercept() const {
        if (!has_intercept())
            return 0;
        const double mean = std::accumulate(y.data, y.data + y.size, 0.) / y.size;
        if (b.options.loss == 0)
            return mean;
        if (b.options.loss == 1) {
            if (mean == 0 || mean == 1)
                throw std::invalid_argument("Binary logistic loss has no finite intercept for a constant class");
            return std::log(mean) - std::log1p(-mean);
        }
        if (b.options.loss == 2) {
            if (mean == 0)
                throw std::invalid_argument("Poisson loss has no finite intercept for all-zero counts");
            return std::log(mean);
        }
        auto ends = std::minmax_element(y.data, y.data + y.size);
        double lo = *ends.first, hi = *ends.second;
        for (size_t iter = 0; iter < 80 && lo < hi; ++iter) {
            double mid = std::midpoint(lo, hi), sum = 0;
            for (size_t i = 0; i < y.size; ++i)
                sum += std::clamp(mid - y.data[i], -1., 1.);
            if (sum == 0)
                return mid;
            if (sum < 0)
                lo = mid;
            else
                hi = mid;
        }
        return std::midpoint(lo, hi);
    }
};
} // namespace

Result fit(Backend& b, uint32_t mode, const std::vector<double>& alphas, const std::vector<Word>& initial) {
    const auto& o = b.options;
    if (o.loss > 3 || !o.samples || !o.length || !o.dimension || !o.degree)
        throw std::invalid_argument("Invalid working-set options");
    const size_t degree = (mode && o.dimension == 1) ? 1 : o.degree;
    b.prepare();
    auto search = b.make_search();
    b.stats.engine = 3;
    b.stats.preparation_seconds = elapsed(b.started);
    Result result;
    result.targets = o.targets;
    std::map<Word, Column, decltype(&word_less)> support(&word_less);
    std::vector<std::vector<std::pair<Word, double>>> coefs(o.targets);
    for (size_t target = 0; target < o.targets; ++target) {
        const auto failures = b.stats.numerical_failures;
        Restricted s{b, b.response(target), b.alloc(o.samples), b.alloc(o.samples), b.alloc(o.samples), {}, alphas[target]};
        const double tau = std::min(o.tolerance, 1e-4 * s.alpha);
        s.intercept = s.initial_intercept();
        for (const auto& w : initial)
            s.add(w, search->selected(w)->values);
        uint64_t updates = 0;
        size_t capacity = 16;
        bool converged = false;
        double inactive = 0, raw_inactive = 0;
        for (;;) {
            auto start = Clock::now();
            bool solved = s.solve(tau);
            b.stats.solve_seconds += elapsed(start);
            size_t before = s.terms.size();
            std::erase_if(s.terms, [](const Term& t) { return !t.beta; });
            b.stats.drops += before - s.terms.size();
            std::unordered_map<Word, size_t, WordHash> active;
            for (size_t j = 0; j < s.terms.size(); ++j)
                active.emplace(s.terms[j].column.word, j);
            bool last = !solved || updates >= o.max_iter;
            start = Clock::now();
            auto found = search->query(s.residual, s.alpha + tau * .9, active, capacity);
            double query_seconds = elapsed(start);
            if (last || found.candidates.empty()) {
                if (o.single_precision) {
                    s.refresh(true);
                    found = search->query(s.residual, s.alpha + tau * .9, active, capacity);
                }
                inactive = found.omitted;
                for (auto& c : found.candidates)
                    inactive = std::max(inactive, guard(c.score));
                if (!std::isfinite(inactive))
                    inactive = search->query(s.residual, s.alpha + tau * .9, active, 0).omitted;
                raw_inactive = inactive;
                if (o.intercept && !mode)
                    raw_inactive += s.intercept_error;
                if (o.intercept && mode && s.intercept_error > 0) {
                    auto raw = b.alloc(o.samples);
                    for (size_t i = 0; i < o.samples; ++i)
                        raw.data[i] = -s.gradient.data[i] * o.samples;
                    raw_inactive = search->query(raw, s.alpha + tau * .9, active, 0).omitted;
                }
                double sum = std::accumulate(s.gradient.data, s.gradient.data + o.samples, 0.);
                for (const auto& t : s.terms) {
                    double corr = b.dot(t.column.values.data, s.gradient.data, o.samples) + t.column.mean * sum;
                    s.active_error = std::max(s.active_error, t.beta ? std::abs(corr + std::copysign(s.alpha, t.beta))
                                                                     : std::max(0., std::abs(corr) - s.alpha));
                }
                converged = solved && !b.uncertain_features && b.stats.numerical_failures == failures &&
                            s.intercept_error <= o.tolerance && s.active_error <= tau && raw_inactive <= s.alpha + tau;
                if (!solved)
                    ++b.stats.numerical_failures;
                b.stats.certification_seconds += elapsed(start);
                ++b.stats.certificate_passes;
                break;
            }
            b.stats.search_seconds += query_seconds;
            start = Clock::now();
            size_t groups = 0;
            double previous = 0;
            const double tie = 1e-10 * found.candidates.front().score;
            for (auto& c : found.candidates) {
                if (!groups || previous - c.score > tie) {
                    if (++groups <= 16) {
                        s.add(c.word, c.values);
                        ++b.stats.activations;
                    }
                }
                previous = c.score;
            }
            size_t count = found.candidates.size();
            capacity = count == capacity ? std::min<size_t>(64, (16 * count + groups - 1) / groups) : 16;
            ++updates;
            b.stats.insertion_seconds += elapsed(start);
        }
        result.storage.push_back(b.account(multiply(s.terms.size(), 256 + 32 * degree)));
        coefs[target].reserve(s.terms.size());
        double intercept = s.intercept, max_score = inactive;
        for (auto& t : s.terms) {
            if (!t.beta)
                continue;
            intercept -= t.beta * t.column.mean;
            coefs[target].emplace_back(t.column.word, t.beta);
            support.try_emplace(t.column.word, t.column);
            max_score = std::max(max_score, guard(std::abs(b.dot(t.column.values.data, s.residual.data, o.samples)) / o.samples));
        }
        result.kkt.push_back(std::max({s.intercept_error, s.active_error, raw_inactive - s.alpha, 0.}));
        double gap = 0;
        if (o.loss == 0) {
            double scale = 1. / o.samples;
            if (max_score > s.alpha)
                scale *= s.alpha / max_score;
            double dual = 0;
            for (size_t i = 0; i < o.samples; ++i) {
                double theta = s.residual.data[i] * scale;
                dual += theta * (s.y.data[i] - .5 * o.samples * theta);
            }
            gap = guard(std::max(0., s.value - dual));
        }
        result.gaps.push_back(gap);
        result.intercepts.push_back(intercept);
        result.objectives.push_back(s.value);
        result.converged.push_back(converged);
        result.iterations.push_back(updates);
    }
    std::map<Word, size_t, decltype(&word_less)> index(&word_less);
    for (auto& [w, col] : support) {
        index.emplace(w, result.words.size());
        result.words.push_back(w);
        result.selected_columns.push_back(std::move(col));
    }
    const size_t coefficients = multiply(index.size(), o.targets);
    result.storage.push_back(b.account(multiply(coefficients, sizeof(double))));
    result.coefficients.resize(coefficients);
    for (size_t t = 0; t < o.targets; ++t)
        for (auto& [w, beta] : coefs[t])
            result.coefficients[index.at(w) * o.targets + t] = beta;
    b.stats.columns = index.size();
    return result;
}
} // namespace siglasso
