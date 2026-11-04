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
#include <LBFGSB.h>
#include <map>

namespace siglasso {
namespace {
using Vector = Eigen::VectorXd;
struct Term {
    Column column;
    double beta, scale;
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
        if (norm2 > 0)
            terms.push_back({{word, values, mean, norm2, b.account(192 + 16 * word.size())}, 0, std::sqrt(y.size / norm2)});
    }
    double loss(const double* base = nullptr) {
        double total = 0;
        const double inv = 1. / y.size;
        for (size_t i = 0; i < y.size; ++i) {
            const double delta = eta.data[i], origin = base ? base[i] : 0., z = origin + delta;
            const double target = y.data[i], e = z - target;
            double g;
            switch (b.options.loss) {
            case 0:
                total += base ? delta * (origin - target + .5 * delta) : .5 * e * e;
                g = e;
                break;
            case 1: {
                const double t = std::exp(-std::abs(z));
                if (base && std::abs(delta) < .5) {
                    const double t0 = std::exp(-std::abs(origin));
                    const double p0 = origin >= 0 ? 1 / (1 + t0) : t0 / (1 + t0);
                    total += std::log1p(p0 * std::expm1(delta)) - target * delta;
                } else {
                    total += std::max(z, 0.) - target * z + std::log1p(t);
                    if (base)
                        total -= std::max(origin, 0.) - target * origin + std::log1p(std::exp(-std::abs(origin)));
                }
                g = (z >= 0 ? 1 / (1 + t) : t / (1 + t)) - target;
                break;
            }
            case 2: {
                double mu = std::exp(z);
                total +=
                    base ? (std::abs(delta) < .5 ? std::exp(origin) * std::expm1(delta) : mu - std::exp(origin)) - target * delta
                         : mu - target * z;
                g = mu - target;
                break;
            }
            default: {
                g = std::clamp(e, -1., 1.);
                const double e0 = origin - target, s0 = std::clamp(e0, -1., 1.);
                if (base && s0 == g)
                    total += delta * g;
                else if (base && e0 == s0 && e == g)
                    total += delta * (e0 + .5 * delta);
                else
                    total += g * (e - .5 * g) - (base ? s0 * (e0 - .5 * s0) : 0.);
                break;
            }
            }
            gradient.data[i] = g * inv;
        }
        return total * inv;
    }
    double evaluate(const Vector& u, Vector& deriv, const Vector& origin, const Array& prediction) {
        const size_t k = terms.size(), offset = has_intercept();
        b.fill(eta, offset ? u[0] - origin[0] : 0);
        double penalty = 0;
        for (size_t j = 0; j < k; ++j) {
            auto& t = terms[j];
            const double plus = u[offset + j] - origin[offset + j], minus = u[offset + k + j] - origin[offset + k + j];
            b.axpy(t.scale * (plus - minus), t.column.values.data, eta.data, y.size);
            penalty += t.scale * (plus + minus);
        }
        double result = loss(prediction.data) + alpha * penalty;
        if (offset)
            deriv[0] = std::accumulate(gradient.data, gradient.data + y.size, 0.);
        for (size_t j = 0; j < k; ++j) {
            auto& t = terms[j];
            double corr = b.dot(t.column.values.data, gradient.data, y.size);
            deriv[offset + j] = t.scale * (alpha + corr);
            deriv[offset + k + j] = t.scale * (alpha - corr);
        }
        return result;
    }
    void refresh(bool rounded) {
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
        value = finite(loss() + alpha * penalty);
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
    bool solve(double tau) {
        size_t k = terms.size(), offset = has_intercept(), m = 2 * k + offset;
        refresh(false);
        if (!m || (intercept_error <= b.options.tolerance * .5 && active_error <= tau * .5))
            return true;
        auto workspace = b.account(multiply(160 * m + 4096, sizeof(double)));
        LBFGSpp::LBFGSBParam<double> param;
        param.m = 10;
        param.past = 0;
        param.epsilon = std::min({1e-9, b.options.tolerance * .05, tau * .1});
        param.epsilon_rel = 0;
        param.max_iterations = 2000;
        param.max_linesearch = 60;
        LBFGSpp::LBFGSBSolver<double> solver(param);
        Vector u(m), low = Vector::Zero(m), high = Vector::Constant(m, std::numeric_limits<double>::infinity());
        if (offset)
            low[0] = -std::numeric_limits<double>::infinity();
        auto prediction = b.alloc(y.size);
        for (size_t attempt = 0; attempt < 3; ++attempt) {
            if (offset)
                u[0] = intercept;
            for (size_t j = 0; j < k; ++j) {
                u[offset + j] = std::max(0., terms[j].beta) / terms[j].scale;
                u[offset + k + j] = std::max(0., -terms[j].beta) / terms[j].scale;
            }
            refresh(false);
            if (intercept_error <= b.options.tolerance * .5 && active_error <= tau * .5)
                return true;
            b.copy(eta.data, prediction.data, y.size);
            const Vector origin = u;
            auto objective = [&](const Vector& x, Vector& g) { return evaluate(x, g, origin, prediction); };
            double fx = 0;
            try {
                solver.minimize(objective, u, fx, low, high);
            } catch (const std::runtime_error&) { /* Check the last finite iterate before restarting. */
            }
            if (!u.allFinite())
                return false;
            if (offset)
                intercept = u[0];
            for (size_t j = 0; j < k; ++j)
                terms[j].beta = terms[j].scale * (u[offset + j] - u[offset + k + j]);
        }
        refresh(false);
        return intercept_error <= b.options.tolerance * .5 && active_error <= tau * .5;
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

Result fit(Backend& b, Array& paths, uint32_t mode, const std::vector<double>& alphas, const std::vector<Word>& initial) {
    const auto& o = b.options;
    if (o.loss > 3 || !o.samples || !o.length || !o.dimension || !o.degree)
        throw std::invalid_argument("Invalid working-set options");
    const size_t degree = (mode && o.dimension == 1) ? 1 : o.degree;
    bool packed = false;
    std::unique_ptr<FeatureSearch> full = make_design(b, paths, degree, mode, packed);
    std::unique_ptr<FeatureSearch> preliminary;
    if (!packed && degree >= 4 && paths.owner) {
        const size_t d = o.dimension;
        size_t stage = d <= (2 << 20) / 8 && d * (1 + d) <= (2 << 20) / 8 / o.samples ? 2 : 1;
        bool small;
        if (std::all_of(initial.begin(), initial.end(), [&](const Word& w) { return w.size() <= stage; })) {
            auto engine = b.stats.engine;
            preliminary = make_design(b, paths, stage, mode, small, 2u << 20);
            b.stats.engine = engine;
        }
    }
    if (full)
        paths = {};
    else {
        b.prepare();
        full = b.make_search();
        b.stats.engine = 3;
    }
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
            s.add(w, full->selected(w)->values);
        FeatureSearch* search = preliminary ? preliminary.get() : full.get();
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
            if (last)
                search = full.get();
            start = Clock::now();
            auto found = search->query(s.residual, s.alpha + tau * .9, active, capacity);
            double query_seconds = elapsed(start);
            if (found.candidates.empty() && search != full.get()) {
                b.stats.search_seconds += query_seconds;
                search = full.get();
                if (target + 1 == o.targets)
                    preliminary.reset();
                continue;
            }
            if (last || found.candidates.empty()) {
                if (o.single_precision) {
                    s.refresh(true);
                    found = full->query(s.residual, s.alpha + tau * .9, active, capacity);
                }
                inactive = found.omitted;
                for (auto& c : found.candidates)
                    inactive = std::max(inactive, guard(c.score));
                if (!std::isfinite(inactive))
                    inactive = full->query(s.residual, s.alpha + tau * .9, active, 0).omitted;
                raw_inactive = inactive;
                const double mean_bound = mode ? full->mean_bound() : 1.;
                if (o.intercept && s.intercept_error > 0 && std::isfinite(mean_bound))
                    raw_inactive += mean_bound * s.intercept_error;
                if (o.intercept && s.intercept_error > 0 && !std::isfinite(mean_bound)) {
                    auto raw = b.alloc(o.samples);
                    for (size_t i = 0; i < o.samples; ++i)
                        raw.data[i] = -s.gradient.data[i] * o.samples;
                    raw_inactive = full->query(raw, s.alpha + tau * .9, active, 0).omitted;
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
