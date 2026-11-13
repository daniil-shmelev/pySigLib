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
#include "cp_lasso_acceleration.h"
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
    Array previous_gram;
    std::vector<Word> gram_words;
    std::shared_ptr<void> gram_metadata;
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
    double priority(double coef, double g, double h, double penalty) const {
        if (!(h > 0)) return 0;
        const double u = h * coef - g;
        const double next = std::copysign(std::max(0., std::abs(u) - penalty), u) / h;
        const double delta = next - coef;
        const double decrease = -g * delta - .5 * h * delta * delta + penalty * (std::abs(coef) - std::abs(next));
        return std::isfinite(decrease) ? std::max(0., decrease) : std::numeric_limits<double>::infinity();
    }
    void save_iterate(double* out) const {
        if (has_intercept()) *out++ = intercept;
        for (const auto& t : terms) *out++ = t.beta;
    }
    void restore_iterate(const double* in) {
        if (has_intercept()) intercept = *in++;
        for (auto& t : terms) t.beta = *in++;
    }
    Array acceleration_history(size_t k) {
        if (!(b.options.restricted_solver & 1) || k < 2 || multiply(12 * sizeof(double), k) > b.budget->available()) return {};
        auto history = b.alloc(multiply(12, k));
        save_iterate(history.data);
        return history;
    }
    void accelerate(const Array& history, size_t& age, const Array& curvature = {}, const Array& gram_values = {}, const Array& corr = {}) {
        if (!history.owner) return;
        const size_t k = terms.size() + has_intercept();
        save_iterate(history.at(++age * k));
        if (age < 5) return;
        ++b.optimization_stats[AndersonAttempts];
        age = 0;
        double* differences = history.at(6 * k);
        for (size_t j = 0; j < 5 * k; ++j) differences[j] = history.data[j + k] - history.data[j];
        std::array<double, 25> gram{};
        std::array<double, 5> weights{};
        for (size_t i = 0; i < 5; ++i) for (size_t j = 0; j <= i; ++j)
            gram[i * 5 + j] = gram[j * 5 + i] = b.dot(differences + i * k, differences + j * k, k);
        if (anderson_weights(gram.data(), weights.data())) {
            double* candidate = history.at(11 * k);
            std::fill_n(candidate, k, 0.);
            for (size_t j = 0; j < 5; ++j) b.axpy(weights[j], history.at((j + 1) * k), candidate, k);
            if (std::all_of(candidate, candidate + k, [](double x) { return std::isfinite(x); })) {
                if (gram_values.owner) {
                    double* delta = differences;
                    double* change = differences + k;
                    double penalty = 0, magnitude = 0;
                    for (size_t j = 0; j < k; ++j) {
                        delta[j] = candidate[j] - history.data[5 * k + j];
                        if (j >= static_cast<size_t>(has_intercept())) {
                            const double term = alpha * (std::abs(candidate[j]) - std::abs(history.data[5 * k + j]));
                            penalty += term; magnitude += std::abs(term);
                        }
                    }
                    for (size_t j = 0; j < k; ++j) {
                        change[j] = b.dot(gram_values.at(j * k), delta, k);
                        magnitude += std::abs(delta[j] * corr.data[j]) + .5 * std::abs(delta[j] * change[j]);
                    }
                    const double decrease = b.dot(delta, corr.data, k) + .5 * b.dot(delta, change, k) + penalty;
                    if (std::isfinite(decrease) && decrease < -128. * k * std::numeric_limits<double>::epsilon() * magnitude) {
                        restore_iterate(candidate);
                        b.axpy(1., change, corr.data, k);
                        ++b.optimization_stats[AndersonAccepted];
                    }
                    save_iterate(history.data);
                    return;
                }
                refresh(false, curvature);
                const double previous_value = value;
                restore_iterate(candidate);
                bool accepted = false;
                try {
                    refresh(false, curvature);
                    accepted = value < previous_value;
                } catch (const NonfiniteError&) {}
                if (accepted) ++b.optimization_stats[AndersonAccepted];
                else {
                    restore_iterate(history.at(5 * k));
                    refresh(false, curvature);
                }
            }
        }
        save_iterate(history.data);
    }
    bool covariance_sweeps(size_t k, double feature_tol, double intercept_tol) {
        const auto started = Clock::now();
        auto gram = b.alloc(multiply(k, k));
        auto corr = b.alloc(k);
        const size_t offset = has_intercept();
        const double inv = 1. / y.size;
        auto index_metadata = previous_gram.owner ? b.account(multiply(k, sizeof(size_t))) : std::shared_ptr<void>{};
        std::vector<size_t> previous(previous_gram.owner ? k : 0, gram_words.size());
        if (previous_gram.owner) for (size_t j = 0; j < k; ++j) {
            const Word& word = j < offset ? Word{} : terms[j - offset].column.word;
            auto found = std::find(gram_words.begin(), gram_words.end(), word);
            previous[j] = static_cast<size_t>(found - gram_words.begin());
        }
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l <= j; ++l) {
                double entry;
                if (!previous.empty() && previous[j] < gram_words.size() && previous[l] < gram_words.size()) {
                    entry = previous_gram.data[previous[j] * gram_words.size() + previous[l]];
                    ++b.optimization_stats[GramReusedEntries];
                } else if (l < offset)
                    entry = j < offset ? 1. : inv * std::accumulate(terms[j-offset].column.values.data,
                                                                  terms[j-offset].column.values.data+y.size, 0.);
                else
                    entry = inv * b.dot(terms[j-offset].column.values.data, terms[l-offset].column.values.data, y.size);
                gram.data[j*k+l] = gram.data[l*k+j] = entry;
            }
        }
        ++b.optimization_stats[GramBuilds];
        b.optimization_stats[GramSeconds] += elapsed(started);
        if (b.options.restricted_solver & 2) {
            previous_gram = gram;
            gram_words.clear();
            gram_metadata = b.account(multiply(k, 64 + 8 * b.options.degree));
            if (offset) gram_words.push_back({});
            for (const auto& t : terms) gram_words.push_back(t.column.word);
        }
        auto history = acceleration_history(k);
        const bool ordered = b.options.restricted_solver & 8;
        auto order_metadata = ordered ? b.account(multiply(k, sizeof(std::pair<double, size_t>))) : std::shared_ptr<void>{};
        std::vector<std::pair<double, size_t>> order(ordered ? k : 0);
        size_t age = 0;
        for (size_t sweep = 0; sweep < 250000;) {
            corr.data[0] = std::accumulate(gradient.data, gradient.data+y.size, 0.);
            for (size_t j = offset; j < k; ++j)
                corr.data[j] = b.dot(terms[j-offset].column.values.data, gradient.data, y.size);
            for (size_t inner = 0; inner < 256 && sweep < 250000; ++inner) {
                ++sweep;
                ++b.experiment_stats[InnerSweeps];
                if (ordered) {
                    for (size_t j = 0; j < k; ++j)
                        order[j] = {-priority(j < offset ? intercept : terms[j-offset].beta, corr.data[j], gram.data[j*k+j], j < offset ? 0. : alpha), j};
                    std::sort(order.begin(), order.end());
                }
                for (size_t index = 0; index < k; ++index) {
                    const size_t j = ordered ? order[index].second : index;
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
                accelerate(history, age, {}, gram, corr);
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
    bool solve(double tau, double intercept_tol) {
        const auto curvature = b.options.loss == 1 || b.options.loss == 2 ? b.alloc(y.size) : Array{};
        const double fraction = b.options.single_precision ? .1 : .5;
        auto history = acceleration_history(terms.size() + has_intercept());
        const bool ordered = b.options.restricted_solver & 8;
        auto order_metadata = ordered ? b.account(multiply(terms.size(), sizeof(std::pair<double, size_t>))) : std::shared_ptr<void>{};
        std::vector<std::pair<double, size_t>> order(ordered ? terms.size() : 0);
        size_t age = 0;
        for (size_t sweeps = 0;;) {
            refresh(false, curvature);
            if (intercept_error <= intercept_tol * fraction && active_error <= tau * fraction)
                return true;
            const size_t k = terms.size() + has_intercept();
            const bool switch_gram = b.options.restricted_solver & 16 ? sweeps >= (k + 1) / 2 : sweeps == 65;
            if ((switch_gram || (previous_gram.owner && (b.options.restricted_solver & 4))) && b.options.loss == 0 && k > 0 && k <= y.size &&
                k + 1 <= b.budget->available() / sizeof(double) / k) {
                history = {};
                return covariance_sweeps(k, tau * fraction, intercept_tol * fraction);
            }
            if (sweeps == 250000)
                return false;
            if (ordered) {
                for (size_t j = 0; j < terms.size(); ++j) {
                    const auto& t = terms[j];
                    double h = t.column.norm2 / y.size;
                    if (curvature.owner) {
                        h = 0;
                        for (size_t i = 0; i < y.size; ++i) h += curvature.data[i] * t.column.values.data[i] * t.column.values.data[i];
                        h /= y.size;
                    }
                    order[j] = {-priority(t.beta, b.dot(t.column.values.data, gradient.data, y.size), h, alpha), j};
                }
                std::sort(order.begin(), order.end());
            }
            const size_t count = sweeps ? std::min<size_t>(4, 250000 - sweeps) : 1;
            for (size_t pass = 0; pass < count; ++pass) {
                if (has_intercept())
                    coordinate(intercept, nullptr, 1., curvature);
                for (size_t index = 0; index < terms.size(); ++index) {
                    auto& t = terms[ordered ? order[index].second : index];
                    coordinate(t.beta, &t.column, t.max_abs, curvature);
                }
                accelerate(history, age, curvature);
            }
            sweeps += count;
            b.experiment_stats[InnerSweeps] += count;
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
    if (((o.experiment & 5) && mode) || ((o.experiment & 4) && o.loss))
        throw std::invalid_argument("Experimental certificates require signature features; permanent screening requires squared loss");
    if ((o.experiment & 4) && experimental_candidate_pool.load())
        throw std::invalid_argument("Permanent screening requires complete correlation queries without candidate pooling");
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
        search->new_target();
        const auto failures = b.stats.numerical_failures;
        Restricted s{b, b.response(target), b.alloc(o.samples), b.alloc(o.samples), b.alloc(o.samples), {}, alphas[target]};
        const double tau = std::min(o.tolerance, 1e-4 * s.alpha);
        double inner_tol = o.experiment & 2 ? std::max(tau, .1 * s.alpha) : tau;
        double best_dual = 0, best_scale = 0, best_stamp = 0, best_offset = 0;
        s.intercept = s.initial_intercept();
        for (const auto& w : initial)
            s.add(w, search->selected(w)->values);
        uint64_t updates = 0;
        size_t capacity = 16;
        const size_t candidate_bytes = multiply(o.samples, sizeof(double)) + 256 + 32 * degree;
        size_t pool_capacity = std::min<size_t>(experimental_candidate_pool.load(), b.budget->available() / candidate_bytes / 4);
        if (pool_capacity < 16) pool_capacity = 0;
        const double pool_refresh = experimental_pool_refresh.load();
        double last_excess = 0;
        std::vector<Candidate> pool;
        auto pool_metadata = pool_capacity ? b.account(multiply(pool_capacity, 256 + 32 * degree)) : std::shared_ptr<void>{};
        bool converged = false;
        double inactive = 0, raw_inactive = 0;
        for (;;) {
            auto start = Clock::now();
            if (updates >= o.max_iter) inner_tol = tau;
            bool solved = s.solve(inner_tol, std::max(o.tolerance, inner_tol));
            b.stats.solve_seconds += elapsed(start);
            size_t before = s.terms.size();
            std::erase_if(s.terms, [](const Term& t) { return !t.beta; });
            b.stats.drops += before - s.terms.size();
            std::unordered_map<Word, size_t, WordHash> active;
            for (size_t j = 0; j < s.terms.size(); ++j)
                active.emplace(s.terms[j].column.word, j);
            bool last = !solved || updates >= o.max_iter;
            b.active_corr_cap = guard(s.alpha + s.active_error + s.intercept_error);
            start = Clock::now();
            SearchResult found;
            if (pool_capacity && !last && !pool.empty()) {
                std::vector<Column> columns;
                for (const auto& c : pool) columns.push_back({{}, c.values, 0, 0});
                std::vector<double> scores;
                b.correlations(columns, s.residual, scores);
                for (size_t j = 0; j < pool.size(); ++j) {
                    auto& c = pool[j];
                    c.score = finite(std::abs(scores[j]) * b.weights[c.word.size()]);
                    if (!active.count(c.word) && c.score > s.alpha + tau * .9) found.candidates.push_back(c);
                }
                std::sort(found.candidates.begin(), found.candidates.end(), better);
                if (!found.candidates.empty() && found.candidates.front().score - s.alpha < pool_refresh * last_excess)
                    found.candidates.clear();
            }
            if (found.candidates.empty()) {
                found = search->query(s.residual, s.alpha + tau * .9, active, pool_capacity ? pool_capacity : capacity);
                if (pool_capacity) pool = found.candidates;
                last_excess = found.candidates.empty() ? 0 : std::max(0., found.candidates.front().score - s.alpha);
            }
            if (o.experiment & 4) {
                double maximum = std::max(found.omitted, b.active_corr_cap);
                for (const auto& c : found.candidates) maximum = std::max(maximum, guard(c.score));
                const double mean = o.intercept ? std::accumulate(s.residual.data, s.residual.data + o.samples, 0.) / o.samples : 0;
                const double scale = s.alpha / std::max(s.alpha, guard(maximum + std::abs(mean)));
                double dual = 0, magnitude = 0;
                for (size_t i = 0; i < o.samples; ++i) {
                    const double theta = scale * (s.residual.data[i] - mean) / o.samples;
                    const double term = theta * (s.y.data[i] - .5 * o.samples * theta);
                    dual += term; magnitude += std::abs(term);
                }
                dual -= 512 * std::numeric_limits<double>::epsilon() * (1 + magnitude + std::abs(s.value));
                if (dual > best_dual) {
                    best_dual = dual; best_scale = scale; best_stamp = search->stamp();
                    best_offset = scale * std::abs(mean) / std::sqrt(static_cast<double>(o.samples));
                }
                const double gap = guard(std::max(0., s.value - best_dual));
                search->screen(best_scale, guard(std::sqrt(2 * gap / o.samples) + best_offset), best_stamp, s.alpha);
            }
            if (!last && found.candidates.empty() && inner_tol > tau) {
                inner_tol = tau; ++b.experiment_stats[AdaptiveRefinements];
                b.stats.search_seconds += elapsed(start);
                continue;
            }
            if ((o.experiment & 4) && (last || found.candidates.empty())) {
                if (o.single_precision) { s.refresh(true); b.active_corr_cap = guard(s.alpha + s.active_error + s.intercept_error); }
                ++b.experiment_stats[FinalAudits];
                found = search->verify(s.residual, s.alpha + tau * .9, active, capacity);
                if (!last && !found.candidates.empty()) ++b.experiment_stats[AuditRepairs];
            }
            double query_seconds = elapsed(start);
            if (last || found.candidates.empty()) {
                if (o.single_precision) {
                    s.refresh(true);
                    b.active_corr_cap = guard(s.alpha + s.active_error + s.intercept_error);
                    found = o.experiment & 4 ? search->verify(s.residual, s.alpha + tau * .9, active, capacity)
                                            : search->query(s.residual, s.alpha + tau * .9, active, capacity);
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
            if (o.experiment & 2) {
                const double violation = std::max({found.candidates.front().score - s.alpha, s.active_error, s.intercept_error});
                inner_tol = std::max(tau, std::min(inner_tol, .1 * violation));
            }
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
