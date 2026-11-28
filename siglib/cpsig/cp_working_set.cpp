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
#include <coroutine>

namespace siglasso {
namespace {
struct TargetFit {
    struct promise_type {
        Separation* request = nullptr;
        std::exception_ptr error;
        TargetFit get_return_object() { return TargetFit(std::coroutine_handle<promise_type>::from_promise(*this)); }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { error = std::current_exception(); }
    };
    std::coroutine_handle<promise_type> handle;
    explicit TargetFit(std::coroutine_handle<promise_type> h) : handle(h) {}
    TargetFit(TargetFit&& other) noexcept : handle(std::exchange(other.handle, {})) {}
    ~TargetFit() { if (handle) handle.destroy(); }
};
struct AwaitSeparation : Separation {
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<TargetFit::promise_type> target) noexcept { target.promise().request = this; }
    SearchResult await_resume() { return std::move(found); }
};
struct Term {
    Column column;
    double beta, max_abs;
    size_t gram_index = SIZE_MAX;
    double gradient_correlation = 0.;
};

// A bounded direct-mapped cache shares Gram entries for live feature columns
// across response restrictions. Equality of both IDs is required on a hit.
struct SharedGram {
    struct Entry { uint64_t key = 0; double value = 0.; };
    std::vector<Entry> slots;
    std::shared_ptr<void> storage;
    uint64_t next_id = 1;
    double entry(Backend& b, const Column& a, const Column& c, size_t n) {
        if (slots.empty()) {
            constexpr size_t capacity = 32768;
            storage = b.account(capacity * sizeof(Entry));
            slots.resize(capacity);
        }
        const auto low = std::min(a.id, c.id), high = std::max(a.id, c.id);
        if (!low || high > UINT32_MAX) return b.gram_dot(a.values.data, c.values.data, n) / n;
        const uint64_t key = (low << 32) | high;
        const size_t index = ((low * 0x9e3779b97f4a7c15ULL) ^ (high * 0xbf58476d1ce4e5b9ULL)) >> 49;
        auto& slot = slots[index];
        if (slot.key == key) return slot.value;
        slot = {key, b.gram_dot(a.values.data, c.values.data, n) / n};
        return slot.value;
    }
};

struct Restricted {
    Backend& b;
    Array y, eta, gradient, residual;
    std::vector<Term> terms;
    double alpha, intercept = 0, value = 0, intercept_error = 0, active_error = 0;
    SharedGram* shared_gram = nullptr;
    Array previous_gram;
    size_t previous_order = 0;
    bool packed_gram = false;
    bool has_intercept() const {
        return b.options.intercept != 0;
    }

    void add(const Word& word, const Array& raw) {
        auto values = b.alloc(y.size);
        b.copy(raw.data, values.data, y.size);
        b.scale(b.weights[word.size()], values.data, y.size);
        const double mean = b.center(values.data, y.size, has_intercept());
        const double norm2 = finite(b.dot(values.data, values.data, y.size));
        if (norm2 > 0) add_column({word, values, mean, norm2, b.account(192 + 16 * word.size()), shared_gram ? shared_gram->next_id++ : 0});
    }
    void add_column(const Column& column) {
        double max_abs = 0;
        if (b.options.loss == 1 || b.options.loss == 2)
            for (size_t i = 0; i < y.size; ++i)
                max_abs = std::max(max_abs, std::abs(column.values.data[i]));
        terms.push_back({column, 0, max_abs});
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
            if (!t.beta) continue;
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
        for (auto& t : terms) {
            const double c = b.dot(t.column.values.data, gradient.data, y.size);
            t.gradient_correlation = c;
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
    void save_iterate(double* out) const {
        if (has_intercept()) *out++ = intercept;
        for (const auto& t : terms) *out++ = t.beta;
    }
    void restore_iterate(const double* in) {
        if (has_intercept()) intercept = *in++;
        for (auto& t : terms) t.beta = *in++;
    }
    Array acceleration_history(size_t k) {
        if (!b.options.accelerate_solver || k < 2 || multiply(12 * sizeof(double), k) > b.budget->available()) return {};
        auto history = b.alloc(multiply(12, k));
        save_iterate(history.data);
        return history;
    }
    void accelerate(const Array& history, size_t& age, const Array& curvature = {}, const Array& gram_values = {}, const Array& corr = {}) {
        if (!history.owner) return;
        const size_t k = terms.size() + has_intercept();
        save_iterate(history.at(++age * k));
        if (age < 5) return;
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
                if (!accepted) {
                    restore_iterate(history.at(5 * k));
                    refresh(false, curvature);
                }
            }
        }
        save_iterate(history.data);
    }
    // Along the Newton direction the smooth loss is quadratic. Each sign
    // crossing increases the one-dimensional penalty derivative by 2 alpha |d_i|.
    bool polish(const Array& gram, const Array& corr) {
        const size_t offset = has_intercept(), k = corr.size;
        const size_t count = offset + std::count_if(terms.begin(), terms.end(), [](const Term& t) { return t.beta != 0; });
        if (!count || count > 512 || (count * count + 4 * count + k) * sizeof(double) > b.budget->available()) return false;
        auto index_memory = b.account(multiply(count, sizeof(size_t) + sizeof(std::pair<double,size_t>)));
        std::vector<size_t> ids; ids.reserve(count);
        if (offset) ids.push_back(0);
        for (size_t j = offset; j < k; ++j) if (terms[j - offset].beta) ids.push_back(j);
        auto matrix = b.alloc(count * count), direction = b.alloc(count);
        for (size_t i = 0; i < count; ++i) {
            const size_t row = ids[i];
            direction.data[i] = -corr.data[row] - (row < offset ? 0. : std::copysign(alpha, terms[row - offset].beta));
            // The column-major Cholesky solver reads only its lower triangle.
            for (size_t j = i; j < count; ++j) matrix.data[i * count + j] = gram.data[row * k + ids[j]];
        }
        if (!solve_positive_definite(matrix.data, direction.data, count)) return false;
        double step = 1.;
        size_t hit = count;
        std::vector<std::pair<double, size_t>> crossings; crossings.reserve(count);
        double curvature = 0.;
        for (size_t i = 0; i < count; ++i) {
            const size_t j = ids[i];
            const double penalty = i < offset ? 0. : std::copysign(alpha, terms[j - offset].beta);
            curvature -= direction.data[i] * (corr.data[j] + penalty);
            if (i < offset) continue;
            const double beta = terms[j - offset].beta, delta = direction.data[i];
            if (std::signbit(beta) != std::signbit(delta) && std::abs(delta) >= std::abs(beta))
                crossings.emplace_back(-beta / delta, i);
        }
        if (!(curvature > 0.) || !std::isfinite(curvature)) return false;
        std::sort(crossings.begin(), crossings.end());
        for (const auto& [time, i] : crossings) {
            if (step < time) break;
            step -= 2 * alpha * std::abs(direction.data[i]) / curvature;
            if (step <= time) { step = time; hit = i; break; }
        }
        auto change = b.alloc(k); b.fill(change, 0.);
        double linear = 0.;
        for (size_t i = 0; i < count; ++i) {
            const size_t j = ids[i];
            const double beta = j < offset ? intercept : terms[j - offset].beta;
            const double next = i == hit ? 0. : beta + step * direction.data[i];
            const double delta = next - beta;
            direction.data[i] = delta;
            linear += delta * corr.data[j] + (j < offset ? 0. : alpha * (std::abs(next) - std::abs(beta)));
            b.axpy(delta, gram.at(j * k), change.data, k);
        }
        double quadratic = 0;
        for (size_t i = 0; i < count; ++i) quadratic += direction.data[i] * change.data[ids[i]];
        const double difference = linear + .5 * quadratic;
        if (!std::isfinite(difference) || difference >= 0) return false;
        for (size_t i = 0; i < count; ++i) {
            double& beta = ids[i] < offset ? intercept : terms[ids[i] - offset].beta;
            beta += direction.data[i];
        }
        b.axpy(1., change.data, corr.data, k);
        return true;
    }
    // A suspended response only needs the symmetric Gram entries for surviving
    // columns. The index follows each Term when zero coefficients are erased.
    void remember_gram(const Array& gram, size_t k) {
        if (!b.options.accelerate_solver) return;
        const size_t offset = has_intercept();
        const size_t live = offset + std::count_if(terms.begin(), terms.end(), [](const Term& t) { return t.beta != 0.; });
        const size_t entries = multiply(live, live + 1) / 2;
        packed_gram = k >= 16 && entries <= b.budget->available() / sizeof(double);
        if (!packed_gram) {
            previous_gram = gram;
            previous_order = k;
            for (size_t j = offset; j < k; ++j) terms[j - offset].gram_index = j;
            return;
        }
        previous_order = live;
        previous_gram = b.alloc(entries);
        if (offset) previous_gram.data[0] = gram.data[0];
        size_t row = offset;
        for (size_t j = offset; j < k; ++j) {
            auto& term = terms[j - offset];
            term.gram_index = term.beta ? row : SIZE_MAX;
            if (!term.beta) continue;
            double* out = previous_gram.data + row * (row + 1) / 2;
            if (offset) *out++ = gram.data[j * k];
            for (size_t l = offset; l <= j; ++l)
                if (terms[l - offset].beta) *out++ = gram.data[j * k + l];
            ++row;
        }
    }
    bool covariance_sweeps(size_t k, double feature_tol, double intercept_tol) {
        auto gram = b.alloc(multiply(k, k));
        auto corr = b.alloc(k);
        const size_t offset = has_intercept();
        const double inv = 1. / y.size;
        for (size_t j = 0; j < k; ++j) {
            const size_t row = j < offset ? 0 : terms[j - offset].gram_index;
            for (size_t l = 0; l <= j; ++l) {
                const size_t col = l < offset ? 0 : terms[l - offset].gram_index;
                double entry;
                if (previous_gram.owner && row < previous_order && col < previous_order) {
                    // All surviving terms retain their relative order.
                    const size_t index = packed_gram ? row * (row + 1) / 2 + col : row * previous_order + col;
                    entry = previous_gram.data[index];
                } else if (l < offset)
                    entry = j < offset ? 1. : inv * std::accumulate(terms[j-offset].column.values.data,
                                                                  terms[j-offset].column.values.data+y.size, 0.);
                else if (j == l) entry = inv * terms[j-offset].column.norm2;
                else if (shared_gram && k >= 112) entry = shared_gram->entry(b, terms[j-offset].column, terms[l-offset].column, y.size);
                else entry = inv * b.gram_dot(terms[j-offset].column.values.data, terms[l-offset].column.values.data, y.size);
                gram.data[j*k+l] = gram.data[l*k+j] = entry;
            }
        }
        previous_gram = {};
        previous_order = 0;
        auto history = acceleration_history(k);
        size_t age = 0;
        for (size_t sweep = 0; sweep < 250000;) {
            corr.data[0] = std::accumulate(gradient.data, gradient.data+y.size, 0.);
            for (size_t j = offset; j < k; ++j)
                corr.data[j] = terms[j-offset].gradient_correlation;
            for (size_t inner = 0; inner < 256 && sweep < 250000; ++inner) {
                ++sweep;
                bool stable_signs = true;
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
                        stable_signs &= (next > 0.) == (coef > 0.) && (next < 0.) == (coef < 0.);
                        coef = next;
                        b.axpy(delta, gram.at(j*k), corr.data, k);
                    }
                }
                if (b.options.accelerate_solver && ((stable_signs && inner < 3) || inner % 8 == 3) && polish(gram, corr)) {
                    age = 0;
                    if (history.owner) save_iterate(history.data);
                } else accelerate(history, age, {}, gram, corr);
                bool done = !offset || std::abs(corr.data[0]) <= intercept_tol;
                for (size_t j = offset; done && j < k; ++j) {
                    const double coef = terms[j-offset].beta, g = corr.data[j];
                    done = (coef ? std::abs(g + std::copysign(alpha, coef)) : std::abs(g)-alpha) <= feature_tol;
                }
                if (done)
                    break;
            }
            refresh(false);
            if (intercept_error <= intercept_tol && active_error <= feature_tol) {
                remember_gram(gram, k);
                return true;
            }
        }
        return false;
    }
    bool solve(double tau, double intercept_tol) {
        const auto curvature = b.options.loss == 1 || b.options.loss == 2 ? b.alloc(y.size) : Array{};
        const double fraction = b.options.single_precision ? .1 : .5;
        auto history = acceleration_history(terms.size() + has_intercept());
        size_t age = 0;
        for (size_t sweeps = 0;;) {
            refresh(false, curvature);
            if (intercept_error <= intercept_tol * fraction && active_error <= tau * fraction)
                return true;
            const size_t k = terms.size() + has_intercept();
            const bool switch_gram = sweeps >= (b.options.accelerate_solver ? 1u : 65u);
            if ((switch_gram || (previous_gram.owner && b.options.accelerate_solver)) && b.options.loss == 0 && k > 0 && k <= y.size &&
                k + 1 <= b.budget->available() / sizeof(double) / k) {
                history = {};
                return covariance_sweeps(k, tau * fraction, intercept_tol * fraction);
            }
            if (sweeps == 250000)
                return false;
            const size_t count = sweeps ? std::min<size_t>(4, 250000 - sweeps) : 1;
            for (size_t pass = 0; pass < count; ++pass) {
                if (has_intercept())
                    coordinate(intercept, nullptr, 1., curvature);
                for (auto& t : terms) {
                    coordinate(t.beta, &t.column, t.max_abs, curvature);
                }
                accelerate(history, age, curvature);
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

Result fit(Backend& b, bool log_features, const std::vector<double>& alphas, const std::vector<Word>& initial,
           FeatureSearch* prepared, const Result* previous) {
    const auto& o = b.options;
    if (o.loss > 3 || !o.samples || !o.length || !o.dimension || !o.degree)
        throw std::invalid_argument("Invalid working-set options");
    const size_t degree = (log_features && o.dimension == 1) ? 1 : o.degree;
    std::unique_ptr<FeatureSearch> owned_search;
    if (!prepared) {
        b.prepare();
        owned_search = b.make_search();
        b.stats.preparation_seconds = elapsed(b.started);
    }
    auto* search = prepared ? prepared : owned_search.get();
    b.stats.engine = 3;
    Result result;
    result.targets = o.targets;
    std::map<Word, Column, decltype(&word_less)> support(&word_less);
    struct SharedColumn {
        std::weak_ptr<void> data, metadata;
        double* values;
        double mean, norm2;
        uint64_t id;
        std::shared_ptr<void> storage;
        Column acquire(const Word& word, size_t n) const {
            auto owner = data.lock();
            if (!owner) return {};
            return {word, {values, n, std::move(owner)}, mean, norm2, metadata.lock(), id};
        }
    };
    std::map<Word, SharedColumn, decltype(&word_less)> live(&word_less);
    const bool share = search->response_batch() > 1;
    SharedGram shared_gram;
    if (previous) shared_gram.next_id = previous->words.size() + 1;
    std::vector<std::vector<std::pair<Word, double>>> coefs(o.targets);
    result.kkt.resize(o.targets); result.gaps.resize(o.targets); result.intercepts.resize(o.targets);
    result.objectives.resize(o.targets); result.converged.resize(o.targets); result.iterations.resize(o.targets);
    auto fit_target = [&](size_t target) -> TargetFit {
        const auto failures = b.stats.numerical_failures;
        Restricted s{b, b.response(target), b.alloc(o.samples), b.alloc(o.samples), b.alloc(o.samples), {}, alphas[target]};
        if (share) s.shared_gram = &shared_gram;
        // Joint shortlists can reach a weakly active coordinate by different
        // routes. Resolve small violations relative to the individual penalty.
        const double certificate_tol = std::min(o.tolerance, 1e-4 * s.alpha);
        const double tau = share ? std::min(certificate_tol, 1e-6 * s.alpha) : certificate_tol;
        s.intercept = s.initial_intercept();
        if (previous) {
            s.intercept = previous->intercepts[target];
            for (size_t j = 0; j < previous->words.size(); ++j) {
                const double beta = previous->coefficients[j * o.targets + target];
                if (!beta) continue;
                auto column = previous->selected_columns[j];
                if (o.order == SignatureOrder::suffix && o.degree > 1)
                    std::reverse(column.word.begin(), column.word.end());
                column.id = j + 1;
                s.add_column(column);
                s.terms.back().beta = beta;
                s.intercept += beta * column.mean;
            }
        } else {
            for (const auto& w : initial)
                s.add(w, search->selected(w)->values);
        }
        uint64_t updates = 0;
        constexpr size_t admission_limit = 16;
        const size_t candidate_bytes = multiply(o.samples, sizeof(double)) + 256 + 32 * degree;
        size_t pool_capacity = std::min<size_t>(256, b.budget->available() / candidate_bytes / 4);
        if (degree == 1) pool_capacity = std::min<size_t>(pool_capacity, std::max<size_t>(admission_limit, o.dimension));
        if (share && o.dimension < 24) pool_capacity = std::min<size_t>(pool_capacity, degree == 2 ? 96 : 160);
        if (share && log_features && o.dimension < 96 && degree >= 3) pool_capacity = std::min<size_t>(pool_capacity, 128);
        if (pool_capacity < admission_limit) pool_capacity = 0;
        std::vector<Candidate> pool;
        auto pool_metadata = pool_capacity ? b.account(multiply(pool_capacity, 256 + 32 * degree)) : std::shared_ptr<void>{};
        bool converged = false;
        double inactive = 0, raw_inactive = 0;
        for (;;) {
            auto start = Clock::now();
            const double inner_tol = updates >= o.max_iter ? certificate_tol : tau;
            bool solved = s.solve(inner_tol, std::max(o.tolerance, inner_tol));
            b.stats.solve_seconds += elapsed(start);
            size_t before = s.terms.size();
            std::erase_if(s.terms, [](const Term& t) { return !t.beta; });
            b.stats.drops += before - s.terms.size();
            ActiveWords active;
            for (size_t j = 0; j < s.terms.size(); ++j)
                active.emplace(s.terms[j].column.word, j);
            bool last = !solved || updates >= o.max_iter;
            start = Clock::now();
            SearchResult found;
            double separation_seconds = 0.;
            if (pool_capacity && !last && !pool.empty()) {
                std::vector<Column> columns;
                for (const auto& c : pool) if (!c.proposal) columns.push_back({{}, c.values, 0, 0});
                std::vector<double> scores;
                if (!columns.empty()) b.correlations(columns, s.residual, scores);
                const double mass = columns.size() == pool.size() ? 0. : b.absolute_sum(s.residual) / o.samples;
                const double mean = columns.size() == pool.size() ? 0. : std::accumulate(s.residual.data, s.residual.data + o.samples, 0.) / o.samples;
                size_t exact = 0;
                for (auto& c : pool) {
                    if (c.proposal) {
                        const auto& packed = *c.proposal;
                        double correlation = 0.;
                        for (size_t i = 0; i < o.samples; ++i) correlation += packed.samples[i] * s.residual.data[i];
                        c.score = std::max(0., std::abs(correlation / o.samples * packed.unit + mean * packed.offset) - mass * packed.error) * b.weights[c.word.size()];
                    } else c.score = finite(std::abs(scores[exact++]) * b.weights[c.word.size()]);
                    if (!active.count(c.word) && c.score > s.alpha + tau * .9) found.candidates.push_back(c);
                }
                std::sort(found.candidates.begin(), found.candidates.end(), better);
            }
            // Previously fitted targets already own centred, scaled columns. Re-score
            // their sparse support before paying for another full tree traversal.
            if (found.candidates.empty() && !last && pool_capacity && !support.empty()) {
                for (const auto& [word, column] : support) {
                    if (active.count(word)) continue;
                    const double score = std::abs(b.dot(column.values.data, s.residual.data, o.samples)) / o.samples;
                    if (score > s.alpha + tau * .9)
                        shortlist(found.candidates, pool_capacity, word, score, [] { return Array{}; });
                }
                std::sort(found.candidates.begin(), found.candidates.end(), better);
            }
            if (share && found.candidates.empty() && !last && pool_capacity) {
                for (auto it = live.begin(); it != live.end();) {
                    const auto& [word, entry] = *it;
                    if (entry.data.expired()) { it = live.erase(it); continue; }
                    ++it;
                    if (active.count(word) || support.count(word)) continue;
                    const double score = std::abs(b.dot(entry.values, s.residual.data, o.samples)) / o.samples;
                    if (score > s.alpha + tau * .9)
                        shortlist(found.candidates, pool_capacity, word, score, [] { return Array{}; });
                }
                std::sort(found.candidates.begin(), found.candidates.end(), better);
            }
            if (found.candidates.empty()) {
                pool.clear();
                AwaitSeparation request{{s.residual, s.alpha, s.alpha + tau * .9, active, pool_capacity ? pool_capacity : admission_limit, {}}};
                request.allow_lower_degree = !last;
                const double local_seconds = elapsed(start);
                found = co_await request;
                separation_seconds = local_seconds + request.seconds;
                start = Clock::now();
                if (pool_capacity) pool = found.candidates;
            }
            double query_seconds = separation_seconds + elapsed(start);
            if (last || found.candidates.empty()) {
                if (o.single_precision) {
                    s.refresh(true);
                    found = search->query(s.residual, s.alpha + tau * .9, active, admission_limit);
                }
                inactive = found.omitted;
                for (auto& c : found.candidates)
                    inactive = std::max(inactive, guard(c.score));
                if (!std::isfinite(inactive))
                    inactive = search->query(s.residual, s.alpha + tau * .9, active, 0).omitted;
                raw_inactive = inactive;
                if (o.intercept && !log_features)
                    raw_inactive += s.intercept_error;
                if (o.intercept && log_features && s.intercept_error > 0)
                    raw_inactive += search->feature_cap() * s.intercept_error;
                if (o.intercept && log_features && s.intercept_error > 0 && raw_inactive > s.alpha + tau) {
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
                converged = solved && (share || b.stats.numerical_failures == failures) &&
                            s.intercept_error <= o.tolerance && s.active_error <= certificate_tol && raw_inactive <= s.alpha + certificate_tol;
                if (!solved)
                    ++b.stats.numerical_failures;
                b.stats.certification_seconds += separation_seconds + elapsed(start);
                ++b.stats.certificate_passes;
                break;
            }
            b.stats.search_seconds += query_seconds;
            start = Clock::now();
            const size_t count = std::min(admission_limit, found.candidates.size());
            for (size_t j = 0; j < count; ++j) {
                auto& c = found.candidates[j];
                const auto known = support.find(c.word);
                if (known != support.end()) s.add_column(known->second);
                else {
                    const auto it = live.find(c.word);
                    Column shared = it == live.end() ? Column{} : it->second.acquire(c.word, o.samples);
                    if (shared.values.owner) s.add_column(shared);
                    else {
                        const size_t before_add = s.terms.size();
                        s.add(c.word, c.values.owner ? c.values : search->selected(c.word, false)->values);
                        if (share && before_add < s.terms.size()) {
                            const auto& col = s.terms.back().column;
                            auto& entry = live[c.word];
                            if (!entry.storage) entry.storage = b.account(sizeof(SharedColumn) + 96 + c.word.size() * sizeof(uint64_t));
                            entry = {col.values.owner, col.metadata, col.values.data, col.mean, col.norm2, col.id, std::move(entry.storage)};
                        }
                    }
                }
                ++b.stats.activations;
            }
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
        result.kkt[target] = std::max({s.intercept_error, s.active_error, raw_inactive - s.alpha, 0.});
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
        result.gaps[target] = gap;
        result.intercepts[target] = intercept;
        result.objectives[target] = s.value;
        result.converged[target] = converged;
        result.iterations[target] = updates;
    };
    const size_t batch = search->response_batch();
    for (size_t first = 0; first < o.targets; first += batch) {
        std::vector<TargetFit> fits;
        for (size_t t = first; t < std::min<size_t>(o.targets, first + batch); ++t) fits.push_back(fit_target(t));
        for (;;) {
            std::vector<Separation*> pending;
            for (auto& fit : fits) if (!fit.handle.done()) {
                fit.handle.resume();
                if (fit.handle.promise().error) std::rethrow_exception(fit.handle.promise().error);
                if (!fit.handle.done()) pending.push_back(fit.handle.promise().request);
            }
            if (pending.empty()) break;
            const auto start = Clock::now();
            search->query_batch(pending);
            const double seconds = elapsed(start) / pending.size();
            for (auto* request : pending) request->seconds = seconds;
        }
    }
    owned_search.reset();
    if (o.order == SignatureOrder::suffix && o.degree > 1) {
        decltype(support) restored(&word_less);
        while (!support.empty()) {
            auto entry = support.extract(support.begin());
            std::reverse(entry.key().begin(), entry.key().end());
            entry.mapped().word = entry.key();
            restored.insert(std::move(entry));
        }
        support.swap(restored);
        for (auto& target : coefs) for (auto& term : target) std::reverse(term.first.begin(), term.first.end());
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
