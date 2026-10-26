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
#include "sig_lasso_core.h"

namespace siglasso {
inline void append_column(Backend& b, std::vector<Column>& columns, const Word& word, const Array& raw) {
    auto metadata = b.account(1024 + word.size() * 64 + b.options.targets * 32);
    Column column{word, b.column_storage()}; column.metadata = std::move(metadata);
    b.copy(raw.data, column.values.data, raw.size);
    b.scale(b.weights[word.size()], column.values.data, column.values.size);
    column.mean = b.center(column.values.data, column.values.size, b.options.intercept != 0);
    column.norm2 = finite(b.dot(column.values.data, column.values.data, column.values.size));
    columns.push_back(std::move(column));
}

inline Result fit(Backend& backend, const std::vector<double>& alphas, const std::vector<Word>& initial) {
    std::vector<Column> columns;
    std::unordered_map<Word, size_t, WordHash> known;
    std::unique_ptr<Search> search;
    if (backend.options.full) backend.full_columns(columns);
    else { backend.prepare(); search = std::make_unique<Search>(backend); }
    for (size_t j = 0; j < columns.size(); ++j) known.emplace(columns[j].word, j);
    if (search) for (const auto& word : initial) if (!known.count(word)) {
        auto node = search->selected(word);
        known.emplace(word, columns.size()); append_column(backend, columns, word, node->values);
    }
    backend.finish(); backend.stats.preparation_seconds = elapsed(backend.started);
    Result result; result.targets = backend.options.targets;
    result.intercepts.resize(result.targets); result.objectives.resize(result.targets); result.gaps.resize(result.targets);
    result.converged.resize(result.targets); result.iterations.resize(result.targets);
    std::vector<std::vector<size_t>> selected_slots(result.targets);
    std::vector<std::vector<double>> coefficients(result.targets);
    struct Response {
        std::unique_ptr<Backend> worker;
        Backend* backend = nullptr;
        Array y, dual, best_theta;
        std::unique_ptr<ActiveSet> state;
        std::vector<size_t> pool;
        std::unordered_set<size_t> offered;
        double mean = 0, alpha = 0, tolerance = 0;
        double objective = 0, gap = std::numeric_limits<double>::infinity();
        uint64_t iterations = 0;
        bool converged = false, pending = false, force_rebuild = false, stopped = false, done = false;
        size_t repairs = 0;
    };
    std::vector<Response> responses(result.targets);
    for (size_t target = 0; target < result.targets; ++target) {
        auto& response = responses[target];
        response.worker = backend.solver_worker(); response.backend = response.worker ? response.worker.get() : &backend;
        auto& b = *response.backend;
        response.y = b.response(target); response.alpha = alphas[target];
        response.mean = b.center(response.y.data, response.y.size, b.options.intercept != 0);
        response.tolerance = b.options.tolerance * std::max(1.0, b.dot(response.y.data, response.y.data, response.y.size) / response.y.size);
        response.state = b.active_set(response.y, response.alpha);
        response.dual = b.alloc(response.y.size); response.best_theta = b.alloc(response.y.size);
        if (search) for (size_t j = 0; j < columns.size(); ++j) { response.pool.push_back(j); response.offered.insert(j); }
        else for (const auto& word : initial) if (response.offered.insert(known.at(word)).second) response.pool.push_back(known.at(word));
    }
    auto certify = [&](Response& response) {
        auto& b = backend;
        auto& [worker, owner, y, dual, best_theta, state, pool, offered, mean, alpha, tolerance, objective, gap,
            iterations, converged, pending, force_rebuild, stopped, done, repairs] = response;
        auto maximum = [&](const Array& vector, double threshold) {
            double bound = b.correlation_max(columns, vector);
            if (search) bound = std::max(bound, search->query(vector, threshold, known, 0).omitted);
            ++b.stats.certificate_passes;
            return finite(bound);
        };
        auto started = Clock::now();
        state->refresh(true);
        const auto& residual = state->residual();
        objective = finite(b.dot(residual.data, residual.data, y.size) / (2 * y.size) + alpha * state->penalty());
        double best_value = 0;
        b.fill(best_theta, 0);
        for (size_t candidate = 0; candidate < 2; ++candidate) {
            if (!candidate) b.copy(residual.data, dual.data, y.size);
            else state->dual_residual(dual);
            if (b.options.intercept) b.center(dual.data, dual.size, true);
            double bound = std::max(alpha, maximum(dual, alpha));
            b.scale(alpha / (bound * y.size), dual.data, dual.size);
            bool feasible = false;
            for (size_t attempt = 0; attempt < 3; ++attempt) {
                bound = maximum(dual, alpha / y.size) * y.size;
                if (bound <= alpha) { feasible = true; break; }
                b.scale(alpha / (bound * (1 + 32 * std::numeric_limits<double>::epsilon())), dual.data, dual.size);
            }
            if (feasible) {
                const double value = finite(b.dot(y.data, dual.data, y.size) - 0.5 * y.size * b.dot(dual.data, dual.data, y.size));
                if (value > best_value) { best_value = value; b.copy(dual.data, best_theta.data, y.size); }
            }
            gap = std::max(0.0, objective - best_value) + 16 * std::numeric_limits<double>::epsilon() * (std::abs(objective) + std::abs(best_value));
            if (gap <= tolerance) break;
        }
        b.stats.certification_seconds += elapsed(started);
        return gap <= tolerance;
    };
    size_t remaining = responses.size();
    while (remaining) {
        const auto phase_started = Clock::now();
        backend.for_targets([&](size_t target) {
            auto& response = responses[target];
            if (response.done || response.stopped) return;
            auto& b = *response.backend;
            auto& [worker, owner, y, dual, best_theta, state, pool, offered, mean, alpha, tolerance, objective, gap,
                iterations, converged, pending, force_rebuild, stopped, done, repairs] = response;
            for (;;) {
                if (pending) {
                    if (iterations == b.options.max_iter) { stopped = true; break; }
                    const auto status = state->correct(force_rebuild);
                    ++iterations; force_rebuild = false;
                    if (status == Correction::failed) { stopped = true; repairs = SIZE_MAX; ++b.stats.numerical_failures; break; }
                    if (status == Correction::boundary) continue;
                    pending = false;
                }
                Selection chosen = state->select(columns, pool);
                if (chosen.slot == static_cast<size_t>(-1)) break;
                if (iterations == b.options.max_iter) { stopped = true; break; }
                state->activate(columns[chosen.slot], chosen.slot, chosen.correlation);
                pending = true;
            }
        });
        backend.stats.solve_seconds += elapsed(phase_started);
        for (size_t target = 0; target < responses.size(); ++target) {
            auto& response = responses[target];
            if (response.done) continue;
            auto& [worker, owner, y, dual, best_theta, state, pool, offered, mean, alpha, tolerance, objective, gap,
                iterations, converged, pending, force_rebuild, stopped, done, repairs] = response;
            if (!stopped) {
                if (state->select(columns, pool).slot != static_cast<size_t>(-1)) continue;
                auto started = Clock::now();
                if (search) {
                    auto found = search->query(state->residual(), alpha, known, backend.options.batch_size);
                    backend.stats.search_seconds += elapsed(started); started = Clock::now();
                    for (const auto& candidate : found.candidates) if (!known.count(candidate.word)) {
                        const size_t slot = columns.size();
                        known.emplace(candidate.word, slot); append_column(backend, columns, candidate.word, candidate.values);
                        for (auto& other : responses) if (!other.done) { other.pool.push_back(slot); other.offered.insert(slot); }
                    }
                    backend.stats.insertion_seconds += elapsed(started);
                } else {
                    const auto found = backend.dense_candidates(columns, state->residual(), offered, alpha, backend.options.batch_size);
                    for (auto slot : found) { pool.push_back(slot); offered.insert(slot); }
                    backend.stats.search_seconds += elapsed(started);
                }
                if (state->select(columns, pool).slot != static_cast<size_t>(-1)) continue;
            }
            converged = certify(response) && repairs != SIZE_MAX;
            if (!stopped && !converged && state->size() && !repairs++ && iterations < backend.options.max_iter) {
                pending = force_rebuild = true;
                continue;
            }
            state->output(selected_slots[target], coefficients[target]);
            double intercept = mean;
            for (size_t j = 0; j < selected_slots[target].size(); ++j) intercept -= columns[selected_slots[target][j]].mean * coefficients[target][j];
            if (backend.options.single_precision) {
                const double rounded = finite(static_cast<float>(intercept));
                const double loss = 0.5 * (rounded - intercept) * (rounded - intercept);
                objective += loss; gap += loss; intercept = rounded;
                converged = converged && gap <= tolerance;
            }
            result.intercepts[target] = intercept; result.objectives[target] = objective; result.gaps[target] = gap;
            result.converged[target] = converged; result.iterations[target] = iterations;
            state.reset(); dual = {}; best_theta = {}; y = {}; done = true; --remaining;
        }
    }
    for (const auto& response : responses) if (response.worker) {
        const auto& counter = response.worker->stats;
        backend.stats.activations += counter.activations; backend.stats.drops += counter.drops;
        backend.stats.qr_appends += counter.qr_appends; backend.stats.qr_deletions += counter.qr_deletions;
        backend.stats.qr_rebuilds += counter.qr_rebuilds; backend.stats.singular_steps += counter.singular_steps;
        backend.stats.factorization_seconds += counter.factorization_seconds;
        backend.stats.numerical_failures += counter.numerical_failures;
    }
    std::vector<size_t> selected;
    for (size_t target = 0; target < result.targets; ++target) for (size_t j = 0; j < selected_slots[target].size(); ++j)
        if (coefficients[target][j]) selected.push_back(selected_slots[target][j]);
    std::sort(selected.begin(), selected.end()); selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    std::sort(selected.begin(), selected.end(), [&](size_t a, size_t b) { return word_less(columns[a].word, columns[b].word); });
    std::unordered_map<size_t, size_t> positions;
    for (size_t j = 0; j < selected.size(); ++j) { positions.emplace(selected[j], j); result.words.push_back(columns[selected[j]].word); }
    result.coefficients.resize(multiply(selected.size(), result.targets));
    for (size_t target = 0; target < result.targets; ++target) for (size_t j = 0; j < selected_slots[target].size(); ++j)
        if (coefficients[target][j]) result.coefficients[positions.at(selected_slots[target][j]) * result.targets + target] = coefficients[target][j];
    backend.finish(); backend.stats.columns = columns.size(); backend.stats.peak_bytes = backend.budget->peak(); backend.stats.total_seconds = elapsed(backend.started);
    result.stats = backend.stats;
    return result;
}
} // namespace siglasso
