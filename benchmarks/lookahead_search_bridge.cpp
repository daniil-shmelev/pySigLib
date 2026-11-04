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

// Benchmark-only access to a persistent AVX2 CPU feature search.
#include "cp_sig_lasso_impl.cpp"

namespace {
struct Oracle {
    cpsig::HWY_NAMESPACE::LassoCPU backend;
    std::unique_ptr<siglasso::FeatureSearch> search;
    siglasso::Array residual;

    Oracle(const SigLassoOptions& o, const double* paths, uint32_t mode)
        : backend(o, paths, static_cast<const double*>(nullptr), mode) {
        backend.prepare();
        search = backend.make_search();
        residual = backend.alloc(o.samples);
    }
};
}

extern "C" __declspec(dllexport) void* lookahead_open(const SigLassoOptions* options, const double* paths, uint32_t mode) noexcept {
    try { return new Oracle(*options, paths, mode); } catch (...) { return nullptr; }
}

extern "C" __declspec(dllexport) int64_t lookahead_query(void* handle, const double* residual, double threshold,
    const uint64_t* excluded, size_t count, uint64_t* output, size_t capacity) noexcept {
    try {
        auto& o = *static_cast<Oracle*>(handle);
        const auto d = o.backend.options.dimension;
        std::unordered_map<siglasso::Word, size_t, siglasso::WordHash> active;
        for (size_t j = 0; j < count; ++j) {
            siglasso::Word w;
            for (auto code = excluded[j]; code; code = (code - 1) / d) w.push_back((code - 1) % d);
            std::reverse(w.begin(), w.end());
            active.emplace(std::move(w), j);
        }
        o.backend.copy(residual, o.residual.data, o.residual.size);
        o.backend.center(o.residual.data, o.residual.size, true);
        auto found = o.search->query(o.residual, threshold, active, capacity);
        for (size_t j = 0; j < found.candidates.size(); ++j) {
            uint64_t code = 0;
            for (const auto a : found.candidates[j].word) code = code * d + (a + 1);
            output[j] = code;
        }
        return static_cast<int64_t>(found.candidates.size());
    } catch (...) { return -1; }
}

extern "C" __declspec(dllexport) void lookahead_close(void* handle, SigLassoStats* stats) noexcept {
    std::unique_ptr<Oracle> o(static_cast<Oracle*>(handle));
    *stats = o->backend.stats;
    stats->peak_bytes = o->backend.budget->peak();
}

// Return normalized columns from the search itself. Words have a fixed stride of degree.
extern "C" __declspec(dllexport) int64_t working_set_query(void* handle, const double* gradient, double threshold,
    const uint64_t* words, const uint64_t* lengths, size_t count, uint64_t* out_words, uint64_t* out_lengths,
    double* columns, size_t capacity) noexcept {
    try {
        auto& o = *static_cast<Oracle*>(handle);
        const size_t n = o.backend.options.samples, degree = o.backend.options.degree;
        std::unordered_map<siglasso::Word, size_t, siglasso::WordHash> active;
        for (size_t j = 0; j < count; ++j) {
            if (!lengths[j] || lengths[j] > degree) return -1;
            const auto* start = words + degree * j;
            active.emplace(siglasso::Word(start, start + lengths[j]), j);
        }
        o.backend.copy(gradient, o.residual.data, n);
        o.backend.center(o.residual.data, n, true);
        auto found = o.search->query(o.residual, threshold, active, capacity);
        if (o.backend.uncertain_features || o.backend.stats.numerical_failures) return -1;
        for (size_t j = 0; j < found.candidates.size(); ++j) {
            const auto& c = found.candidates[j];
            const size_t k = c.word.size();
            std::copy(c.word.begin(), c.word.end(), out_words + j * degree);
            out_lengths[j] = k;
            for (size_t i = 0; i < n; ++i) columns[j * n + i] = c.values.data[i] * o.backend.weights[k];
        }
        return static_cast<int64_t>(found.candidates.size());
    } catch (...) { return -1; }
}
