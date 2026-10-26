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
#include "sig_lasso_math.h"
#include <cfloat>
#include <math_constants.h>

namespace lasso_device {

struct Interval { uint64_t low, high, end; };
struct Active { uint64_t hash, length, offset; };
struct Seen { uint64_t *hashes, *lengths, *words; unsigned long long* count; size_t capacity, degree; };
__device__ bool record(Seen table, const uint64_t* word, size_t length) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t j = 0; j < length; ++j) hash = (hash ^ word[j]) * 0x100000001b3ULL;
    if (!hash) hash = 1;
    for (size_t probe = 0; probe < table.capacity; ++probe) {
        const size_t slot = (hash + probe) & (table.capacity - 1);
        const auto previous = atomicCAS(reinterpret_cast<unsigned long long*>(table.hashes + slot), 0ULL, hash);
        if (!previous) {
            for (size_t j = 0; j < length; ++j) table.words[slot * table.degree + j] = word[j];
            __threadfence(); atomicExch(reinterpret_cast<unsigned long long*>(table.lengths + slot), length);
            atomicAdd(table.count, 1ULL); return true;
        }
        if (previous == hash) {
            uint64_t stored;
            do { stored = atomicAdd(reinterpret_cast<unsigned long long*>(table.lengths + slot), 0ULL); } while (!stored);
            bool same = stored == length;
            if (same) for (size_t j = 0; j < length; ++j) same = same && table.words[slot * table.degree + j] == word[j];
            if (same) return true;
        }
    }
    return false;
}
__global__ void rehash(Seen previous, Seen next) {
    for (size_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < previous.capacity; slot += static_cast<size_t>(gridDim.x) * blockDim.x)
        if (previous.lengths[slot]) record(next, previous.words + slot * previous.degree, previous.lengths[slot]);
}
__global__ void seed_seen(Seen table, const Active* words, const uint64_t* letters, size_t count) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += static_cast<size_t>(gridDim.x) * blockDim.x)
        record(table, letters + words[j].offset, words[j].length);
}
struct Lane {
    uint64_t root, depth, count, evaluated, pruned, queries, cache_hits, requested;
    double omitted;
};
struct SearchData {
    size_t n, l, d, m, lanes, capacity, blocks;
    const double *increments, *ratios, *weights, *residual;
    const double* const* factors;
    const double* const* grams;
    const Interval* intervals;
    Seen exclusions;
    double *coefficients, *moments, *integrals, *values, *candidate_values, *scores;
    uint64_t *words, *next, *candidate_words, *candidate_lengths;
    Lane* states;
    Seen seen;
    double* prefix_cache;
    uint64_t *cache_words, *cache_lengths, *cache_ticks, *cache_clocks;
    size_t cache_slots;
    uint64_t* lane_ids;
    bool segment_scan;
};

struct CandidateBatch { double *values, *scores; uint64_t *words, *lengths, *count; };

__device__ bool word_before(const uint64_t* a, size_t na, const uint64_t* b, size_t nb) {
    if (na != nb) return na < nb;
    for (size_t j = 0; j < na; ++j) if (a[j] != b[j]) return a[j] < b[j];
    return false;
}

__global__ void merge_candidates(SearchData data, size_t lanes, CandidateBatch previous, CandidateBatch out) {
    __shared__ double scores[128];
    __shared__ size_t sources[128];
    if (!threadIdx.x) *out.count = 0;
    for (size_t rank = 0; rank < data.capacity; ++rank) {
        size_t best = SIZE_MAX; double score = -CUDART_INF;
        for (size_t source = threadIdx.x; source < (lanes + 1) * data.capacity; source += blockDim.x) {
            const bool old = source < data.capacity;
            const size_t slot = old ? source : source - data.capacity;
            if (old ? slot >= *previous.count : slot % data.capacity >= data.states[slot / data.capacity].count) continue;
            const double value = old ? previous.scores[slot] : data.scores[slot];
            if (!isfinite(value)) continue;
            bool better = value > score;
            if (!better && value == score && best != SIZE_MAX) {
                const bool old_best = best < data.capacity; const size_t index = old_best ? best : best - data.capacity;
                better = word_before((old ? previous.words : data.candidate_words) + slot * data.m,
                    (old ? previous.lengths : data.candidate_lengths)[slot],
                    (old_best ? previous.words : data.candidate_words) + index * data.m,
                    (old_best ? previous.lengths : data.candidate_lengths)[index]);
            }
            if (better) { best = source; score = value; }
        }
        sources[threadIdx.x] = best; scores[threadIdx.x] = score; __syncthreads();
        for (size_t stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                const size_t a = sources[threadIdx.x], b = sources[threadIdx.x + stride];
                bool better = scores[threadIdx.x + stride] > scores[threadIdx.x];
                if (!better && b != SIZE_MAX && scores[threadIdx.x + stride] == scores[threadIdx.x]) {
                    const bool oa = a < data.capacity, ob = b < data.capacity;
                    const size_t ia = oa ? a : a - data.capacity, ib = ob ? b : b - data.capacity;
                    better = a == SIZE_MAX || word_before((ob ? previous.words : data.candidate_words) + ib * data.m,
                        (ob ? previous.lengths : data.candidate_lengths)[ib], (oa ? previous.words : data.candidate_words) + ia * data.m,
                        (oa ? previous.lengths : data.candidate_lengths)[ia]);
                }
                if (better) { sources[threadIdx.x] = b; scores[threadIdx.x] = scores[threadIdx.x + stride]; }
            }
            __syncthreads();
        }
        const size_t source = sources[0];
        if (source == SIZE_MAX) break;
        const bool old = source < data.capacity; const size_t slot = old ? source : source - data.capacity;
        const size_t length = (old ? previous.lengths : data.candidate_lengths)[slot];
        for (size_t i = threadIdx.x; i < data.n; i += blockDim.x)
            out.values[rank * data.n + i] = (old ? previous.values : data.candidate_values)[slot * data.n + i];
        for (size_t j = threadIdx.x; j < length; j += blockDim.x)
            out.words[rank * data.m + j] = (old ? previous.words : data.candidate_words)[slot * data.m + j];
        if (!threadIdx.x) {
            out.scores[rank] = scores[0]; out.lengths[rank] = length; *out.count = rank + 1;
            (old ? previous.scores : data.scores)[slot] = -CUDART_INF;
        }
        __syncthreads();
    }
}

__device__ double sum(double value, double* scratch) {
    scratch[threadIdx.x] = value; __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const double total = scratch[0]; __syncthreads(); return total;
}
__device__ bool is_active(const SearchData& data, const uint64_t* word, size_t length) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < length; ++k) hash = (hash ^ word[k]) * 0x100000001b3ULL;
    if (!hash) hash = 1;
    const auto table = data.exclusions;
    for (size_t probe = 0; probe < table.capacity; ++probe) {
        const size_t slot = (hash + probe) & (table.capacity - 1);
        if (!table.hashes[slot]) return false;
        if (table.hashes[slot] != hash || table.lengths[slot] != length) continue;
        bool same = true;
        for (size_t k = 0; k < length; ++k) same = same && word[k] == table.words[slot * table.degree + k];
        if (same) return true;
    }
    return false;
}

__global__ void initialise(SearchData data, size_t first_root) {
    const size_t lane = blockIdx.x;
    const size_t nl = data.n * data.l;
    double* coefficients = data.coefficients + lane * data.m * (data.m + 1) / 2 * nl;
    double* integrals = data.integrals + lane * data.m * nl;
    double* moments = data.moments + lane * data.m * (data.m + 1) / 2 * nl;
    for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
        coefficients[index] = 1; integrals[index] = 1;
        for (size_t q = 0; q < data.m; ++q) moments[q * nl + index] = data.factors[0][q * nl + index];
    }
    if (!threadIdx.x) {
        data.lane_ids[lane] = lane;
        data.states[lane] = {first_root + lane, 1, 0, 0, 0, 0, 0, 0, 0};
        size_t block = 0;
        while (data.intervals[block].high - data.intervals[block].low > 1) {
            const auto interval = data.intervals[block];
            const size_t middle = (interval.low + interval.high) / 2;
            block += first_root + lane < middle ? 1 : 2 * (middle - interval.low);
        }
        data.next[lane * data.m] = block;
    }
}

__global__ void compact_lanes(SearchData data, size_t count) {
    const unsigned int lane = threadIdx.x;
    const bool active = lane < count && data.states[lane].depth != 0;
    const unsigned int mask = __ballot_sync(0xffffffffU, active);
    if (active) data.lane_ids[__popc(mask & ((1U << lane) - 1))] = lane;
}

__global__ void traverse(SearchData data, double alpha, size_t steps) {
    __shared__ double scratch[128], threshold;
    __shared__ size_t depth, block, slot, cache_slot;
    __shared__ bool accepted, cache_hit;
    const size_t lane = data.lane_ids[blockIdx.x], nl = data.n * data.l;
    const size_t triangular = data.m * (data.m + 1) / 2;
    double* coefficients = data.coefficients + lane * triangular * nl;
    double* moments = data.moments + lane * triangular * nl;
    double* integrals = data.integrals + lane * data.m * nl;
    double* values = data.values + lane * data.n;
    uint64_t* word = data.words + lane * data.m;
    uint64_t* next = data.next + lane * data.m;
    Lane* state = data.states + lane;
    if (!threadIdx.x) { depth = state->depth; state->requested = 0; }
    __syncthreads();
    for (size_t step = 0; step < steps && depth; ++step) {
        if (!threadIdx.x) {
            while (depth && next[depth - 1] >= data.blocks) --depth;
            if (depth) block = next[depth - 1];
            threshold = alpha;
            if (data.capacity && state->count == data.capacity) {
                for (size_t j = 0; j < state->count; ++j) {
                    if (!j || data.scores[lane * data.capacity + j] < threshold) threshold = data.scores[lane * data.capacity + j];
                }
                threshold = fmax(alpha, threshold);
            }
        }
        __syncthreads();
        if (!depth) break;
        const auto interval = data.intervals[block];
        const size_t k = depth - 1;
        const double* prefix = coefficients + k * (k + 1) / 2 * nl;
        const double* prefix_integrals = integrals + k * nl;
        const double* prefix_moments = moments + k * (2 * data.m - k + 1) / 2 * nl;
        double bound = 0;
        for (size_t q = data.m - k; q-- > 0;) {
            double total = 0;
            for (size_t index = threadIdx.x; index < nl; index += blockDim.x)
                total += fabs(data.residual[index % data.n]) * prefix_moments[q * nl + index] * data.ratios[block * nl + index];
            total = sum(total, scratch);
            if (q == 0 && total >= threshold && interval.high - interval.low > 1) {
                double energy = 0;
                for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) energy += fabs(data.residual[i]) * siglasso::energy_sample(
                    prefix_integrals, data.grams[block], data.increments, interval.low, interval.high, i, data.n, data.l, data.d);
                const double tightened = sum(energy, scratch) * data.weights[k + 1] / data.n;
                total = isfinite(tightened) ? fmin(total, tightened) : CUDART_INF;
            }
            bound = isfinite(total) ? fmax(bound, total) : CUDART_INF;
            if (bound >= threshold) break;
        }
        bound = bound * (1 + 1e-10) + 1e-15;
        if (!threadIdx.x) {
            ++state->queries;
            if (!isfinite(bound)) { state->omitted = CUDART_INF; depth = 0; }
            else if (bound < threshold) {
                ++state->pruned; state->omitted = fmax(state->omitted, bound);
                next[k] = depth == 1 ? data.blocks : interval.end;
            } else next[k] = depth == 1 ? data.blocks : block + 1;
        }
        __syncthreads();
        if (!depth) break;
        if (bound < threshold || interval.high - interval.low > 1) continue;
        if (depth < data.m && !data.factors[depth]) {
            if (!threadIdx.x) { next[k] = block; state->requested = depth; }
            break;
        }
        if (!threadIdx.x) {
            word[k] = interval.low; cache_hit = false; cache_slot = lane * data.cache_slots;
            if (data.cache_slots && depth < data.m) for (size_t j = lane * data.cache_slots; j < (lane + 1) * data.cache_slots; ++j) {
                bool same = data.cache_lengths[j] == depth;
                if (same) for (size_t h = 0; h < depth; ++h) same = same && data.cache_words[j * data.m + h] == word[h];
                if (same) { cache_slot = j; cache_hit = true; break; }
                if (data.cache_ticks[j] < data.cache_ticks[cache_slot]) cache_slot = j;
            }
        }
        __syncthreads();
        double* child = depth < data.m ? coefficients + depth * (depth + 1) / 2 * nl : nullptr;
        const size_t cache_stride = (data.m + 2) * nl + data.n;
        double* cached = data.prefix_cache ? data.prefix_cache + cache_slot * cache_stride : nullptr;
        if (cache_hit) {
            for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) values[i] = cached[cache_stride - data.n + i];
            for (size_t index = threadIdx.x; index < (depth + 1) * nl; index += blockDim.x) child[index] = cached[index];
        } else if (data.segment_scan) {
            const unsigned int warp_lane = threadIdx.x % 32;
            for (size_t i = threadIdx.x / 32; i < data.n; i += blockDim.x / 32) {
                double carry = 0;
                for (size_t begin = 0; begin < data.l; begin += 32) {
                    const size_t j = begin + warp_lane;
                    double inclusive = j < data.l ? data.increments[(j * data.d + interval.low) * data.n + i] * prefix_integrals[j * data.n + i] : 0;
                    for (unsigned int offset = 1; offset < 32; offset *= 2) {
                        const double previous = __shfl_up_sync(0xffffffffU, inclusive, offset);
                        if (warp_lane >= offset) inclusive += previous;
                    }
                    const double previous = __shfl_up_sync(0xffffffffU, inclusive, 1);
                    double value = carry + (warp_lane ? previous : 0);
                    if (child && j < data.l) {
                        child[j * (k + 2) * data.n + i] = value;
                        const double increment = data.increments[(j * data.d + interval.low) * data.n + i] / (k + 1);
                        for (size_t a = 0; a <= k; ++a) {
                            value += increment * prefix[(j * (k + 1) + a) * data.n + i];
                            child[(j * (k + 2) + a + 1) * data.n + i] = value;
                        }
                    }
                    carry += __shfl_sync(0xffffffffU, inclusive, 31);
                }
                if (!warp_lane) values[i] = carry;
            }
        } else for (size_t i = threadIdx.x; i < data.n; i += blockDim.x)
            siglasso::child_sample(data.increments, prefix, prefix_integrals, child, values, interval.low, k, i, data.n, data.l, data.d);
        __syncthreads();
        double correlation = 0;
        for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) correlation += values[i] * data.residual[i];
        correlation = fabs(sum(correlation, scratch)) * data.weights[depth] / data.n;
        if (!threadIdx.x) {
            if (cache_hit) ++state->cache_hits; else ++state->evaluated;
            accepted = false;
            slot = SIZE_MAX - 1;
            if (!isfinite(correlation)) { state->omitted = CUDART_INF; depth = 0; }
            else if (!record(data.seen, word, depth)) { next[k] = block; slot = SIZE_MAX; }
            else if (!is_active(data, word, depth)) {
                state->omitted = fmax(state->omitted, correlation * (1 + 64 * DBL_EPSILON) + 1e-15);
                if (data.capacity && correlation > alpha && correlation >= threshold) {
                    slot = state->count;
                    const bool fresh = slot < data.capacity;
                    if (slot < data.capacity) ++state->count;
                    else {
                        slot = 0;
                        for (size_t j = 1; j < data.capacity; ++j) {
                            const size_t a = lane * data.capacity + j, b = lane * data.capacity + slot;
                            if (data.scores[a] < data.scores[b] || (data.scores[a] == data.scores[b] && word_before(
                                data.candidate_words + b * data.m, data.candidate_lengths[b], data.candidate_words + a * data.m, data.candidate_lengths[a]))) slot = j;
                        }
                    }
                    const size_t index = lane * data.capacity + slot;
                    if (fresh || correlation > data.scores[index] || word_before(word, depth, data.candidate_words + index * data.m, data.candidate_lengths[index])) {
                        data.scores[index] = correlation; data.candidate_lengths[index] = depth;
                        for (size_t j = 0; j < depth; ++j) data.candidate_words[index * data.m + j] = word[j];
                        accepted = true;
                    }
                }
            }
        }
        __syncthreads();
        if (!depth) break;
        if (slot == SIZE_MAX) break;
        if (accepted) for (size_t i = threadIdx.x; i < data.n; i += blockDim.x)
            data.candidate_values[(lane * data.capacity + slot) * data.n + i] = values[i];
        if (depth < data.m) {
            double* child_integrals = integrals + depth * nl;
            double* child_moments = moments + depth * (2 * data.m - depth + 1) / 2 * nl;
            if (cache_hit) {
                for (size_t index = threadIdx.x; index < nl; index += blockDim.x) child_integrals[index] = cached[(depth + 1) * nl + index];
                for (size_t index = threadIdx.x; index < (data.m - depth) * nl; index += blockDim.x) child_moments[index] = cached[(depth + 2) * nl + index];
            } else for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
                const size_t i = index % data.n, j = index / data.n;
                double integral = 0;
                for (size_t a = 0; a <= depth; ++a) integral += child[(j * (depth + 1) + a) * data.n + i];
                child_integrals[index] = integral / (depth + 1);
                for (size_t q = 0; q < data.m - depth; ++q) child_moments[q * nl + index] = siglasso::moment_value(
                    child, data.factors[depth], depth, q, j, i, data.n, data.l);
            }
            __syncthreads();
            if (data.cache_slots) {
                if (!cache_hit) {
                    for (size_t index = threadIdx.x; index < (depth + 1) * nl; index += blockDim.x) cached[index] = child[index];
                    for (size_t index = threadIdx.x; index < nl; index += blockDim.x) cached[(depth + 1) * nl + index] = child_integrals[index];
                    for (size_t index = threadIdx.x; index < (data.m - depth) * nl; index += blockDim.x) cached[(depth + 2) * nl + index] = child_moments[index];
                    for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) cached[cache_stride - data.n + i] = values[i];
                }
                __syncthreads();
                if (!threadIdx.x) {
                    data.cache_lengths[cache_slot] = depth;
                    for (size_t j = 0; j < depth; ++j) data.cache_words[cache_slot * data.m + j] = word[j];
                    data.cache_ticks[cache_slot] = ++data.cache_clocks[lane];
                }
            }
            __syncthreads();
            if (!threadIdx.x) { next[depth] = 0; ++depth; }
        }
        __syncthreads();
    }
    if (!threadIdx.x) state->depth = depth;
}

} // namespace lasso_device
