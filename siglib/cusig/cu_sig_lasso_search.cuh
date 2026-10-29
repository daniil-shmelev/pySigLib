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
    uint64_t root, depth, count, evaluated, pruned, queries, cache_hits, requested, cache_writes;
    double omitted;
    uint64_t terminal_queries, terminal_pruned, terminal_correlations;
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
    uint32_t search_method;
    bool intercept;
    const double *variation, *after;
    bool exact;
    double *time_values, *linear_integrals, *integral_scratch;
    size_t scratch_size, cache_stride;
    siglasso::TerminalCache terminal_cache;
    double* terminal_residual;
    double* terminal_products;
    const uint64_t* frontier;
    size_t root_groups;
    const double* reciprocals;
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
    const unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (unsigned int stride = 16; stride; stride /= 2) value += __shfl_down_sync(0xffffffffU, value, stride);
    if (!lane) scratch[warp] = value;
    __syncthreads();
    if (!warp) {
        value = lane < blockDim.x / 32 ? scratch[lane] : 0;
        for (unsigned int stride = 16; stride; stride /= 2) value += __shfl_down_sync(0xffffffffU, value, stride);
        if (!lane) scratch[0] = value;
    }
    __syncthreads();
    const double total = scratch[0]; __syncthreads(); return total;
}
__global__ void prepare_terminal_residual(SearchData data) {
    __shared__ double scratch[128];
    double mean = 0, lower = 0, upper = 0;
    if (data.intercept) {
        for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) mean += data.residual[i];
        mean = sum(mean, scratch) / data.n;
    }
    for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) {
        const double r = fabs(data.residual[i] - mean), v = data.variation[i] + data.after[i];
        if (data.m > 1) lower += r * pow(v, static_cast<double>(data.m - 1));
        upper += r * pow(v, static_cast<double>(data.m));
    }
    lower = sum(lower, scratch); upper = sum(upper, scratch);
    if (!threadIdx.x) {
        data.terminal_residual[0] = mean;
        data.terminal_residual[1] = lower; data.terminal_residual[2] = upper;
    }
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
    double* coefficients = data.coefficients;
    double* endpoints = data.time_values;
    double* integrals = data.integrals;
    double* moments = data.moments;
    for (size_t index = threadIdx.x; lane == 0 && index < nl; index += blockDim.x) {
        if (coefficients) coefficients[index] = 1;
        if (endpoints) endpoints[index] = 1;
        if (integrals) integrals[index] = 1;
        if (data.linear_integrals) data.linear_integrals[index] = .5;
        const double residual = fabs(data.residual[index % data.n]);
        if (data.exact) for (size_t q = 0; q < data.m; ++q) moments[q * nl + index] = residual * data.factors[0][q * nl + index];
        else {
            siglasso::trapezoidal_moments(1, 1, data.variation[index], data.after[index], data.weights[1] / data.n, 0, data.m, nl, moments + index, data.reciprocals);
            for (size_t q = 0; q < data.m; ++q) moments[q * nl + index] *= residual;
        }
    }
    if (endpoints && lane == 0) for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) endpoints[nl + i] = 1;
    if (!threadIdx.x) {
        data.lane_ids[lane] = lane;
        data.states[lane] = {first_root + lane, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        const size_t root = (first_root + lane) / data.root_groups;
        size_t block = (data.search_method & 1) ? 0 : root;
        while (data.intervals[block].high - data.intervals[block].low > 1) {
            const auto interval = data.intervals[block];
            const size_t middle = (interval.low + interval.high) / 2;
            block += root < middle ? 1 : 2 * (middle - interval.low);
        }
        data.next[lane * data.m] = block;
    }
}

__global__ void compact_lanes(SearchData data, size_t count) {
    __shared__ unsigned int counts[4];
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = lane / 32;
    const bool active = lane < count && data.states[lane].depth != 0;
    const unsigned int mask = __ballot_sync(0xffffffffU, active);
    if (lane % 32 == 0) counts[warp] = __popc(mask);
    __syncthreads();
    unsigned int offset = 0;
    for (unsigned int j = 0; j < warp; ++j) offset += counts[j];
    if (active) data.lane_ids[offset + __popc(mask & ((1U << (lane % 32)) - 1))] = lane;
}

template<bool Exact>
__global__ __launch_bounds__(128, Exact ? 2 : 4) void traverse(SearchData data, double alpha, size_t steps) {
    __shared__ double scratch[128], threshold;
    __shared__ size_t depth, block, slot, cache_slot;
    __shared__ bool accepted, cache_hit, terminal_hits[4];
    __shared__ double terminal_bounds[4];
    __shared__ size_t terminal_slots[4];
    const size_t lane = data.lane_ids[blockIdx.x], nl = data.n * data.l;
    const size_t triangular = data.m * (data.m + 1) / 2;
    auto terminal_cache = data.terminal_cache;
    if (terminal_cache.capacity) terminal_cache.data += lane * terminal_cache.capacity * (data.m + 3);
    double* coefficients = Exact ? data.coefficients + lane * (triangular - 1) * nl : nullptr;
    double* endpoints = !Exact ? data.time_values + lane * (data.m - 1) * data.n * (data.l + 1) : nullptr;
    double* moments = data.moments + lane * (triangular - data.m) * nl;
    double* integrals = data.integrals ? data.integrals + lane * (data.m - 1) * nl : nullptr;
    double* linears = data.linear_integrals ? data.linear_integrals + lane * (data.m - 1) * nl : nullptr;
    double* values = data.values + lane * data.n;
    double* products = data.terminal_products ? data.terminal_products + lane * nl : nullptr;
    uint64_t* word = data.words + lane * data.m;
    uint64_t* next = data.next + lane * data.m;
    Lane* state = data.states + lane;
    const size_t group_root = data.frontier[state->root % data.root_groups];
    const size_t group_end = data.root_groups > 1 ? data.intervals[group_root].end : data.blocks;
    if (!threadIdx.x) { depth = state->depth; state->requested = 0; }
    __syncthreads();
    for (size_t step = 0; step < steps && depth; ++step) {
        if (!threadIdx.x) {
            while (depth && next[depth - 1] >= (depth == 2 ? group_end : data.blocks)) --depth;
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
        const double* prefix = coefficients ? (k ? coefficients + k * (k + 1) / 2 * nl : data.coefficients) : nullptr;
        const double* prefix_integrals = integrals ? (k ? integrals + k * nl : data.integrals) : nullptr;
        const double* prefix_linear = linears ? (k ? linears + k * nl : data.linear_integrals) : nullptr;
        const double* prefix_moments = k ? moments + k * (2 * data.m - k + 1) / 2 * nl : data.moments;
        double bound = 0;
        for (size_t q = data.m - k; q-- > 0;) {
            double total = 0;
            bool uncertain = false;
            for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
                const double moment = prefix_moments[q * nl + index];
                uncertain |= moment < 0;
                total += fmax(0.0, moment) * data.ratios[block * nl + index];
            }
            const bool expand = __syncthreads_or(uncertain);
            total = sum(total, scratch);
            if (expand) { bound = threshold; break; }
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
        if ((data.search_method & 2) && data.m - k <= 2 && bound >= threshold) {
            if (!threadIdx.x) ++state->terminal_queries;
            const double mean = data.terminal_residual[0];
            const double error = siglasso::terminal_roundoff(k, data.n, data.l);
            double signed_bound = 0;
            bool complete = true;
            for (size_t q = 1; q <= data.m - k && complete; ++q) {
                const double cap = data.terminal_residual[1 + (k + q == data.m)];
                const size_t suffixes = q == 2 ? data.d : 1, count = interval.high - interval.low;
                const unsigned int width = count >= 4 ? 32 : blockDim.x, groups = blockDim.x / width;
                const unsigned int group = threadIdx.x / width, rank = threadIdx.x % width;
                const size_t tiles = (count + groups - 1) / groups;
                const bool reuse = count >= (q == 2 ? 2 : 4);
                bool prepared = false;
                double product_magnitude = 0;
                for (size_t tile = 0; tile < tiles * suffixes && complete; ++tile) {
                    const size_t start = tile % tiles * groups, index = start + group, a = interval.low + index, b = tile / tiles;
                    if (start == 0) prepared = false;
                    if (!rank) {
                        terminal_hits[group] = false; terminal_bounds[group] = 0;
                        if (index < count && terminal_cache.capacity) {
                            terminal_slots[group] = terminal_cache.slot(word, k, a, b, q);
                            terminal_hits[group] = terminal_cache.get(terminal_slots[group], word, k, a, b, q, terminal_bounds[group]);
                        }
                    }
                    __syncthreads();
                    if (reuse && !prepared && __syncthreads_or(index < count && !terminal_hits[group])) {
                        // One variation majorant covers all first letters in this block.
                        double absolute = 0;
                        for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) {
                            double tail = 0;
                            for (size_t j = data.l; j-- > 0;) {
                                const double other = q == 2 ? data.increments[(j * data.d + b) * data.n + i] : 0; tail += other;
                                const double integral = prefix_integrals[j * data.n + i];
                                const double left = q == 2 ? integral * tail : integral;
                                const double right = q == 2 ? prefix_linear[j * data.n + i] * other : 0;
                                const double r = data.residual[i] - mean;
                                products[j * data.n + i] = r * (left - right);
                                absolute += fabs(r) * (fabs(left) + fabs(right)) * data.variation[j * data.n + i];
                            }
                        }
                        product_magnitude = sum(absolute, scratch) * (1 + siglasso::terminal_roundoff(k, nl, data.l));
                        __syncthreads(); prepared = true;
                    }
                    if (index < count && !terminal_hits[group]) {
                        double correlation = 0, magnitude = 0;
                        if (reuse) {
                            for (size_t j = 0; j < data.l; ++j) for (size_t i = rank; i < data.n; i += width) {
                                const double step = data.increments[(j * data.d + a) * data.n + i];
                                correlation += step * products[j * data.n + i];
                            }
                        } else for (size_t i = rank; i < data.n; i += width) {
                            double absolute;
                            const double value = siglasso::terminal_sample(prefix_integrals, prefix_linear, data.increments,
                                a, b, q, k, i, data.n, data.l, data.d, absolute);
                            correlation += (data.residual[i] - mean) * value;
                            magnitude += fabs(data.residual[i] - mean) * absolute;
                        }
                        if (width == 32) {
                            for (unsigned int stride = 16; stride; stride /= 2) {
                                correlation += __shfl_down_sync(0xffffffffU, correlation, stride);
                                if (!reuse) magnitude += __shfl_down_sync(0xffffffffU, magnitude, stride);
                            }
                        } else { correlation = sum(correlation, scratch); if (!reuse) magnitude = sum(magnitude, scratch); }
                        if (reuse) magnitude = product_magnitude;
                        const double rounding = reuse ? siglasso::terminal_roundoff(k, nl, data.l) : error;
                        if (!rank) terminal_bounds[group] = (fabs(correlation) * data.weights[k + q]
                            + rounding * (magnitude * data.weights[k + q] + cap)) / data.n + 1e-15;
                    }
                    __syncthreads();
                    for (size_t j = 0; j < groups && start + j < count; ++j) {
                        const double upper = terminal_bounds[j];
                        if (!threadIdx.x && !terminal_hits[j]) {
                            ++state->terminal_correlations;
                            if (terminal_cache.capacity && isfinite(upper)) terminal_cache.put(terminal_slots[j], word, k,
                                interval.low + start + j, b, q, upper);
                        }
                        if (!isfinite(upper) || upper * (1 + 1e-10) >= threshold) complete = false;
                        signed_bound = fmax(signed_bound, upper);
                    }
                    __syncthreads();
                }
            }
            if (complete && signed_bound * (1 + 1e-10) + 1e-15 < threshold) {
                bound = fmin(bound, signed_bound);
                if (!threadIdx.x) ++state->terminal_pruned;
            }
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
        if (Exact && depth < data.m && !data.factors[depth]) {
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
        double* child = Exact && depth < data.m ? coefficients + depth * (depth + 1) / 2 * nl : nullptr;
        double* child_times = !Exact && depth < data.m ? endpoints + depth * data.n * (data.l + 1) : nullptr;
        double* child_integrals = depth < data.m && integrals && (Exact || (data.search_method & 1) || data.m - depth <= 2) ? integrals + depth * nl : nullptr;
        double* child_linears = depth < data.m && linears && data.m - depth <= 2 ? linears + depth * nl : nullptr;
        const size_t state_size = Exact ? (depth + 1) * nl : data.n * (data.l + 1);
        const size_t integral_size = child_integrals ? nl : 0, linear_size = child_linears ? nl : 0;
        const size_t moment_offset = state_size + integral_size + linear_size;
        const size_t cache_stride = data.cache_stride;
        double* cached = data.prefix_cache ? data.prefix_cache + cache_slot * cache_stride : nullptr;
        if (cache_hit) {
            for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) values[i] = cached[cache_stride - data.n + i];
            for (size_t index = threadIdx.x; index < state_size; index += blockDim.x) (Exact ? child : child_times)[index] = cached[index];
        } else if (!Exact && depth == data.m && prefix_integrals) {
            for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) {
                double value = 0;
                for (size_t j = 0; j < data.l; ++j)
                    value += data.increments[(j * data.d + interval.low) * data.n + i] * prefix_integrals[j * data.n + i];
                values[i] = value;
            }
        } else if (!Exact && data.segment_scan) {
            const unsigned int warp_lane = threadIdx.x % 32;
            for (size_t i = threadIdx.x / 32; i < data.n; i += blockDim.x / 32) {
                double carry = 0;
                if (child_times && !warp_lane) child_times[i] = 0;
                for (size_t begin = 0; begin < data.l; begin += 32) {
                    const size_t j = begin + warp_lane;
                    double change = 0, integral = 0, linear = 0, product = 1;
                    if (j < data.l) for (size_t h = 1; h <= depth; ++h) {
                        product *= data.increments[(j * data.d + word[depth - h]) * data.n + i] * data.reciprocals[h];
                        const double term = (depth == h ? 1 : endpoints[((depth - h) * (data.l + 1) + j) * data.n + i]) * product;
                        change += term;
                        if (child_integrals) integral += term * data.reciprocals[h + 1];
                        if (child_linears) linear += term * data.reciprocals[h + 2];
                    }
                    double inclusive = change;
                    for (unsigned int offset = 1; offset < 32; offset *= 2) {
                        const double previous = __shfl_up_sync(0xffffffffU, inclusive, offset);
                        if (warp_lane >= offset) inclusive += previous;
                    }
                    const double previous = __shfl_up_sync(0xffffffffU, inclusive, 1);
                    const double start = carry + (warp_lane ? previous : 0);
                    if (child_times && j < data.l) {
                        child_times[(j + 1) * data.n + i] = carry + inclusive;
                        if (child_integrals) child_integrals[j * data.n + i] = start + integral;
                        if (child_linears) child_linears[j * data.n + i] = start / 2 + linear;
                    }
                    carry += __shfl_sync(0xffffffffU, inclusive, 31);
                }
                if (!warp_lane) values[i] = carry;
            }
        } else if (!Exact) {
            for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) {
                double value = 0;
                if (child_times) child_times[i] = 0;
                for (size_t j = 0; j < data.l; ++j) {
                    double product = 1, integral = value, linear = value / 2;
                    for (size_t h = 1; h <= depth; ++h) {
                        product *= data.increments[(j * data.d + word[depth - h]) * data.n + i] * data.reciprocals[h];
                        const double term = (depth == h ? 1 : endpoints[((depth - h) * (data.l + 1) + j) * data.n + i]) * product;
                        value += term;
                        if (child_integrals) integral += term * data.reciprocals[h + 1];
                        if (child_linears) linear += term * data.reciprocals[h + 2];
                    }
                    if (child_times) {
                        child_times[(j + 1) * data.n + i] = value;
                        if (child_integrals) child_integrals[j * data.n + i] = integral;
                        if (child_linears) child_linears[j * data.n + i] = linear;
                    }
                }
                values[i] = value;
            }
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
            if (cache_hit) { if (state->cache_hits != UINT64_MAX) ++state->cache_hits; }
            else ++state->evaluated;
            accepted = false;
            slot = SIZE_MAX - 1;
            if (!isfinite(correlation)) { state->omitted = CUDART_INF; depth = 0; }
            else if (!record(data.seen, word, depth)) { next[k] = block; slot = SIZE_MAX; }
            else if (!is_active(data, word, depth)) {
                state->omitted = fmax(state->omitted, correlation * (1 + 64 * DBL_EPSILON) + 1e-15);
                if (data.capacity && correlation > alpha && correlation >= threshold && (depth > 1 || state->root % data.root_groups == 0)) {
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
            double* child_moments = moments + depth * (2 * data.m - depth + 1) / 2 * nl;
            if (cache_hit) {
                for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
                    if (child_integrals) child_integrals[index] = cached[state_size + index];
                    if (child_linears) child_linears[index] = cached[state_size + integral_size + index];
                }
                for (size_t index = threadIdx.x; index < (data.m - depth) * nl; index += blockDim.x) child_moments[index] = cached[moment_offset + index];
            } else for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
                const size_t i = index % data.n, j = index / data.n;
                if (Exact) {
                    double integral = 0, linear = 0;
                    for (size_t a = 0; a <= depth; ++a) {
                        const double c = child[(j * (depth + 1) + a) * data.n + i]; integral += c;
                        if (child_linears) linear += (a + 1) * c;
                    }
                    child_integrals[index] = integral / (depth + 1);
                    if (child_linears) child_linears[index] = linear / ((depth + 1) * (depth + 2));
                    siglasso::absolute_moments(child, data.factors[depth], data.variation[index], data.after[index], depth, data.m - depth, j, i, data.n, data.l,
                        data.integral_scratch + (lane * blockDim.x + threadIdx.x) * data.scratch_size, child_moments + index,
                        (data.search_method & 2) && data.m - depth <= 2);
                } else siglasso::trapezoidal_moments(child_times[index], child_times[index + data.n], data.variation[index], data.after[index],
                    data.weights[depth + 1] / data.n, depth, data.m - depth, nl, child_moments + index, data.reciprocals);
            }
            __syncthreads();
            if (data.cache_slots) {
                if (!cache_hit) {
                    for (size_t index = threadIdx.x; index < state_size; index += blockDim.x) cached[index] = (Exact ? child : child_times)[index];
                    for (size_t index = threadIdx.x; index < nl; index += blockDim.x) {
                        if (child_integrals) cached[state_size + index] = child_integrals[index];
                        if (child_linears) cached[state_size + integral_size + index] = child_linears[index];
                    }
                    for (size_t index = threadIdx.x; index < (data.m - depth) * nl; index += blockDim.x) cached[moment_offset + index] = child_moments[index];
                    for (size_t i = threadIdx.x; i < data.n; i += blockDim.x) cached[cache_stride - data.n + i] = values[i];
                }
                __syncthreads();
                if (!threadIdx.x) {
                    data.cache_lengths[cache_slot] = depth;
                    for (size_t j = 0; j < depth; ++j) data.cache_words[cache_slot * data.m + j] = word[j];
                    data.cache_ticks[cache_slot] = ++data.cache_clocks[lane];
                    if (!cache_hit && state->cache_writes != UINT64_MAX) ++state->cache_writes;
                }
            }
            for (size_t index = threadIdx.x; index < (data.m - depth) * nl; index += blockDim.x)
                if (child_moments[index] >= 0) child_moments[index] *= fabs(data.residual[index % data.n]);
            __syncthreads();
            if (!threadIdx.x) { next[depth] = depth == 1 ? group_root : 0; ++depth; }
        }
        __syncthreads();
    }
    if (!threadIdx.x) state->depth = depth;
}

} // namespace lasso_device
