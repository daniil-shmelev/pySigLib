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
#include "cp_simd_config.h"
#include "cp_lasso_features.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cp_lasso_features_impl.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace cpsig { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template<typename D>
void log_polynomial_block(D tag, const double* parent, const double* increments, double* out, double* values,
    size_t n, size_t l, size_t d, size_t degree, size_t levels, size_t letter, size_t i) {
    const size_t stride = n * l;
    for (size_t p = 0; p <= levels; ++p) {
        auto value = hn::Zero(tag);
        for (size_t j = 0; j < l; ++j) {
            const size_t sample = j * n + i;
            const auto step = hn::Div(hn::LoadU(tag, increments + (j * d + letter) * n + i), hn::Set(tag, static_cast<double>(degree)));
            if (p < levels) hn::StoreU(value, tag, out + p * (degree + 1) * stride + sample);
            for (size_t k = 0; k < degree; ++k) {
                value = hn::Add(value, hn::Mul(step, hn::LoadU(tag, parent + (p * degree + k) * stride + sample)));
                if (p < levels) hn::StoreU(value, tag, out + (p * (degree + 1) + k + 1) * stride + sample);
            }
        }
        if (!p) {
            if (!hn::AllTrue(tag, hn::IsFinite(value))) throw siglasso::NonfiniteError("Nonfinite log coefficient");
            hn::StoreU(value, tag, values + i);
        } else for (size_t k = 0; k <= degree; ++k) for (size_t j = 0; j < l; ++j) {
            double* destination = out + ((p - 1) * (degree + 1) + k) * stride + j * n + i;
            hn::StoreU(hn::Add(hn::LoadU(tag, destination), value), tag, destination);
        }
    }
}

void log_polynomial_(const double* parent, const double* increments, double* out, double* values,
    uint64_t n, uint64_t l, uint64_t d, uint64_t degree, uint64_t levels, uint64_t letter, uint64_t begin, uint64_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) log_polynomial_block(tag, parent, increments, out, values, n, l, d, degree, levels, letter, i);
    for (; i < end; ++i) log_polynomial_block(hn::CappedTag<double, 1>(), parent, increments, out, values, n, l, d, degree, levels, letter, i);
}

template<typename D>
void log_moments_block(D tag, const siglasso::LogMomentPlan& plan, const double* polynomial, const double* variation,
    const double* after, double* out, size_t n, size_t l, size_t i, double* scratch) {
    const size_t h = plan.degree, levels = plan.levels, stride = n * l;
    double* powers_v = scratch, *powers_r = scratch + levels * 4, *sums = scratch + levels * 8;
    auto power = hn::Set(tag, 1.);
    const auto total = hn::Add(hn::LoadU(tag, variation + i), hn::LoadU(tag, after + i));
    for (size_t k = 0; k < levels; ++k) {
        hn::StoreU(power, tag, powers_v + k * 4); power = hn::Mul(power, total);
    }
    for (size_t j = 0; j < l; ++j) {
        std::fill_n(sums, levels * 4, 0.);
        for (size_t g = 0; g < plan.nodes.size(); ++g) {
            const auto r = hn::Add(hn::LoadU(tag, after + j * n + i),
                hn::Mul(hn::Set(tag, 1 - plan.nodes[g]), hn::LoadU(tag, variation + j * n + i)));
            power = hn::Set(tag, 1.);
            for (size_t k = 0; k < levels; ++k) {
                hn::StoreU(hn::Div(power, hn::Set(tag, plan.factorials[k])), tag, powers_r + k * 4); power = hn::Mul(power, r);
            }
            for (size_t p = 0; p < levels; ++p) {
                auto value = hn::Zero(tag);
                for (size_t k = 0; k <= h; ++k)
                    value = hn::Add(value, hn::Mul(hn::Abs(hn::LoadU(tag, polynomial + (p * (h + 1) + k) * stride + j * n + i)),
                        hn::Set(tag, plan.basis[g * (h + 1) + k])));
                value = hn::Mul(value, hn::Set(tag, plan.weights[g]));
                for (size_t q = p + 1; q <= levels; ++q) {
                    auto weight = hn::Zero(tag);
                    for (size_t k = p; k < q; ++k)
                        weight = hn::Add(weight, hn::Mul(hn::Mul(hn::LoadU(tag, powers_v + k * 4),
                            hn::LoadU(tag, powers_r + (q - 1 - k) * 4)), hn::Set(tag, plan.coefficients[p * levels + k])));
                    hn::StoreU(hn::Add(hn::LoadU(tag, sums + (q - 1) * 4), hn::Mul(weight, value)), tag, sums + (q - 1) * 4);
                }
            }
        }
        for (size_t q = 1; q <= levels; ++q) {
            auto value = hn::Mul(hn::LoadU(tag, sums + (q - 1) * 4), hn::Set(tag, plan.factorials[h + q]));
            value = hn::IfThenElse(hn::IsFinite(value), value, hn::Set(tag, std::numeric_limits<double>::infinity()));
            hn::StoreU(value, tag, out + (q - 1) * stride + j * n + i);
        }
    }
}

void log_moments_(const siglasso::LogMomentPlan& plan, const double* polynomial, const double* variation, const double* after,
    double* out, uint64_t n, uint64_t l, uint64_t begin, uint64_t end, double* scratch) {
    const hn::CappedTag<double, 4> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) log_moments_block(tag, plan, polynomial, variation, after, out, n, l, i, scratch);
    for (; i < end; ++i) log_moments_block(hn::CappedTag<double, 1>(), plan, polynomial, variation, after, out, n, l, i, scratch);
}

template<typename D>
void log_terminal_integrals_block(D tag, const double* polynomial, double* out, size_t n, size_t l, size_t degree, size_t levels, size_t i) {
    const size_t stride = n * l;
    for (size_t j = 0; j < l; ++j) {
        auto first = hn::Zero(tag), time = hn::Zero(tag), second = hn::Zero(tag);
        for (size_t k = 0; k <= degree; ++k) {
            const auto value = hn::Mul(hn::LoadU(tag, polynomial + k * stride + j * n + i), hn::Set(tag, 1. / (degree + 1)));
            first = hn::Add(first, value);
            time = hn::Add(time, hn::Mul(value, hn::Set(tag, static_cast<double>(k + 1) / (degree + 2))));
            if (levels == 2) second = hn::Add(second, hn::Mul(hn::LoadU(tag, polynomial + (degree + 1 + k) * stride + j * n + i), hn::Set(tag, 1. / (degree + 1))));
        }
        hn::StoreU(first, tag, out + j * n + i);
        hn::StoreU(time, tag, out + stride + j * n + i);
        hn::StoreU(second, tag, out + 2 * stride + j * n + i);
    }
}

void log_terminal_integrals_(const double* polynomial, double* out, uint64_t n, uint64_t l, uint64_t degree, uint64_t levels, uint64_t begin, uint64_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) log_terminal_integrals_block(tag, polynomial, out, n, l, degree, levels, i);
    for (; i < end; ++i) log_terminal_integrals_block(hn::CappedTag<double, 1>(), polynomial, out, n, l, degree, levels, i);
}

template<typename D>
void log_terminal_products_block(D tag, const double* integrals, const double* increments, const double* residual,
    double* out, size_t n, size_t l, size_t d, size_t letter, size_t suffix, size_t i) {
    const size_t stride = n * l;
    auto displacement = hn::Zero(tag);
    if (suffix == 2) for (size_t j = 0; j < l; ++j)
        displacement = hn::Add(displacement, hn::LoadU(tag, increments + (j * d + letter) * n + i));
    auto tail = displacement, squared = hn::Zero(tag), magnitude = hn::Zero(tag);
    for (size_t j = 0; j < l; ++j) {
        auto value = hn::LoadU(tag, integrals + j * n + i), absolute = hn::Abs(value);
        if (suffix == 2) {
            const auto step = hn::LoadU(tag, increments + (j * d + letter) * n + i);
            const auto left = hn::Mul(value, tail);
            const auto right = hn::Mul(step, hn::LoadU(tag, integrals + stride + j * n + i));
            const auto extra = hn::Mul(displacement, hn::LoadU(tag, integrals + 2 * stride + j * n + i));
            value = hn::Add(hn::Sub(left, right), extra);
            absolute = hn::Add(hn::Add(hn::Abs(left), hn::Abs(right)), hn::Abs(extra));
            tail = hn::Sub(tail, step);
        }
        hn::StoreU(value, tag, out + j * n + i);
        squared = hn::MulAdd(value, value, squared); magnitude = hn::Add(magnitude, absolute);
    }
    const auto weight = hn::Abs(hn::LoadU(tag, residual + i));
    const auto norm = hn::Sqrt(hn::Add(squared, hn::Set(tag, 32. * (l + 1) * std::numeric_limits<double>::min())));
    hn::StoreU(hn::Mul(norm, weight), tag, out + stride + i);
    hn::StoreU(hn::Mul(magnitude, weight), tag, out + stride + n + i);
}

void log_terminal_products_(const double* integrals, const double* increments, const double* residual, double* out,
    uint64_t n, uint64_t l, uint64_t d, uint64_t letter, uint64_t suffix, uint64_t begin, uint64_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) log_terminal_products_block(tag, integrals, increments, residual, out, n, l, d, letter, suffix, i);
    for (; i < end; ++i) log_terminal_products_block(hn::CappedTag<double, 1>(), integrals, increments, residual, out, n, l, d, letter, suffix, i);
}

template<typename D>
void log_terminal_values_block(D tag, const double* products, const double* increments, double* out,
    size_t n, size_t l, size_t d, size_t letter, size_t i) {
    auto value = hn::Zero(tag), correction = hn::Zero(tag);
    for (size_t j = 0; j < l; ++j) {
        const auto term = hn::Mul(hn::LoadU(tag, products + j * n + i), hn::LoadU(tag, increments + (j * d + letter) * n + i));
        const auto adjusted = hn::Sub(term, correction), next = hn::Add(value, adjusted);
        correction = hn::Sub(hn::Sub(next, value), adjusted); value = next;
    }
    if (!hn::AllTrue(tag, hn::IsFinite(value))) throw siglasso::NonfiniteError("Nonfinite log coefficient");
    hn::StoreU(value, tag, out + i);
}

void log_terminal_values_(const double* products, const double* increments, double* out,
    uint64_t n, uint64_t l, uint64_t d, uint64_t letter, uint64_t begin, uint64_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) log_terminal_values_block(tag, products, increments, out, n, l, d, letter, i);
    for (; i < end; ++i) log_terminal_values_block(hn::CappedTag<double, 1>(), products, increments, out, n, l, d, letter, i);
}

template<typename D>
void hall_terminal_block(D tag, const siglasso::HallKernelPlan& plan, const double* left, const double* right,
    double* values, size_t n, size_t l, size_t i, double run) {
    const size_t stride = n * l;
    auto sum = hn::Zero(tag), correction = hn::Zero(tag);
    for (size_t j = 0; j < l; ++j) for (const auto& term : plan.terminal_terms) {
        const size_t u = term.left * stride + j * n + i, v = term.right * stride + j * n + i;
        const auto derivative = hn::Sub(hn::LoadU(tag, left + u + stride), hn::LoadU(tag, left + u));
        const auto value = hn::Mul(hn::Mul(derivative, hn::LoadU(tag, right + v)), hn::Set(tag, term.factor / run));
        const auto adjusted = hn::Sub(value, correction), next = hn::Add(sum, adjusted);
        correction = hn::Sub(hn::Sub(next, sum), adjusted); sum = next;
    }
    if (!hn::AllTrue(tag, hn::IsFinite(sum))) throw siglasso::NonfiniteError("Nonfinite Hall coefficient");
    hn::StoreU(sum, tag, values + i);
}

void hall_terminal_(const siglasso::HallKernelPlan& plan, const double* left, const double* right, double* values,
    uint64_t n, uint64_t l, uint64_t begin, uint64_t end, double run) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) hall_terminal_block(tag, plan, left, right, values, n, l, i, run);
    for (; i < end; ++i) hall_terminal_block(hn::CappedTag<double, 1>(), plan, left, right, values, n, l, i, run);
}

template<typename D>
void hall_polynomial_block(D tag, const siglasso::HallKernelPlan& plan, const double* left, const double* right,
    double* out, double* values, size_t n, size_t l, size_t i, double run, double* scratch) {
    const size_t degree = plan.degree, product_count = degree * (degree + 1) / 2;
    const size_t stride = n * l;
    double* product = scratch;
    double* jump = product + product_count * 4;
    double* initial = jump + (degree + 1) * 4;
    std::fill_n(jump, (degree + 1) * 8, 0.);
    for (size_t j = 0; j < l; ++j) {
        std::fill_n(product, product_count * 4, 0.);
        for (const auto& term : plan.terms) {
            const size_t u = term.left * stride + j * n + i, v = term.right * stride + j * n + i;
            const auto derivative = hn::Sub(hn::LoadU(tag, left + u + stride), hn::LoadU(tag, left + u));
            const auto value = hn::Mul(hn::Mul(derivative, hn::LoadU(tag, right + v)), hn::Set(tag, term.factor / run));
            double* destination = product + term.output * 4;
            hn::StoreU(hn::Add(hn::LoadU(tag, destination), value), tag, destination);
        }
        for (size_t p = 0, base = 0, source = 0; p <= degree; ++p) {
            const size_t td = degree - p;
            auto value = hn::Zero(tag);
            if (out) hn::StoreU(value, tag, out + base * stride + j * n + i);
            for (size_t k = 0; k < td; ++k) {
                value = hn::Add(value, hn::Mul(hn::LoadU(tag, product + (source + k) * 4), hn::Set(tag, 1. / td)));
                if (out) hn::StoreU(value, tag, out + (base + k + 1) * stride + j * n + i);
            }
            hn::StoreU(hn::Add(hn::LoadU(tag, jump + p * 4), value), tag, jump + p * 4);
            base += td + 1; source += td;
        }
    }
    for (size_t p = degree; p > 0; --p) {
        auto value = hn::LoadU(tag, jump + (p - 1) * 4);
        for (size_t k = p + 1; k <= degree; ++k)
            value = hn::Sub(value, hn::Mul(hn::Set(tag, plan.choose[k * (degree + 1) + p - 1]), hn::LoadU(tag, initial + k * 4)));
        hn::StoreU(hn::Mul(value, hn::Set(tag, 1. / p)), tag, initial + p * 4);
    }
    auto coefficient = hn::Zero(tag), constant = hn::Zero(tag);
    double power = 0.5;
    for (size_t p = 1; p <= degree; ++p) {
        const auto value = hn::LoadU(tag, initial + p * 4);
        constant = hn::Sub(constant, hn::Mul(value, hn::Set(tag, power)));
        coefficient = hn::Add(coefficient, hn::Mul(value, hn::Set(tag, p * 2 * power)));
        power *= 0.5;
    }
    coefficient = hn::Mul(coefficient, hn::Set(tag, degree % 2 ? 1. : -1.));
    if (!hn::AllTrue(tag, hn::IsFinite(coefficient))) throw siglasso::NonfiniteError("Nonfinite Hall coefficient");
    hn::StoreU(coefficient, tag, values + i);
    hn::StoreU(constant, tag, initial);
    if (!out) return;
    for (size_t p = 0, base = 0; p <= degree; ++p) {
        const size_t td = degree - p;
        auto prefix = hn::LoadU(tag, initial + p * 4);
        for (size_t j = 0; j < l; ++j) {
            const auto integral = hn::LoadU(tag, out + (base + td) * stride + j * n + i);
            for (size_t k = 0; k <= td; ++k) {
                double* destination = out + (base + k) * stride + j * n + i;
                const auto value = hn::Add(hn::LoadU(tag, destination), prefix);
                if (!hn::AllTrue(tag, hn::IsFinite(value))) throw siglasso::NonfiniteError("Nonfinite Hall polynomial");
                hn::StoreU(value, tag, destination);
            }
            prefix = hn::Add(prefix, integral);
        }
        base += td + 1;
    }
}

void hall_polynomial_(const siglasso::HallKernelPlan& plan, const double* left, const double* right,
    double* out, double* values, uint64_t n, uint64_t l, uint64_t begin, uint64_t end, double run, double* scratch) {
    const hn::CappedTag<double, 4> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) hall_polynomial_block(tag, plan, left, right, out, values, n, l, i, run, scratch);
    for (; i < end; ++i) hall_polynomial_block(hn::CappedTag<double, 1>(), plan, left, right, out, values, n, l, i, run, scratch);
}

template<typename D>
double hall_group_sum(D tag, const double* global, const double* side, const double* residual,
    const siglasso::HallBoundFactor* factors, size_t count, size_t q, size_t n, size_t begin, size_t end) {
    auto sum = hn::Zero(tag);
    for (size_t i = begin; i < end; i += hn::Lanes(tag)) {
        auto product = hn::Zero(tag);
        for (size_t v = 0; v < count; ++v) {
            const auto& factor = factors[v]; if (factor.degree > q) continue;
            auto value = hn::Mul(hn::Mul(hn::LoadU(tag, side + factor.slot * n + i), hn::Set(tag, factor.scale)),
                hn::LoadU(tag, global + (q - factor.degree + 1) * n + i));
            value = hn::IfThenElse(hn::IsFinite(value), value, hn::Set(tag, std::numeric_limits<double>::infinity()));
            product = hn::Max(product, value);
        }
        sum = hn::Add(sum, hn::Mul(hn::Mul(hn::Abs(hn::LoadU(tag, residual + i)), hn::LoadU(tag, global + i)), product));
    }
    return hn::ReduceSum(tag, sum);
}

template<typename D>
void hall_envelope_block(D tag, const siglasso::HallEnvelopePlan& plan, const double* polynomial, double* out, size_t n, size_t l, size_t i) {
    const size_t h = plan.degree, degree = h - plan.derivative, stride = n * l;
    auto total = hn::Zero(tag);
    for (size_t g = 0; g < plan.weights.size(); ++g) for (size_t j = 0; j < l; ++j) {
        auto envelope = hn::Zero(tag);
        for (size_t k = 0; k <= degree; ++k) {
            auto real = hn::Zero(tag), imaginary = hn::Zero(tag);
            for (size_t p = 0; p <= h; ++p) {
                auto elevated = hn::Zero(tag);
                const size_t group = k * (h + 1) + p;
                for (size_t t = plan.offsets[group]; t < plan.offsets[group + 1]; ++t) {
                    const auto& term = plan.terms[t];
                    auto value = hn::LoadU(tag, polynomial + term.offset * stride + j * n + i);
                    if (plan.derivative) value = hn::Sub(hn::LoadU(tag, polynomial + (term.offset + 1) * stride + j * n + i), value);
                    elevated = hn::Add(elevated, hn::Mul(value, hn::Set(tag, term.factor)));
                }
                const auto value = hn::Mul(elevated, hn::Set(tag, plan.powers[g * (h + 1) + p]));
                if (p % 2) imaginary = hn::Add(imaginary, value); else real = hn::Add(real, value);
            }
            const auto hi = hn::Max(hn::Abs(real), hn::Abs(imaginary)), lo = hn::Min(hn::Abs(real), hn::Abs(imaginary));
            const auto ratio = hn::Div(lo, hn::IfThenElse(hn::Eq(hi, hn::Zero(tag)), hn::Set(tag, 1.), hi));
            auto magnitude = hn::Mul(hi, hn::Sqrt(hn::Add(hn::Set(tag, 1.), hn::Mul(ratio, ratio))));
            magnitude = hn::IfThenElse(hn::And(hn::IsFinite(real), hn::IsFinite(imaginary)), magnitude, hn::Set(tag, std::numeric_limits<double>::infinity()));
            if (plan.derivative) envelope = hn::Add(envelope, hn::Mul(magnitude, hn::Set(tag, 1. / (degree + 1))));
            else envelope = hn::Max(envelope, magnitude);
        }
        if (plan.derivative) total = hn::Add(total, hn::Mul(envelope, hn::Set(tag, plan.weights[g])));
        else total = hn::Max(total, envelope);
    }
    hn::StoreU(total, tag, out + i);
}

void hall_envelope_(const siglasso::HallEnvelopePlan& plan, const double* polynomial, double* out, uint64_t n, uint64_t l, uint64_t begin, uint64_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    size_t i = begin;
    for (; i + lanes <= end; i += lanes) hall_envelope_block(tag, plan, polynomial, out, n, l, i);
    for (; i < end; ++i) hall_envelope_block(hn::CappedTag<double, 1>(), plan, polynomial, out, n, l, i);
}

double hall_group_bound_(const double* global, const double* side, const double* residual,
    const siglasso::HallBoundFactor* factors, uint64_t count, const double* weights, uint64_t degree, uint64_t levels, uint64_t n) {
    const hn::CappedTag<double, 4> tag;
    const size_t end = n - n % hn::Lanes(tag);
    double best = 0;
    for (size_t q = 1; q <= levels; ++q) {
        const double sum = hall_group_sum(tag, global, side, residual, factors, count, q, n, 0, end)
            + hall_group_sum(hn::CappedTag<double, 1>(), global, side, residual, factors, count, q, n, end, n);
        const double value = sum * weights[degree + q] / n;
        if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
        best = std::max(best, value);
    }
    return best;
}

} }
HWY_AFTER_NAMESPACE();
