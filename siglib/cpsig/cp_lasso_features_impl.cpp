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
