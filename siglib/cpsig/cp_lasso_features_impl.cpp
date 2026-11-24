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
                const auto coefficient = parent ? hn::LoadU(tag, parent + (p * degree + k) * stride + sample)
                    : hn::Set(tag, (p % 2 ? -1. : 1.) / (p + 1));
                value = hn::Add(value, hn::Mul(step, coefficient));
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

// Integrate a child directly into the three moments used by the final two
// letters. Its Bernstein control points only live in registers. The next
// log-series level contributes a constant shift on every segment.
template<size_t fixed_degree, typename D>
HWY_INLINE void log_child_integrals_block(D tag, const double* parent, const double* dx, double* out, double* values,
                              size_t n, size_t l, size_t stride, size_t degree, size_t i) {
    const size_t h = fixed_degree ? fixed_degree : degree;
    const size_t plane = n * l;
    auto first = hn::Zero(tag), second = hn::Zero(tag), third = hn::Zero(tag);
    const auto mean = hn::Set(tag, 1. / (h + 1)), moment = hn::Set(tag, 1. / ((h + 1) * (h + 2)));
    for (size_t j = 0; j < l; ++j) {
        const size_t index = j * n + i;
        const auto step = hn::Mul(hn::LoadU(tag, dx + j * stride * n + i), hn::Set(tag, 1. / h));
        auto integral = first, linear = first, extra = second;
        for (size_t k = 0; k < h; ++k) {
            const auto a = parent ? hn::LoadU(tag, parent + k * plane + index) : hn::Set(tag, 1.);
            const auto b = parent ? hn::LoadU(tag, parent + (h + k) * plane + index) : hn::Set(tag, -.5);
            const auto c = parent ? hn::LoadU(tag, parent + (2 * h + k) * plane + index) : hn::Set(tag, 1. / 3.);
            first = hn::MulAdd(step, a, first); second = hn::MulAdd(step, b, second); third = hn::MulAdd(step, c, third);
            integral = hn::Add(integral, first); linear = hn::MulAdd(hn::Set(tag, double(k + 2)), first, linear);
            extra = hn::Add(extra, second);
        }
        hn::StoreU(hn::Mul(integral, mean), tag, out + index);
        hn::StoreU(hn::Mul(linear, moment), tag, out + plane + index);
        hn::StoreU(hn::Mul(extra, mean), tag, out + 2 * plane + index);
    }
    hn::StoreU(first, tag, values + i);
    for (size_t j = 0; j < l; ++j) {
        double* base = out + j * n + i;
        hn::StoreU(hn::Add(hn::LoadU(tag, base), second), tag, base);
        hn::StoreU(hn::MulAdd(hn::Set(tag, .5), second, hn::LoadU(tag, base + plane)), tag, base + plane);
        hn::StoreU(hn::Add(hn::LoadU(tag, base + 2 * plane), third), tag, base + 2 * plane);
    }
}
void log_child_integrals_(const double* parent, const double* dx, double* out, double* values,
                         size_t n, size_t l, size_t stride, size_t h, size_t begin, size_t end) {
    const hn::ScalableTag<double> tag;
    const size_t lanes = hn::Lanes(tag);
    const auto apply = [&]<size_t degree>() {
        size_t i = begin;
        for (; i + lanes <= end; i += lanes) log_child_integrals_block<degree>(tag, parent, dx, out, values, n, l, stride, h, i);
        for (; i < end; ++i) log_child_integrals_block<degree>(hn::CappedTag<double, 1>(), parent, dx, out, values, n, l, stride, h, i);
    };
    if (h == 1) apply.template operator()<1>();
    else if (h == 2) apply.template operator()<2>();
    else apply.template operator()<0>();
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
        auto value = integrals ? hn::LoadU(tag, integrals + j * n + i) : hn::Set(tag, 1.);
        auto absolute = hn::Abs(value);
        if (suffix == 2) {
            const auto step = hn::LoadU(tag, increments + (j * d + letter) * n + i);
            const auto left = hn::Mul(value, tail);
            const auto right = hn::Mul(step, integrals ? hn::LoadU(tag, integrals + stride + j * n + i) : hn::Set(tag, .5));
            const auto extra = hn::Mul(displacement, integrals ? hn::LoadU(tag, integrals + 2 * stride + j * n + i) : hn::Set(tag, -.5));
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
        const auto step = hn::LoadU(tag, increments + (j * d + letter) * n + i);
        const auto term = products ? hn::Mul(hn::LoadU(tag, products + j * n + i), step) : step;
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

} }
HWY_AFTER_NAMESPACE();
