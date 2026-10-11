// Copyright 2026. Licensed under the Apache License, Version 2.0.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// A[a,b] integrates the b-th Lagrange basis polynomial from 0 to node a.
// Legendre orthogonality constructs A without expanding high-degree monomials;
// the appended weights integrate the same basis over the whole interval [0,1].
template<typename T>
struct BranchedPolynomialTables {
    uint64_t q;
    std::vector<T> data; // q*q integration matrix followed by q weights
    explicit BranchedPolynomialTables(uint64_t order) : q(order + 1), data(q*q + q) {
        std::vector<long double> x(q), w(q), p(q * (q + 1));
        const auto legendre = [](long double z, uint64_t n) {
            long double a = 1, b = z;
            for (uint64_t k = 1; k < n; ++k) {
                const auto c = ((2*k + 1)*z*b - k*a)/(k + 1);
                a = b; b = c;
            }
            return std::pair<long double, long double>(b, a);
        };
        for (uint64_t j = 0; j < q; ++j) {
            long double z = -std::cos(std::acos(-1.L) * (j + .75L)/(q + .5L));
            for (int iteration = 0; iteration < 100; ++iteration) {
                const auto [pn, pm] = legendre(z, q);
                const auto derivative = q * (z*pn - pm)/(z*z - 1);
                const auto delta = pn/derivative;
                z -= delta;
                if (std::abs(delta) < 8*std::numeric_limits<long double>::epsilon()) break;
            }
            const auto [pn, pm] = legendre(z, q);
            const auto derivative = q * (z*pn - pm)/(z*z - 1);
            x[j] = z; w[j] = 1/((1-z*z)*derivative*derivative);
            p[j*(q+1)] = 1;
            p[j*(q+1)+1] = z;
            for (uint64_t k = 1; k < q; ++k)
                p[j*(q+1)+k+1] = ((2*k+1)*z*p[j*(q+1)+k]-k*p[j*(q+1)+k-1])/(k+1);
            data[q*q+j] = static_cast<T>(w[j]);
        }
        for (uint64_t a = 0; a < q; ++a)
            for (uint64_t b = 0; b < q; ++b) {
                long double val = (x[a]+1)/2;
                for (uint64_t k = 1; k < q; ++k)
                    val += p[b*(q+1)+k]*(p[a*(q+1)+k+1]-p[a*(q+1)+k-1])/2;
                data[a*q+b] = static_cast<T>(val*w[b]);
            }
    }
};
