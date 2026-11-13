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

#include "cp_lasso_acceleration.h"
#include <Eigen/Dense>
#include <cmath>

namespace siglasso {
bool anderson_weights(const double* gram, double* weights) {
    using Matrix = Eigen::Matrix<double, 5, 5>;
    using Vector = Eigen::Matrix<double, 5, 1>;
    const Eigen::Map<const Matrix> input(gram);
    const double scale = input.diagonal().maxCoeff();
    if (!(scale > 0) || !std::isfinite(scale)) return false;
    const Matrix matrix = input / scale;
    const Eigen::FullPivLU<Matrix> solver(matrix);
    if (!solver.isInvertible()) return false;
    Vector result = solver.solve(Vector::Ones());
    const double sum = result.sum();
    if (!result.allFinite() || !std::isfinite(sum) || std::abs(sum) < 1e-14) return false;
    result /= sum;
    if (result.cwiseAbs().maxCoeff() > 1e6) return false;
    Eigen::Map<Vector> output(weights);
    output = result;
    return true;
}

void terminal_matrix_product(const double* left, const double* right, double* out, size_t samples, size_t rows, size_t columns) {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const Eigen::Map<const Matrix> a(left, samples, rows), b(right, samples, columns);
    Eigen::Map<Matrix> c(out, rows, columns);
    c.noalias() += a.transpose() * b;
}
}
