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
#include <cstddef>
#include <memory>
#include <vector>

namespace siglasso {
class Backend;
struct Array;
struct Column;

enum class Correction { stationary, boundary, failed };
struct Selection { size_t slot = static_cast<size_t>(-1); double correlation = 0; };

class ActiveSet {
public:
    virtual ~ActiveSet() = default;
    virtual void activate(const Column&, size_t slot, double correlation) = 0;
    virtual Correction correct(bool rebuild = false) = 0;
    virtual void refresh(bool rounded) = 0;
    virtual const Array& residual() const = 0;
    virtual double penalty() = 0;
    virtual void dual_residual(Array&) = 0;
    virtual Selection select(const std::vector<Column>&, const std::vector<size_t>&) = 0;
    virtual void output(std::vector<size_t>&, std::vector<double>&) = 0;
    virtual size_t size() const = 0;
};

std::unique_ptr<ActiveSet> make_cpu_active(Backend&, const Array&, double alpha);
} // namespace siglasso
