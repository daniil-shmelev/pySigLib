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
struct HallProductTerm {
    size_t left, right, output;
    double factor;
};
struct HallKernelPlan {
    size_t degree;
    std::vector<HallProductTerm> terms;
    std::vector<double> choose;
    std::shared_ptr<void> metadata;
};
struct HallBoundFactor {
    size_t degree, slot;
    double scale;
};
std::unique_ptr<FeatureSearch> make_cpu_feature_search(Backend&, uint32_t, bool selected_only = false);
size_t audit_cpu_feature_bounds(Backend&, uint32_t);
}
