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
#include "cppch.h"
#include "preparation/polynomial_sig_kernel/polynomial_sig_kernel_tables.h"
#include "preparation/polynomial_sig_kernel/branched_polynomial_tables.h"

template<std::floating_point T>
using sig_poly_tables = PolynomialSigKernelTables<T>;

template<std::floating_point T>
std::shared_ptr<const sig_poly_tables<T>> get_sig_poly_tables_(uint64_t order);
void clear_sig_poly_table_cache();

template<typename T>
std::shared_ptr<const BranchedPolynomialTables<T>> get_branched_poly_tables(uint64_t order);
void clear_branched_poly_table_cache();
