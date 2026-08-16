/* Copyright 2025 Daniil Shmelev
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

#include "preparation/log_sig_cache.h"

#include <cstdint>
#include <memory>

struct BchCache;

void prepare_basis_cache(
	uint64_t dimension,
	uint64_t degree,
	int method,
	bool use_disk = false);
const BasisCache& get_basis_cache(
	uint64_t dimension,
	uint64_t degree,
	int method);
const LogSigCache& get_log_sig_cache(
	uint64_t dimension,
	uint64_t degree,
	int method);
LogSigCache& get_log_sig_cache_mutable(
	uint64_t dimension,
	uint64_t degree,
	int method);
void clear_cache_(bool use_disk);
