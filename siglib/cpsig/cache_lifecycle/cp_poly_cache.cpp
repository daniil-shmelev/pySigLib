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

#include "cp_poly_cache.h"

template<std::floating_point T>
struct sig_poly_table_cache {
	std::mutex mutex;
	std::vector<std::shared_ptr<const sig_poly_tables<T>>> tables;
};

template<std::floating_point T>
sig_poly_table_cache<T>& sig_poly_table_cache_() {
	static sig_poly_table_cache<T> cache;
	return cache;
}

template<std::floating_point T>
std::shared_ptr<const sig_poly_tables<T>> get_sig_poly_tables_(uint64_t order) {
	auto& cache = sig_poly_table_cache_<T>();
	std::lock_guard<std::mutex> lock(cache.mutex);
	if (cache.tables.empty())
		cache.tables.resize(65);
	auto& tables = cache.tables[order];
	if (!tables)
		tables = std::make_shared<const sig_poly_tables<T>>(order);
	return tables;
}

template<std::floating_point T>
void clear_sig_poly_table_cache_() {
	auto& cache = sig_poly_table_cache_<T>();
	std::lock_guard<std::mutex> lock(cache.mutex);
	if (cache.tables.empty())
		return;
	decltype(cache.tables)().swap(cache.tables);
}


void clear_sig_poly_table_cache() {
	clear_sig_poly_table_cache_<float>();
	clear_sig_poly_table_cache_<double>();
}

template std::shared_ptr<const sig_poly_tables<float>> get_sig_poly_tables_<float>(uint64_t);
template std::shared_ptr<const sig_poly_tables<double>> get_sig_poly_tables_<double>(uint64_t);
template<typename T> struct TableCache {
    std::mutex mutex;
    std::vector<std::shared_ptr<const BranchedPolynomialTables<T>>> entries;
};
template<typename T> TableCache<T>& table_cache() { static TableCache<T> c; return c; }
template<typename T> std::shared_ptr<const BranchedPolynomialTables<T>> get_branched_poly_tables(uint64_t order) {
    auto& c = table_cache<T>();
    std::lock_guard<std::mutex> lock(c.mutex);
    if (c.entries.empty()) c.entries.resize(65);
    auto& entry = c.entries[order];
    if (!entry) entry = std::make_shared<const BranchedPolynomialTables<T>>(order);
    return entry;
}
void clear_branched_poly_table_cache() {
    { auto& c = table_cache<float>(); std::lock_guard<std::mutex> lock(c.mutex); decltype(c.entries)().swap(c.entries); }
    { auto& c = table_cache<double>(); std::lock_guard<std::mutex> lock(c.mutex); decltype(c.entries)().swap(c.entries); }
}

template std::shared_ptr<const BranchedPolynomialTables<float>> get_branched_poly_tables<float>(uint64_t);
template std::shared_ptr<const BranchedPolynomialTables<double>> get_branched_poly_tables<double>(uint64_t);
