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

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

inline constexpr uint64_t cache_magic_number = 0x70797369676C6962ULL;
inline constexpr uint64_t MAX_CACHE_VECTOR_SIZE = 1'000'000'000ULL;

inline void check_stream_has_bytes(
	std::istream& in,
	uint64_t need,
	const char* label
) {
	const std::streampos here = in.tellg();
	in.seekg(0, std::ios::end);
	const std::streampos end = in.tellg();
	in.seekg(here);
	if (here < 0 || end < 0 || static_cast<uint64_t>(end - here) < need)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}

template<typename T>
inline void serialize_cache_vector(
	std::ostream& out,
	const std::vector<T>& values
) {
	static_assert(std::is_trivially_copyable_v<T>);
	const uint64_t size = values.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(size));
	if (size != 0) {
		out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(size * sizeof(T)));
	}
}

template<typename T>
inline void deserialize_cache_vector(
	std::istream& in,
	std::vector<T>& values,
	const char* label
) {
	static_assert(std::is_trivially_copyable_v<T>);
	uint64_t size = 0;
	in.read(reinterpret_cast<char*>(&size), sizeof(size));
	if (!in || size > MAX_CACHE_VECTOR_SIZE)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
	if (size == 0) {
		values.clear();
		return;
	}
	check_stream_has_bytes(in, size * sizeof(T), label);
	values.resize(size);
	in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size * sizeof(T)));
	if (!in)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}
