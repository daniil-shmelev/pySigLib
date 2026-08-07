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

#include "cupch.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr uint64_t kCuMaxCacheVectorSize = 1'000'000'000ULL;
inline constexpr uint64_t cu_cache_magic_number = 0x70797369676C6962;
inline constexpr const char* cu_cache_folder_name = "pysiglib_cache";

inline void cu_check_stream_has_bytes_(std::istream& in, uint64_t need, const char* label) {
	const std::streampos here = in.tellg();
	in.seekg(0, std::ios::end);
	const std::streampos end = in.tellg();
	in.seekg(here);
	if (here < 0 || end < 0 || static_cast<uint64_t>(end - here) < need)
		throw std::runtime_error(std::string("Tried to read an invalid cache file: ") + label);
}

inline std::filesystem::path& get_cuda_cache_dir_() {
	static std::filesystem::path dir;
	return dir;
}

inline void cu_serialize_vector_(std::ostream& out, const std::vector<uint64_t>& v) {
	const uint64_t size = v.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(size));
	if (size > 0)
		out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(uint64_t));
}

inline void cu_deserialize_vector_(std::istream& in, std::vector<uint64_t>& out) {
	uint64_t size;
	in.read(reinterpret_cast<char*>(&size), sizeof(size));
	if (!in)
		throw std::runtime_error("Tried to read an invalid cache file: vector size header");
	if (size > kCuMaxCacheVectorSize)
		throw std::runtime_error("Tried to read an invalid cache file: vector size exceeds limit");

	if (size == 0) {
		out.clear();
		return;
	}

	cu_check_stream_has_bytes_(in, size * sizeof(uint64_t), "vector body");
	out.resize(size);
	in.read(reinterpret_cast<char*>(out.data()), size * sizeof(uint64_t));
	if (!in)
		throw std::runtime_error("Tried to read an invalid cache file: vector body read");
}

inline void set_cache_dir_cuda_(const char* dir) {
	std::filesystem::path dir_path = dir;
	if (!std::filesystem::exists(dir_path))
		throw directory_not_found_error("Directory " + std::string(dir) + " does not exist.");
	std::filesystem::path cache_path = dir_path / cu_cache_folder_name;
	if (!std::filesystem::exists(cache_path))
		std::filesystem::create_directories(cache_path);
	get_cuda_cache_dir_() = dir_path;
}

inline void set_default_cuda_cache_dir_() {
#ifdef _WIN32
	char* dir = nullptr;
	size_t len;
	const errno_t err = _dupenv_s(&dir, &len, "LOCALAPPDATA");
	if (err || !dir)
		throw default_cache_dir_error("Failed to get default cache directory.");
#elif __APPLE__
	const char* home = std::getenv("HOME");
	if (!home)
		throw default_cache_dir_error("$HOME is not set; cannot determine default cache directory.");
	std::string dir_str = std::string(home) + "/Library/Caches";
	const char* dir = dir_str.c_str();
#else
	const char* home = std::getenv("HOME");
	if (!home)
		throw default_cache_dir_error("$HOME is not set; cannot determine default cache directory.");
	std::string dir_str = std::string(home) + "/.cache";
	const char* dir = dir_str.c_str();
#endif
	set_cache_dir_cuda_(dir);
#ifdef _WIN32
	free(dir);
#endif
}

inline void ensure_cuda_cache_dir_() {
	auto& dir = get_cuda_cache_dir_();
	if (dir.empty())
		set_default_cuda_cache_dir_();
	auto cache_path = dir / cu_cache_folder_name;
	if (!std::filesystem::exists(cache_path))
		std::filesystem::create_directories(cache_path);
}
