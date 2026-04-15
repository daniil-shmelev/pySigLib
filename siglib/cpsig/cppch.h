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

// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <span>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <utility>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <concepts>
#include <variant>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <cmath>

#ifdef VEC
#ifndef __APPLE__
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif
#endif
