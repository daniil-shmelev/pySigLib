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
#include <atomic>
#include <cstdint>

namespace coefficient_workspace {
// Native benchmark instrumentation. Disabled during normal calls.
inline std::atomic<bool> enabled{false};
inline std::atomic<uint64_t> live{0}, peak{0};

class Allocation {
    uint64_t bytes_;
public:
    explicit Allocation(uint64_t bytes) : bytes_(enabled.load(std::memory_order_relaxed) ? bytes : 0) {
        if (!bytes_) return;
        const uint64_t current = live.fetch_add(bytes_, std::memory_order_relaxed) + bytes_;
        uint64_t previous = peak.load(std::memory_order_relaxed);
        while (previous < current && !peak.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {}
    }
    ~Allocation() { if (bytes_) live.fetch_sub(bytes_, std::memory_order_relaxed); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};
}
