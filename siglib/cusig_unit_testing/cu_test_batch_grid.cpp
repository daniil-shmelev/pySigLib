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

#include <gtest/gtest.h>

#include "cu_runtime_utils.h"

TEST(cudaBatchGridTest, SplitsAcrossLaunches) {
	const uint64_t batch_size = CUDA_BATCH_GRID_CAPACITY + 1;
	const auto first = make_cuda_batch_grid_chunk(7, batch_size, 0);
	EXPECT_EQ(first.grid.x, 7u);
	EXPECT_EQ(first.grid.y, CUDA_BATCH_GRID_LIMIT);
	EXPECT_EQ(first.grid.z, CUDA_BATCH_GRID_LIMIT);
	EXPECT_EQ(first.offset, 0u);
	EXPECT_EQ(first.size, CUDA_BATCH_GRID_CAPACITY);

	const auto second = make_cuda_batch_grid_chunk(
		7, batch_size, first.offset + first.size);
	EXPECT_EQ(second.grid.x, 7u);
	EXPECT_EQ(second.grid.y, 1u);
	EXPECT_EQ(second.grid.z, 1u);
	EXPECT_EQ(second.offset, CUDA_BATCH_GRID_CAPACITY);
	EXPECT_EQ(second.size, 1u);
}

TEST(cudaBatchGridTest, BalancesGridDimensions) {
	const uint64_t batch_size = CUDA_BATCH_GRID_LIMIT + 1;
	const auto chunk = make_cuda_batch_grid_chunk(3, batch_size, 0);
	EXPECT_EQ(chunk.grid.x, 3u);
	EXPECT_EQ(chunk.grid.y, 32768u);
	EXPECT_EQ(chunk.grid.z, 2u);
	EXPECT_EQ(chunk.size, batch_size);
}
