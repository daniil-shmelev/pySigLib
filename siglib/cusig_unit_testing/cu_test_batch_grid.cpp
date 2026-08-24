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
#include "cu_utils.h"

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

TEST(cudaBatchGridTest, RespectsWorkspaceCapacity) {
	const auto chunk = make_cuda_batch_grid_chunk(1, 1000, 200, 123);
	EXPECT_EQ(chunk.offset, 200u);
	EXPECT_EQ(chunk.size, 123u);
	EXPECT_EQ(chunk.grid.y, 123u);
	EXPECT_EQ(chunk.grid.z, 1u);
}

TEST(cudaWorkspacePlanTest, UsesHalfFreeMemoryAndAlwaysTriesOne) {
	EXPECT_EQ(cuda_workspace_initial_capacity(100, 100, 1000), 5u);
	EXPECT_EQ(cuda_workspace_initial_capacity(100, 100, 50), 1u);
	EXPECT_EQ(cuda_workspace_initial_capacity(2, 100, 10000), 2u);
	EXPECT_EQ(cuda_workspace_initial_capacity(0, 100, 10000), 0u);
	EXPECT_THROW(
		cuda_workspace_initial_capacity(1, 0, 10000), std::invalid_argument);
}

TEST(cudaWorkspacePlanTest, HalvesAllocationRetries) {
	EXPECT_EQ(cuda_workspace_retry_capacity(9), 4u);
	EXPECT_EQ(cuda_workspace_retry_capacity(4), 2u);
	EXPECT_EQ(cuda_workspace_retry_capacity(2), 1u);
	EXPECT_EQ(cuda_workspace_retry_capacity(1), 0u);
}

TEST(cudaWorkspacePlanTest, ChecksLocalAndGlobalIndexes) {
	EXPECT_EQ(cuda_global_batch_index(500, 12), 512u);
	EXPECT_THROW(
		cuda_global_batch_index(UINT64_MAX, 1), std::overflow_error);
}

TEST(cudaCheckedSizeTest, DetectsOverflow) {
	EXPECT_EQ(checked_cuda_size_add(12, 30, "test"), 42u);
	EXPECT_EQ(checked_cuda_size_mul(6, 7, "test"), 42u);
	EXPECT_THROW(
		checked_cuda_size_add(SIZE_MAX, 1, "test"), std::overflow_error);
	EXPECT_THROW(
		checked_cuda_size_mul(SIZE_MAX, 2, "test"), std::overflow_error);
}

TEST(cudaSharedMemoryPlanTest, IncludesStaticMemory) {
	const CudaSharedMemoryLimits limits{48 * 1024, 64 * 1024};
	EXPECT_TRUE(cuda_shared_memory_fits(1024, 63 * 1024, limits));
	EXPECT_FALSE(cuda_shared_memory_fits(1025, 63 * 1024, limits));
}
