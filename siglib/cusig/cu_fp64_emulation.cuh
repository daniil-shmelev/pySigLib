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

// Included after the prefix-suffix cuBLAS error helper.
struct Fp64Emulation {
    uint64_t limit;
    cudaStream_t stream = nullptr;
#if CUBLAS_VERSION >= 130100
    struct State {
        cublasLtHandle_t handle = nullptr;
        cublasLtMatmulDesc_t operation = nullptr;
        cublasLtEmulationDesc_t emulation = nullptr;
        cublasLtMatmulPreference_t preference = nullptr;
        cublasLtMatrixLayout_t matrices[3] = {};
        ~State() {
            for (auto matrix : matrices) if (matrix) (void)cublasLtMatrixLayoutDestroy(matrix);
            if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
            if (operation) (void)cublasLtMatmulDescDestroy(operation);
            if (emulation) (void)cublasLtEmulationDescDestroy(emulation);
            if (handle) (void)cublasLtDestroy(handle);
        }
    } state;
    CudaBuf<std::byte> workspace;
    uint64_t reserved = 0;
    uint64_t parallelism;

    Fp64Emulation(int device, uint64_t bytes) : limit(bytes), parallelism(device_parallelism(device)) {
        blas_check(cublasLtCreate(&state.handle));
        blas_check(cublasLtMatmulDescCreate(&state.operation, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F));
        blas_check(cublasLtEmulationDescCreate(&state.emulation));
        const auto strategy = CUBLAS_EMULATION_STRATEGY_PERFORMANT;
        const auto control = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
        const int offset = 0;
        blas_check(cublasLtEmulationDescSetAttribute(state.emulation, CUBLASLT_EMULATION_DESC_STRATEGY, &strategy, sizeof(strategy)));
        blas_check(cublasLtEmulationDescSetAttribute(state.emulation, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL, &control, sizeof(control)));
        blas_check(cublasLtEmulationDescSetAttribute(state.emulation, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET, &offset, sizeof(offset)));
        blas_check(cublasLtEmulationDescSetAttribute(state.emulation, CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &offset, sizeof(offset)));
        blas_check(cublasLtMatmulDescSetAttribute(state.operation, CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &state.emulation, sizeof(state.emulation)));
        blas_check(cublasLtMatmulPreferenceCreate(&state.preference));
        blas_check(cublasLtMatmulPreferenceSetAttribute(state.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
        for (auto& matrix : state.matrices) blas_check(cublasLtMatrixLayoutCreate(&matrix, CUDA_R_64F, 1, 1, 1));
    }

    uint64_t gemm(Matrix<double> a, Matrix<double> b, Matrix<double> c, uint64_t batch, bool first) {
        if (std::min({a.rows, b.cols, a.cols}) < 128 || !limit) return 0;
        // Four 128 x 128 output tiles per SM; cuBLAS selects the actual kernel.
        const uint64_t tiles = mul((a.rows - 1) / 128 + 1, (b.cols - 1) / 128 + 1);
        uint64_t count = std::min(batch, (parallelism - 1) / tiles + 1);
        const cublasOperation_t operations[] = {b.rs == 1 ? CUBLAS_OP_T : CUBLAS_OP_N, a.rs == 1 ? CUBLAS_OP_T : CUBLAS_OP_N};
        blas_check(cublasLtMatmulDescSetAttribute(state.operation, CUBLASLT_MATMUL_DESC_TRANSA, operations, sizeof(operations[0])));
        blas_check(cublasLtMatmulDescSetAttribute(state.operation, CUBLASLT_MATMUL_DESC_TRANSB, operations + 1, sizeof(operations[1])));
        const Matrix<double> matrices[] = {b, a, c};
        constexpr cublasLtMatmulPreferenceAttributes_t alignments[] = {
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES};
        for (int i = 0; i < 3; ++i) {
            const auto& matrix = matrices[i];
            const bool transposed = i < 2 && operations[i] == CUBLAS_OP_T;
            const uint64_t rows = i == 2 ? c.cols : transposed ? matrix.rows : matrix.cols;
            const uint64_t cols = i == 2 ? c.rows : transposed ? matrix.cols : matrix.rows;
            const int64_t leading = matrix.leading_dimension(), stride = matrix.bs;
            blas_check(cublasLtMatrixLayoutSetAttribute(state.matrices[i], CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)));
            blas_check(cublasLtMatrixLayoutSetAttribute(state.matrices[i], CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
            blas_check(cublasLtMatrixLayoutSetAttribute(state.matrices[i], CUBLASLT_MATRIX_LAYOUT_LD, &leading, sizeof(leading)));
            blas_check(cublasLtMatrixLayoutSetAttribute(state.matrices[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
            const uintptr_t address = reinterpret_cast<uintptr_t>(matrix.data) | (count > 1 ? mul(matrix.bs, sizeof(double)) : 0);
            uint32_t alignment = 256;
            while (address % alignment) alignment /= 2;
            blas_check(cublasLtMatmulPreferenceSetAttribute(state.preference, alignments[i], &alignment, sizeof(alignment)));
            if (i == 2) blas_check(cublasLtMatmulPreferenceSetAttribute(state.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignment, sizeof(alignment)));
        }
        cublasLtMatmulHeuristicResult_t choice{};
        for (;;) {
            const int32_t group = static_cast<int32_t>(count);
            for (auto matrix : state.matrices)
                blas_check(cublasLtMatrixLayoutSetAttribute(matrix, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &group, sizeof(group)));
            int returned = 0;
            const auto status = cublasLtMatmulAlgoGetHeuristic(state.handle, state.operation,
                state.matrices[0], state.matrices[1], state.matrices[2], state.matrices[2], state.preference, 1, &choice, &returned);
            if (status != CUBLAS_STATUS_NOT_SUPPORTED) blas_check(status);
            if (status == CUBLAS_STATUS_SUCCESS && returned && choice.state == CUBLAS_STATUS_SUCCESS && choice.workspaceSize <= limit) break;
            if (count == 1) return 0;
            count = (count + 1) / 2;
        }
        if (choice.workspaceSize > reserved) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            workspace.reset(); reserved = 0;
            workspace = CudaBuf<std::byte>(choice.workspaceSize);
            reserved = choice.workspaceSize;
        }
        const double alpha = 1, beta = first ? 0 : 1;
        blas_check(cublasLtMatmul(state.handle, state.operation, &alpha, b.data, state.matrices[0],
            a.data, state.matrices[1], &beta, c.data, state.matrices[2], c.data, state.matrices[2],
            &choice.algo, workspace.get(), reserved, stream));
        return count;
    }
#else
    Fp64Emulation(int, uint64_t bytes) : limit(bytes) {
        throw std::runtime_error("fp64_emulation requires a CUDA plugin built with cuBLAS 13.1 or later");
    }
    uint64_t gemm(Matrix<double>, Matrix<double>, Matrix<double>, uint64_t, bool) { return 0; }
#endif
};
