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

// test_app.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#if defined(_WIN32)
    #include <Windows.h>
    #include <strsafe.h>
#else
    #include <stdlib.h>
    #include <stdio.h>
    #include <dlfcn.h>
    #include <float.h>
#endif

#include "dll_funcs.h"
#include "tests.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>


int main(int argc, char* argv[])
{
    std::string dir_path(".");

    if (argc >= 2) {
        dir_path = argv[1];
    }

    load_cpsig(dir_path);
    get_cpsig_fn_ptrs();

    if (argc >= 3 && (std::string(argv[2]) == "sig-kernel-profile"
        || std::string(argv[2]) == "sig-kernel-backprop-profile")) {
        if (argc != 7) {
            std::cerr << "Usage: pysiglib_test_app <dll-dir> "
                << "<sig-kernel-profile|sig-kernel-backprop-profile> "
                << "<polynomial|finite_difference> <segments> <parameter> <iterations>"
                << std::endl;
            unload_cpsig();
            return 2;
        }

        const std::string method(argv[3]);
        const uint64_t segments = std::stoull(argv[4]);
        const uint64_t order = std::stoull(argv[5]);
        const uint64_t iterations = std::stoull(argv[6]);
        if (segments == 0 || iterations == 0) {
            std::cerr << "segments and iterations must be positive" << std::endl;
            unload_cpsig();
            return 2;
        }

        if (method != "polynomial" && method != "finite_difference") {
            std::cerr << "method must be polynomial or finite_difference" << std::endl;
            unload_cpsig();
            return 2;
        }
        std::mt19937_64 rng(25022025);
        std::normal_distribution<double> normal(0.0, 1.0 / std::sqrt(static_cast<double>(segments)));
        std::vector<double> dx(2 * segments);
        std::vector<double> dy(2 * segments);
        for (double& value : dx)
            value = normal(rng);
        for (double& value : dy)
            value = normal(rng);

        std::vector<double> gram(segments * segments);
        for (uint64_t i = 0; i < segments; ++i) {
            for (uint64_t j = 0; j < segments; ++j) {
                gram[i * segments + j] = dx[2 * i] * dy[2 * j]
                    + dx[2 * i + 1] * dy[2 * j + 1];
            }
        }

		double out = 0;
		int status = 0;
		const bool backprop = std::string(argv[2]) == "sig-kernel-backprop-profile";
		std::vector<double> gram_derivs(segments * segments);
		std::vector<double> solver_state;
		if (method == "polynomial") {
			solver_state.resize(2 * segments * segments * (order + 1));
			status = polysig_kernel_d(
				gram.data(), &out, backprop ? solver_state.data() : nullptr,
				1, 2, segments + 1, segments + 1, order, false, 1);
		}
		else {
			solver_state.resize((segments + 1) * (segments + 1));
			status = sig_kernel_d(
				gram.data(), backprop ? solver_state.data() : &out,
				1, 2, segments + 1, segments + 1, order, order, backprop, 1);
		}

		const double output_deriv = 1.0;
		const auto start = std::chrono::steady_clock::now();
		for (uint64_t i = 0; i < iterations && status == 0; ++i) {
			if (!backprop && method == "polynomial")
				status = polysig_kernel_d(
					gram.data(), &out, nullptr, 1, 2, segments + 1,
					segments + 1, order, false, 1);
			else if (!backprop)
				status = sig_kernel_d(
					gram.data(), &out, 1, 2, segments + 1, segments + 1,
					order, order, false, 1);
			else if (method == "polynomial")
				status = polysig_kernel_backprop_d(
					gram.data(), gram_derivs.data(), &output_deriv,
					solver_state.data(), 1, 2, segments + 1, segments + 1,
					order, false, 1);
			else
				status = sig_kernel_backprop_d(
					gram.data(), gram_derivs.data(), &output_deriv,
					solver_state.data(), 1, 2, segments + 1, segments + 1,
					order, order, false, 1);
		}
		const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

		std::cout << method << " segments=" << segments << " order=" << order
			<< " iterations=" << iterations << " seconds=" << elapsed
			<< " seconds_per_iteration=" << elapsed / static_cast<double>(iterations)
			<< " result=" << (backprop ? gram_derivs.back() : out) << std::endl;
        unload_cpsig();
        return status;
    }

    load_cusig(dir_path);
    get_cusig_fn_ptrs();

    //example_batch_signature_d(1000, 6, 100, 6, false, false, true, -1, 10);
    //example_batch_signature_cuda_d(1000, 6, 100, 6, false, false, true, 10); // Min run time: 156ms

    //// ---- Forward pass benchmark: single path, dim=5, len=1000, deg=5 ----
    //example_signature_d(5, 1000, 5);
    //example_signature_cuda_d(5, 1000, 5);

    //// ---- Backprop benchmark: batch=100, dim=6, len=100, deg=6 ----
    //example_batch_sig_backprop_d(100, 6, 100, 6);
    //example_batch_sig_backprop_cuda_d(100, 6, 100, 5);

    //example_batch_sig_to_log_sig_d();
    //example_batch_sig_to_log_sig_cuda_d();

    //example_batch_sig_to_log_sig_backprop_d();
    //example_batch_sig_to_log_sig_backprop_cuda_d();

    //example_batch_sig_combine_cuda_d();

    //example_batch_sig_combine_backprop_d(5200, 6, 6, -1, 50);  // CPU all cores
    //example_batch_sig_combine_backprop_cuda_d();

    //example_batch_sig_coef(100, 5000, 5, 5, 1000, false, false, 1., -1, 10);  // CPU all cores
    //example_batch_sig_coef_cuda_d(100, 5000, 5, 5, 1000, 10);               // CUDA

    // benchmark: sig_coef_backprop_cuda_d (batch=64, dim=8, deg=5, len=1000, num_idx=100)
    //example_batch_sig_coef_backprop_cuda_d(100, 1000, 8, 5, 400, 10);

    //example_batch_log_sig_combine_d(1700, 5, 5, -1, 20);
    //example_batch_log_sig_combine_cuda_d(45000, 5, 5, 20);

    //example_batch_log_sig_combine_backprop_d(13000, 5, 5, -1, 20);
    //example_batch_log_sig_combine_backprop_cuda_d(13000, 5, 5, 20);
    //example_batch_branched_log_sig_d(2, 3, 8, 4, 1);
    example_batch_branched_sig_coef_cuda(32, 64, 3, 4, 256, 10, true);
    example_batch_branched_sig_coef_backprop_cuda(32, 64, 3, 4, 256, 10, true);
    //example_batch_branched_sig_d(1, 3, 1000, 4, 1, 50);
    //example_batch_branched_sig_d(100, 3, 100, 4, -1, 50);

    unload_cpsig();
    unload_cusig();
}
