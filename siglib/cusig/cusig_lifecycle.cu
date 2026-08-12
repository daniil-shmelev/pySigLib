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

#include "cupch.h"
#include "cusig.h"
#include <cuda_runtime.h>

void release_signature_state();
void release_log_sig_state();
void release_exp_sig_state();
void release_sig_combine_state();
void release_branched_sig_gpu_state();
void release_branched_log_sig_gpu_state();
void release_log_sig_combine_state();
void release_sig_kernel_poly_state();

extern "C" {

	CUSIG_API void cusig_shutdown() noexcept {
		// Sync before freeing: in-flight kernels may still hold pointers into
		// the buffers we're about to cudaFree.
		try { (void)cudaDeviceSynchronize(); } catch (...) {}

		try { release_signature_state();        } catch (...) {}
		try { release_log_sig_state();          } catch (...) {}
		try { release_log_sig_combine_state();  } catch (...) {}
		try { release_sig_kernel_poly_state();   } catch (...) {}
		try { release_exp_sig_state();          } catch (...) {}
		try { release_sig_combine_state();      } catch (...) {}
		try { release_branched_sig_gpu_state(); } catch (...) {}
		try { release_branched_log_sig_gpu_state(); } catch (...) {}
	}

}
