# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import timeit

import numpy as np
import torch
import sigkernel

import pysiglib

if __name__ == '__main__':

    batch_size = 10
    length = 100
    dim = 5
    dyadic_order = 0

    X = np.random.uniform(size = (batch_size, length, dim)).astype("double")
    Y = np.random.uniform(size=(batch_size, length, dim)).astype("double")

    X = torch.tensor(X, device = "cuda")
    Y = torch.tensor(Y, device = "cuda")

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    start = timeit.default_timer()
    kernel = signature_kernel.compute_kernel(X, Y)
    end = timeit.default_timer()

    print(end - start)
    print(kernel)

    start = timeit.default_timer()
    kernel = pysiglib.sig_kernel(X, Y, dyadic_order=dyadic_order)
    end = timeit.default_timer()
    print(end - start)
    print(kernel)
