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

from tqdm import tqdm
import torch
from timing_utils import time_sigkernel_kernel_backprop, time_pysiglib_kernel_backprop, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    cfg = {
        'batch_size': 32,
        'dimension': 5,
        'dyadic_order': 0,
        'dtype': "double",
        'device': 'cuda',
        'num_runs': 10
    }

    length_arr = list(range(10, 2100, 200))
    sigkerneltime = []
    pysiglibtime = []


    for length in tqdm(length_arr):
        cfg['length'] = length
        sigkerneltime.append(time_sigkernel_kernel_backprop(cfg))
        pysiglibtime.append(time_pysiglib_kernel_backprop(cfg, -1))

    print(sigkerneltime)
    print(pysiglibtime)

    for scale in ["linear", "log"]:
        plot_times(
                x=length_arr[:9],
                ys= [sigkerneltime[:9], pysiglibtime[:9]],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels Backprop " + ("(CPU)" if cfg['device'] == "cpu" else "(CUDA)"),
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_backprop_times_len_" + scale + "_" + cfg['device'] + "_1",
                linestyles = ["-", "--"]
        )
        plot_times(
                x= length_arr,
                ys= [sigkerneltime, pysiglibtime],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels Backprop " + ("(CPU)" if cfg['device'] == "cpu" else "(CUDA)"),
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_backprop_times_len_" + scale + "_" + cfg['device'] + "_2",
                linestyles = ["-", "--"]
        )
