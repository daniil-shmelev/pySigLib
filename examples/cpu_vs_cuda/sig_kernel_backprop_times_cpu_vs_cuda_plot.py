# Copyright 2026 Daniil Shmelev
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
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from timing_utils import time_pysiglib_kernel_backprop, plot_times
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':
    cfg_cpu = {
        'batch_size': 32,
        'dimension': 5,
        'dyadic_order': 0,
        'dtype': "double",
        'device': 'cpu',
        'num_runs': 5
    }

    cfg_cuda = {
        'batch_size': 32,
        'dimension': 5,
        'dyadic_order': 0,
        'dtype': "double",
        'device': 'cuda',
        'num_runs': 5
    }

    length_arr = list(range(10, 2100, 200))
    cpu_time = []
    cuda_time = []

    for length in tqdm(length_arr):
        cfg_cpu['length'] = length
        cfg_cuda['length'] = length
        cpu_time.append(time_pysiglib_kernel_backprop(cfg_cpu, -1))
        cuda_time.append(time_pysiglib_kernel_backprop(cfg_cuda, 1))

    print("CPU:", cpu_time)
    print("CUDA:", cuda_time)

    for scale in ["linear", "log"]:
        plot_times(
            x=length_arr,
            ys=[cpu_time, cuda_time],
            legend=["CPU", "CUDA"],
            linestyles=["-", "--"],
            title="Sig Kernel Backprop: CPU vs CUDA",
            xlabel="Path Length",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="sig_kernel_backprop_times_cpu_vs_cuda_" + scale
        )
