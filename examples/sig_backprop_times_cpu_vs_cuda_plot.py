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
from timing_utils import time_pysiglib_sig_backprop, plot_times
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':
    cfg_cpu = {
        'batch_size': 100,
        'length': 10,
        'dimension': 5,
        'degree_arr': list(range(1, 8)),
        'dtype': "double",
        'device': 'cpu',
        'num_runs': 50
    }

    cfg_cuda = {
        'batch_size': 100,
        'length': 10,
        'dimension': 5,
        'degree_arr': list(range(1, 8)),
        'dtype': "double",
        'device': 'cuda',
        'num_runs': 50
    }

    cpu_serial_time = []
    cpu_parallel_time = []
    cuda_time = []

    for degree in tqdm(cfg_cpu['degree_arr']):
        cfg_cpu['degree'] = degree
        cfg_cuda['degree'] = degree
        cpu_serial_time.append(time_pysiglib_sig_backprop(cfg_cpu, 1))
        cpu_parallel_time.append(time_pysiglib_sig_backprop(cfg_cpu, -1))
        cuda_time.append(time_pysiglib_sig_backprop(cfg_cuda, 1))

    print("CPU (Serial):", cpu_serial_time)
    print("CPU (Parallel):", cpu_parallel_time)
    print("CUDA:", cuda_time)

    for scale in ["linear", "log"]:
        plot_times(
            x=cfg_cpu['degree_arr'],
            ys=[cpu_serial_time, cpu_parallel_time, cuda_time],
            legend=["CPU (Serial)", "CPU (Parallel)", "CUDA"],
            linestyles=["-", "-", "--"],
            title="Sig Backprop: CPU vs CUDA",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="sig_backprop_times_cpu_vs_cuda_" + scale
        )
