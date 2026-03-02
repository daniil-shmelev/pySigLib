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
from timing_utils import time_pysiglib_sig, plot_times
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':
    cfg_cpu = {
        'batch_size': 1000,
        'length': 10,
        'dimension': 8,
        'degree_arr': list(range(1, 7)),
        'dtype': "float",
        'device': 'cpu',
        'num_runs': 5
    }

    cfg_cuda = {
        'batch_size': 1000,
        'length': 10,
        'dimension': 8,
        'degree_arr': list(range(1, 7)),
        'dtype': "float",
        'device': 'cuda',
        'num_runs': 5
    }

    cpu_direct_time = []
    cpu_horner_time = []
    cuda_direct_time = []
    cuda_horner_time = []

    for degree in tqdm(cfg_cpu['degree_arr']):
        cfg_cpu['degree'] = degree
        cfg_cuda['degree'] = degree
        cpu_direct_time.append(time_pysiglib_sig(cfg_cpu, False, -1))
        cpu_horner_time.append(time_pysiglib_sig(cfg_cpu, True, -1))
        cuda_direct_time.append(time_pysiglib_sig(cfg_cuda, False, 1))
        cuda_horner_time.append(time_pysiglib_sig(cfg_cuda, True, 1))

    print("CPU Direct:", cpu_direct_time)
    print("CPU Horner:", cpu_horner_time)
    print("CUDA Direct:", cuda_direct_time)
    print("CUDA Horner:", cuda_horner_time)

    for scale in ["linear", "log"]:
        plot_times(
            x=cfg_cpu['degree_arr'],
            ys=[cpu_direct_time, cpu_horner_time, cuda_direct_time, cuda_horner_time],
            legend=["CPU (Direct)", "CPU (Horner)", "CUDA (Direct)", "CUDA (Horner)"],
            linestyles=["-", "-", "--", "--"],
            title="Truncated Signatures: CPU vs CUDA",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="signature_times_cpu_vs_cuda_" + scale
        )
