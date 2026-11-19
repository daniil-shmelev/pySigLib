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
from timing_utils import time_pysiglib_sig, plot_times
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':
    cfg_cpu = {
        'batch_size': 1000,
        'length': 10,
        'dimension': 5,
        'degree_arr': list(range(1, 8)),
        'dtype': "float",
        'device': 'cpu',
        'num_runs': 5
    }

    cfg_cuda = {
        'batch_size': 1000,
        'length': 10,
        'dimension': 5,
        'degree_arr': list(range(1, 8)),
        'dtype': "float",
        'device': 'cuda',
        'num_runs': 5
    }

    cpu_time = []
    cuda_time = []

    for degree in tqdm(cfg_cpu['degree_arr']):
        cfg_cpu['degree'] = degree
        cfg_cuda['degree'] = degree
        cpu_time.append(time_pysiglib_sig(cfg_cpu, -1))
        cuda_time.append(time_pysiglib_sig(cfg_cuda, 1))

    print("CPU Horner:", cpu_time)
    print("CUDA Horner:", cuda_time)

    for scale in ["linear", "log"]:
        plot_times(
            x=cfg_cpu['degree_arr'],
            ys=[cpu_time, cuda_time],
            legend=["CPU (Horner)", "CUDA (Horner)"],
            linestyles=["-", "--"],
            title="Truncated Signatures: CPU vs CUDA",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="signature_times_cpu_vs_cuda_" + scale
        )
