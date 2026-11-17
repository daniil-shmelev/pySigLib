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
from timing_utils import time_iisig_sig, time_pysiglib_sig, time_signatory_sig, plot_times
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)
# pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall

if __name__ == '__main__':
    cfg = {
        'batch_size': 32,
        'length': 1024,
        'dimension': 5,
        'degree_arr': list(range(1, 8)),
        'dtype': "float",
        'device': 'cpu',
        'num_runs': 5
    }

    iisigtime = []
    signatorytime = []
    pysiglibtime = []
    pysiglibtimeparallel = []
    pysiglibtimehorner = []
    pysiglibtimehornerparallel = []

    for degree in tqdm(cfg['degree_arr']):
        cfg['degree'] = degree
        iisigtime.append(time_iisig_sig(cfg))
        signatorytime.append(time_signatory_sig(cfg))
        pysiglibtime.append(time_pysiglib_sig(cfg, False, 1))
        pysiglibtimeparallel.append(time_pysiglib_sig(cfg, False, -1))
        pysiglibtimehorner.append(time_pysiglib_sig(cfg, True, 1))
        pysiglibtimehornerparallel.append(time_pysiglib_sig(cfg, True, -1))

    print(iisigtime)
    print(signatorytime)
    print(pysiglibtime)
    print(pysiglibtimeparallel)
    print(pysiglibtimehorner)
    print(pysiglibtimehornerparallel)

    for scale in ["linear", "log"]:
        plot_times(
            x= cfg['degree_arr'],
            ys= [iisigtime, pysiglibtime, pysiglibtimehorner],
            legend = ["iisignature (Direct)", "pySigLib (Direct)", "pySigLib (Horner)"],
            linestyles=["-", "--", "--"],
            title = "Truncated Signatures (Serial)",
            xlabel = "Truncation Level",
            ylabel = "Elapsed Time (s)",
            scale = scale,
            filename = "signature_times_" + scale + "_serial"
        )

        plot_times(
            x=cfg['degree_arr'],
            ys=[signatorytime, pysiglibtimeparallel, pysiglibtimehornerparallel],
            legend=["signatory (Horner)", "pySigLib (Direct)", "pySigLib (Horner)"],
            linestyles=["-", "--", "--"],
            title= "Truncated Signatures (Parallel)",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="signature_times_" + scale + "_parallel"
        )
