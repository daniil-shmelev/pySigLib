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

try:
    import jax
except:
    jax = None

import pysiglib

from timing_utils import time_iisig_sig, time_signatory_sig, time_pysiglib_sig, time_esig_sig, time_pysiglib_sig_jax

if __name__ == '__main__':
    cfg = {
        'batch_size': 128,
        'length': 256,
        'dimension': 4,
        'degree': 6,
        'dtype': "float",
        'device': 'cpu',
        'num_runs': 5
    }

    #print("\nesig (serial): ", time_esig_sig(cfg, True))
    print("\niisignature (serial): ", time_iisig_sig(cfg, False))
    print("\npysiglib (serial): ", time_pysiglib_sig(cfg, True, 1, False))
    if jax is not None and pysiglib.BUILT_WITH_JAX_FFI:
        print("\npysiglib (serial jax): ", time_pysiglib_sig_jax(cfg, True, 1, False))
    else:
        print("\npysiglib (jax): skipped")

    # print("\nsignatory (parallel): ", time_signatory_sig(cfg, True))
    print("\npysiglib (parallel): ", time_pysiglib_sig(cfg, True, -1, False))
    if jax is not None and pysiglib.BUILT_WITH_JAX_FFI:
        print("\npysiglib (parallel jax): ", time_pysiglib_sig_jax(cfg, True, -1, False))
