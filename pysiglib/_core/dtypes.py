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

from __future__ import annotations


from ctypes import c_float, c_double
import numpy as np
from ..load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA

######################################################
# Some dicts to simplify dtype cases
######################################################

DTYPES = {
    "float32": c_float,
    "float64": c_double
}

SUPPORTED_DTYPES = [
    np.float32,
    np.float64,
]

SUPPORTED_DTYPES_STR = "float or double"

CPSIG_TRANSFORM_PATH = {
    "float32": CPSIG.transform_path_f,
    "float64": CPSIG.transform_path_d
}

CPSIG_TRANSFORM_PATH_BACKPROP = {
    "float32": CPSIG.transform_path_backprop_f,
    "float64": CPSIG.transform_path_backprop_d
}

CUSIG_TRANSFORM_PATH = None
CUSIG_TRANSFORM_PATH_BACKPROP = None
CUSIG_SIG_KERNEL = None
CUSIG_SIG_KERNEL_BACKPROP = None
CUSIG_SIG_KERNEL_POLY = None
CUSIG_SIG_KERNEL_POLY_BACKPROP = None
CUSIG_BRANCHED_SIG_KERNEL = None
CUSIG_BRANCHED_SIG_KERNEL_BACKPROP = None
CUSIG_SIGNATURE = None
CUSIG_SIG_BACKPROP = None
CUSIG_SIG_COMBINE = None
CUSIG_SIG_COMBINE_BACKPROP = None
CUSIG_SIG_TO_LOG_SIG = None
CUSIG_SIG_TO_LOG_SIG_BACKPROP = None
CUSIG_SIG_COEF = None
CUSIG_SIG_COEF_BACKPROP = None
CUSIG_LOG_SIG_COMBINE = None
CUSIG_LOG_SIG_COMBINE_BACKPROP = None
CUSIG_LOG_SIG_FROM_PATH = None
CUSIG_LOG_SIG_FROM_PATH_BACKPROP = None
CUSIG_LOGSIG_TO_SIG = None
CUSIG_LOGSIG_TO_SIG_BACKPROP = None
CUSIG_LINEAR_SIG = None
CUSIG_SIG_JOIN = None
CUSIG_SIG_JOIN_BACKPROP = None
CUSIG_LOG_SIG_JOIN = None
CUSIG_LOG_SIG_JOIN_BACKPROP = None
CUSIG_BRANCHED_SIG = None
CUSIG_BRANCHED_SIG_COEF = None
CUSIG_BRANCHED_SIG_COEF_BACKPROP = None
CUSIG_BRANCHED_SIG_COMBINE = None
CUSIG_BRANCHED_SIG_COMBINE_BACKPROP = None
CUSIG_BRANCHED_SIG_BACKPROP = None
CUSIG_BRANCHED_SIG_TO_LOG_SIG = None
CUSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP = None
CUSIG_BRANCHED_LOG_SIG_FROM_PATH = None
CUSIG_BRANCHED_LOG_SIG_FROM_PATH_BACKPROP = None
if BUILT_WITH_CUDA:
    CUSIG_TRANSFORM_PATH = {
        "float32": CUSIG.transform_path_cuda_f,
        "float64": CUSIG.transform_path_cuda_d
    }

    CUSIG_TRANSFORM_PATH_BACKPROP = {
        "float32": CUSIG.transform_path_backprop_cuda_f,
        "float64": CUSIG.transform_path_backprop_cuda_d
    }

    CUSIG_SIG_KERNEL = {
        "float32": CUSIG.sig_kernel_cuda_f,
        "float64": CUSIG.sig_kernel_cuda_d
    }

    CUSIG_SIG_KERNEL_POLY = {
        "float32": CUSIG.sig_kernel_poly_cuda_f,
        "float64": CUSIG.sig_kernel_poly_cuda_d
    }

    CUSIG_SIG_KERNEL_POLY_BACKPROP = {
        "float32": CUSIG.sig_kernel_poly_backprop_cuda_f,
        "float64": CUSIG.sig_kernel_poly_backprop_cuda_d
    }

    CUSIG_SIG_KERNEL_BACKPROP = {
        "float32": CUSIG.sig_kernel_backprop_cuda_f,
        "float64": CUSIG.sig_kernel_backprop_cuda_d
    }

    CUSIG_BRANCHED_SIG_KERNEL = {
        "float32": CUSIG.branched_sig_kernel_cuda_f,
        "float64": CUSIG.branched_sig_kernel_cuda_d
    }

    CUSIG_BRANCHED_SIG_KERNEL_BACKPROP = {
        "float32": CUSIG.branched_sig_kernel_backprop_cuda_f,
        "float64": CUSIG.branched_sig_kernel_backprop_cuda_d
    }

    CUSIG_SIGNATURE = {
        "float32": CUSIG.signature_cuda_f,
        "float64": CUSIG.signature_cuda_d
    }

    CUSIG_SIG_BACKPROP = {
        "float32": CUSIG.sig_backprop_cuda_f,
        "float64": CUSIG.sig_backprop_cuda_d
    }

    CUSIG_SIG_COMBINE = {
        "float32": CUSIG.sig_combine_cuda_f,
        "float64": CUSIG.sig_combine_cuda_d
    }

    CUSIG_SIG_COMBINE_BACKPROP = {
        "float32": CUSIG.sig_combine_backprop_cuda_f,
        "float64": CUSIG.sig_combine_backprop_cuda_d
    }

    CUSIG_SIG_TO_LOG_SIG = {
        "float32": CUSIG.sig_to_log_sig_cuda_f,
        "float64": CUSIG.sig_to_log_sig_cuda_d
    }

    CUSIG_SIG_TO_LOG_SIG_BACKPROP = {
        "float32": CUSIG.sig_to_log_sig_backprop_cuda_f,
        "float64": CUSIG.sig_to_log_sig_backprop_cuda_d
    }

    CUSIG_SIG_COEF = {
        "float32": CUSIG.sig_coef_cuda_f,
        "float64": CUSIG.sig_coef_cuda_d
    }

    CUSIG_SIG_COEF_BACKPROP = {
        "float32": CUSIG.sig_coef_backprop_cuda_f,
        "float64": CUSIG.sig_coef_backprop_cuda_d
    }

    CUSIG_LOG_SIG_COMBINE = {
        "float32": CUSIG.log_sig_combine_cuda_f,
        "float64": CUSIG.log_sig_combine_cuda_d
    }

    CUSIG_LOG_SIG_COMBINE_BACKPROP = {
        "float32": CUSIG.log_sig_combine_backprop_cuda_f,
        "float64": CUSIG.log_sig_combine_backprop_cuda_d
    }

    CUSIG_LOG_SIG_FROM_PATH = {
        "float32": CUSIG.log_sig_from_path_cuda_f,
        "float64": CUSIG.log_sig_from_path_cuda_d
    }

    CUSIG_LOG_SIG_FROM_PATH_BACKPROP = {
        "float32": CUSIG.log_sig_from_path_backprop_cuda_f,
        "float64": CUSIG.log_sig_from_path_backprop_cuda_d
    }

    CUSIG_LOGSIG_TO_SIG = {
        "float32": CUSIG.logsig_to_sig_cuda_f,
        "float64": CUSIG.logsig_to_sig_cuda_d
    }

    CUSIG_LOGSIG_TO_SIG_BACKPROP = {
        "float32": CUSIG.logsig_to_sig_backprop_cuda_f,
        "float64": CUSIG.logsig_to_sig_backprop_cuda_d
    }

    CUSIG_LINEAR_SIG = {
        "float32": CUSIG.linear_sig_cuda_f,
        "float64": CUSIG.linear_sig_cuda_d
    }

    CUSIG_SIG_JOIN = {
        "float32": CUSIG.sig_join_cuda_f,
        "float64": CUSIG.sig_join_cuda_d
    }

    CUSIG_SIG_JOIN_BACKPROP = {
        "float32": CUSIG.sig_join_backprop_cuda_f,
        "float64": CUSIG.sig_join_backprop_cuda_d
    }

    CUSIG_LOG_SIG_JOIN = {
        "float32": CUSIG.log_sig_join_cuda_f,
        "float64": CUSIG.log_sig_join_cuda_d
    }

    CUSIG_LOG_SIG_JOIN_BACKPROP = {
        "float32": CUSIG.log_sig_join_backprop_cuda_f,
        "float64": CUSIG.log_sig_join_backprop_cuda_d
    }

CPSIG_LOG_SIG_COMBINE = {
    "float32": CPSIG.log_sig_combine_f,
    "float64": CPSIG.log_sig_combine_d
}

CPSIG_LOG_SIG_FROM_PATH = {
    "float32": CPSIG.log_sig_from_path_f,
    "float64": CPSIG.log_sig_from_path_d
}

CPSIG_LOG_SIG_FROM_PATH_BACKPROP = {
    "float32": CPSIG.log_sig_from_path_backprop_f,
    "float64": CPSIG.log_sig_from_path_backprop_d
}

CPSIG_LOG_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.log_sig_combine_backprop_f,
    "float64": CPSIG.log_sig_combine_backprop_d
}

CPSIG_SIG_COEF = {
    "float32": CPSIG.sig_coef_f,
    "float64": CPSIG.sig_coef_d
}

CPSIG_SIG_COEF_BACKPROP = {
    "float32": CPSIG.sig_coef_backprop_f,
    "float64": CPSIG.sig_coef_backprop_d
}

CPSIG_SIGNATURE = {
    "float32": CPSIG.signature_f,
    "float64": CPSIG.signature_d
}

CPSIG_SIG_BACKPROP = {
    "float32": CPSIG.sig_backprop_f,
    "float64": CPSIG.sig_backprop_d
}

CPSIG_SIG_COMBINE = {
    "float32": CPSIG.sig_combine_f,
    "float64": CPSIG.sig_combine_d
}

CPSIG_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.sig_combine_backprop_f,
    "float64": CPSIG.sig_combine_backprop_d
}

CPSIG_SIG_TO_LOG_SIG = {
    "float32": CPSIG.sig_to_log_sig_f,
    "float64": CPSIG.sig_to_log_sig_d
}

CPSIG_SIG_TO_LOG_SIG_BACKPROP = {
    "float32": CPSIG.sig_to_log_sig_backprop_f,
    "float64": CPSIG.sig_to_log_sig_backprop_d
}

CPSIG_LOGSIG_TO_SIG = {
    "float32": CPSIG.logsig_to_sig_f,
    "float64": CPSIG.logsig_to_sig_d
}

CPSIG_LOGSIG_TO_SIG_BACKPROP = {
    "float32": CPSIG.logsig_to_sig_backprop_f,
    "float64": CPSIG.logsig_to_sig_backprop_d
}

CPSIG_LINEAR_SIG = {
    "float32": CPSIG.linear_sig_f,
    "float64": CPSIG.linear_sig_d
}

CPSIG_SIG_JOIN = {
    "float32": CPSIG.sig_join_f,
    "float64": CPSIG.sig_join_d
}

CPSIG_SIG_JOIN_BACKPROP = {
    "float32": CPSIG.sig_join_backprop_f,
    "float64": CPSIG.sig_join_backprop_d
}

CPSIG_LOG_SIG_JOIN = {
    "float32": CPSIG.log_sig_join_f,
    "float64": CPSIG.log_sig_join_d
}

CPSIG_LOG_SIG_JOIN_BACKPROP = {
    "float32": CPSIG.log_sig_join_backprop_f,
    "float64": CPSIG.log_sig_join_backprop_d
}

CPSIG_SIG_KERNEL = {
    "float32": CPSIG.sig_kernel_f,
    "float64": CPSIG.sig_kernel_d
}

CPSIG_SIG_KERNEL_POLY = {
    "float32": CPSIG.sig_kernel_poly_f,
    "float64": CPSIG.sig_kernel_poly_d
}

CPSIG_SIG_KERNEL_POLY_BACKPROP = {
    "float32": CPSIG.sig_kernel_poly_backprop_f,
    "float64": CPSIG.sig_kernel_poly_backprop_d
}

CPSIG_SIG_KERNEL_BACKPROP = {
    "float32": CPSIG.sig_kernel_backprop_f,
    "float64": CPSIG.sig_kernel_backprop_d
}

CPSIG_BRANCHED_SIG_KERNEL = {
    "float32": CPSIG.branched_sig_kernel_f,
    "float64": CPSIG.branched_sig_kernel_d
}

CPSIG_BRANCHED_SIG_KERNEL_BACKPROP = {
    "float32": CPSIG.branched_sig_kernel_backprop_f,
    "float64": CPSIG.branched_sig_kernel_backprop_d
}

CPSIG_BRANCHED_SIG = {
    "float32": CPSIG.branched_sig_f,
    "float64": CPSIG.branched_sig_d
}

CPSIG_BRANCHED_SIG_COEF = {
    "float32": CPSIG.branched_sig_coef_f,
    "float64": CPSIG.branched_sig_coef_d
}

CPSIG_BRANCHED_SIG_COEF_BACKPROP = {
    "float32": CPSIG.branched_sig_coef_backprop_f,
    "float64": CPSIG.branched_sig_coef_backprop_d
}

CPSIG_BRANCHED_SIG_COMBINE = {
    "float32": CPSIG.branched_sig_combine_f,
    "float64": CPSIG.branched_sig_combine_d
}

CPSIG_BRANCHED_SIG_TO_LOG_SIG = {
    "float32": CPSIG.branched_sig_to_log_sig_f,
    "float64": CPSIG.branched_sig_to_log_sig_d
}

CPSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP = {
    "float32": CPSIG.branched_sig_to_log_sig_backprop_f,
    "float64": CPSIG.branched_sig_to_log_sig_backprop_d
}

CPSIG_BRANCHED_LOG_SIG_FROM_PATH = {
    "float32": CPSIG.branched_log_sig_from_path_f,
    "float64": CPSIG.branched_log_sig_from_path_d
}

CPSIG_BRANCHED_LOG_SIG_FROM_PATH_BACKPROP = {
    "float32": CPSIG.branched_log_sig_from_path_backprop_f,
    "float64": CPSIG.branched_log_sig_from_path_backprop_d
}

CPSIG_BRANCHED_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.branched_sig_combine_backprop_f,
    "float64": CPSIG.branched_sig_combine_backprop_d
}

CPSIG_BRANCHED_SIG_BACKPROP = {
    "float32": CPSIG.branched_sig_backprop_f,
    "float64": CPSIG.branched_sig_backprop_d
}

if BUILT_WITH_CUDA:
    CUSIG_BRANCHED_SIG = {
        "float32": CUSIG.branched_sig_cuda_f,
        "float64": CUSIG.branched_sig_cuda_d
    }

    CUSIG_BRANCHED_SIG_COEF = {
        "float32": CUSIG.branched_sig_coef_cuda_f,
        "float64": CUSIG.branched_sig_coef_cuda_d
    }

    CUSIG_BRANCHED_SIG_COEF_BACKPROP = {
        "float32": CUSIG.branched_sig_coef_backprop_cuda_f,
        "float64": CUSIG.branched_sig_coef_backprop_cuda_d
    }

    CUSIG_BRANCHED_SIG_COMBINE = {
        "float32": CUSIG.branched_sig_combine_cuda_f,
        "float64": CUSIG.branched_sig_combine_cuda_d
    }
    CUSIG_BRANCHED_SIG_COMBINE_BACKPROP = {
        "float32": CUSIG.branched_sig_combine_backprop_cuda_f,
        "float64": CUSIG.branched_sig_combine_backprop_cuda_d
    }

    CUSIG_BRANCHED_SIG_BACKPROP = {
        "float32": CUSIG.branched_sig_backprop_cuda_f,
        "float64": CUSIG.branched_sig_backprop_cuda_d
    }

    CUSIG_BRANCHED_SIG_TO_LOG_SIG = {
        "float32": CUSIG.branched_sig_to_log_sig_cuda_f,
        "float64": CUSIG.branched_sig_to_log_sig_cuda_d
    }

    CUSIG_BRANCHED_SIG_TO_LOG_SIG_BACKPROP = {
        "float32": CUSIG.branched_sig_to_log_sig_backprop_cuda_f,
        "float64": CUSIG.branched_sig_to_log_sig_backprop_cuda_d
    }

    CUSIG_BRANCHED_LOG_SIG_FROM_PATH = {
        "float32": CUSIG.branched_log_sig_from_path_cuda_f,
        "float64": CUSIG.branched_log_sig_from_path_cuda_d
    }

    CUSIG_BRANCHED_LOG_SIG_FROM_PATH_BACKPROP = {
        "float32": CUSIG.branched_log_sig_from_path_backprop_cuda_f,
        "float64": CUSIG.branched_log_sig_from_path_backprop_cuda_d
    }
