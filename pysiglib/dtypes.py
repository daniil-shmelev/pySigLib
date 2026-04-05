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

from ctypes import c_float, c_double
import numpy as np
import torch
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA

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
    torch.float32,
    torch.float64
]

SUPPORTED_DTYPES_STR = "float or double"

CPSIG_TRANSFORM_PATH = {
    "float32": CPSIG.transform_path_f,
    "float64": CPSIG.transform_path_d
}

CPSIG_BATCH_TRANSFORM_PATH = {
    "float32": CPSIG.batch_transform_path_f,
    "float64": CPSIG.batch_transform_path_d
}

CPSIG_TRANSFORM_PATH_BACKPROP = {
    "float32": CPSIG.transform_path_backprop_f,
    "float64": CPSIG.transform_path_backprop_d
}

CPSIG_BATCH_TRANSFORM_PATH_BACKPROP = {
    "float32": CPSIG.batch_transform_path_backprop_f,
    "float64": CPSIG.batch_transform_path_backprop_d
}

CUSIG_TRANSFORM_PATH_CUDA = None
CUSIG_BATCH_TRANSFORM_PATH_CUDA = None
CUSIG_TRANSFORM_PATH_BACKPROP_CUDA = None
CUSIG_BATCH_TRANSFORM_PATH_BACKPROP_CUDA = None
CUSIG_BATCH_SIG_KERNEL_CUDA = None
CUSIG_BATCH_SIG_KERNEL_BACKPROP_CUDA = None
CUSIG_SIGNATURE_CUDA = None
CUSIG_BATCH_SIGNATURE_CUDA = None
CUSIG_SIG_BACKPROP_CUDA = None
CUSIG_BATCH_SIG_BACKPROP_CUDA = None
CUSIG_SIG_COMBINE_CUDA = None
CUSIG_BATCH_SIG_COMBINE_CUDA = None
CUSIG_SIG_COMBINE_BACKPROP_CUDA = None
CUSIG_BATCH_SIG_COMBINE_BACKPROP_CUDA = None
CUSIG_SIG_TO_LOG_SIG_CUDA = None
CUSIG_BATCH_SIG_TO_LOG_SIG_CUDA = None
CUSIG_SIG_TO_LOG_SIG_BACKPROP_CUDA = None
CUSIG_BATCH_SIG_TO_LOG_SIG_BACKPROP_CUDA = None
CUSIG_SIG_COEF_CUDA = None
CUSIG_BATCH_SIG_COEF_CUDA = None
CUSIG_SIG_COEF_BACKPROP_CUDA = None
CUSIG_BATCH_SIG_COEF_BACKPROP_CUDA = None
CUSIG_BATCH_LOG_SIG_COMBINE_CUDA = None
CUSIG_BATCH_LOG_SIG_COMBINE_BACKPROP_CUDA = None
CUSIG_BATCH_LOG_SIG_FROM_PATH_CUDA = None
CUSIG_BATCH_LOG_SIG_FROM_PATH_BACKPROP_CUDA = None
if BUILT_WITH_CUDA:
    CUSIG_TRANSFORM_PATH_CUDA = {
        "float32": CUSIG.transform_path_cuda_f,
        "float64": CUSIG.transform_path_cuda_d
    }

    CUSIG_BATCH_TRANSFORM_PATH_CUDA = {
        "float32": CUSIG.batch_transform_path_cuda_f,
        "float64": CUSIG.batch_transform_path_cuda_d
    }

    CUSIG_TRANSFORM_PATH_BACKPROP_CUDA = {
        "float32": CUSIG.transform_path_backprop_cuda_f,
        "float64": CUSIG.transform_path_backprop_cuda_d
    }

    CUSIG_BATCH_TRANSFORM_PATH_BACKPROP_CUDA = {
        "float32": CUSIG.batch_transform_path_backprop_cuda_f,
        "float64": CUSIG.batch_transform_path_backprop_cuda_d
    }

    CUSIG_BATCH_SIG_KERNEL_CUDA = {
        "float32": CUSIG.batch_sig_kernel_cuda_f,
        "float64": CUSIG.batch_sig_kernel_cuda_d
    }

    CUSIG_BATCH_SIG_KERNEL_BACKPROP_CUDA = {
        "float32": CUSIG.batch_sig_kernel_backprop_cuda_f,
        "float64": CUSIG.batch_sig_kernel_backprop_cuda_d
    }

    CUSIG_SIGNATURE_CUDA = {
        "float32": CUSIG.signature_cuda_f,
        "float64": CUSIG.signature_cuda_d
    }

    CUSIG_BATCH_SIGNATURE_CUDA = {
        "float32": CUSIG.batch_signature_cuda_f,
        "float64": CUSIG.batch_signature_cuda_d
    }

    CUSIG_SIG_BACKPROP_CUDA = {
        "float32": CUSIG.sig_backprop_cuda_f,
        "float64": CUSIG.sig_backprop_cuda_d
    }

    CUSIG_BATCH_SIG_BACKPROP_CUDA = {
        "float32": CUSIG.batch_sig_backprop_cuda_f,
        "float64": CUSIG.batch_sig_backprop_cuda_d
    }

    CUSIG_SIG_COMBINE_CUDA = {
        "float32": CUSIG.sig_combine_cuda_f,
        "float64": CUSIG.sig_combine_cuda_d
    }

    CUSIG_BATCH_SIG_COMBINE_CUDA = {
        "float32": CUSIG.batch_sig_combine_cuda_f,
        "float64": CUSIG.batch_sig_combine_cuda_d
    }

    CUSIG_SIG_COMBINE_BACKPROP_CUDA = {
        "float32": CUSIG.sig_combine_backprop_cuda_f,
        "float64": CUSIG.sig_combine_backprop_cuda_d
    }

    CUSIG_BATCH_SIG_COMBINE_BACKPROP_CUDA = {
        "float32": CUSIG.batch_sig_combine_backprop_cuda_f,
        "float64": CUSIG.batch_sig_combine_backprop_cuda_d
    }

    CUSIG_SIG_TO_LOG_SIG_CUDA = {
        "float32": CUSIG.sig_to_log_sig_cuda_f,
        "float64": CUSIG.sig_to_log_sig_cuda_d
    }

    CUSIG_BATCH_SIG_TO_LOG_SIG_CUDA = {
        "float32": CUSIG.batch_sig_to_log_sig_cuda_f,
        "float64": CUSIG.batch_sig_to_log_sig_cuda_d
    }

    CUSIG_SIG_TO_LOG_SIG_BACKPROP_CUDA = {
        "float32": CUSIG.sig_to_log_sig_backprop_cuda_f,
        "float64": CUSIG.sig_to_log_sig_backprop_cuda_d
    }

    CUSIG_BATCH_SIG_TO_LOG_SIG_BACKPROP_CUDA = {
        "float32": CUSIG.batch_sig_to_log_sig_backprop_cuda_f,
        "float64": CUSIG.batch_sig_to_log_sig_backprop_cuda_d
    }

    CUSIG_SIG_COEF_CUDA = {
        "float32": CUSIG.sig_coef_cuda_f,
        "float64": CUSIG.sig_coef_cuda_d
    }

    CUSIG_BATCH_SIG_COEF_CUDA = {
        "float32": CUSIG.batch_sig_coef_cuda_f,
        "float64": CUSIG.batch_sig_coef_cuda_d
    }

    CUSIG_SIG_COEF_BACKPROP_CUDA = {
        "float32": CUSIG.sig_coef_backprop_cuda_f,
        "float64": CUSIG.sig_coef_backprop_cuda_d
    }

    CUSIG_BATCH_SIG_COEF_BACKPROP_CUDA = {
        "float32": CUSIG.batch_sig_coef_backprop_cuda_f,
        "float64": CUSIG.batch_sig_coef_backprop_cuda_d
    }

    CUSIG_BATCH_LOG_SIG_COMBINE_CUDA = {
        "float32": CUSIG.batch_log_sig_combine_cuda_f,
        "float64": CUSIG.batch_log_sig_combine_cuda_d
    }

    CUSIG_BATCH_LOG_SIG_COMBINE_BACKPROP_CUDA = {
        "float32": CUSIG.batch_log_sig_combine_backprop_cuda_f,
        "float64": CUSIG.batch_log_sig_combine_backprop_cuda_d
    }

    CUSIG_BATCH_LOG_SIG_FROM_PATH_CUDA = {
        "float32": CUSIG.batch_log_sig_from_path_cuda_f,
        "float64": CUSIG.batch_log_sig_from_path_cuda_d
    }

    CUSIG_BATCH_LOG_SIG_FROM_PATH_BACKPROP_CUDA = {
        "float32": CUSIG.batch_log_sig_from_path_backprop_cuda_f,
        "float64": CUSIG.batch_log_sig_from_path_backprop_cuda_d
    }

CPSIG_BATCH_LOG_SIG_COMBINE = {
    "float32": CPSIG.batch_log_sig_combine_f,
    "float64": CPSIG.batch_log_sig_combine_d
}

CPSIG_BATCH_LOG_SIG_FROM_PATH = {
    "float32": CPSIG.batch_log_sig_from_path_f,
    "float64": CPSIG.batch_log_sig_from_path_d
}

CPSIG_BATCH_LOG_SIG_FROM_PATH_BACKPROP = {
    "float32": CPSIG.batch_log_sig_from_path_backprop_f,
    "float64": CPSIG.batch_log_sig_from_path_backprop_d
}

CPSIG_LOG_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.log_sig_combine_backprop_f,
    "float64": CPSIG.log_sig_combine_backprop_d
}

CPSIG_BATCH_LOG_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.batch_log_sig_combine_backprop_f,
    "float64": CPSIG.batch_log_sig_combine_backprop_d
}

CPSIG_SIG_COEF = {
    "float32": CPSIG.sig_coef_f,
    "float64": CPSIG.sig_coef_d
}

CPSIG_BATCH_SIG_COEF = {
    "float32": CPSIG.batch_sig_coef_f,
    "float64": CPSIG.batch_sig_coef_d
}

CPSIG_SIG_COEF_BACKPROP = {
    "float32": CPSIG.sig_coef_backprop_f,
    "float64": CPSIG.sig_coef_backprop_d
}

CPSIG_BATCH_SIG_COEF_BACKPROP = {
    "float32": CPSIG.batch_sig_coef_backprop_f,
    "float64": CPSIG.batch_sig_coef_backprop_d
}

CPSIG_SIGNATURE = {
    "float32": CPSIG.signature_f,
    "float64": CPSIG.signature_d
}

CPSIG_BATCH_SIGNATURE = {
    "float32": CPSIG.batch_signature_f,
    "float64": CPSIG.batch_signature_d
}

CPSIG_SIG_BACKPROP = {
    "float32": CPSIG.sig_backprop_f,
    "float64": CPSIG.sig_backprop_d
}

CPSIG_BATCH_SIG_BACKPROP = {
    "float32": CPSIG.batch_sig_backprop_f,
    "float64": CPSIG.batch_sig_backprop_d
}

CPSIG_SIG_COMBINE = {
    "float32": CPSIG.sig_combine_f,
    "float64": CPSIG.sig_combine_d
}

CPSIG_BATCH_SIG_COMBINE = {
    "float32": CPSIG.batch_sig_combine_f,
    "float64": CPSIG.batch_sig_combine_d
}

CPSIG_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.sig_combine_backprop_f,
    "float64": CPSIG.sig_combine_backprop_d
}

CPSIG_BATCH_SIG_COMBINE_BACKPROP = {
    "float32": CPSIG.batch_sig_combine_backprop_f,
    "float64": CPSIG.batch_sig_combine_backprop_d
}

CPSIG_SIG_TO_LOG_SIG = {
    "float32": CPSIG.sig_to_log_sig_f,
    "float64": CPSIG.sig_to_log_sig_d
}

CPSIG_BATCH_SIG_TO_LOG_SIG = {
    "float32": CPSIG.batch_sig_to_log_sig_f,
    "float64": CPSIG.batch_sig_to_log_sig_d
}

CPSIG_SIG_TO_LOG_SIG_BACKPROP = {
    "float32": CPSIG.sig_to_log_sig_backprop_f,
    "float64": CPSIG.sig_to_log_sig_backprop_d
}

CPSIG_BATCH_SIG_TO_LOG_SIG_BACKPROP = {
    "float32": CPSIG.batch_sig_to_log_sig_backprop_f,
    "float64": CPSIG.batch_sig_to_log_sig_backprop_d
}

CPSIG_SIG_KERNEL = {
    "float32": CPSIG.sig_kernel_f,
    "float64": CPSIG.sig_kernel_d
}

CPSIG_BATCH_SIG_KERNEL = {
    "float32": CPSIG.batch_sig_kernel_f,
    "float64": CPSIG.batch_sig_kernel_d
}

CPSIG_SIG_KERNEL_BACKPROP = {
    "float32": CPSIG.sig_kernel_backprop_f,
    "float64": CPSIG.sig_kernel_backprop_d
}

CPSIG_BATCH_SIG_KERNEL_BACKPROP = {
    "float32": CPSIG.batch_sig_kernel_backprop_f,
    "float64": CPSIG.batch_sig_kernel_backprop_d
}

CPSIG_BRANCHED_SIG = {
    "float32": CPSIG.branched_sig_f,
    "float64": CPSIG.branched_sig_d
}

CPSIG_BATCH_BRANCHED_SIG = {
    "float32": CPSIG.batch_branched_sig_f,
    "float64": CPSIG.batch_branched_sig_d
}

CPSIG_BRANCHED_SIG_COMBINE = {
    "float32": CPSIG.branched_sig_combine_f,
    "float64": CPSIG.branched_sig_combine_d
}

CPSIG_BATCH_BRANCHED_SIG_COMBINE = {
    "float32": CPSIG.batch_branched_sig_combine_f,
    "float64": CPSIG.batch_branched_sig_combine_d
}

if BUILT_WITH_CUDA:
    CUSIG_BRANCHED_SIG = {
        "float32": CUSIG.branched_sig_cuda_f,
        "float64": CUSIG.branched_sig_cuda_d
    }

    CUSIG_BATCH_BRANCHED_SIG = {
        "float32": CUSIG.batch_branched_sig_cuda_f,
        "float64": CUSIG.batch_branched_sig_cuda_d
    }
