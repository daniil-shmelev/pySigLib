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

import os.path
import timeit

try:
    import signatory
except:
    signatory = None

# iisignature requires numpy < 2.0
# esig requires numpy >= 2.0
# Will have to run timings separately, installing numpy<2.0 first for iisig
# and then numpy>=2.0 for esig
try:
    import iisignature
except:
    iisignature = None

try:
    import esig
except:
    esig = None

try:
    import sigkernel
except:
    sigkernel = None

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

import pysiglib

def get_dtype(dtype, module):
    if dtype == "float":
        return module.float32
    if dtype == "double":
        return module.float64
    raise ValueError("invalid dtype")

def np_dtype(dtype):
    return get_dtype(dtype, np)

def torch_dtype(dtype):
    return get_dtype(dtype, torch)

def check_esig_cfg(cfg):
    if cfg['device'] != "cpu":
        raise ValueError("esig only supports cpu computation")

def check_iisig_cfg(cfg):
    if cfg['device'] != "cpu":
        raise ValueError("iisignature only supports cpu computation")
    if 'method' in cfg and cfg['method'] == 1:
        raise ValueError("iisignature doesn not support method=1")

def iisig_method(method):
    if method == 0:
        return "x"
    if method == 2:
        return "s"

def signatory_method(method):
    if method == 0:
        return "expand"
    if method == 1:
        return "words"
    if method == 2:
        return "brackets"

def plot_times(
        x,
        ys,
        legend,
        title,
        xlabel,
        ylabel,
        scale,
        filename,
        linestyles = None
):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    if linestyles is None:
        linestyles = ["-"] * len(ys)

    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y, ls in zip(ys, linestyles):
        plt.plot(x, y, linestyle = ls)
    plt.legend(legend)
    plt.yscale(scale)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("plots/" + filename + ".png", dpi=300)
    plt.savefig("plots/" + filename + ".pdf", dpi=300)
    plt.show()

def time_iisig_sig(cfg, progress_bar = False):
    check_iisig_cfg(cfg)
    dtype = np_dtype(cfg['dtype'])
    X = np.random.uniform(size=(cfg['batch_size'], cfg['length'], cfg['dimension'])).astype(dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        start = timeit.default_timer()
        iisignature.sig(X, cfg['degree'])
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_sig(cfg, horner, n_jobs, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), dtype=dtype, device=cfg['device'])
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig(X, cfg['degree'], horner = horner, n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def time_signatory_sig(cfg, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        start = timeit.default_timer()
        signatory.signature(X, cfg['degree'])
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def time_esig_sig(cfg, progress_bar = False):
    check_esig_cfg(cfg)
    dtype = np_dtype(cfg['dtype'])
    X = np.random.uniform(size=(cfg['batch_size'], cfg['length'], cfg['dimension'])).astype(dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        start = timeit.default_timer()
        for i in range(cfg['batch_size']): # esig cannot handle batches, so loop
            esig.stream2sig(X[i], cfg['degree'])
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_iisig_log_sig(cfg, progress_bar = False):
    check_iisig_cfg(cfg)
    dtype = np_dtype(cfg['dtype'])
    method = iisig_method(cfg['method'])
    X = np.random.uniform(size=(cfg['batch_size'], cfg['length'], cfg['dimension'])).astype(dtype)
    s = iisignature.prepare(cfg['dimension'], cfg['degree'], method)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        start = timeit.default_timer()
        iisignature.logsig(X, s, method)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_log_sig(cfg, n_jobs, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), dtype=dtype)
    pysiglib.prepare_log_sig(cfg['dimension'], cfg['degree'], cfg['method'])
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.log_sig(X, cfg['degree'], method=cfg['method'], n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def time_signatory_log_sig(cfg, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    method = signatory_method(cfg['method'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        start = timeit.default_timer()
        signatory.logsignature(X, cfg['degree'], mode=method)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_sigkernel_kernel(cfg, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, cfg['dyadic_order'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)
    Y = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)

    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        try:
            if cfg['device'] == "cuda":
                torch.cuda.empty_cache()
            start = timeit.default_timer()
            signature_kernel.compute_kernel(X, Y)
            end = timeit.default_timer()
            time_ = end - start
            best_time = min(best_time, time_)
        except:
            continue
    return best_time

def time_sigkernel_kernel_backprop(cfg, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, cfg['dyadic_order'])
    derivs = torch.ones(cfg['batch_size'], device = cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), dtype=dtype, device=cfg['device'], requires_grad=True)
        Y = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), dtype=dtype, device=cfg['device'], requires_grad=True)
        try:
            if cfg['device'] == "cuda":
                torch.cuda.empty_cache()
            K = signature_kernel.compute_kernel(X, Y)
            start = timeit.default_timer()
            K.backward(derivs)
            end = timeit.default_timer()
            time_ = end - start
            best_time = min(best_time, time_)
        except:
            continue
    return best_time

def time_pysiglib_kernel(cfg, n_jobs, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)
    Y = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        if cfg['device'] == "cuda":
            torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig_kernel(X, Y, cfg['dyadic_order'], n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_kernel_backprop(cfg, n_jobs, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    derivs = torch.ones(cfg['batch_size'], device = cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device = cfg['device'], dtype=dtype, requires_grad=True)
        Y = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device = cfg['device'], dtype=dtype, requires_grad=True)
        if cfg['device'] == "cuda":
            torch.cuda.empty_cache()
        K = pysiglib.torch_api.sig_kernel(X, Y, cfg['dyadic_order'], n_jobs = n_jobs)
        start = timeit.default_timer()
        K.backward(derivs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_sig_backprop(cfg, n_jobs, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    sig_len = pysiglib.sig_length(cfg['dimension'], cfg['degree'])
    X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), dtype=dtype, device=cfg['device'])
    s = torch.rand(size=(cfg['batch_size'], sig_len), dtype=dtype, device=cfg['device'])
    sig_derivs = torch.rand(size=(cfg['batch_size'], sig_len), dtype=dtype, device=cfg['device'])
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig_backprop(X, s, sig_derivs, cfg['degree'], n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_iisig_sig_backprop(cfg, progress_bar = False):
    check_iisig_cfg(cfg)
    dtype = np_dtype(cfg['dtype'])
    X = np.random.uniform(size=(cfg['batch_size'], cfg['length'], cfg['dimension'])).astype(dtype=dtype)
    sig_derivs = np.random.uniform(size=(cfg['batch_size'], pysiglib.sig_length(cfg['dimension'], cfg['degree']) - 1))
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        iisignature.sigbackprop(sig_derivs, X, cfg['degree'])
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_signatory_sig_backprop(cfg, progress_bar = False):
    dtype = torch_dtype(cfg['dtype'])
    sig_derivs = torch.rand(size=(cfg['batch_size'], pysiglib.sig_length(cfg['dimension'], cfg['degree']) - 1), device=cfg['device'], dtype=dtype)
    best_time = float('inf')
    loop = tqdm(range(cfg['num_runs'])) if progress_bar else range(cfg['num_runs'])
    for _ in loop:
        X = torch.rand(size=(cfg['batch_size'], cfg['length'], cfg['dimension']), device=cfg['device'], dtype=dtype, requires_grad=True)
        torch.cuda.empty_cache()
        s = signatory.signature(X, cfg['degree'])
        start = timeit.default_timer()
        s.backward(sig_derivs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time
