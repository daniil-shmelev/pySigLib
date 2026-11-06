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

"""Validate and compare fixed prefix-suffix splits."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from endpoint_gemm.benchmark import Native, measure


def backward(native, path, signature, deriv, degree, method, jobs):
    gpu = isinstance(path, torch.Tensor)
    scalar = ctypes.c_float if str(path.dtype).split('.')[-1] == 'float32' else ctypes.c_double
    pointer = ctypes.POINTER(scalar)
    ptr = lambda x: ctypes.cast(x.data_ptr() if gpu else x.ctypes.data, pointer)
    output = torch.empty_like(path) if gpu else np.empty_like(path)
    lib = native.libs['cuda' if gpu else 'cpu']
    suffix = 'f' if scalar == ctypes.c_float else 'd'
    name = 'sig_backprop' if method == 'horner' else 'signature_prefix_suffix_backprop'
    fn = getattr(lib, name + ('_cuda_' if gpu else '_') + suffix)
    types = [pointer] * (4 if method == 'horner' else 3)
    args = (ptr(path), ptr(output), ptr(deriv)) + ((ptr(signature),) if method == 'horner' else ())
    batch, points, d = path.shape
    types += [ctypes.c_uint64] * 4 + [ctypes.c_bool, ctypes.c_bool, scalar, ctypes.c_bool]
    args += (batch, d, points, degree, False, False, scalar(1), True)
    if method == 'horner':
        types += ([] if gpu else [ctypes.c_int]) + [pointer] + [ctypes.c_uint64] * 3
        args += (() if gpu else (jobs,)) + (None, 0, 0, 0)
    else:
        types += [ctypes.c_void_p if gpu else ctypes.c_int]
        args += (torch.cuda.current_stream().cuda_stream if gpu else jobs,)
    fn.argtypes = types
    fn.restype = ctypes.c_int
    def run():
        status = fn(*args)
        if status: raise RuntimeError(f'backward status {status}')
        return output
    run.inputs = (path, signature, deriv)
    return run


def validate(native):
    count = 0
    for gpu in (False, True):
        for dtype in ('float32', 'float64'):
            for d, m in ((2, 3), (2, 6), (3, 4), (4, 5), (8, 3)):
                rng = np.random.default_rng(d * 100 + m)
                array = rng.normal(scale=0.1, size=(3, 10, d)).astype(dtype)
                path = torch.as_tensor(array, device='cuda') if gpu else array
                os.environ['PYSIGLIB_PS_SPLIT_A'] = '0'
                expected = native.operation(path, m, 'horner')()
                seeds = rng.normal(size=expected.shape).astype(dtype)
                deriv = torch.as_tensor(seeds, device='cuda') if gpu else seeds
                grad = backward(native, path, expected, deriv, m, 'horner', 2)()
                for a in range(1, m + 1):
                    os.environ['PYSIGLIB_PS_SPLIT_A'] = str(a)
                    value = native.operation(path, m, 'prefix_suffix', 2)()
                    actual = backward(native, path, expected, deriv, m, 'prefix_suffix', 2)()
                    for result, ref in ((value, expected), (actual, grad)):
                        if gpu: result, ref = result.cpu().numpy(), ref.cpu().numpy()
                        tolerance = 2e-5 if dtype == 'float32' else 2e-11
                        np.testing.assert_allclose(result, ref, atol=tolerance, rtol=tolerance,
                                                   err_msg=f'{gpu=} {dtype=} {d=} {m=} {a=}')
                    count += 1
    os.environ.pop('PYSIGLIB_PS_SPLIT_A', None)
    print(f'{count} forward/backward split checks passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--case', nargs=4, type=int, action='append')
    parser.add_argument('--output', type=Path, default=Path('tmp/prefix_suffix_dispatch/tuning.json'))
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    native = Native()
    if args.validate_only:
        validate(native)
        return
    gpu = args.device == 'cuda'
    jobs = 1 if gpu else 20
    report = {'device': args.device, 'cases': []}
    for batch, n, d, m in args.case or [(32, n, d, m) for n in (16, 256, 4096)
                                      for d, m in ((2, 8), (3, 5), (4, 4), (4, 6), (8, 3), (8, 4), (16, 3), (32, 3))]:
        rng = np.random.default_rng(719)
        array = np.concatenate([np.zeros((batch, 1, d)), np.cumsum(rng.normal(size=(batch, n, d)), axis=1) / np.sqrt(n)], axis=1).astype('float32')
        path = torch.as_tensor(array, device='cuda') if gpu else array
        os.environ['PYSIGLIB_PS_SPLIT_A'] = '0'
        signature = native.operation(path, m, 'horner', jobs)()
        deriv_np = (rng.normal(size=signature.shape) / np.sqrt(signature.shape[-1])).astype('float32')
        deriv = torch.as_tensor(deriv_np, device='cuda') if gpu else deriv_np
        reference_grad = backward(native, path, signature, deriv, m, 'horner', jobs)()
        case = {'shape': [batch, n, d, m], 'candidates': []}
        for a in range(m + 1):
            scan = m // 2 if not a else max(a - 1, m - a)
            estimated = batch * 4 * (4 * (n + 1) * sum(d ** k for k in range(scan + 1)))
            if estimated > 1536 * 1024 ** 2:
                case['candidates'].append({'a': a, 'excluded': 'scan memory estimate exceeds 1.5 GiB'})
                continue
            os.environ['PYSIGLIB_PS_SPLIT_A'] = str(a)
            result = {'a': a}
            for phase in ('forward', 'backward'):
                run = (native.operation(path, m, 'prefix_suffix', jobs) if phase == 'forward'
                       else backward(native, path, signature, deriv, m, 'prefix_suffix', jobs))
                started = time.perf_counter()
                actual = run()
                ref = signature if phase == 'forward' else reference_grad
                if gpu: actual, ref = actual.cpu().numpy(), ref.cpu().numpy()
                np.testing.assert_allclose(actual, ref, atol=1e-4, rtol=1e-4)
                if time.perf_counter() - started > 10:
                    result[phase] = {'excluded': 'initial call exceeds ten seconds'}
                    continue
                run()
                values = [measure(run, 1, gpu) for _ in range(args.samples)]
                result[phase] = {'samples_ms': values, 'median_ms': float(np.median(values))}
            case['candidates'].append(result)
        report['cases'].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='ascii', newline='\r\n')
        print(case, flush=True)
    os.environ.pop('PYSIGLIB_PS_SPLIT_A', None)


if __name__ == '__main__':
    main()
