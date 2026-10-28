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

from contextlib import nullcontext
from ctypes import POINTER, byref, c_double, c_uint64, c_void_p, cast
import warnings

import numpy as np
import torch

from ._lasso_types import LassoOptions, LassoShape, LassoAllocator
from .data_handlers import PathInputHandler
from .dtypes import DTYPES, CPSIG_SIG_LASSO, CUSIG_SIG_LASSO
from .error_codes import err_msg
from .param_checks import check_type, check_dtype, check_n_jobs, ensure_own_contiguous_storage
from .transform_path import transform_path


def _pointer(array, scalar):
    if isinstance(array, np.ndarray):
        return array.ctypes.data_as(POINTER(scalar))
    return cast(array.data_ptr(), POINTER(scalar))


def _check_status(status, device):
    if not status:
        return
    message = err_msg(status, device)
    if status == 1:
        raise MemoryError(message)
    if status == 2:
        raise ValueError(message)
    if status == 3:
        raise OverflowError(message)
    raise RuntimeError(message)


def sig_lasso(path, y, degree, alpha, *, method="sparse",
              fit_intercept=True, tol=None, max_iter=1000,
              batch_size=128, initial_words=(), cache_bytes=256 * 1024**2,
              memory_limit=None, time_aug=False, lead_lag=False, end_time=1.0,
              n_jobs=1):
    r"""Fit Lasso on weighted signature coordinates on CPU or CUDA.

    Minimise ``||y - intercept - Z @ coef||**2 / (2*n) + alpha*||coef||_1``,
    where ``Z[i, w] = k! * S(path[i])^w / V(path[i])**k`` for ``k=len(w)``.
    ``V`` is the sum of the infinity norms of the path increments, computed
    after path transforms. Constant paths have zero nonempty features.
    Columns are centred when fitting an intercept; they are not standardised.
    ``method='full'`` computes the
    complete dictionary; ``method='sparse'`` searches it with integral bounds
    and Bernstein prefix coefficients. Both use the same native solver.

    ``path`` has shape ``(*batch, length, dimension)``. Leading batch axes
    are flattened into samples. ``y`` has shape ``batch`` or
    ``(*batch, n_targets)`` (also ``(n_samples,)`` or ``(n_samples, n_targets)``).
    Responses are fitted independently. ``alpha`` is positive and scalar
    or has one entry per response. ``y`` must match the array type, dtype
    and device of ``path``. Only float32 and float64 are supported. ``degree``
    must be at most 170 so that its factorial is finite in float64.

    ``initial_words`` contains zero-based channel tuples in the transformed
    alphabet and seeds the candidate pool. ``batch_size`` limits candidates
    added per search. Both methods use feature-sign search with QR updates.
    ``max_iter`` limits signed correction steps per response. ``tol`` defaults to
    1e-8 for float64 or 1e-5 for float32. Convergence uses a full-dictionary
    primal-dual gap scaled by ``max(1, mean(centered_y**2))``.

    ``cache_bytes`` bounds optional caches. ``memory_limit`` optionally
    limits owned native fitting allocations, including factors and library
    workspace allowances, in bytes. Reuse lasts for this call
    only. ``n_jobs`` follows the usual CPU threading convention. CUDA uses
    the current stream on the input device. Transform parameters have the
    same meaning as in :func:`sig`.

    Returns a dictionary with ``words``, ``coef``, ``intercept``, ``objective``,
    ``dual_gap``, ``converged``, ``n_iter`` and ``stats``. Words are the union
    of nonzero coordinates, sorted by degree then lexicographically.
    Coefficients have shape ``(n_words,)`` or ``(n_words, n_targets)``.
    Prediction is ``intercept + normalized_signature[:, words] @ coef``.
    For new paths, apply the same transforms, divide each path by its
    variation, and multiply each selected degree-k signature by ``k!``.
    Numeric arrays preserve the input type, dtype and device. The result
    has no autograd connection. ``n_iter`` counts correction steps and is
    zero for an empty optimum. ``stats`` includes active-set changes, QR
    operations, singular steps, certificate passes, phase times, constructed
    columns and peak accounted memory. Counters sum over responses. Factors
    need O(n_samples*n_active + n_active**2) memory.
    Iteration limits and persistent numerical failures give a warning and
    ``converged=False``; invalid inputs and allocation failures raise errors.

    This forward-only function is available in the base API only. Numerical
    agreement does not imply a unique or bitwise-identical coefficient vector.
    """
    for name, value in (('degree', degree), ('max_iter', max_iter), ('batch_size', batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 < value < 2**31:
            raise ValueError(name + ' must be a positive integer below 2**31')
    if degree > 170:
        raise ValueError('degree must be at most 170 because factorial weights must fit in float64')
    for name, value in (('cache_bytes', cache_bytes), ('memory_limit', memory_limit)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**64):
            raise ValueError(name + ' must be a nonnegative integer or None')
    if cache_bytes is None:
        raise ValueError('cache_bytes must be an integer')
    if memory_limit == 0:
        raise MemoryError('memory_limit must allow the native fitting workspace')
    if method not in ('full', 'sparse'):
        raise ValueError("method must be 'full' or 'sparse'")
    for name, value in (('fit_intercept', fit_intercept), ('time_aug', time_aug), ('lead_lag', lead_lag)):
        check_type(value, name, bool)
    check_n_jobs(n_jobs)
    if not np.isfinite(end_time):
        raise ValueError('end_time must be finite')
    inputs = PathInputHandler(path, time_aug, lead_lag, end_time, 'path')
    if not inputs.batch_size or inputs.data_length < 1:
        raise ValueError('sig_lasso requires at least one sample and one path point')
    if type(y) is not type(path):
        raise TypeError('y and path must have the same array type')
    check_dtype(y, 'y')
    if y.dtype != path.dtype or (isinstance(path, torch.Tensor) and y.device != path.device):
        raise ValueError('y and path must have the same dtype and device')
    shape = tuple(y.shape)
    if shape in (inputs.batch_shape, (inputs.batch_size,)):
        single = True
        targets = 1
    elif ((len(shape) == len(inputs.batch_shape) + 1 and shape[:-1] == inputs.batch_shape)
          or (len(shape) == 2 and shape[0] == inputs.batch_size)) and shape[-1] > 0:
        single = False
        targets = shape[-1]
    else:
        raise ValueError('y shape must match the path sample axes, with an optional target axis')
    if isinstance(path, torch.Tensor):
        valid = bool(torch.isfinite(path).all()) and bool(torch.isfinite(y).all())
    else:
        valid = np.isfinite(path).all() and np.isfinite(y).all()
    if not valid:
        raise ValueError('path and y must contain only finite values')
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    alphas = np.asarray(alpha, dtype=np.float64)
    if alphas.ndim == 0:
        alphas = np.full(targets, alphas.item())
    if alphas.shape != (targets,) or not np.all(np.isfinite(alphas)) or np.any(alphas <= 0):
        raise ValueError('alpha must be positive, finite and scalar or have one entry per target')
    alphas = np.ascontiguousarray(alphas)
    words = []
    for word in initial_words:
        if not 1 <= len(word) <= degree or any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= a < inputs.dimension for a in word):
            raise ValueError('initial_words contains an invalid transformed-channel word')
        words.append(tuple(int(a) for a in word))
    words = sorted(set(words), key=lambda word: (len(word), word))
    offsets = np.zeros(len(words) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(word) for word in words], dtype=np.uint64)
    letters = np.array([a for word in words for a in word], dtype=np.uint64)
    if tol is None:
        tol = 1e-5 if inputs.dtype == 'float32' else 1e-8
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be finite and positive')
    device = inputs.device
    functions = CUSIG_SIG_LASSO if device == 'cuda' else CPSIG_SIG_LASSO
    if functions is None:
        raise RuntimeError('The native library does not support the current sig_lasso interface; rebuild pysiglib and its CUDA plugin')
    context = torch.cuda.device(path.device) if device == 'cuda' else nullcontext()
    with context, torch.no_grad():
        transformed = transform_path(inputs.path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if time_aug or lead_lag else inputs.path
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        response = ensure_own_contiguous_storage(y.detach() if isinstance(y, torch.Tensor) else y)
        options = LassoOptions(data.batch_size, data.length, data.dimension, degree, targets,
                               max_iter, batch_size, cache_bytes, memory_limit or 0,
                               tol, n_jobs, method == 'full', fit_intercept, inputs.dtype == 'float32')
        scalar = DTYPES[data.dtype]
        result_shape = None
        out_letters = out_offsets = coefficients = values = converged = iterations = None
        allocation_error = None

        @LassoAllocator
        def allocate_output(context, shape, output):
            nonlocal result_shape, out_letters, out_offsets, coefficients, values, converged, iterations, allocation_error
            try:
                result_shape = LassoShape.from_buffer_copy(shape.contents)
                out_letters = np.empty(result_shape.letters, dtype=np.uint64)
                out_offsets = np.empty(result_shape.words + 1, dtype=np.uint64)
                if isinstance(path, np.ndarray):
                    coefficients = np.empty((result_shape.words, targets), dtype=path.dtype)
                    values = [np.empty(targets, dtype=path.dtype) for _ in range(3)]
                else:
                    coefficients = torch.empty((result_shape.words, targets), dtype=path.dtype, device=path.device)
                    values = [torch.empty(targets, dtype=path.dtype, device=path.device) for _ in range(3)]
                converged = np.empty(targets, dtype=np.uint64)
                iterations = np.empty(targets, dtype=np.uint64)
                for name, array in zip(('letters', 'offsets', 'converged', 'iterations'),
                                       (out_letters, out_offsets, converged, iterations)):
                    setattr(output.contents, name, _pointer(array, c_uint64))
                for name, array in zip(('coefficients', 'intercepts', 'objectives', 'gaps'), (coefficients, *values)):
                    setattr(output.contents, name, cast(_pointer(array, scalar), c_void_p).value)
                return 0
            except BaseException as error:
                allocation_error = error
                return 1

        args = [data.data_ptr, _pointer(response, scalar), _pointer(alphas, c_double),
                _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), byref(options)]
        if device == 'cuda':
            args.extend((c_void_p(torch.cuda.current_stream(path.device).cuda_stream), path.device.index))
        args.extend((allocate_output, None))
        status = functions[data.dtype](*args)
        if allocation_error is not None:
            raise allocation_error
        _check_status(status, device)
        selected = [tuple(int(a) for a in out_letters[out_offsets[j]:out_offsets[j + 1]]) for j in range(result_shape.words)]
        converged = converged.astype(bool)
        if not converged.all():
            reason = 'numerical accuracy limit' if np.any((~converged) & (iterations < max_iter)) else 'max_iter limit'
            if result_shape.stats.numerical_failures:
                reason = 'active-equation or objective-descent accuracy limit after a QR rebuild'
            warnings.warn('sig_lasso reached the ' + reason + ' before the requested global gap; inspect converged and dual_gap', RuntimeWarning, stacklevel=2)
        stats = {name: getattr(result_shape.stats, name) for name, _ in result_shape.stats._fields_}
        stats['dictionary_size'] = sum(data.dimension**k for k in range(1, degree + 1))
        if isinstance(path, torch.Tensor):
            converged = torch.as_tensor(converged, device=path.device)
            iterations = torch.as_tensor(iterations.astype(np.int64), device=path.device)
        return dict(words=selected, coef=coefficients[:, 0] if single else coefficients,
                    intercept=values[0][0] if single else values[0], objective=values[1][0] if single else values[1],
                    dual_gap=values[2][0] if single else values[2], converged=converged[0] if single else converged,
                    n_iter=iterations[0] if single else iterations, stats=stats)
