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
import math
import warnings

import numpy as np
import torch

from ._lasso_types import LassoOptions, LassoShape
from .data_handlers import PathInputHandler
from .dtypes import DTYPES, CPSIG_SIG_LASSO, CUSIG_SIG_LASSO
from .error_codes import err_msg
from .load_siglib import CPSIG, CUSIG, CPSIG_SIG_LASSO_RESULT, CUSIG_SIG_LASSO_RESULT
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


def sig_lasso(path, y, degree, alpha, *, method="sparse", degree_weights=None,
              fit_intercept=True, tol=None, max_iter=1000, max_epochs=100000,
              batch_size=128, initial_words=(), cache_bytes=256 * 1024**2,
              memory_limit=None, time_aug=False, lead_lag=False, end_time=1.0,
              n_jobs=1):
    r"""Fit Lasso on weighted signature coordinates on CPU or CUDA.

    Minimise ``||y - intercept - Z @ coef||**2 / (2*n) + alpha*||coef||_1``,
    where ``Z[:, w] = degree_weights[len(w)-1] * S(path)^w``. The default
    degree-k weight is ``sqrt(k!)``. Columns are centred when fitting an
    intercept; they are not standardised. ``method='full'`` computes the
    complete dictionary; ``method='sparse'`` searches it with integral bounds
    and Bernstein prefix coefficients. Both use the same native solver.

    ``path`` has shape ``(*batch, length, dimension)``. Leading batch axes
    are flattened into samples. ``y`` has shape ``batch`` or
    ``(*batch, n_targets)`` (also ``(n_samples,)`` or ``(n_samples, n_targets)``).
    Responses are fitted independently. ``alpha`` is positive and scalar
    or has one entry per response. ``y`` must match the array type, dtype
    and device of ``path``. Only float32 and float64 are supported.

    ``degree_weights`` is a length-degree vector of finite positive weights.
    ``initial_words`` contains zero-based channel tuples in the transformed
    alphabet. ``batch_size`` limits words added per sparse search.
    ``max_iter`` limits outer passes per response, and ``max_epochs`` limits
    total restricted coordinate sweeps per response. ``tol`` defaults to
    1e-8 for float64 or 1e-5 for float32. Convergence uses a full-dictionary
    primal-dual gap scaled by ``max(1, mean(centered_y**2))``.

    ``cache_bytes`` bounds optional caches. ``memory_limit`` optionally
    limits native fitting allocations, in bytes. Reuse lasts for this call
    only. ``n_jobs`` follows the usual CPU threading convention. CUDA uses
    the current stream on the input device. Transform parameters have the
    same meaning as in :func:`sig`.

    Returns a dictionary with ``words``, ``coef``, ``intercept``, ``objective``,
    ``dual_gap``, ``converged``, ``n_iter`` and ``stats``. Words are the union
    of nonzero coordinates, sorted by degree then lexicographically.
    Coefficients have shape ``(n_words,)`` or ``(n_words, n_targets)``.
    Prediction is ``intercept + weighted_signature[:, words] @ coef``.
    Numeric arrays preserve the input type, dtype and device. The result
    has no autograd connection. Iteration limits give a warning and
    ``converged=False``; invalid inputs and allocation failures raise errors.

    This forward-only function is available in the base API only. Numerical
    agreement does not imply a unique or bitwise-identical coefficient vector.
    """
    for name, value in (('degree', degree), ('max_iter', max_iter), ('max_epochs', max_epochs), ('batch_size', batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 < value < 2**31:
            raise ValueError(name + ' must be a positive integer below 2**31')
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
    if degree_weights is None:
        try:
            weights = np.array([math.exp(math.lgamma(k + 1) / 2) for k in range(degree + 1)])
        except OverflowError as exc:
            raise ValueError('Default degree weights overflow; supply finite degree_weights') from exc
    else:
        if isinstance(degree_weights, torch.Tensor):
            degree_weights = degree_weights.detach().cpu().numpy()
        supplied = np.asarray(degree_weights, dtype=np.float64)
        if supplied.shape != (degree,):
            raise ValueError('degree_weights must have length degree')
        weights = np.concatenate(([1.0], supplied))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('degree_weights must be finite and positive')
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
        raise RuntimeError('The native library does not include sig_lasso; rebuild pysiglib and its CUDA plugin')
    context = torch.cuda.device(path.device) if device == 'cuda' else nullcontext()
    with context, torch.no_grad():
        transformed = transform_path(inputs.path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if time_aug or lead_lag else inputs.path
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        response = ensure_own_contiguous_storage(y.detach() if isinstance(y, torch.Tensor) else y)
        options = LassoOptions(data.batch_size, data.length, data.dimension, degree, targets,
                               max_iter, max_epochs, batch_size, cache_bytes, memory_limit or 0,
                               tol, n_jobs, method == 'full', fit_intercept, inputs.dtype == 'float32')
        scalar = DTYPES[data.dtype]
        library = CUSIG if device == 'cuda' else CPSIG
        suffix = '_cuda' if device == 'cuda' else ''
        handle = c_uint64()
        args = [data.data_ptr, _pointer(response, scalar), _pointer(weights, c_double), _pointer(alphas, c_double),
                _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), byref(options)]
        if device == 'cuda':
            args.extend((c_void_p(torch.cuda.current_stream(path.device).cuda_stream), path.device.index))
        args.append(byref(handle))
        _check_status(functions[data.dtype](*args), device)
        try:
            result_shape = LassoShape()
            _check_status(getattr(library, 'sig_lasso_shape' + suffix)(handle, byref(result_shape)), device)
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
            result_functions = CUSIG_SIG_LASSO_RESULT if device == 'cuda' else CPSIG_SIG_LASSO_RESULT
            _check_status(result_functions[data.dtype](handle, _pointer(out_letters, c_uint64), _pointer(out_offsets, c_uint64),
                _pointer(coefficients, scalar), *[_pointer(v, scalar) for v in values],
                _pointer(converged, c_uint64), _pointer(iterations, c_uint64)), device)
            selected = [tuple(int(a) for a in out_letters[out_offsets[j]:out_offsets[j + 1]]) for j in range(result_shape.words)]
            converged = converged.astype(bool)
            if not converged.all():
                warnings.warn('sig_lasso did not reach the requested global gap; inspect converged, dual_gap and the iteration limits', RuntimeWarning, stacklevel=2)
            stats = {name: getattr(result_shape.stats, name) for name, _ in result_shape.stats._fields_}
            stats['dictionary_size'] = sum(data.dimension**k for k in range(1, degree + 1))
            if isinstance(path, torch.Tensor):
                converged = torch.as_tensor(converged, device=path.device)
                iterations = torch.as_tensor(iterations.astype(np.int64), device=path.device)
            return dict(words=selected, coef=coefficients[:, 0] if single else coefficients,
                        intercept=values[0][0] if single else values[0], objective=values[1][0] if single else values[1],
                        dual_gap=values[2][0] if single else values[2], converged=converged[0] if single else converged,
                        n_iter=iterations[0] if single else iterations, stats=stats)
        finally:
            _check_status(getattr(library, 'sig_lasso_release' + suffix)(handle), device)
