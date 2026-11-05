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

from ctypes import POINTER, byref, c_double, c_uint64, c_void_p, cast, pointer
import warnings

import numpy as np
import torch

from ._lasso_types import LassoOptions, LassoShape, LassoAllocator
from .data_handlers import PathInputHandler
from .dtypes import DTYPES
from .load_siglib import CPSIG_WORKING_SET, CPSIG_SIG_LASSO_PREDICT
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


def sig_lasso(path, y, degree, alpha, *, feature_type="sig", loss="squared", fit_intercept=True,
              tol=None, max_iter=1000, initial_words=(), memory_limit=None, time_aug=False,
              lead_lag=False, end_time=1.0, n_jobs=1):
    r"""Fit a sparse linear predictor on normalized path features on CPU.

    Minimise the mean loss at ``intercept + X @ coef`` plus ``alpha * ||coef||_1``.
    Select squared error (with factor 1/2), binary logistic, Poisson count, or
    unit-threshold Huber loss with ``loss='squared'|'logistic'|'poisson'|'huber'``.
    An unpenalised intercept requires both classes for logistic regression and
    at least one positive count for Poisson regression.

    ``feature_type='sig'`` selects signature words, ``'log'`` selects tensor log
    coefficients at Lyndon words, and ``'hall'`` selects reverse-degree Hall
    coefficients. Each degree k is scaled by k! / V**k after path transforms,
    where V is the sum of infinity norms of increments. Constant paths have
    zero features. Columns are centred only when fitting an intercept.

    NumPy arrays and CPU Torch tensors support float32 and float64. Paths have
    shape (*batch, length, dimension). Responses have shape batch or
    (*batch, targets), with optional flattening of batch axes. Fit each target
    separately with a scalar or per-target positive penalty. ``initial_words``
    seeds the working set with words in the transformed alphabet.

    The native solver uses cyclic coordinate descent on each working set.
    Every fit searches the feature tree by branch and bound. It evaluates
    coordinates without constructing a full truncated signature.
    ``memory_limit`` bounds owned native workspace and output
    in bytes. ``n_jobs`` controls CPU workers. ``max_iter`` limits working-set
    updates per response. ``tol`` defaults to 1e-5 for float32 or 1e-8 for
    float64. Feature stationarity is checked at min(tol, 1e-4 * alpha).

    Return sorted union support in ``words``, uncentred columns in ``signatures``,
    ``coef``, ``intercept``, ``objective``, ``loss``, per-target ``kkt_violation``,
    ``converged``, ``n_iter`` (working-set updates), timings and memory in
    ``stats``, and prediction metadata. ``dual_gap`` is a squared-loss duality
    gap and is None for other losses. Prediction returns the linear predictor.
    This forward-only function belongs to the base API.
    """
    for name, value in (('degree', degree), ('max_iter', max_iter)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 < value < 2**31:
            raise ValueError(name + ' must be a positive integer below 2**31')
    if degree > 170:
        raise ValueError('degree must be at most 170 because factorial weights must fit in float64')
    if memory_limit is not None and (isinstance(memory_limit, bool) or not isinstance(memory_limit, (int, np.integer)) or not 0 <= memory_limit < 2**64):
        raise ValueError('memory_limit must be a nonnegative integer or None')
    if memory_limit == 0:
        raise MemoryError('memory_limit must allow the native fitting workspace')
    losses = {'squared': 0, 'logistic': 1, 'poisson': 2, 'huber': 3}
    if not isinstance(loss, str) or loss not in losses:
        raise ValueError("loss must be 'squared', 'logistic', 'poisson' or 'huber'")
    feature_types = {'sig': 0, 'log': 1, 'hall': 2}
    if not isinstance(feature_type, str) or feature_type not in feature_types:
        raise ValueError("feature_type must be 'sig', 'log', or 'hall'")
    for name, value in (('fit_intercept', fit_intercept), ('time_aug', time_aug), ('lead_lag', lead_lag)):
        check_type(value, name, bool)
    check_n_jobs(n_jobs)
    if not np.isfinite(end_time):
        raise ValueError('end_time must be finite')
    if any(isinstance(value, torch.Tensor) and value.device.type != 'cpu' for value in (path, y, alpha)):
        raise ValueError('sig_lasso requires CPU inputs')
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
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().numpy()
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
    if CPSIG_WORKING_SET is None:
        raise RuntimeError('The CPU library does not support branch-and-bound Lasso; rebuild pysiglib')
    with torch.no_grad():
        transformed = transform_path(inputs.path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if time_aug or lead_lag else inputs.path
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        response = ensure_own_contiguous_storage(y.detach() if isinstance(y, torch.Tensor) else y)
        options = LassoOptions(data.batch_size, data.length, data.dimension, degree, targets,
                               max_iter, memory_limit or 0, tol, n_jobs, fit_intercept, inputs.dtype == 'float32',
                               losses[loss])
        scalar = DTYPES[data.dtype]
        result_shape = None
        out_letters = out_offsets = coefficients = signatures = values = converged = iterations = None
        allocation_error = None

        @LassoAllocator
        def allocate_output(context, shape, output):
            nonlocal result_shape, out_letters, out_offsets, coefficients, signatures, values, converged, iterations, allocation_error
            try:
                result_shape = LassoShape.from_buffer_copy(shape.contents)
                out_letters = np.empty(result_shape.letters, dtype=np.uint64)
                out_offsets = np.empty(result_shape.words + 1, dtype=np.uint64)
                if isinstance(path, np.ndarray):
                    coefficients = np.empty((result_shape.words, targets), dtype=path.dtype)
                    signatures = np.empty((*inputs.batch_shape, result_shape.words), dtype=path.dtype)
                    values = [np.empty(targets, dtype=path.dtype) for _ in range(4)]
                else:
                    coefficients = torch.empty((result_shape.words, targets), dtype=path.dtype, device=path.device)
                    signatures = torch.empty((*inputs.batch_shape, result_shape.words), dtype=path.dtype, device=path.device)
                    values = [torch.empty(targets, dtype=path.dtype, device=path.device) for _ in range(4)]
                converged = np.empty(targets, dtype=np.uint64)
                iterations = np.empty(targets, dtype=np.uint64)
                for name, array in zip(('letters', 'offsets', 'converged', 'iterations'),
                                       (out_letters, out_offsets, converged, iterations)):
                    setattr(output.contents, name, _pointer(array, c_uint64))
                for name, array in zip(('coefficients', 'intercepts', 'objectives', 'gaps', 'kkt', 'signatures'), (coefficients, *values, signatures)):
                    setattr(output.contents, name, cast(_pointer(array, scalar), c_void_p).value)
                output.contents.stats = pointer(result_shape.stats)
                return 0
            except BaseException as error:
                allocation_error = error
                return 1

        args = [data.data_ptr, _pointer(response, scalar), _pointer(alphas, c_double),
                _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), feature_types[feature_type],
                byref(options), allocate_output, None]
        status = CPSIG_WORKING_SET[data.dtype](*args)
        if allocation_error is not None:
            raise allocation_error
        _check_status(status, 'cpu')
        selected = [tuple(int(a) for a in out_letters[out_offsets[j]:out_offsets[j + 1]]) for j in range(result_shape.words)]
        converged = converged.astype(bool)
        if not converged.all():
            warnings.warn('sig_lasso did not pass the global KKT check; inspect n_iter and kkt_violation',
                          RuntimeWarning, stacklevel=2)
        stats = {name: getattr(result_shape.stats, name) for name, _ in result_shape.stats._fields_}
        if stats['engine'] != 3 or stats['full_transform_calls']:
            raise RuntimeError('The CPU library did not use branch and bound')
        stats['engine'] = 'tree'
        stats['dictionary_size'] = sum(data.dimension**k for k in range(1, degree + 1))
        if feature_type != 'sig':
            primitive = [0] * (degree + 1)
            for k in range(1, degree + 1):
                primitive[k] = data.dimension**k - sum(primitive[j] for j in range(1, k) if k % j == 0)
            stats['dictionary_size'] = sum(primitive[k] // k for k in range(1, degree + 1))
        if isinstance(path, torch.Tensor):
            converged = torch.as_tensor(converged, device=path.device)
            iterations = torch.as_tensor(iterations.astype(np.int64), device=path.device)
        return dict(words=selected, coef=coefficients[:, 0] if single else coefficients, signatures=signatures,
                    intercept=values[0][0] if single else values[0], objective=values[1][0] if single else values[1],
                    dual_gap=(values[2][0] if single else values[2]) if loss == 'squared' else None,
                    loss=loss, kkt_violation=values[3][0] if single else values[3], converged=converged[0] if single else converged,
                    n_iter=iterations[0] if single else iterations, stats=stats,
                    feature_type=feature_type, degree=degree, dimension=path.shape[-1],
                    time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
                    feature_convention='reverse_degree_hall_v1' if feature_type == 'hall' else feature_type,
                    normalization='factorial_variation')


def sig_lasso_predict(path, model, *, n_jobs=1, memory_limit=None):
    """Predict on CPU paths using selected features from :func:`sig_lasso`.

    The model stores the feature convention and path transforms. Input paths
    must have the same array type, dtype and channel dimension as the model.
    The result preserves leading batch dimensions and has an optional target
    axis. Computation has no autograd connection and uses only selected features.
    Private state reuse is limited to 64 MiB and one quarter of an explicit
    ``memory_limit``. All native prediction workspace obeys that limit.
    """
    check_n_jobs(n_jobs)
    if memory_limit is not None and (isinstance(memory_limit, bool) or not isinstance(memory_limit, (int, np.integer)) or not 0 < memory_limit < 2**64):
        raise ValueError('memory_limit must be a positive integer or None')
    feature_types = {'sig': 0, 'log': 1, 'hall': 2}
    mode = model['feature_type']
    if mode not in feature_types or model.get('normalization') != 'factorial_variation':
        raise ValueError('Unsupported Lasso model convention')
    if mode == 'hall' and model.get('feature_convention') != 'reverse_degree_hall_v1':
        raise ValueError('Unsupported Hall basis convention')
    coef = model['coef']
    if any(isinstance(value, torch.Tensor) and value.device.type != 'cpu' for value in (path, coef)):
        raise ValueError('sig_lasso_predict supports CPU arrays only')
    if isinstance(model['intercept'], torch.Tensor) and model['intercept'].device.type != 'cpu':
        raise ValueError('sig_lasso_predict requires a CPU intercept')
    inputs = PathInputHandler(path, False, False, 1.0, 'path')
    if type(path) is not type(coef) or path.dtype != coef.dtype or path.shape[-1] != model['dimension']:
        raise ValueError('Paths must match the model array type, dtype and dimension')
    degree = model['degree']
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 170:
        raise ValueError('Invalid model degree')
    if coef.ndim not in (1, 2) or coef.shape[0] != len(model['words']) or (coef.ndim == 2 and coef.shape[1] == 0):
        raise ValueError('Invalid model coefficient shape')
    for word in model['words']:
        if not 1 <= len(word) <= degree or any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or not 0 <= a < inputs.dimension * (2 if model['lead_lag'] else 1) + int(model['time_aug']) for a in word):
            raise ValueError('Invalid model feature word')
    if not inputs.length:
        raise ValueError('Paths must contain at least one point')
    if not (bool(torch.isfinite(path).all()) if isinstance(path, torch.Tensor) else np.isfinite(path).all()):
        raise ValueError('path must contain only finite values')
    single = coef.ndim == 1
    targets = 1 if single else coef.shape[1]
    if tuple(np.shape(model['intercept'])) not in ((), (targets,)) or (targets > 1 and tuple(np.shape(model['intercept'])) != (targets,)):
        raise ValueError('Invalid model intercept shape')
    shape = (*inputs.batch_shape,) if single else (*inputs.batch_shape, targets)
    output = np.empty(shape, dtype=path.dtype) if isinstance(path, np.ndarray) else torch.empty(shape, dtype=path.dtype)
    if not inputs.batch_size:
        return output
    if not model['words']:
        if isinstance(path, np.ndarray):
            intercept = np.asarray(model['intercept'], dtype=path.dtype)
            output[...] = intercept.reshape(()) if single else intercept
        else:
            intercept = torch.as_tensor(model['intercept'], dtype=path.dtype, device='cpu').detach()
            output.copy_(intercept.reshape(()) if single else intercept)
        return output
    if CPSIG_SIG_LASSO_PREDICT is None:
        raise RuntimeError('The native library does not support Lasso prediction; rebuild pysiglib')
    words = model['words']
    offsets = np.zeros(len(words) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(w) for w in words], dtype=np.uint64)
    letters = np.array([a for w in words for a in w], dtype=np.uint64)
    with torch.no_grad():
        transformed = (transform_path(inputs.path, time_aug=model['time_aug'], lead_lag=model['lead_lag'], end_time=model['end_time'], n_jobs=n_jobs)
                       if model['time_aug'] or model['lead_lag'] else inputs.path)
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        coefficients = ensure_own_contiguous_storage(coef.detach() if isinstance(coef, torch.Tensor) else coef)
        intercept = (np.atleast_1d(np.asarray(model['intercept'], dtype=path.dtype)) if isinstance(path, np.ndarray)
                     else torch.as_tensor(model['intercept'], dtype=path.dtype, device='cpu').detach().reshape(-1).contiguous())
        selected_degree = max((len(w) for w in words), default=1)
        options = LassoOptions(data.batch_size, data.length, data.dimension, selected_degree, targets, 1, memory_limit or 0, 1e-8, n_jobs, 0, inputs.dtype == 'float32', 0)
        scalar = DTYPES[data.dtype]
        status = CPSIG_SIG_LASSO_PREDICT[data.dtype](data.data_ptr, _pointer(coefficients, scalar), _pointer(intercept, scalar), _pointer(output, scalar),
            _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), feature_types[mode], byref(options))
        _check_status(status, 'cpu')
    return output
