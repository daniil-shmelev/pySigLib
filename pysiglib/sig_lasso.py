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


_FEATURE_TYPES = {'sig': 0, 'log': 1}
_TREE_ORDERS = ('prefix', 'lyndon', 'suffix', 'energy', 'profile', 'forward', 'grouped', 'outside', 'balanced', 'adaptive',
                'pivotweak', 'pivotstrong', 'pivotalternating')
_TERMINAL_BOUNDS = ('norm', 'projected', 'primitive')
_SIGNATURE_SEARCH = {(order, bound): 16 + 4 * i + j for i, order in enumerate(_TREE_ORDERS)
                     for j, bound in enumerate(_TERMINAL_BOUNDS)}
_SIGNATURE_SEARCH.update({('prefix', 'norm'): 0, ('lyndon', 'projected'): 2,
                          ('prefix', 'projected'): 3, ('lyndon', 'norm'): 4})


def _pack_words(words):
    offsets = np.zeros(len(words) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(word) for word in words], dtype=np.uint64)
    letters = np.array([letter for word in words for letter in word], dtype=np.uint64)
    return letters, offsets


def _valid_word(word, degree, dimension):
    return 1 <= len(word) <= degree and all(
        isinstance(a, (int, np.integer)) and not isinstance(a, bool) and 0 <= a < dimension
        for a in word
    )


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


def sig_lasso(path, y, degree, alpha, *, loss="squared", fit_intercept=True,
              tol=None, max_iter=1000, initial_words=(), memory_limit=None, time_aug=False,
              lead_lag=False, end_time=1.0, n_jobs=1, tree_order="prefix", terminal_bound="norm"):
    r"""Fit a sparse linear predictor on normalized signature coordinates on CPU.

    Minimise the mean loss at ``intercept + X @ coef`` plus ``alpha * ||coef||_1``.
    Select squared error (with factor 1/2), binary logistic, Poisson count, or
    unit-threshold Huber loss with ``loss='squared'|'logistic'|'poisson'|'huber'``.
    An unpenalised intercept requires both classes for logistic regression and
    at least one positive count for Poisson regression.

    Each degree ``k`` is scaled by ``k! / V**k`` after path transforms,
    where ``V`` is the sum of the infinity norms of the increments. Constant paths have
    zero features. Columns are centred only when fitting an intercept.

    NumPy arrays and CPU Torch tensors support float32 and float64. Paths have
    shape ``(*batch, length, dimension)``. Responses have shape ``batch`` or
    ``(*batch, targets)``, with optional flattening of batch axes. Each target
    has its own objective with a scalar or per-target positive penalty. ``initial_words``
    seeds the working set with words in the transformed alphabet.

    The native solver uses cyclic coordinate descent with safeguarded
    acceleration on each working set.
    Every fit searches the feature tree by branch and bound. Multiple squared-loss
    responses can share this traversal and receive individual certificates.
    Coordinates are evaluated without constructing a full truncated signature.
    ``tree_order="prefix"`` searches by successive letters. ``"lyndon"`` groups
    ordinary signature words by their nonincreasing Lyndon factorisation.
    ``"suffix"`` grows suffixes, ``"energy"`` and ``"profile"`` group channels
    by training increment norms, and ``"forward"`` fixes the penultimate
    letter first in terminal pairs. ``"grouped"`` uses input-order terminal
    groups. ``"outside"`` screens first and last letters jointly before
    filling their interior. ``"balanced"`` splits parallel work by channel
    energy in input order. ``"adaptive"`` selects channel groups by norm
    envelope slack and fixes the more restricted terminal letter first.
    The experimental orders "pivotweak", "pivotstrong" and "pivotalternating"
    add a degree-four positional pivot screen to outside-in search, using
    weak-first, strong-first and alternating channel elimination respectively.
    All orders fit the same dictionary and objective.
    The default terminal bound uses norms. Set ``terminal_bound="projected"``
    to additionally screen channel groups using a signed time projection,
    or ``"primitive"`` for a summation-by-parts bound.
    ``memory_limit`` bounds owned native workspace and output
    in bytes. ``n_jobs`` controls CPU workers. ``max_iter`` limits working-set
    updates per response. ``tol`` defaults to 1e-5 for float32 or 1e-8 for
    float64. Feature stationarity is checked at ``min(tol, 1e-4 * alpha)``.
    Joint squared-loss searches can refine candidates to ``min(tol, 1e-6 * alpha)``
    before returning the certificate.

    Return sorted union support in ``words``, uncentred columns in ``signatures``,
    ``coef``, ``intercept``, ``objective``, ``loss``, per-target ``kkt_violation``,
    ``converged``, ``n_iter`` (working-set updates), timings and memory in
    ``stats``, and prediction metadata. ``dual_gap`` is a squared-loss duality
    gap and is None for other losses. Prediction returns the linear predictor.
    The result is independent of any autograd graph.
    """
    if not isinstance(tree_order, str) or tree_order not in _TREE_ORDERS:
        raise ValueError("tree_order must be one of " + ", ".join(_TREE_ORDERS))
    if not isinstance(terminal_bound, str) or terminal_bound not in _TERMINAL_BOUNDS:
        raise ValueError("terminal_bound must be one of " + ", ".join(_TERMINAL_BOUNDS))
    return _fit_lasso(path, y, degree, alpha, 'sig', tree_order=tree_order, terminal_bound=terminal_bound,
                      loss=loss, fit_intercept=fit_intercept, tol=tol,
                      max_iter=max_iter, initial_words=initial_words, memory_limit=memory_limit,
                      time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)


def log_sig_lasso(path, y, degree, alpha, *, loss="squared", fit_intercept=True,
                  tol=None, max_iter=1000, initial_words=(), memory_limit=None, time_aug=False,
                  lead_lag=False, end_time=1.0, n_jobs=1):
    """Fit a sparse linear predictor on tensor log-signature coordinates on CPU.

    Features are coefficients of the tensor logarithm indexed by Lyndon words.
    A word of degree ``k`` is scaled by ``k! / V**k``, where ``V`` is the path's
    variation after the requested transforms. Initial words must be Lyndon words
    in the transformed alphabet.

    The arguments, losses and returned model fields follow :func:`sig_lasso`.
    Use :func:`sig_lasso_predict` to evaluate the selected coordinates on new paths.
    """
    return _fit_lasso(path, y, degree, alpha, 'log', loss=loss, fit_intercept=fit_intercept, tol=tol,
                      max_iter=max_iter, initial_words=initial_words, memory_limit=memory_limit,
                      time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)


def _fit_lasso(path, y, degree, alpha, feature_type, *, loss, fit_intercept, tol, max_iter,
               initial_words, memory_limit, time_aug, lead_lag, end_time, n_jobs, tree_order=None, terminal_bound=None):
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
        if not _valid_word(word, degree, inputs.dimension):
            raise ValueError('initial_words contains an invalid transformed-channel word')
        words.append(tuple(int(a) for a in word))
    words = sorted(set(words), key=lambda word: (len(word), word))
    letters, offsets = _pack_words(words)
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

        native_mode = _SIGNATURE_SEARCH[tree_order, terminal_bound] if feature_type == 'sig' else _FEATURE_TYPES[feature_type]
        args = [data.data_ptr, _pointer(response, scalar), _pointer(alphas, c_double),
                _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words),
                native_mode,
                byref(options), allocate_output, None]
        status = CPSIG_WORKING_SET[data.dtype](*args)
        if allocation_error is not None:
            raise allocation_error
        _check_status(status, 'cpu')
        selected = [tuple(int(a) for a in out_letters[out_offsets[j]:out_offsets[j + 1]]) for j in range(result_shape.words)]
        converged = converged.astype(bool)
        if not converged.all():
            warnings.warn('Lasso fit did not pass the global KKT check. Inspect n_iter and kkt_violation.',
                          RuntimeWarning, stacklevel=3)
        stats = {name: getattr(result_shape.stats, name) for name, _ in result_shape.stats._fields_}
        if stats['engine'] != 3 or stats['full_transform_calls']:
            raise RuntimeError('The CPU library did not use branch and bound')
        stats['engine'] = 'tree'
        if feature_type == 'sig':
            stats['tree_order'] = tree_order
            stats['terminal_bound'] = terminal_bound
            stats['dictionary_size'] = sum(data.dimension**k for k in range(1, degree + 1))
        else:
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
                    feature_convention=feature_type,
                    normalization='factorial_variation')


def sig_lasso_predict(path, model, *, n_jobs=1, memory_limit=None):
    """Predict on CPU paths using a model from :func:`sig_lasso` or :func:`log_sig_lasso`.

    The model stores the feature convention and path transforms. Input paths
    must have the same array type, dtype and channel dimension as the model.
    The result preserves leading batch dimensions and has an optional target
    axis. Computation has no autograd connection and uses only selected features.
    Signature and tensor-log prediction process sample tiles and share common
    prefixes of selected words. ``memory_limit`` bounds native prediction workspace.
    """
    check_n_jobs(n_jobs)
    if memory_limit is not None and (isinstance(memory_limit, bool) or not isinstance(memory_limit, (int, np.integer)) or not 0 < memory_limit < 2**64):
        raise ValueError('memory_limit must be a positive integer or None')
    mode = model['feature_type']
    if mode not in _FEATURE_TYPES or model.get('normalization') != 'factorial_variation':
        raise ValueError('Unsupported Lasso model convention')
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
    dimension = inputs.dimension * (2 if model['lead_lag'] else 1) + int(model['time_aug'])
    for word in model['words']:
        if not _valid_word(word, degree, dimension):
            raise ValueError('Invalid model feature word')
    if not inputs.length:
        raise ValueError('Paths must contain at least one point')
    if not (bool(torch.isfinite(path).all()) if isinstance(path, torch.Tensor) else np.isfinite(path).all()):
        raise ValueError('path must contain only finite values')
    single = coef.ndim == 1
    targets = 1 if single else coef.shape[1]
    intercept_shape = tuple(np.shape(model['intercept']))
    if intercept_shape != (targets,) and (targets != 1 or intercept_shape != ()):
        raise ValueError('Invalid model intercept shape')
    model_intercept = model['intercept']
    finite_intercept = torch.isfinite(model_intercept).all() if isinstance(model_intercept, torch.Tensor) else np.isfinite(model_intercept).all()
    if not finite_intercept:
        raise ValueError('Model intercept must be finite')
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
    letters, offsets = _pack_words(words)
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
            _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), _FEATURE_TYPES[mode], byref(options))
        _check_status(status, 'cpu')
    return output
