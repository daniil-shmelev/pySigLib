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
from ctypes import POINTER, byref, c_double, c_uint64, c_void_p, cast, pointer
import warnings

import numpy as np
import torch

from ._lasso_types import LassoOptions, LassoShape, LassoAllocator
from .data_handlers import PathInputHandler
from .dtypes import DTYPES, CPSIG_SIG_LASSO, CUSIG_SIG_LASSO
from .load_siglib import CPSIG_SIG_LASSO_FEATURES, CPSIG_SIG_LASSO_PREDICT
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


def sig_lasso(path, y, degree, alpha, *, feature_type="sig", method="sparse", search_method="grouped",
              bound_method="approximate",
              fit_intercept=True, tol=None, max_iter=1000,
              batch_size=128, initial_words=(), cache_bytes=256 * 1024**2,
              memory_limit=None, time_aug=False, lead_lag=False, end_time=1.0,
              n_jobs=1):
    r"""Fit Lasso on weighted signature or log-signature coordinates.

    ``feature_type='sig'`` uses signature words on CPU or CUDA. ``'log'``
    uses tensor log coefficients at Lyndon words, matching ``log_sig(method=1)``.
    ``'hall'`` uses canonical reverse-degree Hall log coefficients. Log and
    Hall fitting are CPU-only. All modes use the same factorial/variation
    scale and solver, but define different dictionaries and L1 penalties.
    In Hall mode, ``words`` and ``initial_words`` are foliage words in the
    fixed Hall set; they need not be Lyndon. ``signatures`` contains the
    selected features in every mode. :func:`sig_lasso_predict` evaluates them
    on new CPU paths using the returned convention and transform metadata.

    Minimise ``||y - intercept - Z @ coef||**2 / (2*n) + alpha*||coef||_1``,
    where signature mode uses ``Z[i, w] = k! * S(path[i])^w / V(path[i])**k``
    for ``k=len(w)``. Log and Hall modes apply this scale to their coordinates.
    ``V`` is the sum of the infinity norms of the path increments, computed
    after path transforms. Constant paths have zero nonempty features.
    Columns are centred when fitting an intercept; they are not standardised.
    ``method='full'`` computes the
    complete dictionary; ``method='sparse'`` searches it using the absolute
    prefix and remaining variation. Both use the same native solver.

    ``bound_method='approximate'`` uses node-based trapezoidal integration
    throughout search and stopping. ``dual_gap`` is then an estimate and
    ``converged`` means that the approximate stopping test passed, without
    an exact check or repair pass. ``bound_method='exact'`` integrates the
    absolute Bernstein prefix polynomial with conservative numerical padding.
    Full fits ignore this option. Both modes compute exact signature columns
    up to floating-point error, directly in native code. Log exact bounds
    use nonnegative Bernstein polynomial envelopes. Hall exact bounds use
    Bernstein modulus envelopes and a factorial product dynamic program.
    Normalized log and Hall coordinates can exceed one in magnitude.

    ``search_method`` selects sparse-search optimisations: ``'plain'`` uses
    individual child bounds, ``'grouped'`` also tests channel groups,
    ``'signed'`` adds signed terminal bounds to plain search, and
    ``'grouped_signed'`` enables both. Signed checks cover the last two
    levels and retain cancellation across samples. Full fits ignore this
    setting. Hall grouping uses right factors, and signed Hall checks use
    small terminal blocks. Channel order is unchanged.

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
    1e-8 for float64 or 1e-5 for float32. The stopping gap is scaled by
    ``max(1, mean(centered_y**2))`` and is estimated in approximate sparse mode.

    ``cache_bytes`` bounds optional caches. ``memory_limit`` optionally
    limits owned native fitting allocations, including factors and library
    workspace allowances, in bytes. Reuse lasts for this call
    only. ``n_jobs`` follows the usual CPU threading convention. CUDA uses
    the current stream on the input device. Transform parameters have the
    same meaning as in :func:`sig`.

    Returns a dictionary with ``words``, ``coef``, ``signatures``, ``intercept``, ``objective``,
    ``dual_gap``, ``converged``, ``n_iter`` and ``stats``. Words are the union
    of nonzero coordinates, sorted by degree then lexicographically.
    Coefficients have shape ``(n_words,)`` or ``(n_words, n_targets)``.
    ``signatures`` contains uncentred normalized features for the returned
    words, with shape ``(*batch, n_words)``. Training prediction is
    ``intercept + signatures @ coef``. The feature matrix is shared across
    targets, and its storage is included in ``memory_limit`` and ``peak_bytes``.
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
    feature_types = {'sig': 0, 'log': 1, 'hall': 2}
    if not isinstance(feature_type, str) or feature_type not in feature_types:
        raise ValueError("feature_type must be 'sig', 'log', or 'hall'")
    search_methods = {'plain': 0, 'grouped': 1, 'signed': 2, 'grouped_signed': 3}
    if not isinstance(search_method, str) or search_method not in search_methods:
        raise ValueError("search_method must be 'plain', 'grouped', 'signed', or 'grouped_signed'")
    if not isinstance(bound_method, str) or bound_method not in ('approximate', 'exact'):
        raise ValueError("bound_method must be 'approximate' or 'exact'")
    for name, value in (('fit_intercept', fit_intercept), ('time_aug', time_aug), ('lead_lag', lead_lag)):
        check_type(value, name, bool)
    check_n_jobs(n_jobs)
    if not np.isfinite(end_time):
        raise ValueError('end_time must be finite')
    if feature_type != 'sig' and isinstance(path, torch.Tensor) and path.device.type != 'cpu':
        raise ValueError('Log and Hall Lasso features are supported on CPU only')
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
        if feature_type != 'sig' and alpha.device.type != 'cpu':
            raise ValueError('Log and Hall Lasso inputs must be on CPU only')
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
    if feature_type != 'sig':
        functions = CPSIG_SIG_LASSO_FEATURES
    if functions is None:
        if feature_type != 'sig':
            raise RuntimeError('The CPU native library does not support log/Hall Lasso; rebuild pysiglib')
        raise RuntimeError('The native library does not support the current sig_lasso interface; rebuild pysiglib and its CUDA plugin')
    if feature_type != 'sig' and method == 'full':
        primitive = [0] * (degree + 1)
        for k in range(1, degree + 1):
            primitive[k] = inputs.dimension**k - sum(primitive[j] for j in range(1, k) if k % j == 0)
        required = sum((primitive[k] // k) * (inputs.batch_size * 8 + 1024 + k * 64 + targets * 32) for k in range(1, degree + 1))
        if required > np.iinfo(np.intp).max:
            raise OverflowError('Full log/Hall dictionary workspace exceeds addressable storage')
        if memory_limit is not None and required > memory_limit:
            raise MemoryError('memory_limit cannot hold the full log/Hall dictionary')
    context = torch.cuda.device(path.device) if device == 'cuda' else nullcontext()
    with context, torch.no_grad():
        transformed = transform_path(inputs.path, time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs) if time_aug or lead_lag else inputs.path
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        response = ensure_own_contiguous_storage(y.detach() if isinstance(y, torch.Tensor) else y)
        options = LassoOptions(data.batch_size, data.length, data.dimension, degree, targets,
                               max_iter, batch_size, cache_bytes, memory_limit or 0,
                               tol, n_jobs, method == 'full', fit_intercept, inputs.dtype == 'float32', search_methods[search_method], bound_method == 'exact')
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
                    values = [np.empty(targets, dtype=path.dtype) for _ in range(3)]
                else:
                    coefficients = torch.empty((result_shape.words, targets), dtype=path.dtype, device=path.device)
                    signatures = torch.empty((*inputs.batch_shape, result_shape.words), dtype=path.dtype, device=path.device)
                    values = [torch.empty(targets, dtype=path.dtype, device=path.device) for _ in range(3)]
                converged = np.empty(targets, dtype=np.uint64)
                iterations = np.empty(targets, dtype=np.uint64)
                for name, array in zip(('letters', 'offsets', 'converged', 'iterations'),
                                       (out_letters, out_offsets, converged, iterations)):
                    setattr(output.contents, name, _pointer(array, c_uint64))
                for name, array in zip(('coefficients', 'intercepts', 'objectives', 'gaps', 'signatures'), (coefficients, *values, signatures)):
                    setattr(output.contents, name, cast(_pointer(array, scalar), c_void_p).value)
                output.contents.stats = pointer(result_shape.stats)
                return 0
            except BaseException as error:
                allocation_error = error
                return 1

        args = [data.data_ptr, _pointer(response, scalar), _pointer(alphas, c_double),
                _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), byref(options)]
        if feature_type != 'sig':
            args.insert(-1, feature_types[feature_type])
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
                reason = ('active-equation or objective-descent accuracy limit after a QR rebuild' if feature_type == 'sig'
                          else 'native feature or active-equation accuracy limit')
            criterion = 'estimated gap' if method == 'sparse' and bound_method == 'approximate' else 'global gap'
            warnings.warn('sig_lasso reached the ' + reason + ' before the requested ' + criterion + '; inspect converged and dual_gap', RuntimeWarning, stacklevel=2)
        stats = {name: getattr(result_shape.stats, name) for name, _ in result_shape.stats._fields_}
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
                    dual_gap=values[2][0] if single else values[2], converged=converged[0] if single else converged,
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
    if isinstance(path, torch.Tensor) and path.device.type != 'cpu':
        raise ValueError('sig_lasso_predict supports CPU arrays only')
    inputs = PathInputHandler(path, False, False, 1.0, 'path')
    coef = model['coef']
    if inputs.device == 'cuda' or (isinstance(coef, torch.Tensor) and coef.device.type != 'cpu'):
        raise ValueError('sig_lasso_predict supports CPU arrays only')
    if isinstance(model['intercept'], torch.Tensor) and model['intercept'].device.type != 'cpu':
        raise ValueError('sig_lasso_predict requires a CPU intercept')
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
        transformed = transform_path(inputs.path, time_aug=model['time_aug'], lead_lag=model['lead_lag'], end_time=model['end_time'], n_jobs=n_jobs)
        data = PathInputHandler(transformed, False, False, 1.0, 'path')
        coefficients = ensure_own_contiguous_storage(coef.detach() if isinstance(coef, torch.Tensor) else coef)
        intercept = (np.atleast_1d(np.asarray(model['intercept'], dtype=path.dtype)) if isinstance(path, np.ndarray)
                     else torch.as_tensor(model['intercept'], dtype=path.dtype, device='cpu').detach().reshape(-1).contiguous())
        selected_degree = max((len(w) for w in words), default=1)
        cache_bytes = min(64 * 1024**2, memory_limit // 4) if memory_limit else 64 * 1024**2
        options = LassoOptions(data.batch_size, data.length, data.dimension, selected_degree, targets, 1, 1, cache_bytes, memory_limit or 0, 1e-8, n_jobs, 0, 0, inputs.dtype == 'float32', 0, 1)
        scalar = DTYPES[data.dtype]
        status = CPSIG_SIG_LASSO_PREDICT[data.dtype](data.data_ptr, _pointer(coefficients, scalar), _pointer(intercept, scalar), _pointer(output, scalar),
            _pointer(letters, c_uint64), _pointer(offsets, c_uint64), len(words), feature_types[mode], byref(options))
        _check_status(status, 'cpu')
    return output
