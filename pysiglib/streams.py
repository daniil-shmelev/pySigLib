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

from typing import Union, List, Tuple
import numpy as np
import torch

from .param_checks import check_pos, check_type, check_n_jobs, resolve_scalar_term
from .sig_length import sig_length, log_sig_length
from .sig_join import sig_join
from .sig import sig_combine, sig
from .log_sig_join import log_sig_join
from .log_sig_combine import log_sig_combine
from .log_sig import log_sig

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _is_jax(x):
    """Return True if ``x`` is a JAX array. Returns False if JAX is not installed."""
    return _HAS_JAX and isinstance(x, jax.Array)


def _make_zero(length, batch_shape, like_arr):
    """Create a zero array of shape ``(*batch_shape, length)`` matching dtype/device of ``like_arr``."""
    full_shape = (*batch_shape, length)
    if isinstance(like_arr, torch.Tensor):
        return torch.zeros(full_shape, dtype=like_arr.dtype, device=like_arr.device)
    if _is_jax(like_arr):
        return jnp.zeros(full_shape, dtype=like_arr.dtype)
    return np.zeros(full_shape, dtype=like_arr.dtype)


def _make_identity_sig(sig_len, batch_shape, like_arr, scalar_term=True):
    """Create an identity signature of shape ``(*batch_shape, sig_len)``.

    With ``scalar_term=True`` the result is ``[1, 0, ..., 0]``; with ``scalar_term=False``
    the leading 1 is stripped and the identity becomes all zeros (the sig of a
    zero-length path is zero in every tensor level >= 1).
    """
    identity = _make_zero(sig_len, batch_shape, like_arr)
    if scalar_term:
        # JAX arrays are immutable; use functional update.
        if _is_jax(identity):
            return identity.at[..., 0].set(1.0)
        identity[..., 0] = 1.0
    return identity


def _stack(arrays, like_arr):
    """Stack arrays along a new leading dimension."""
    if isinstance(like_arr, torch.Tensor):
        return torch.stack(arrays)
    if _is_jax(like_arr):
        return jnp.stack(arrays)
    return np.stack(arrays)


def _validate_push_point(point, expected_dim, expected_batch_shape):
    """Validate a ``push(point)`` argument and return its inferred batch shape.

    Point must have shape ``(*batch, dimension)``. If the stream already has a
    locked-in batch shape, the input shape must match it.
    """
    if point.ndim < 1 or point.shape[-1] != expected_dim:
        raise ValueError(
            f"push expects a point of shape (*batch, {expected_dim}); "
            f"got shape {tuple(point.shape)}")
    batch = tuple(point.shape[:-1])
    if expected_batch_shape is not None and batch != expected_batch_shape:
        raise ValueError(
            f"push batch shape {batch} does not match the shape {expected_batch_shape} "
            f"locked in by the first push")
    return batch


def _validate_push_batch(points, expected_dim, expected_batch_shape):
    """Validate a ``push_batch(points)`` argument and return ``(batch_shape, n_points)``.

    Points must have shape ``(*batch, n_points, dimension)``. If the stream already
    has a locked-in batch shape, the input shape must match it.
    """
    if points.ndim < 2 or points.shape[-1] != expected_dim:
        raise ValueError(
            f"push_batch expects points of shape (*batch, n_points, {expected_dim}); "
            f"got shape {tuple(points.shape)}")
    batch = tuple(points.shape[:-2])
    n_points = points.shape[-2]
    if expected_batch_shape is not None and batch != expected_batch_shape:
        raise ValueError(
            f"push_batch batch shape {batch} does not match the shape {expected_batch_shape} "
            f"locked in by the first push")
    return batch, n_points


def _cat_time(a, b):
    """Concatenate along the time axis (axis ``-2``)."""
    if isinstance(a, torch.Tensor):
        return torch.cat([a, b], dim=-2)
    if _is_jax(a):
        return jnp.concatenate([a, b], axis=-2)
    return np.concatenate([a, b], axis=-2)


def _expand_time(point):
    """Insert a length-1 time axis at position ``-2``: ``(*batch, dim) -> (*batch, 1, dim)``."""
    if isinstance(point, torch.Tensor):
        return point.unsqueeze(-2)
    if _is_jax(point):
        return jnp.expand_dims(point, axis=-2)
    return point[..., np.newaxis, :]


def _flip_time(path):
    """Reverse along the time axis (axis ``-2``) and return a contiguous copy."""
    if isinstance(path, torch.Tensor):
        return path.flip(-2).contiguous()
    if _is_jax(path):
        # JAX arrays are immutable; jnp.flip returns a fresh array.
        return jnp.flip(path, axis=-2)
    return np.flip(path, axis=-2).copy()


def _last_time_step(points):
    """Extract the last time-step, preserving batch shape. Returns an owning copy."""
    if isinstance(points, torch.Tensor):
        return points[..., -1, :].contiguous()
    if _is_jax(points):
        # JAX arrays are immutable; the slice already owns its data.
        return points[..., -1, :]
    return points[..., -1, :].copy()


def _copy_sig(s):
    """Return a fresh copy of a signature tensor (not a view)."""
    if isinstance(s, torch.Tensor):
        return s.clone()
    if _is_jax(s):
        # JAX arrays are immutable; returning the same reference is safe.
        return s
    return s.copy()


class SigStream:
    """
    A stateful stream that maintains precomputed cumulative signatures over a growing
    path, supporting efficient push/pop operations and O(1) arbitrary interval queries.

    Cumulative signatures ``S(0, t)`` and their inverses ``S(0, t)^{-1}`` are stored
    for each point. Any interval signature is computed via Chen's identity:
    ``S(a, b) = S(0, a)^{-1} * S(0, b)``.

    Supports numpy arrays, torch tensors (with autograd via ``pysiglib.torch_api``),
    and JAX arrays (via ``pysiglib.jax_api``). Accepts a single path or a batch of
    independent paths - the batch shape is inferred from the first ``push`` /
    ``push_batch`` call and locked in for the rest of the stream's lifetime. A single
    ``SigStream`` instance can therefore track many independent paths in parallel.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int
    :param scalar_term: If True (default), stored signatures include the leading constant
        1 at index 0. If False, the leading element is stripped. The default will change
        to False in pySigLib v4.0.
    :type scalar_term: bool
    :param n_jobs: Number of threads to run in parallel in the internal ``sig``,
        ``sig_join`` and ``sig_combine`` calls. If ``n_jobs = 1`` the computation is
        serial. If ``-1``, all available threads are used. For ``n_jobs < -1``,
        ``max_threads + 1 + n_jobs`` threads are used.
    :type n_jobs: int

    Example::

        import pysiglib
        import numpy as np

        # Single path
        stream = pysiglib.SigStream(dimension=3, degree=4)
        path = np.random.randn(50, 3)
        stream.push_batch(path)
        s = stream.sig(10, 30)  # shape (sig_length,)

        # Batch of 8 independent paths tracked in parallel
        stream = pysiglib.SigStream(dimension=3, degree=4)
        paths = np.random.randn(8, 50, 3)
        stream.push_batch(paths)
        s = stream.sig(10, 30)  # shape (8, sig_length)
    """

    def __init__(self, dimension: int, degree: int,
                 scalar_term=None, n_jobs: int = 1,
                 _sig_join=None, _sig_combine=None, _sig=None):
        scalar_term = resolve_scalar_term(scalar_term)
        check_n_jobs(n_jobs)
        self._dimension = dimension
        self._degree = degree
        self._scalar_term = scalar_term
        self._sig_len = sig_length(dimension, degree, scalar_term=scalar_term)
        raw_sig = _sig or sig
        raw_sig_join = _sig_join or sig_join
        raw_sig_combine = _sig_combine or sig_combine
        self._sig_fn = lambda path, deg: raw_sig(
            path, deg, scalar_term=scalar_term, n_jobs=n_jobs)
        self._sig_combine_fn = lambda s1, s2, dim, deg: raw_sig_combine(
            s1, s2, dim, deg, scalar_term=scalar_term, n_jobs=n_jobs)
        self._sig_join_fn = lambda s, disp, dim, deg, prepend=False: raw_sig_join(
            s, disp, dim, deg, prepend=prepend, scalar_term=scalar_term, n_jobs=n_jobs)
        self._sigs = []      # cumulative forward sigs at each checkpoint
        self._inv_sigs = []  # cumulative inverse sigs at each checkpoint
        self._last_point = None
        self._start = 0
        self._batch_shape = None  # locked in by the first push

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point (or batch of points, one per tracked path) and update
        the cumulative signature.

        :param point: Shape ``(*batch, dimension)``. ``batch`` is either empty
            (single-path stream) or matches the batch shape locked in by the first push.
        :type point: numpy.ndarray | torch.Tensor
        """
        batch = _validate_push_point(point, self._dimension, self._batch_shape)
        if self._last_point is None:
            self._batch_shape = batch
            self._last_point = point
            identity = _make_identity_sig(self._sig_len, batch, point, self._scalar_term)
            self._sigs.append(identity)
            self._inv_sigs.append(identity)
            return
        displacement = point - self._last_point
        new_sig = self._sig_join_fn(
            self._sigs[-1], displacement, self._dimension, self._degree)
        new_inv = self._sig_join_fn(
            self._inv_sigs[-1], -displacement, self._dimension, self._degree, prepend=True)
        self._sigs.append(new_sig)
        self._inv_sigs.append(new_inv)
        self._last_point = point

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points to the stream. Computes the batch signature in a
        single batched call rather than per-point sequential joins.

        :param points: Shape ``(*batch, n_points, dimension)``. ``batch`` is either
            empty (single-path stream) or matches the batch shape locked in by the first push.
        :type points: numpy.ndarray | torch.Tensor
        """
        batch, n_points = _validate_push_batch(points, self._dimension, self._batch_shape)
        if n_points == 0:
            return
        if self._last_point is None:
            self.push(points[..., 0, :])
            if n_points == 1:
                return
            points = points[..., 1:, :]
            n_points -= 1

        last_expanded = _expand_time(self._last_point)
        batch_path = _cat_time(last_expanded, points)
        batch_path_rev = _flip_time(batch_path)

        batch_sig = self._sig_fn(batch_path, self._degree)
        new_cumulative = self._sig_combine_fn(
            self._sigs[-1], batch_sig, self._dimension, self._degree)

        batch_sig_rev = self._sig_fn(batch_path_rev, self._degree)
        new_inv = self._sig_combine_fn(
            batch_sig_rev, self._inv_sigs[-1], self._dimension, self._degree)

        self._sigs.append(new_cumulative)
        self._inv_sigs.append(new_inv)
        self._last_point = _last_time_step(points)

    def pop_front(self) -> None:
        """Remove the oldest cumulative signature from the stream."""
        if len(self._sigs) <= 1:
            raise ValueError("Cannot pop_front: stream has 1 or fewer entries")
        self._sigs.pop(0)
        self._inv_sigs.pop(0)
        self._start += 1

    def sig(self, start: int, end: int) -> Union[np.ndarray, torch.Tensor]:
        """
        Query the signature over an interval via Chen's identity.

        :param start: Start index (absolute, inclusive).
        :type start: int
        :param end: End index (absolute, inclusive).
        :type end: int
        :return: The signature of shape ``(*batch, sig_length)`` for the interval
            ``path[start:end+1]``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        si = start - self._start
        ei = end - self._start
        if si < 0 or ei >= len(self._sigs):
            raise IndexError(f"Indices [{start}, {end}] out of range [{self._start}, {self._start + len(self._sigs) - 1}]")
        # Fast path: no pops + si==0 means combine is a no-op. Return a fresh
        # copy so the caller can't mutate internal state.
        if si == 0 and self._start == 0:
            return _copy_sig(self._sigs[ei])
        return self._sig_combine_fn(
            self._inv_sigs[si], self._sigs[ei], self._dimension, self._degree)

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Query signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked signatures of shape ``(K, *batch, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._sigs[0])

    def sig_all(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the expanding (cumulative) signatures ``S(0, 0), S(0, 1), ..., S(0, t)``.

        :return: Stacked signatures of shape ``(n, *batch, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        return _stack(self._sigs, self._sigs[0])

    @property
    def size(self) -> int:
        """Number of time-steps currently in the stream."""
        return len(self._sigs)

    @property
    def start_index(self) -> int:
        """Absolute index of the first point in the stream."""
        return self._start

    @property
    def end_index(self) -> int:
        """Absolute index of the last point in the stream."""
        return self._start + len(self._sigs) - 1

    @property
    def batch_shape(self) -> Union[tuple, None]:
        """Batch shape locked in by the first push, or ``None`` if nothing has been pushed."""
        return self._batch_shape


class LogSigStream:
    """
    A stateful stream that maintains precomputed cumulative log-signatures over a growing
    path, supporting efficient push/pop operations and O(1) arbitrary interval queries.

    Cumulative log-signatures are stored for each point. Any interval log-signature
    is computed via BCH: ``L(a, b) = BCH(-L(0, a), L(0, b))``, since the inverse of a
    log-signature is its negation.

    Supports numpy arrays, torch tensors (with autograd via ``pysiglib.torch_api``),
    and JAX arrays (via ``pysiglib.jax_api``). Accepts a single path or a batch of
    independent paths - the batch shape is inferred from the first ``push`` /
    ``push_batch`` call and locked in for the rest of the stream's lifetime.

    .. note::

        You must call ``pysiglib.prepare_log_sig(dimension, degree)`` before creating
        a ``LogSigStream``. This precomputes the Lyndon basis and BCH coefficients.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signature, :math:`N`.
    :type degree: int
    :param method: Method to use for internal log-signature computation
        (``2`` or ``3``). Method ``2`` uses the Lyndon bracket basis via the
        signature-to-log-signature projection; method ``3`` computes log-sigs
        directly from the path via BCH.
    :type method: int
    :param n_jobs: Number of threads to run in parallel in internal ``log_sig``,
        ``log_sig_join`` and ``log_sig_combine`` calls. ``-1`` uses all
        available threads; for ``n_jobs < -1``, ``max_threads + 1 + n_jobs``
        threads are used.
    :type n_jobs: int

    Example::

        import pysiglib
        import numpy as np

        pysiglib.prepare_log_sig(3, 4, method=2)

        # Single path
        stream = pysiglib.LogSigStream(dimension=3, degree=4)
        path = np.random.randn(50, 3)
        stream.push_batch(path)
        ls = stream.sig(10, 30)  # shape (log_sig_length,)

        # Batch of 8 independent paths
        stream = pysiglib.LogSigStream(dimension=3, degree=4)
        paths = np.random.randn(8, 50, 3)
        stream.push_batch(paths)
        ls = stream.sig(10, 30)  # shape (8, log_sig_length)
    """

    def __init__(self, dimension: int, degree: int, method: int = 2,
                 n_jobs: int = 1,
                 _log_sig_join=None, _log_sig_combine=None, _log_sig=None):
        if method not in (2, 3):
            raise ValueError(
                f"LogSigStream requires method=2 or method=3 (Lyndon basis); "
                f"got method={method}. Method 1 uses the Hall basis which is "
                f"incompatible with log_sig_combine/log_sig_join.")
        check_n_jobs(n_jobs)
        self._dimension = dimension
        self._degree = degree
        self._ls_len = log_sig_length(dimension, degree)
        raw_log_sig = _log_sig or log_sig
        raw_log_sig_join = _log_sig_join or log_sig_join
        raw_log_sig_combine = _log_sig_combine or log_sig_combine
        self._log_sig_fn = lambda path, deg: raw_log_sig(
            path, deg, method=method, n_jobs=n_jobs)
        self._log_sig_combine_fn = lambda s1, s2, dim, deg: raw_log_sig_combine(
            s1, s2, dim, deg, n_jobs=n_jobs)
        self._log_sig_join_fn = lambda ls, disp, dim, deg: raw_log_sig_join(
            ls, disp, dim, deg, n_jobs=n_jobs)
        self._log_sigs = []
        self._last_point = None
        self._start = 0
        self._batch_shape = None

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point (or batch of points, one per tracked path) and update
        the cumulative log-signature.

        :param point: Shape ``(*batch, dimension)``.
        :type point: numpy.ndarray | torch.Tensor
        """
        batch = _validate_push_point(point, self._dimension, self._batch_shape)
        if self._last_point is None:
            self._batch_shape = batch
            self._last_point = point
            self._log_sigs.append(_make_zero(self._ls_len, batch, point))
            return
        displacement = point - self._last_point
        new_ls = self._log_sig_join_fn(
            self._log_sigs[-1], displacement, self._dimension, self._degree)
        self._log_sigs.append(new_ls)
        self._last_point = point

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points to the stream. Computes the batch log-signature in a
        single batched call rather than per-point sequential joins.

        :param points: Shape ``(*batch, n_points, dimension)``.
        :type points: numpy.ndarray | torch.Tensor
        """
        batch, n_points = _validate_push_batch(points, self._dimension, self._batch_shape)
        if n_points == 0:
            return
        if self._last_point is None:
            self.push(points[..., 0, :])
            if n_points == 1:
                return
            points = points[..., 1:, :]
            n_points -= 1

        last_expanded = _expand_time(self._last_point)
        batch_path = _cat_time(last_expanded, points)

        batch_ls = self._log_sig_fn(batch_path, self._degree)
        new_cumulative = self._log_sig_combine_fn(
            self._log_sigs[-1], batch_ls, self._dimension, self._degree)

        self._log_sigs.append(new_cumulative)
        self._last_point = _last_time_step(points)

    def pop_front(self) -> None:
        """Remove the oldest cumulative log-signature from the stream."""
        if len(self._log_sigs) <= 1:
            raise ValueError("Cannot pop_front: stream has 1 or fewer entries")
        self._log_sigs.pop(0)
        self._start += 1

    def sig(self, start: int, end: int) -> Union[np.ndarray, torch.Tensor]:
        """
        Query the log-signature over an interval via BCH.

        :param start: Start index (absolute, inclusive).
        :type start: int
        :param end: End index (absolute, inclusive).
        :type end: int
        :return: The log-signature of shape ``(*batch, log_sig_length)`` for the
            interval ``path[start:end+1]``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        si = start - self._start
        ei = end - self._start
        if si < 0 or ei >= len(self._log_sigs):
            raise IndexError(f"Indices [{start}, {end}] out of range [{self._start}, {self._start + len(self._log_sigs) - 1}]")
        # Fast path: no pops + si==0 means BCH is a no-op. Return a fresh copy.
        if si == 0 and self._start == 0:
            return _copy_sig(self._log_sigs[ei])
        return self._log_sig_combine_fn(
            -self._log_sigs[si], self._log_sigs[ei],
            self._dimension, self._degree)

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Query log-signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked log-signatures of shape ``(K, *batch, log_sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._log_sigs[0])

    def sig_all(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the expanding (cumulative) log-signatures.

        :return: Stacked log-signatures of shape ``(n, *batch, log_sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        return _stack(self._log_sigs, self._log_sigs[0])

    @property
    def size(self) -> int:
        """Number of time-steps currently in the stream."""
        return len(self._log_sigs)

    @property
    def start_index(self) -> int:
        """Absolute index of the first point in the stream."""
        return self._start

    @property
    def end_index(self) -> int:
        """Absolute index of the last point in the stream."""
        return self._start + len(self._log_sigs) - 1

    @property
    def batch_shape(self) -> Union[tuple, None]:
        """Batch shape locked in by the first push, or ``None`` if nothing has been pushed."""
        return self._batch_shape


class _WindowStream:
    """Base class for windowed stream classes. Buffers points (along the time axis)
    and computes each window's (log-)signature directly from the path slice.
    Supports arbitrary leading batch dimensions; batch shape is locked in on first push.
    """

    def __init__(self, sig_fn, dimension: int, degree: int, window_size: int, stride: int):
        self._sig_fn = sig_fn
        self._dimension = dimension
        self._degree = degree
        self._window_size = window_size
        self._stride = stride
        self._pending = []   # unbatched points awaiting consolidation
        self._buffer = None  # consolidated array of points, shape (*batch, buf_len, dim)
        self._buf_len = 0
        # Set when `stride > window_size` creates a gap between windows
        # wider than the buffer currently holds; counts points to drop.
        self._skip_next = 0
        self._windows = []
        self._next_window_end = window_size - 1
        self._batch_shape = None

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point (or batch of points). If a new window is completed,
        its (log-)signature is computed and stored.

        :param point: Shape ``(*batch, dimension)``.
        :type point: numpy.ndarray | torch.Tensor
        """
        batch = _validate_push_point(point, self._dimension, self._batch_shape)
        if self._batch_shape is None:
            self._batch_shape = batch
        if self._skip_next > 0:
            self._skip_next -= 1
            return
        self._pending.append(point)
        self._buf_len += 1
        self._emit_windows()

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points, emitting windows as they become complete.

        :param points: Shape ``(*batch, n_points, dimension)``.
        :type points: numpy.ndarray | torch.Tensor
        """
        batch, n_points = _validate_push_batch(points, self._dimension, self._batch_shape)
        if n_points == 0:
            return
        if self._batch_shape is None:
            self._batch_shape = batch
        if self._skip_next > 0:
            skip_count = min(self._skip_next, n_points)
            self._skip_next -= skip_count
            points = points[..., skip_count:, :]
            n_points -= skip_count
            if n_points == 0:
                return
        self._flush_pending()
        if self._buffer is None:
            self._buffer = points
        else:
            self._buffer = _cat_time(self._buffer, points)
        self._buf_len += n_points
        self._emit_windows()

    def _flush_pending(self):
        """Consolidate pending single-point pushes into the buffer."""
        if not self._pending:
            return
        first = self._pending[0]
        if isinstance(first, torch.Tensor):
            batch = torch.stack(self._pending, dim=-2)
        else:
            batch = np.stack(self._pending, axis=-2)
        if self._buffer is None:
            self._buffer = batch
        else:
            self._buffer = _cat_time(self._buffer, batch)
        self._pending.clear()

    def _emit_windows(self):
        self._flush_pending()
        if self._buffer is None:
            return
        while self._buf_len - 1 >= self._next_window_end:
            w_start = self._next_window_end - self._window_size + 1
            w_end = self._next_window_end + 1
            window_path = self._buffer[..., w_start:w_end, :]
            self._windows.append(self._sig_fn(window_path, self._degree))
            self._next_window_end += self._stride

        # When `stride > window_size`, `earliest_needed` can exceed `_buf_len`;
        # drop the whole buffer and defer the remainder to `_skip_next`.
        earliest_needed = self._next_window_end - self._window_size + 1
        if earliest_needed > 0:
            if earliest_needed > self._buf_len:
                self._skip_next += earliest_needed - self._buf_len
                trim = self._buf_len
            else:
                trim = earliest_needed
            self._buffer = self._buffer[..., trim:, :]
            self._buf_len -= trim
            self._next_window_end -= earliest_needed

    def sig(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the stacked (log-)signatures of all complete windows.

        :return: Array of shape ``(num_windows, *batch, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        if not self._windows:
            raise ValueError("No complete windows yet")
        return _stack(self._windows, self._windows[0])

    @property
    def num_windows(self) -> int:
        """Number of complete windows emitted so far."""
        return len(self._windows)

    @property
    def batch_shape(self) -> Union[tuple, None]:
        """Batch shape locked in by the first push, or ``None`` if nothing has been pushed."""
        return self._batch_shape


class SigWindowStream(_WindowStream):
    """
    A fixed-width sliding window over a stream of incoming points that emits
    windowed signatures. A new window is emitted every ``stride`` points once
    enough points have been accumulated to fill the window.

    Accepts a single path or a batch of independent paths - the batch shape is
    inferred from the first ``push`` / ``push_batch`` call. Windows are extracted
    along the time axis only; each batch element is windowed independently.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int
    :param window_size: Number of points per window.
    :type window_size: int
    :param stride: Number of points between successive window starts. Default 1.
    :type stride: int
    :param scalar_term: If True (default), each emitted window signature includes
        the leading constant 1. If False, the leading element is stripped. The
        default will change to False in pySigLib v4.0.
    :type scalar_term: bool
    :param n_jobs: Number of threads to run in parallel in the internal per-window
        ``sig`` calls. ``-1`` uses all available threads.
    :type n_jobs: int

    Example::

        import pysiglib
        import numpy as np

        ws = pysiglib.SigWindowStream(dimension=3, degree=4, window_size=20, stride=5)

        # Single path
        path = np.random.randn(100, 3)
        ws.push_batch(path)
        window_sigs = ws.sig()  # shape (num_windows, sig_length)

        # Batch of 8 independent paths
        ws = pysiglib.SigWindowStream(dimension=3, degree=4, window_size=20, stride=5)
        paths = np.random.randn(8, 100, 3)
        ws.push_batch(paths)
        window_sigs = ws.sig()  # shape (num_windows, 8, sig_length)
    """

    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1,
                 scalar_term=None, n_jobs: int = 1, _sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        scalar_term = resolve_scalar_term(scalar_term)
        check_n_jobs(n_jobs)
        raw_sig = _sig or sig
        sig_fn = lambda path, deg: raw_sig(path, deg, scalar_term=scalar_term, n_jobs=n_jobs)
        super().__init__(sig_fn, dimension, degree, window_size, stride)


class LogSigWindowStream(_WindowStream):
    """
    A fixed-width sliding window over a stream of incoming points that emits
    windowed log-signatures. A new window is emitted every ``stride`` points once
    enough points have been accumulated to fill the window.

    Accepts a single path or a batch of independent paths - the batch shape is
    inferred from the first ``push`` / ``push_batch`` call.

    .. note::

        You must call ``pysiglib.prepare_log_sig(dimension, degree, method=2)`` before
        creating a ``LogSigWindowStream``.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signature, :math:`N`.
    :type degree: int
    :param window_size: Number of points per window.
    :type window_size: int
    :param stride: Number of points between successive window starts. Default 1.
    :type stride: int
    :param method: Method used for per-window log-signature computation (``2`` or ``3``).
    :type method: int
    :param n_jobs: Number of threads to run in parallel in the internal per-window
        ``log_sig`` calls. ``-1`` uses all available threads.
    :type n_jobs: int

    Example::

        import pysiglib
        import numpy as np

        pysiglib.prepare_log_sig(3, 4, method=2)
        ws = pysiglib.LogSigWindowStream(dimension=3, degree=4, window_size=20, stride=5)

        path = np.random.randn(100, 3)
        ws.push_batch(path)
        window_logsigs = ws.sig()  # shape (num_windows, log_sig_length)
    """

    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1,
                 method: int = 2, n_jobs: int = 1, _log_sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        check_n_jobs(n_jobs)
        sig_fn = _log_sig or (lambda path, deg: log_sig(path, deg, method=method, n_jobs=n_jobs))
        super().__init__(sig_fn, dimension, degree, window_size, stride)
