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

from __future__ import annotations
from array_api_compat import array_namespace, device
from ._array import ArrayT, copy_array, require_array, set_first

from typing import Generic, Union, List, Tuple
import numpy as np

from .param_checks import check_pos, check_type, check_n_jobs
from ..sig_length import sig_length, log_sig_length
from .sig_join import sig_join
from .sig import sig_combine, sig
from .log_sig_join import log_sig_join
from .log_sig_combine import log_sig_combine
from .log_sig import log_sig


def _make_zero(length, batch_shape, like_arr):
    xp = array_namespace(like_arr)
    return xp.zeros((*batch_shape, length), dtype=like_arr.dtype, device=device(like_arr))


def _make_identity_sig(sig_len, batch_shape, like_arr, scalar_term=True):
    identity = _make_zero(sig_len, batch_shape, like_arr)
    return set_first(identity, 1.0) if scalar_term else identity


def _stack(arrays, like_arr):
    return array_namespace(like_arr).stack(arrays)


def _validate_push_point(point, expected_dim, expected_batch_shape):
    """Validate a ``push(point)`` argument and return its inferred batch shape.

    Point must have shape ``(..., dimension)``. If the stream already has a
    locked-in batch shape, the input shape must match it.
    """
    if point.ndim < 1 or point.shape[-1] != expected_dim:
        raise ValueError(
            f"push expects a point of shape (..., {expected_dim}); "
            f"got shape {tuple(point.shape)}")
    batch = tuple(point.shape[:-1])
    if expected_batch_shape is not None and batch != expected_batch_shape:
        raise ValueError(
            f"push batch shape {batch} does not match the shape {expected_batch_shape} "
            f"locked in by the first push")
    return batch


def _validate_push_batch(points, expected_dim, expected_batch_shape):
    """Validate a ``push_batch(points)`` argument and return ``(batch_shape, n_points)``.

    Points must have shape ``(..., n_points, dimension)``. If the stream already
    has a locked-in batch shape, the input shape must match it.
    """
    if points.ndim < 2 or points.shape[-1] != expected_dim:
        raise ValueError(
            f"push_batch expects points of shape (..., n_points, {expected_dim}); "
            f"got shape {tuple(points.shape)}")
    batch = tuple(points.shape[:-2])
    n_points = points.shape[-2]
    if expected_batch_shape is not None and batch != expected_batch_shape:
        raise ValueError(
            f"push_batch batch shape {batch} does not match the shape {expected_batch_shape} "
            f"locked in by the first push")
    return batch, n_points


def _cat_time(a, b):
    return array_namespace(a).concat([a, b], axis=-2)


def _expand_time(point):
    return array_namespace(point).expand_dims(point, axis=-2)


def _flip_time(path):
    return array_namespace(path).flip(path, axis=-2)


def _last_time_step(points):
    return copy_array(points[..., -1, :])


def _copy_sig(s):
    return copy_array(s)


class SigStream(Generic[ArrayT]):
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
    :param scalar_term: If True, stored signatures include the leading constant
        1 at index 0. If False (default), the leading element is stripped.
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

    _array_type = np.ndarray

    def __init__(self, dimension: int, degree: int,
                 *,
                 scalar_term: bool = False, n_jobs: int = 1,
                 _sig_join=None, _sig_combine=None, _sig=None):
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
            s1, s2, dim, deg, n_jobs=n_jobs)
        self._sig_join_fn = lambda s, disp, dim, deg, prepend=False: raw_sig_join(
            s, disp, dim, deg, prepend=prepend, n_jobs=n_jobs)
        self._sigs = []      # cumulative forward sigs at each checkpoint
        self._inv_sigs = []  # cumulative inverse sigs at each checkpoint
        self._last_point = None
        self._start = 0
        self._batch_shape = None  # locked in by the first push

    def push(self, point: ArrayT) -> None:
        """
        Append a single point (or batch of points, one per tracked path) and update
        the cumulative signature.

        :param point: Shape ``(..., dimension)``. The leading batch dimensions are
            either empty (single-path stream) or match the batch shape locked in by
            the first push.
        :type point: Array
        """
        require_array(point, self._array_type, "point")
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

    def push_batch(self, points: ArrayT) -> None:
        """
        Append multiple points to the stream. Computes the batch signature in a
        single batched call rather than per-point sequential joins.

        :param points: Shape ``(..., n_points, dimension)``. The leading batch
            dimensions are either empty (single-path stream) or match the batch
            shape locked in by the first push.
        :type points: Array
        """
        require_array(points, self._array_type, "points")
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

    def sig(self, start: int, end: int) -> ArrayT:
        """
        Query the signature over an interval via Chen's identity.

        :param start: Start index (absolute, inclusive).
        :type start: int
        :param end: End index (absolute, inclusive).
        :type end: int
        :return: The signature of shape ``(..., sig_length)`` for the interval
            ``path[start:end+1]``.
        :rtype: Array
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

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> ArrayT:
        """
        Query signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked signatures of shape ``(K, ..., sig_length)``.
        :rtype: Array
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._sigs[0])

    def sig_all(self) -> ArrayT:
        """
        Return the expanding (cumulative) signatures ``S(0, 0), S(0, 1), ..., S(0, t)``.

        :return: Stacked signatures of shape ``(n, ..., sig_length)``.
        :rtype: Array
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


class LogSigStream(Generic[ArrayT]):
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

        The operations of this class require a call to
        ``pysiglib.prepare_log_sig(dimension, degree, method=3)``.

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
        pysiglib.prepare_log_sig(3, 4, method=3)

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

    _array_type = np.ndarray

    def __init__(self, dimension: int, degree: int,
                 *,
                 method: int = 2, n_jobs: int = 1,
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

    def push(self, point: ArrayT) -> None:
        """
        Append a single point (or batch of points, one per tracked path) and update
        the cumulative log-signature.

        :param point: Shape ``(..., dimension)``.
        :type point: Array
        """
        require_array(point, self._array_type, "point")
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

    def push_batch(self, points: ArrayT) -> None:
        """
        Append multiple points to the stream. Computes the batch log-signature in a
        single batched call rather than per-point sequential joins.

        :param points: Shape ``(..., n_points, dimension)``.
        :type points: Array
        """
        require_array(points, self._array_type, "points")
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

    def sig(self, start: int, end: int) -> ArrayT:
        """
        Query the log-signature over an interval via BCH.

        :param start: Start index (absolute, inclusive).
        :type start: int
        :param end: End index (absolute, inclusive).
        :type end: int
        :return: The log-signature of shape ``(..., log_sig_length)`` for the
            interval ``path[start:end+1]``.
        :rtype: Array
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

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> ArrayT:
        """
        Query log-signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked log-signatures of shape ``(K, ..., log_sig_length)``.
        :rtype: Array
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._log_sigs[0])

    def sig_all(self) -> ArrayT:
        """
        Return the expanding (cumulative) log-signatures.

        :return: Stacked log-signatures of shape ``(n, ..., log_sig_length)``.
        :rtype: Array
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


class _WindowStream(Generic[ArrayT]):
    """Base class for windowed stream classes. Buffers points (along the time axis)
    and computes each window's (log-)signature directly from the path slice.
    Supports arbitrary leading batch dimensions; batch shape is locked in on first push.
    """

    _array_type = np.ndarray

    def __init__(self, sig_fn, dimension: int, degree: int, window_size: int, stride: int):
        self._sig_fn = sig_fn
        self._dimension = dimension
        self._degree = degree
        self._window_size = window_size
        self._stride = stride
        self._pending = []   # unbatched points awaiting consolidation
        self._buffer = None  # consolidated array of points, shape (..., buf_len, dim)
        self._buf_len = 0
        # Set when `stride > window_size` creates a gap between windows
        # wider than the buffer currently holds; counts points to drop.
        self._skip_next = 0
        self._windows = []
        self._next_window_end = window_size - 1
        self._batch_shape = None

    def push(self, point: ArrayT) -> None:
        """
        Append a single point (or batch of points). If a new window is completed,
        its (log-)signature is computed and stored.

        :param point: Shape ``(..., dimension)``.
        :type point: Array
        """
        require_array(point, self._array_type, "point")
        batch = _validate_push_point(point, self._dimension, self._batch_shape)
        if self._batch_shape is None:
            self._batch_shape = batch
        if self._skip_next > 0:
            self._skip_next -= 1
            return
        self._pending.append(point)
        self._buf_len += 1
        self._emit_windows()

    def push_batch(self, points: ArrayT) -> None:
        """
        Append multiple points, emitting windows as they become complete.

        :param points: Shape ``(..., n_points, dimension)``.
        :type points: Array
        """
        require_array(points, self._array_type, "points")
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
        batch = array_namespace(first).stack(self._pending, axis=-2)
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

    def sig(self) -> ArrayT:
        """
        Return the stacked (log-)signatures of all complete windows.

        :return: Array of shape ``(num_windows, ..., sig_length)``.
        :rtype: Array
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


class SigWindowStream(_WindowStream[ArrayT]):
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
    :param scalar_term: If True, each emitted window signature includes the leading
        constant 1. If False (default), the leading element is stripped.
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

    def __init__(self, dimension: int, degree: int, window_size: int,
                 *,
                 stride: int = 1, scalar_term: bool = False, n_jobs: int = 1, _sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        check_n_jobs(n_jobs)
        raw_sig = _sig or sig
        sig_fn = lambda path, deg: raw_sig(path, deg, scalar_term=scalar_term, n_jobs=n_jobs)
        super().__init__(sig_fn, dimension, degree, window_size, stride)


class LogSigWindowStream(_WindowStream[ArrayT]):
    """
    A fixed-width sliding window over a stream of incoming points that emits
    windowed log-signatures. A new window is emitted every ``stride`` points once
    enough points have been accumulated to fill the window.

    Accepts a single path or a batch of independent paths - the batch shape is
    inferred from the first ``push`` / ``push_batch`` call.

    .. note::

        Before creating a ``LogSigWindowStream``, prepare method 3 for its BCH
        operations and method 2 for its default log-signature method.

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
        pysiglib.prepare_log_sig(3, 4, method=3)
        ws = pysiglib.LogSigWindowStream(dimension=3, degree=4, window_size=20, stride=5)

        path = np.random.randn(100, 3)
        ws.push_batch(path)
        window_logsigs = ws.sig()  # shape (num_windows, log_sig_length)
    """

    def __init__(self, dimension: int, degree: int, window_size: int,
                 *,
                 stride: int = 1, method: int = 2, n_jobs: int = 1, _log_sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        check_n_jobs(n_jobs)
        sig_fn = _log_sig or (lambda path, deg: log_sig(path, deg, method=method, n_jobs=n_jobs))
        super().__init__(sig_fn, dimension, degree, window_size, stride)
