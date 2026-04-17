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

from .param_checks import check_pos, check_type
from .sig_length import sig_length, log_sig_length
from .sig_join import sig_join
from .sig import sig_combine, sig
from .log_sig_join import log_sig_join
from .log_sig_combine import log_sig_combine
from .log_sig import log_sig


def _make_identity_sig(sig_len, like_arr):
    """Create the identity signature (1, 0, 0, ..., 0) matching dtype/device of like_arr."""
    if isinstance(like_arr, torch.Tensor):
        identity = torch.zeros(sig_len, dtype=like_arr.dtype, device=like_arr.device)
        identity[0] = 1.0
    else:
        identity = np.zeros(sig_len, dtype=like_arr.dtype)
        identity[0] = 1.0
    return identity


def _make_zero(length, like_arr):
    """Create a zero array matching dtype/device of like_arr."""
    if isinstance(like_arr, torch.Tensor):
        return torch.zeros(length, dtype=like_arr.dtype, device=like_arr.device)
    else:
        return np.zeros(length, dtype=like_arr.dtype)


def _stack(arrays, like_arr):
    """Stack arrays along a new first dimension."""
    if isinstance(like_arr, torch.Tensor):
        return torch.stack(arrays)
    else:
        return np.stack(arrays)


def _is_push_batch_empty(points):
    """Raise on bad shape. Return True if the batch is empty."""
    if points.ndim != 2:
        raise ValueError(
            f"push_batch expects a 2D array of shape (n_points, dimension); "
            f"got shape {tuple(points.shape)}")
    return points.shape[0] == 0


def _validate_push_point(point):
    """Raise on bad point shape (must be 1D)."""
    if point.ndim != 1:
        raise ValueError(
            f"push expects a 1D point of shape (dimension,); "
            f"got shape {tuple(point.shape)}")


class SigStream:
    """
    A stateful stream that maintains precomputed cumulative signatures over a growing
    path, supporting efficient push/pop operations and O(1) arbitrary interval queries.

    Cumulative signatures ``S(0, t)`` and their inverses ``S(0, t)^{-1}`` are stored
    for each point. Any interval signature is computed via Chen's identity:
    ``S(a, b) = S(0, a)^{-1} * S(0, b)``.

    Supports both numpy arrays and torch tensors (with autograd).

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int

    Example::

        import pysiglib
        import numpy as np

        stream = pysiglib.SigStream(dimension=3, degree=4)

        # Per-point push: stores a cumulative sig per point
        path = np.random.randn(50, 3)
        for p in path:
            stream.push(p)
        s = stream.sig(10, 30)  # arbitrary interval query

        # Or use push_batch for speed (stores only batch endpoints)
        stream2 = pysiglib.SigStream(dimension=3, degree=4)
        stream2.push_batch(path[:25])
        stream2.push_batch(path[25:])
        s2 = stream2.sig(0, 2)  # query across batch boundaries
    """

    def __init__(self, dimension: int, degree: int, _sig_join=None, _sig_combine=None, _sig=None):
        self._dimension = dimension
        self._degree = degree
        self._sig_len = sig_length(dimension, degree, scalar_term=True)
        self._sig_join_fn = _sig_join or sig_join
        self._sig_combine_fn = _sig_combine or sig_combine
        self._sig_fn = _sig or sig
        self._sigs = []      # cumulative forward sigs at each checkpoint
        self._inv_sigs = []  # cumulative inverse sigs at each checkpoint
        self._last_point = None
        self._start = 0

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point to the stream and update the cumulative signature.

        :param point: A point of shape ``(dimension,)``.
        :type point: numpy.ndarray | torch.Tensor
        """
        if self._last_point is None:
            self._last_point = point
            identity = _make_identity_sig(self._sig_len, point)
            self._sigs.append(identity)
            self._inv_sigs.append(identity)
            return
        displacement = point - self._last_point
        new_sig = self._sig_join_fn(self._sigs[-1], displacement, self._dimension, self._degree)
        new_inv = self._sig_join_fn(self._inv_sigs[-1], -displacement, self._dimension, self._degree, prepend=True)
        self._sigs.append(new_sig)
        self._inv_sigs.append(new_inv)
        self._last_point = point

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points to the stream. Computes the batch signature in a
        single native call rather than per-point sequential joins.

        :param points: Points of shape ``(n, dimension)``.
        :type points: numpy.ndarray | torch.Tensor
        """
        if _is_push_batch_empty(points):
            return
        if self._last_point is None:
            self.push(points[0])
            points = points[1:]
            if points.shape[0] == 0:
                return

        if isinstance(points, torch.Tensor):
            batch_path = torch.cat([self._last_point.unsqueeze(0), points])
            batch_path_rev = batch_path.flip(0).contiguous()
        else:
            batch_path = np.concatenate([self._last_point[np.newaxis], points])
            batch_path_rev = batch_path[::-1].copy()

        batch_sig = self._sig_fn(batch_path, self._degree)
        new_cumulative = self._sig_combine_fn(self._sigs[-1], batch_sig, self._dimension, self._degree)

        batch_sig_rev = self._sig_fn(batch_path_rev, self._degree)
        new_inv = self._sig_combine_fn(batch_sig_rev, self._inv_sigs[-1], self._dimension, self._degree)

        self._sigs.append(new_cumulative)
        self._inv_sigs.append(new_inv)
        self._last_point = points[-1]

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
        :return: The signature ``S(path[start:end+1])``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        si = start - self._start
        ei = end - self._start
        if si < 0 or ei >= len(self._sigs):
            raise IndexError(f"Indices [{start}, {end}] out of range [{self._start}, {self._start + len(self._sigs) - 1}]")
        # Fast path: no pops + si==0 means combine is a no-op. Return a fresh
        # copy so the caller can't mutate internal state.
        if si == 0 and self._start == 0:
            s = self._sigs[ei]
            return s.clone() if isinstance(s, torch.Tensor) else s.copy()
        return self._sig_combine_fn(self._inv_sigs[si], self._sigs[ei], self._dimension, self._degree)

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Query signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked signatures of shape ``(K, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._sigs[0])

    def sig_all(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the expanding (cumulative) signatures ``S(0, 0), S(0, 1), ..., S(0, t)``.

        :return: Stacked signatures of shape ``(n, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        return _stack(self._sigs, self._sigs[0])

    @property
    def size(self) -> int:
        """Number of points currently in the stream."""
        return len(self._sigs)

    @property
    def start_index(self) -> int:
        """Absolute index of the first point in the stream."""
        return self._start

    @property
    def end_index(self) -> int:
        """Absolute index of the last point in the stream."""
        return self._start + len(self._sigs) - 1


class LogSigStream:
    """
    A stateful stream that maintains precomputed cumulative log-signatures over a growing
    path, supporting efficient push/pop operations and O(1) arbitrary interval queries.

    Cumulative log-signatures are stored for each point. Any interval log-signature
    is computed via BCH: ``L(a, b) = BCH(-L(0, a), L(0, b))``, since the inverse of a
    log-signature is its negation.

    Supports both numpy arrays and torch tensors (with autograd).

    .. note::

        You must call ``pysiglib.prepare_log_sig(dimension, degree)`` before creating
        a ``LogSigStream``. This precomputes the Lyndon basis and BCH coefficients.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the log-signature, :math:`N`.
    :type degree: int

    Example::

        import pysiglib
        import numpy as np

        pysiglib.prepare_log_sig(3, 4, method=2)
        stream = pysiglib.LogSigStream(dimension=3, degree=4)

        path = np.random.randn(50, 3)
        stream.push_batch(path)

        # Log-signature over interval [10, 30]
        ls = stream.sig(10, 30)

        # Expanding log-signatures at every prefix
        all_ls = stream.sig_all()
    """

    def __init__(self, dimension: int, degree: int, method: int = 2,
                 _log_sig_join=None, _log_sig_combine=None, _log_sig=None):
        if method not in (2, 3):
            raise ValueError(
                f"LogSigStream requires method=2 or method=3 (Lyndon basis); "
                f"got method={method}. Method 1 uses the Hall basis which is "
                f"incompatible with log_sig_combine/log_sig_join.")
        self._dimension = dimension
        self._degree = degree
        self._ls_len = log_sig_length(dimension, degree)
        self._log_sig_join_fn = _log_sig_join or log_sig_join
        self._log_sig_combine_fn = _log_sig_combine or log_sig_combine
        self._log_sig_fn = _log_sig or (lambda path, deg: log_sig(path, deg, method=method))
        self._log_sigs = []
        self._last_point = None
        self._start = 0

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point to the stream and update the cumulative log-signature.

        :param point: A point of shape ``(dimension,)``.
        :type point: numpy.ndarray | torch.Tensor
        """
        if self._last_point is None:
            self._last_point = point
            self._log_sigs.append(_make_zero(self._ls_len, point))
            return
        displacement = point - self._last_point
        new_ls = self._log_sig_join_fn(self._log_sigs[-1], displacement, self._dimension, self._degree)
        self._log_sigs.append(new_ls)
        self._last_point = point

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points to the stream. Computes the batch log-signature in a
        single native call rather than per-point sequential joins.

        :param points: Points of shape ``(n, dimension)``.
        :type points: numpy.ndarray | torch.Tensor
        """
        if _is_push_batch_empty(points):
            return
        if self._last_point is None:
            self.push(points[0])
            points = points[1:]
            if points.shape[0] == 0:
                return

        if isinstance(points, torch.Tensor):
            batch_path = torch.cat([self._last_point.unsqueeze(0), points])
        else:
            batch_path = np.concatenate([self._last_point[np.newaxis], points])

        batch_ls = self._log_sig_fn(batch_path, self._degree)
        new_cumulative = self._log_sig_combine_fn(self._log_sigs[-1], batch_ls, self._dimension, self._degree)

        self._log_sigs.append(new_cumulative)
        self._last_point = points[-1]

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
        :return: The log-signature ``L(path[start:end+1])``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        si = start - self._start
        ei = end - self._start
        if si < 0 or ei >= len(self._log_sigs):
            raise IndexError(f"Indices [{start}, {end}] out of range [{self._start}, {self._start + len(self._log_sigs) - 1}]")
        # Fast path: no pops + si==0 means BCH is a no-op. Return a fresh copy.
        if si == 0 and self._start == 0:
            ls = self._log_sigs[ei]
            return ls.clone() if isinstance(ls, torch.Tensor) else ls.copy()
        return self._log_sig_combine_fn(-self._log_sigs[si], self._log_sigs[ei],
                                        self._dimension, self._degree)

    def sig_batch(self, intervals: List[Tuple[int, int]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Query log-signatures over multiple intervals at once.

        :param intervals: List of ``(start, end)`` pairs.
        :type intervals: list[tuple[int, int]]
        :return: Stacked log-signatures of shape ``(K, log_sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        results = [self.sig(s, e) for s, e in intervals]
        return _stack(results, self._log_sigs[0])

    def sig_all(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the expanding (cumulative) log-signatures.

        :return: Stacked log-signatures of shape ``(n, log_sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        return _stack(self._log_sigs, self._log_sigs[0])

    @property
    def size(self) -> int:
        """Number of points currently in the stream."""
        return len(self._log_sigs)

    @property
    def start_index(self) -> int:
        """Absolute index of the first point in the stream."""
        return self._start

    @property
    def end_index(self) -> int:
        """Absolute index of the last point in the stream."""
        return self._start + len(self._log_sigs) - 1


class _WindowStream:
    """Base class for windowed stream classes. Buffers points and computes
    each window's (log-)signature directly from the path slice."""

    def __init__(self, sig_fn, degree: int, window_size: int, stride: int):
        self._sig_fn = sig_fn
        self._degree = degree
        self._window_size = window_size
        self._stride = stride
        self._pending = []   # unbatched points awaiting consolidation
        self._buffer = None  # consolidated array of points
        self._buf_len = 0
        # Set when `stride > window_size` creates a gap between windows
        # wider than the buffer currently holds; counts points to drop.
        self._skip_next = 0
        self._windows = []
        self._next_window_end = window_size - 1

    def push(self, point: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append a single point. If a new window is completed, its (log-)signature
        is computed and stored.

        :param point: A point of shape ``(dimension,)``.
        :type point: numpy.ndarray | torch.Tensor
        """
        _validate_push_point(point)
        if self._skip_next > 0:
            self._skip_next -= 1
            return
        self._pending.append(point)
        self._buf_len += 1
        self._emit_windows()

    def push_batch(self, points: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Append multiple points, emitting windows as they become complete.

        :param points: Points of shape ``(n, dimension)``.
        :type points: numpy.ndarray | torch.Tensor
        """
        if _is_push_batch_empty(points):
            return
        if self._skip_next > 0:
            skip_count = min(self._skip_next, points.shape[0])
            self._skip_next -= skip_count
            points = points[skip_count:]
            if points.shape[0] == 0:
                return
        self._flush_pending()
        if self._buffer is None:
            self._buffer = points
        else:
            if isinstance(points, torch.Tensor):
                self._buffer = torch.cat([self._buffer, points])
            else:
                self._buffer = np.concatenate([self._buffer, points])
        self._buf_len += points.shape[0]
        self._emit_windows()

    def _flush_pending(self):
        """Consolidate pending single-point pushes into the buffer."""
        if not self._pending:
            return
        if isinstance(self._pending[0], torch.Tensor):
            batch = torch.stack(self._pending)
        else:
            batch = np.stack(self._pending)
        if self._buffer is None:
            self._buffer = batch
        else:
            if isinstance(batch, torch.Tensor):
                self._buffer = torch.cat([self._buffer, batch])
            else:
                self._buffer = np.concatenate([self._buffer, batch])
        self._pending.clear()

    def _emit_windows(self):
        self._flush_pending()
        if self._buffer is None:
            return
        while self._buf_len - 1 >= self._next_window_end:
            w_start = self._next_window_end - self._window_size + 1
            w_end = self._next_window_end + 1
            window_path = self._buffer[w_start:w_end]
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
            self._buffer = self._buffer[trim:]
            self._buf_len -= trim
            self._next_window_end -= earliest_needed

    def sig(self) -> Union[np.ndarray, torch.Tensor]:
        """
        Return the stacked (log-)signatures of all complete windows.

        :return: Array of shape ``(num_windows, sig_length)``.
        :rtype: numpy.ndarray | torch.Tensor
        """
        if not self._windows:
            raise ValueError("No complete windows yet")
        return _stack(self._windows, self._windows[0])

    @property
    def num_windows(self) -> int:
        """Number of complete windows emitted so far."""
        return len(self._windows)


class SigWindowStream(_WindowStream):
    """
    A fixed-width sliding window over a stream of incoming points that emits
    windowed signatures. A new window is emitted every ``stride`` points once
    enough points have been accumulated to fill the window.

    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`.
    :type degree: int
    :param window_size: Number of points per window.
    :type window_size: int
    :param stride: Number of points between successive window starts. Default 1.
    :type stride: int

    Example::

        import pysiglib
        import numpy as np

        ws = pysiglib.SigWindowStream(dimension=3, degree=4, window_size=20, stride=5)

        path = np.random.randn(100, 3)
        ws.push_batch(path)

        # Signatures of all complete windows
        window_sigs = ws.sig()  # shape (num_windows, sig_length)
    """

    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1,
                 _sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        super().__init__(_sig or sig, degree, window_size, stride)


class LogSigWindowStream(_WindowStream):
    """
    A fixed-width sliding window over a stream of incoming points that emits
    windowed log-signatures. A new window is emitted every ``stride`` points once
    enough points have been accumulated to fill the window.

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

    Example::

        import pysiglib
        import numpy as np

        pysiglib.prepare_log_sig(3, 4, method=2)
        ws = pysiglib.LogSigWindowStream(dimension=3, degree=4, window_size=20, stride=5)

        path = np.random.randn(100, 3)
        ws.push_batch(path)

        # Log-signatures of all complete windows
        window_logsigs = ws.sig()  # shape (num_windows, log_sig_length)
    """

    def __init__(self, dimension: int, degree: int, window_size: int, stride: int = 1,
                 method: int = 2, _log_sig=None):
        check_type(window_size, "window_size", int)
        check_type(stride, "stride", int)
        check_pos(window_size, "window_size")
        check_pos(stride, "stride")
        sig_fn = _log_sig or (lambda path, deg: log_sig(path, deg, method=method))
        super().__init__(sig_fn, degree, window_size, stride)
