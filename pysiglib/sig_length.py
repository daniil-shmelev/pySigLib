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

from .load_siglib import CPSIG
from .param_checks import check_type, check_non_neg, check_pos


def aug_dim(dimension: int, time_aug: bool, lead_lag: bool) -> int:
    """Effective dimension after applying lead-lag and/or time augmentation."""
    return (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)


def sig_length(
        dimension : int,
        degree : int,
        *,
        time_aug : bool = False,
        lead_lag : bool = False,
        scalar_term : bool = False,
) -> int:
    """
    Returns the length of a truncated signature,

    .. math::

        \\sum_{i=0}^N d^i = \\frac{d^{N+1} - 1}{d - 1},

    where :math:`d` is the dimension of the underlying path and :math:`N`
    is the truncation level of the signature.

    :param dimension: Dimension of the underlying path, :math:`d`
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation is applied before computing
        the signature. This flag is provided for convenience, and is equivalent
        to calling ``sig_length(dimension + 1, degree)``.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation is applied before computing
        the signature. This flag is provided for convenience, and is equivalent
        to calling ``sig_length(2 * dimension, degree)``.
    :type lead_lag: bool
    :param scalar_term: If True, the returned length includes the leading
        constant-term entry at index 0. If False (default), the length is one less. Must match
        the ``scalar_term`` value used with :func:`sig`.
    :type scalar_term: bool
    :return: Length of a truncated signature
    :rtype: int

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # Length of a truncated signature for a 3-dimensional path at degree 4
        length = pysiglib.sig_length(3, 4, scalar_term=True)
        print(length) # 121 (= 1 + 3 + 9 + 27 + 81)

    .. code-block:: python

        # Signature length with time augmentation and lead-lag
        import pysiglib

        # lead_lag doubles the dimension (6), time_aug adds one (7)
        length = pysiglib.sig_length(3, 4, time_aug=True, lead_lag=True, scalar_term=True)
        print(length)  # 2801

    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    out = CPSIG.sig_length(aug_dimension, degree)
    if out == 0:
        raise ValueError("Integer overflow encountered in sig_length")
    return out - (0 if scalar_term else 1)

def log_sig_length(
        dimension : int,
        degree : int,
        *,
        time_aug: bool = False,
        lead_lag: bool = False
) -> int:
    """
    Returns the length of a truncated log signature,

    .. math::

        \\sum_{i=0}^N \\frac{1}{i} \\sum_{x | i} \\mu\\left(\\frac{i}{x}\\right) d^x,

    where :math:`d` is the dimension of the underlying path, :math:`N`
    is the truncation level of the log signature and :math:`\\mu` is
    the Mobius function.

    :param dimension: Dimension of the underlying path, :math:`d`
    :type dimension: int
    :param degree: Truncation level of the log signature, :math:`N`
    :type degree: int
    :param time_aug: Whether time augmentation is applied before computing
        the signature. This flag is provided for convenience, and is equivalent
        to calling ``sig_length(dimension + 1, degree)``.
    :type time_aug: bool
    :param lead_lag: Whether the lead lag transformation is applied before computing
        the signature. This flag is provided for convenience, and is equivalent
        to calling ``sig_length(2 * dimension, degree)``.
    :type lead_lag: bool
    :return: Length of a truncated log signature
    :rtype: int

    Example:
    ---------

    .. code-block:: python

        import pysiglib

        # Length of a truncated log signature for a 3-dimensional path at degree 4
        length = pysiglib.log_sig_length(3, 4)
        print(length) # 32

    .. code-block:: python

        # Log signature length with time augmentation and lead-lag
        import pysiglib

        length = pysiglib.log_sig_length(3, 4, time_aug=True, lead_lag=True)
        print(length) # 728

    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_pos(dimension, "dimension")
    check_pos(degree, "degree")
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    aug_dimension = aug_dim(dimension, time_aug, lead_lag)

    out = CPSIG.log_sig_length(aug_dimension, degree)
    if out == 0:
        raise ValueError("Integer overflow encountered in sig_length")
    return out


def _infer_scalar_term(sig, dimension: int, degree: int, time_aug: bool = False, lead_lag: bool = False) -> bool:
    """Return True iff ``sig``'s trailing dimension includes the leading scalar 1.

    Raises ``ValueError`` if the shape matches neither the scalar_term=True nor
    the scalar_term=False signature length for the given ``(dimension, degree,
    time_aug, lead_lag)``. Used by consumer-side functions that accept sigs in
    either format and match their output format to the input.
    """
    full_len = sig_length(dimension, degree, time_aug=time_aug, lead_lag=lead_lag, scalar_term=True)
    actual = sig.shape[-1]
    if actual == full_len:
        return True
    if actual == full_len - 1:
        return False
    raise ValueError(
        "sig has incompatible length " + str(actual) + " for dimension=" + str(dimension) +
        ", degree=" + str(degree) + " (expected " + str(full_len) + " or " + str(full_len - 1) + ")."
    )
