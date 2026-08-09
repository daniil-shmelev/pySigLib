Computing Log Signatures
===========================

For :math:`x \in T(\mathbb{R}^d)`, the logarithm in tensor space is defined by

.. math::

    \log(1 + x) = \sum_{n \geq 1} \frac{(-1)^{n-1} x^n}{n}.

Applying this logarithm map to the signature yields the `log signature`. A
useful property of the log signature is its ability to be `compressed` into
a smaller tensor without losing information, by considering its values at
`Lyndon words`. ``pysiglib`` implements four methods for
log signature computation, controlled by the ``method`` flag, described in
detail below.

Preparing for Log Signature Computations
------------------------------------------

Before computing the log signature, ``pysiglib`` requires a call to
``pysiglib.prepare_log_sig``, which will pre-compute and cache certain
objects which are required for the computation. This function should be
run only once before the computation, for each required ``(dimension, degree, method)``
combination. This call is not thread safe.

.. code-block:: python

    import torch
    import pysiglib

    # We know in advance that we will need log signatures
    # for dimensions 5 and 10 with degree 3 and method 2.
    pysiglib.prepare_log_sig(5, 3, method=2)
    pysiglib.prepare_log_sig(10, 3, method=2)

    for i in range(10):
        X = torch.rand((200, 5))
        Y = torch.rand((100, 10))

        X_ls = pysiglib.log_sig(X, 3, method=2)
        Y_ls = pysiglib.log_sig(Y, 3, method=2)

Preparing for ``method=2`` is also sufficient to run ``method=1`` and
prepares the BCH cache used by log-signature combination. Preparing for
``method=3`` prepares the BCH cache used by the direct method.
Only ``method=0`` does not require a call to
``pysiglib.prepare_log_sig``.

.. code-block:: python

    import torch
    import pysiglib

    X = torch.rand((100, 5))

    pysiglib.log_sig(X, 3, method=0) # No error: method=0 does not require preparation

    pysiglib.prepare_log_sig(5, 3, method=2)
    pysiglib.log_sig(X, 3, method=1) # No error: prepare already called with a higher method

    pysiglib.prepare_log_sig(5, 3, method=3)
    pysiglib.log_sig(X, 3, method=3) # No error: BCH cache is prepared

Methods
--------

We give a brief overview of the methods below. For details concerning the Lyndon
basis, we refer the user to the documentation for the ``iisignature`` and
``signatory`` packages `here <https://github.com/bottler/phd-docs/blob/master/iisignature.pdf>`__
and `here <https://signatory.readthedocs.io/en/latest/>`__, as well as their corresponding papers
`here <https://arxiv.org/pdf/1802.08252>`__ and `here <https://arxiv.org/pdf/2001.00706>`__.

``method = 0``
-------------------

This option corresponds to the full uncompressed log signature, obtained by first
computing the signature and then applying the tensor logarithm. The output is the
same length as a signature.

This method corresponds to ``methods="x"`` in the ``iisignature`` package and
``mode="expand"`` in the ``signatory`` package.

``method = 1``
-------------------

This option computes the log signature as in the ``method=0`` case, and then extracts
those coefficients which are indexed by `Lyndon words`. Whilst this does not represent 
the log signature using a canonical Hall basis of the free Lie algebra, it is 
equivalent up to a linear transformation. This makes it a faster alternative to 
``method=2``, suitable for machine learning applications where the missing linear 
transformation can be learnt implicitly.

This method corresponds to ``mode="words"`` in the ``signatory`` package.

``method = 2``
-------------------

This option computes the log signature as in the ``method=0`` case and projects the 
result to the `Lyndon basis`, yielding the canonical compressed log-signature. This 
method is required whenever canonical coordinates are important, such as for 
interpretability, applying the Log-ODE method to solve a controlled differential 
equation, or consistent comparison across implementations.

This method corresponds to ``methods="s"`` in the ``iisignature`` package and
``mode="brackets"`` in the ``signatory`` package.

``method = 3``
-------------------

This option computes the log signature directly from the path using the
Baker-Campbell-Hausdorff (BCH) formula, without ever computing the full signature.
Each linear segment of the path has a trivial log signature (just the increment),
and these are sequentially combined using

.. math::

    L(x_1 \ast x_2) = \mathrm{BCH}(L(x_1), L(x_2)).

The output is in the Lyndon basis (same format as ``method=2``).

The advantage of this method is **memory**: it never allocates the full signature tensor,
whose size grows exponentially with degree (:math:`\sum_{k=1}^{N} d^k`). This makes it
the only viable option when the signature would be too large to fit in memory (e.g. high
degree or high dimension). For typical dimensions and degrees, ``method=2`` is faster.

This method requires a call to ``pysiglib.prepare_log_sig`` with
``method=3`` for each dimension and degree pair.

This method corresponds to ``methods="o"`` or ``methods="c"`` in the ``iisignature``
package. There is no equivalent in the ``signatory`` package.
