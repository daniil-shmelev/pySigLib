pysiglib.sig_coef
==========================

.. versionadded:: v1.1.0

.. autofunction:: pysiglib.sig_coef

Use ``method="pascal"`` to compute coefficients by Pascal factorisation.
For explicit backpropagation, request every prefix and supply a derivative
for each output entry. For example, this differentiates just the last
coefficient of a single word on a NumPy path::

    word = (0, 1, 0)
    coefs = pysiglib.sig_coef(path, word, method="pascal", prefixes=True)
    seeds = np.zeros_like(coefs)
    seeds[..., -1] = 1
    dx = pysiglib.sig_coef_backprop(path, word, coefs, seeds, method="pascal")

Torch and JAX supply the prefix derivatives automatically. The empty word
contributes one output entry with value one and has zero path derivative.

For coefficients, ``method="auto"`` selects Horner on CUDA. On CPU it selects
Pascal for calibrated single-thread float32 workloads with equal word lengths
from 8 to 16. Other CPUs and workloads use Horner. Both methods support
float32 and float64. CUDA supports word lengths through 20.
