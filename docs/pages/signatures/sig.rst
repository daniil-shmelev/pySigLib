pysiglib.sig
========================

.. autofunction:: pysiglib.sig

Choosing a method
-----------------

Use ``method="ps"`` to select prefix-suffix. By default, pySigLib chooses the
top-level split. To fix the prefix and suffix lengths for a call, pass a pair::

    s = pysiglib.sig(path, 5, method="ps", ps_split=(2, 3))
    dx = pysiglib.sig_backprop(path, s, ds, 5, method="ps", ps_split=(2, 3))

Torch and JAX reuse an explicit split for the gradient. The prefix length must
be positive, the suffix length can be zero, and their sum must equal the degree.

Use ``method="auto"`` to select between Horner and prefix-suffix separately for
forward and backward. The rule uses the size of the top signature level,
:math:`d^m`, where :math:`d` is the dimension and :math:`m` is the degree.
The minimum sizes for prefix-suffix are:

===========  ==============  ==============
Backend      Forward         Backward
===========  ==============  ==============
CPU          :math:`2^{17}`  :math:`2^{11}`
CUDA         :math:`2^{19}`  :math:`2^{16}`
===========  ==============  ==============

These thresholds apply to float32 paths without augmentation or corrections,
with 1 to 32 paths, 16 to 4096 segments per path, dimensions 2 to 32 and
degrees 3 to 16. CUDA also requires at least 512 segments across the batch.
To limit workspace, both :math:`d^m` and :math:`n d^{\lceil m/2\rceil}`
must be at most :math:`2^{20}`, where :math:`n` is the segment count per path.
Outside these ranges, ``"auto"`` uses Horner. The thresholds come from local
benchmarks and the crossover can differ on other processors.
