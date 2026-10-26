Default Behaviours and Conventions
=======================================

CPU and GPU Computation
------------------------

All ``pysiglib`` functions support both CPU and GPU (CUDA) computation.
The computation is performed on the same device as the input data:
pass a CPU tensor for CPU computation, or a CUDA tensor for GPU computation.

Note that CUDA is not necessarily faster than CPU for all operations. Whether
the GPU backend is beneficial depends on the specific function, the problem
parameters (batch size, path length, dimension, truncation degree, etc.), and
the hardware. We recommend benchmarking on your own setup to determine the
best device for your workload.

CPU Parallelism
----------------

For CPU-based computations, the ``n_jobs`` parameter in ``pysiglib`` functions specifies
the number of threads to run in parallel. If ``n_jobs = 1``, the computation is run serially.
This is the default behaviour.
If set to ``-1``, all available threads are used. For ``n_jobs`` below ``-1``, ``(max_threads + 1 + n_jobs)``
threads are used. For example if ``n_jobs = -2``, all threads but one are used.

Parallelising the computation by setting ``n_jobs != 1`` is beneficial when the
workload is large. However, if the workload is too small, it may be faster to set this
to ``1`` and run the computation serially, due to parallelisation overhead.

Floating Point Precision
-------------------------

Data passed to ``pysiglib`` functions should be of type ``float`` or ``double``.
The calculation will be performed in the same type as the data, so that input
data of type ``float`` will lead to faster but less accurate calculations, whilst
inputs of type ``double`` will be slower but more accurate. For most machine
learning applications, ``float``-level accuracy should be sufficient.

Input types
------------

All arrays passed to a given ``pysiglib`` function should be of the same type
and be located on the same device.
For example, for the call ``k = pysiglib.sig_kernel(X, Y, dyadic_order=1)``, if ``X`` is a
``torch`` tensor of type ``float`` located on a GPU, then ``Y`` should be the same.
This will also be the format of the output tensor, ``k``.

Batch dimensions
----------------

Paths have shape ``(..., length, dimension)``. The leading dimensions identify independent batch items and are preserved by path-to-feature operations. A single path has shape ``(length, dimension)`` and a single signature has shape ``(sig_length,)``. Empty batches are supported. Functions that combine paired inputs require matching batch shapes. Gram-matrix functions use the batch axes for all pairs; see their output-shape documentation.

Non-Contiguous Arrays
----------------------

Ideally, any array passed to ``pysiglib`` functions should be both contiguous and own its data.
If this is not the case, ``pysiglib`` will internally create a contiguous copy, which may be
inefficient in some cases. When this happens, ``pysiglib`` will issue a one-time warning.


Combinatorial coordinate labels
------------------------------

The five coordinate families are ``words``, ``lyndon_words``, ``trees``, ``forests``, and ``lyndon_forests``. Their enumerators and index helpers consistently take ``dimension`` and ``degree``. Enumerations return tuples in numerical output order, and full enumerations default to ``scalar_term=False``, matching signatures and indexing. Lyndon families always exclude the scalar. See :doc:`words` and :doc:`trees` for ordering, representations, and the distinction between full and compressed indices.

The combinatorics API intentionally changes the earlier conventions: word helpers use ``dimension`` and ``degree`` instead of ``alphabet_size`` and ``max_length``; word index helpers now require the truncation degree; word enumerations return tuples; and full enumeration requires ``scalar_term=True`` to include the empty element. Replace ``trees(..., planar=True)`` and the corresponding index calls with ``forests``, ``forest_to_idx``, and ``idx_to_forest``. Tree helpers now exclusively describe nonplanar BCK coordinates. Forest index helpers require an explicit tuple of component trees and represent the empty forest as ``()``.

When passing a whole enumeration to coefficient-selection methods, convert it to a list to distinguish the collection from a single word, tree, or forest tuple. For example, use ``sig_coef(path, list(words(dimension, degree)))``.
