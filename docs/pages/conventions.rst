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
For example, for the call ``k = pysiglib.sig_kernel(X, Y, 1)``, if ``X`` is a
``torch`` tensor of type ``float`` located on a GPU, then ``Y`` should be the same.
This will also be the format of the output tensor, ``k``.

Non-Contiguous Arrays
----------------------

Ideally, any array passed to ``pysiglib`` functions should be both contiguous and own its data.
If this is not the case, ``pysiglib`` will internally create a contiguous copy, which may be
inefficient in some cases. When this happens, ``pysiglib`` will issue a one-time warning.

