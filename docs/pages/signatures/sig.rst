pysiglib.sig
========================

.. autofunction:: pysiglib.sig

Optional CUDA FP64 emulation
---------------------------

Set ``fp64_emulation=True`` to allow fixed-point FP64 emulation for eligible
CUDA ``float64`` prefix-suffix matrix products. The default is ``False``.
The option is available in the base, Torch and JAX APIs. Automatic
differentiation preserves it for the backward call. Explicit
``sig_backprop`` calls accept the same option.

This requires a CUDA plugin built with cuBLAS 13.1 or later, first shipped
with CUDA 13.0 Update 2. Older builds report an error when the emulation
backend is requested. CPU, ``float32`` and Horner computations are unchanged.
Dynamic mantissa control uses the default cuBLAS precision settings;
cuBLAS can use native FP64 when emulation is unsuitable.

For an output matrix of size :math:`M \times N`, let
:math:`t = \lceil M/128 \rceil\lceil N/128 \rceil` and let :math:`S` be the
GPU multiprocessor count. Each GEMM call processes at most
:math:`\min(B, \lceil 4S/t \rceil)` paths from the remaining batch of size
:math:`B`. This is a hardware-based estimate; it does not run timing tests.
Products with any matrix dimension below 128 retain native arithmetic.

The cuBLASLt heuristic query limits the supplied emulation workspace to
512 MiB by default. Set ``PYSIGLIB_FP64_EMULATION_WORKSPACE_BYTES`` to change
this limit in bytes. The existing ``PYSIGLIB_PS_WORKSPACE_BYTES`` limit also
reserves space for this buffer. The implementation reduces the group size
if no algorithm fits, and uses native FP64 if a single product cannot fit.
One buffer is reused across groups and calls and is released by
``clear_cache``. Inputs, outputs and other CUDA library allocations are
separate from this workspace limit.

See NVIDIA's `FP64 emulation documentation
<https://docs.nvidia.com/cuda/cublas/index.html#fixed-point>`_ for the precision
and hardware requirements.
