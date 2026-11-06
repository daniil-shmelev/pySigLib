Sig functions
==============

For both passes, set ``method`` to 0 for the standard algorithm, 1 for
prefix-suffix, or 2 for automatic selection. With method 1, set ``split_a``
to the desired prefix length from 1 to ``degree``, or pass zero to select
the split automatically. For all other methods, pass zero for ``split_a``.
The CUDA functions accept a stream pointer, which defaults to the default stream.

signature_cuda
---------------

.. doxygengroup:: signature_cuda_functions
   :content-only:

sig_backprop_cuda
------------------

With explicit prefix-suffix (``method=1``), the backward pass recomputes its
states from the path. The caller can pass ``nullptr`` for ``sig`` in this case.

.. doxygengroup:: sig_backprop_cuda_functions
   :content-only:

linear_sig_cuda
----------------

.. doxygengroup:: linear_sig_cuda_functions
   :content-only:

sig_join_cuda
--------------

.. doxygengroup:: sig_join_cuda_functions
   :content-only:

sig_join_backprop_cuda
-----------------------

.. doxygengroup:: sig_join_backprop_cuda_functions
   :content-only:
