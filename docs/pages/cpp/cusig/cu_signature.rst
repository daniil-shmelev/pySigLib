Sig functions
==============

signature_cuda
---------------

.. doxygengroup:: signature_cuda_functions
   :content-only:

sig_backprop_cuda
------------------

.. doxygengroup:: sig_backprop_cuda_functions
   :content-only:

Prefix-suffix signatures
------------------------

Pass the path, output, and signature derivative buffers in device memory.
The functions use the supplied CUDA stream. The split variants take a prefix
length ``split_a`` from 1 to ``degree``. Pass zero for automatic selection.

.. doxygengroup:: prefix_suffix_cuda_functions
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
