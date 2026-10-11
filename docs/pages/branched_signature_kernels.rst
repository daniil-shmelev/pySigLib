Branched Signature Kernels
========================================

This section covers the BCK branched signature kernel and its Gram matrices.

Both CPU and CUDA support ``method="finite_difference"`` with ``dyadic_order`` and ``method="polynomial"`` with ``order``. The polynomial method applies Gauss–Legendre product integration to each level of the branched depth recursion on the original path segments. An order of ``p`` uses ``p + 1`` nodes per coordinate; supported orders are 0 through 64. The independent ``depth`` parameter continues to control rooted-tree depth.

The NumPy/PyTorch interface is available in ``pysiglib``, with differentiable versions in ``pysiglib.torch_api`` and ``pysiglib.jax_api``. CUDA tensors use the native CUDA solver. Kernel values, original-vertex grids, Gram matrices, normalized kernels, scores and MMD all accept the polynomial method.

For example, compute a differentiable squared MMD between two batches of paths:

.. code-block:: python

   import torch
   import pysiglib.torch_api as ps

   x = (0.1 * torch.randn(8, 33, 3, device="cuda", dtype=torch.float64)).requires_grad_()
   y = 0.1 * torch.randn(8, 33, 3, device="cuda", dtype=torch.float64)
   loss = ps.branched_sig_mmd(x, y, depth=3, method="polynomial", order=3)
   loss.backward()

Use ``device="cpu"`` for the CPU solver. ``order=3`` means four quadrature nodes per coordinate. Increase the order and check convergence for the scale and depth of the paths being used.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   /pages/signature_kernels/branched_sig_kernel
   /pages/signature_kernels/branched_sig_kernel_gram
   /pages/signature_kernels/branched_sig_score
   /pages/signature_kernels/expected_branched_sig_score
   /pages/signature_kernels/branched_sig_mmd
