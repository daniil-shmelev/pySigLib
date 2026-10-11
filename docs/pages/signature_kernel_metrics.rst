Signature Kernel Metrics
========================

Scoring rules and maximum mean discrepancy (MMD) computed with signature kernels.

The branched variants use the depth-truncated BCK kernel and accept both finite-difference and polynomial solvers. As with the geometric metrics, use ``pysiglib.torch_api`` or ``pysiglib.jax_api`` when differentiating these quantities.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   /pages/signature_kernels/sig_score
   /pages/signature_kernels/expected_sig_score
   /pages/signature_kernels/sig_mmd
   /pages/signature_kernels/branched_sig_score
   /pages/signature_kernels/expected_branched_sig_score
   /pages/signature_kernels/branched_sig_mmd
