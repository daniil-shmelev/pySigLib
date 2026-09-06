Torch API
========================

.. meta::
   :description: Use pySigLib path signatures, log signatures, and signature kernels with differentiable PyTorch tensors.

.. versionadded:: v0.2

Install ``pysiglib[torch]`` to use this API. Array inputs and outputs are ``torch.Tensor``, and operations participate in Torch autograd. The base ``pysiglib`` API accepts NumPy arrays; select Torch explicitly:

.. code-block:: python

    import pysiglib.torch_api as pysiglib

