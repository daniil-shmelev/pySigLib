pysiglib.sig_backprop
========================

.. warning::

    Where possible, ``pysiglib.torch_api`` or ``pysiglib.jax_api`` should be used rather than explicitly
    calling backpropagation functions. Explicit backpropagation can introduce subtle errors if called
    incorrectly. In addition, some ``pysiglib`` functions can only be backpropagated through using their
    ``pysiglib.torch_api`` or ``pysiglib.jax_api`` variants and do not expose explicit backpropagation
    functions.

.. autofunction:: pysiglib.sig_backprop