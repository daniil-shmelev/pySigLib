Backpropagation
========================

.. warning::

    Where possible, ``pysiglib.torch_api`` should be used rather than explicitly calling
    backpropagation functions. Explicit backpropagation can introduce subtle errors if called
    incorrectly. In addition, some ``pysiglib`` functions can only be backpropagated through
    using their ``pysiglib.torch_api`` variants and do not expose explicit backpropagation functions.

.. toctree::
   :titlesonly:

   backprop/transform_path_backprop
   backprop/sig_backprop
   backprop/sig_combine_backprop
   backprop/sig_coef_backprop
   backprop/sig_to_log_sig_backprop
   backprop/log_sig_combine_backprop
   backprop/sig_kernel_backprop
   backprop/sig_kernel_gram_backprop
   backprop/branched_sig_backprop
   backprop/branched_sig_combine_backprop
