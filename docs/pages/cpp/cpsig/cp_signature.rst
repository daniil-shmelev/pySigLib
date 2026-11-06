Sig functions
==============

signature
----------

.. doxygengroup:: signature_functions
   :content-only:

sig_backprop
-------------

.. doxygengroup:: sig_backprop_functions
   :content-only:

Prefix-suffix signatures
------------------------

The forward functions write the truncated signature to ``out``. The backward
functions take signature derivatives in ``deriv`` and write path derivatives
to ``out``. The split variants take the optional choice as a prefix length
``split_a`` from 1 to ``degree``. Pass zero for automatic selection.

.. doxygengroup:: prefix_suffix_functions
   :content-only:

linear_sig
-----------

.. doxygengroup:: linear_sig_functions
   :content-only:

sig_join
---------

.. doxygengroup:: sig_join_functions
   :content-only:

sig_join_backprop
------------------

.. doxygengroup:: sig_join_backprop_functions
   :content-only:
