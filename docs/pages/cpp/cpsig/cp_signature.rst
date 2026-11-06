Sig functions
==============

For both passes, set ``method`` to 0 for the standard algorithm, 1 for
prefix-suffix, or 2 for automatic selection. With method 1, set ``split_a``
to the desired prefix length from 1 to ``degree``, or pass zero to select
the split automatically. For all other methods, pass zero for ``split_a``.

signature
----------

.. doxygengroup:: signature_functions
   :content-only:

sig_backprop
-------------

With explicit prefix-suffix (``method=1``), the backward pass recomputes its
states from the path. The caller can pass ``nullptr`` for ``sig`` in this case.

.. doxygengroup:: sig_backprop_functions
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
