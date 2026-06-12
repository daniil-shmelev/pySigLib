Computing Branched Signatures
=============================

Branched signatures are a tree-indexed variant of path signatures. Instead of
recording iterated integrals indexed by words, they record coefficients indexed
by rooted trees. These coefficients are the natural coordinates for branched
rough paths, B-series, and their planar Lie-Butcher analogues.

Non-Planar and Planar Branched Signatures
------------------------------------------

``pysiglib`` currently supports two branched-signature algebras.

With ``planar=False`` (the default), the basis is made of decorated
non-planar rooted trees and the coproduct is the Butcher-Connes-Kreimer
(BCK) coproduct. This is the usual branched rough path setting over Euclidean
space.

With ``planar=True``, the basis is made of decorated ordered forests of
planar rooted trees and the coproduct is the Munthe-Kaas-Wright (MKW)
coproduct. This is the algebra used for planar branched rough paths and
Lie-Butcher series.

The following tables compare the sizes of signatures and branched signatures
with ``scalar_term=False``.

.. rst-class:: branched-size-tables

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item::

      .. list-table:: Signature
         :header-rows: 1

         * - Degree
           - d=1
           - d=2
           - d=3
         * - 1
           - 1
           - 2
           - 3
         * - 2
           - 2
           - 6
           - 12
         * - 3
           - 3
           - 14
           - 39
         * - 4
           - 4
           - 30
           - 120
         * - 5
           - 5
           - 62
           - 363

   .. grid-item::

      .. list-table:: Non-Planar Branched Signature
         :header-rows: 1

         * - Degree
           - d=1
           - d=2
           - d=3
         * - 1
           - 1
           - 2
           - 3
         * - 2
           - 2
           - 6
           - 12
         * - 3
           - 4
           - 20
           - 57
         * - 4
           - 8
           - 72
           - 303
         * - 5
           - 17
           - 286
           - 1788

   .. grid-item::

      .. list-table:: Planar Branched Signature
         :header-rows: 1

         * - Degree
           - d=1
           - d=2
           - d=3
         * - 1
           - 1
           - 2
           - 3
         * - 2
           - 3
           - 10
           - 21
         * - 3
           - 8
           - 50
           - 156
         * - 4
           - 22
           - 274
           - 1290
         * - 5
           - 64
           - 1618
           - 11496

Preparing and indexing
----------------------

Before computing branched signatures, call ``pysiglib.prepare_branched_sig``
once for each ``(dimension, degree, planar)`` combination that will be used.
The preparation step builds the tree or ordered-forest basis and the coproduct
tables used for convolution.

.. code-block:: python

    import numpy as np
    import pysiglib

    pysiglib.prepare_branched_sig(2, 3)
    pysiglib.prepare_branched_sig(2, 3, planar=True)

    path = np.array([[0.0, 0.0], [0.2, -0.3], [0.7, 0.4]])

    # BCK non-planar branched signature.
    bsig = pysiglib.branched_sig(path, 3)

    # MKW planar branched signature.
    planar_bsig = pysiglib.branched_sig(path, 3, planar=True)
