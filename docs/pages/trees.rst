Trees and forests
=================

These helpers enumerate and index the coordinate families used by branched signatures and log signatures. Every enumeration returns a tuple in numerical output order. ``dimension`` is the number of labels after any path augmentation, and ``degree`` is the inclusive maximum total number of nodes. Increasing ``degree`` appends labels without changing existing indices.

.. list-table:: Coordinate families
   :header-rows: 1

   * - Enumeration
     - Index helpers
     - Numerical coordinates
   * - ``trees``
     - ``tree_to_idx``, ``idx_to_tree``
     - Nonplanar BCK signatures and log signatures
   * - ``forests``
     - ``forest_to_idx``, ``idx_to_forest``
     - Planar MKW signatures and expanded log signatures (method 0)
   * - ``lyndon_forests``
     - ``lyndon_forest_to_idx``, ``idx_to_lyndon_forest``
     - Compressed MKW log signatures (methods 1, 2, 3)

A decorated tree is ``(child_1, ..., child_k, root_label)``; a leaf is ``(label,)``. In BCK trees, children are sorted by native tree ID. A forest is a tuple of planar trees in their specified order, including an explicit singleton tuple for a one-tree forest. Tree IDs follow node count, child-ID sequence, and then root label. Forests follow total node count and then lexicographic order of tree-ID sequences. Comparing nested Python tree tuples does not implement this order.

The full enumerators and index helpers default to ``scalar_term=False``. With ``scalar_term=True``, index zero is ``None`` for BCK trees and ``()`` for MKW forests. Thus ``forests(dimension, degree, scalar_term=s)[i] == idx_to_forest(i, dimension, degree, scalar_term=s)``, with the analogous identity for trees. Degree zero returns an empty enumeration, or just the empty element when the scalar is requested. ``trees_of_order`` retains exact-order BCK enumeration with the same scalar convention.

For Lyndon forests, each planar tree is a letter whose weight is its node count. The Lyndon condition compares native tree IDs; degree bounds the sum of their weights. Every singleton forest is Lyndon, including those whose tree has more than one node. The empty forest is always excluded.

.. code-block:: python

   import pysiglib

   leaf = (0,)
   chain = ((0,), 0)
   assert pysiglib.lyndon_forests(1, 2) == ((leaf,), (chain,))
   assert (leaf, chain) in pysiglib.lyndon_forests(1, 3)
   assert pysiglib.forest_to_idx((chain,), 1, 2) == 2
   assert pysiglib.lyndon_forest_to_idx((chain,), 1, 2) == 1

Full and compressed indices are distinct. Method 1 returns the expanded forest coefficients selected by Lyndon labels. Methods 2 and 3 return coefficients in the standard Lyndon bracket basis with the same labels; the values generally differ from method 1. Index helpers reject labels outside the requested family or degree and indices outside the enumeration.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   /pages/trees/trees
   /pages/trees/trees_of_order
   /pages/trees/tree_to_idx
   /pages/trees/idx_to_tree
   /pages/trees/forests
   /pages/trees/forest_to_idx
   /pages/trees/idx_to_forest
   /pages/trees/lyndon_forests
   /pages/trees/lyndon_forest_to_idx
   /pages/trees/idx_to_lyndon_forest
