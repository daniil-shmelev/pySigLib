Words
=====

Words label ordinary signature coordinates and expanded log-signature coordinates. Lyndon words label compressed log signatures. Both enumerations return tuples ordered by length and then lexicographically; ``degree`` is the inclusive maximum length and ``dimension`` is the alphabet size after any path augmentation. Increasing ``degree`` appends labels without changing existing indices.

Enumeration and indexing use the same parameters: ``words(dimension, degree)[i] == idx_to_word(i, dimension, degree)`` and likewise for Lyndon words. Full coordinates default to ``scalar_term=False``; passing ``scalar_term=True`` includes the empty word ``()`` at index zero. Lyndon coordinates always exclude the empty word. At degree zero, the result is ``()`` unless the scalar is requested, in which case the full enumeration returns ``((),)``.

Full and compressed indices are distinct. Method 1 returns the expanded coefficients selected by the Lyndon labels. Methods 2 and 3 use the standard Lyndon bracket basis indexed by the same labels; their values generally differ from method 1.

.. code-block:: python

   import pysiglib

   labels = pysiglib.lyndon_words(dimension=2, degree=3)
   word = labels[2]  # (0, 1)
   compressed_index = pysiglib.lyndon_word_to_idx(word, 2, 3)  # 2
   expanded_index = pysiglib.word_to_idx(word, 2, 3)          # 3

All index helpers validate the requested degree and reject out-of-range indices or labels. Exact-length enumeration remains available through ``words_of_length`` and ``lyndon_words_of_length``; their results are also tuples. ``words_of_length(dimension, 0)`` excludes the empty word unless ``scalar_term=True``.

.. toctree::
   :titlesonly:
   :maxdepth: 2

   /pages/words/words_of_length
   /pages/words/words
   /pages/words/lyndon_words_of_length
   /pages/words/lyndon_words
   /pages/words/is_lyndon
   /pages/words/word_to_idx
   /pages/words/idx_to_word
   /pages/words/lyndon_word_to_idx
   /pages/words/idx_to_lyndon_word
