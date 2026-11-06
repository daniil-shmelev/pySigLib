C++ Documentation
========================

Brief documentation for the ``cpSIG`` and ``cuSIG`` C++ libraries. We omit details
of the mathematical operations, for which we refer the user to the corresponding python documentation.

The headers expose a C ABI through ``extern "C"`` functions. Functions with suffix ``_f`` use ``float``; functions with suffix ``_d`` use ``double``. The caller must allocate contiguous, row-major buffers on the device required by the backend. Status-returning functions return zero on success. Use ``cpsig_last_error_message()`` or ``cusig_last_error_message()`` to read the error from the last failed call on the current thread.

.. toctree::
   :titlesonly:

   cpp/cpsig
   cpp/cusig
