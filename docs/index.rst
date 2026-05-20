.. pysiglib documentation master file, created by
   sphinx-quickstart on Mon Jun  2 17:57:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div style="display: none;">

pySigLib |release|
======================================

.. raw:: html

   </div>

.. meta::
   :description: pySigLib documentation for fast path signatures, log signatures, and signature kernels on CPU and CUDA GPU.

.. raw:: html

   <div style="text-align: center;">
     <img src="_static/logo_light.png" width="400" class="only-light" style="pointer-events: none;" />
     <img src="_static/logo_dark.png" width="400" class="only-dark" style="pointer-events: none;" />
   </div>

.. rst-class:: landing-tagline

Fast path signatures, log signatures, and signature kernels on CPU and CUDA GPU.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`download;1.5em` Installation
      :link: /pages/installation
      :link-type: doc

      Get up and running with pySigLib on Windows, Linux, or macOS, with optional CUDA support.

   .. grid-item-card:: :octicon:`book;1.5em` Conventions
      :link: /pages/conventions
      :link-type: doc

      Default behaviours, CPU/GPU conventions, parallelism, and data format expectations.

   .. grid-item-card:: :octicon:`code;1.5em` API Reference
      :link: /pages/api_reference
      :link-type: doc

      Path signatures, log signatures, signature kernels, and more.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`flame;1.5em` PyTorch API
      :link: /pages/torch_api
      :link-type: doc

      Use pySigLib functions as native PyTorch autograd functions with full gradient support.

   .. grid-item-card:: :octicon:`zap;1.5em` JAX API
      :link: /pages/jax_api
      :link-type: doc

      Use pySigLib functions with JAX transformations: ``jit``, ``grad``, and ``vmap``.

   .. grid-item-card:: :octicon:`terminal;1.5em` C++ Library
      :link: /pages/cpp
      :link-type: doc

      Direct access to the underlying C++/CUDA siglib library.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   /pages/installation
   /pages/conventions

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   /pages/path_transformations
   /pages/words
   /pages/signatures
   /pages/signature_coefficients
   /pages/log_signatures
   /pages/branched_signatures
   /pages/branched_log_signatures
   /pages/signature_kernels
   /pages/streams
   /pages/backprop

.. toctree::
   :maxdepth: 2
   :caption: Interfaces
   :hidden:

   /pages/torch_api
   /pages/jax_api
   /pages/cpp
