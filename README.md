<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.svg" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.svg" width="450" alt="pySigLib logo">
  </picture>
</p>

<h2 align="center">The high-performance toolkit for rough path computation</h2>

<p align="center">
  <a href="https://pysiglib.readthedocs.io">Documentation</a> |   <a href="https://pysiglib.readthedocs.io/en/stable/pages/installation.html">Installation</a> |   <a href="https://pysiglib.readthedocs.io/en/stable/pages/api_reference.html">API reference</a> |   <a href="https://arxiv.org/abs/2509.10613">Paper</a>
</p>

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/pysiglib)](https://pypi.org/project/pysiglib/) [![PyPI - Downloads](https://static.pepy.tech/badge/pysiglib/month)](https://pepy.tech/projects/pysiglib) [![Python Versions](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://pypi.org/project/pysiglib/) [![CI - Test](https://github.com/daniil-shmelev/pySigLib/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/daniil-shmelev/pySigLib/actions/workflows/unit_tests.yml) [![Read the Docs](https://img.shields.io/readthedocs/pysiglib)](https://pysiglib.readthedocs.io) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

pySigLib brings path signatures, log-signatures, branched signatures, and signature kernels into one accelerated toolkit. It provides NumPy, PyTorch, and JAX support, with automatic differentiation for PyTorch and JAX and multithreaded C++ or native CUDA execution.

## Installation

```bash
pip install pysiglib                 # NumPy API
pip install "pysiglib[torch]"        # Add PyTorch
pip install "pysiglib[jax]"          # Add JAX (Python 3.11+)
pip install "pysiglib[torch,jax]"    # Both frameworks
pip install "pysiglib[torch,cuda]"   # PyTorch with native CUDA support
```

The base installation requires neither Torch nor JAX. Extras install dependencies; select the array backend explicitly through its import:

```python
import pysiglib                       # numpy.ndarray inputs and outputs
import pysiglib.torch_api as sig_torch # torch.Tensor inputs and outputs
import pysiglib.jax_api as sig_jax     # jax.Array inputs and outputs
```

Each API accepts its own array type, including optional correction arrays. Torch provides autograd and JAX provides JIT, autodiff, and vectorization. Function names and mathematical parameters are shared, with backend-specific type annotations for editor completion and static checking. Explicit backpropagation functions in the base API operate on NumPy arrays.

**Migration:** code that passed Torch tensors to `pysiglib` should import `pysiglib.torch_api as pysiglib` and install the `[torch]` extra. Import custom static kernels and streaming classes from the same backend namespace. Code that passed NumPy arrays to `torch_api` should use the base API. Convert arrays explicitly when moving between frameworks.

The `[cuda]` extra installs the native CUDA plugin and can be combined with `[torch]` or `[jax]`. JAX GPU use also requires a GPU-capable JAX installation. For source builds and platform-specific guidance, see the [installation guide](https://pysiglib.readthedocs.io/en/stable/pages/installation.html).

## Quick start

```python
import numpy as np
import pysiglib

path = np.random.default_rng().normal(size=(32, 1000, 10))
signature = pysiglib.sig(path, degree=5)
```

Paths have shape `(path length, dimension)` or `(batch size, path length, dimension)`. Computation runs on the device where the input already lives.

## Why pySigLib?

- A unified toolkit for rough path computations - signatures, log-signatures, branched signatures, and signature kernels.
- Accelerated CPU and CUDA implementations for large workloads.
- Native NumPy, PyTorch, and JAX support without moving data between frameworks.
- Automatic differentiation with PyTorch and JAX, including `jit` and `vmap` support in JAX.
- Cross-platform - Windows, Linux and Mac systems supported.

## Capabilities

<table width="100%">
  <tr>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/signatures.html">Signatures</a></strong><br>
      <sub>Truncated signatures and individual coefficients.</sub>
    </td>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/log_signatures.html">Log-signatures</a></strong><br>
      <sub>Truncated log signatures in full or compact Lyndon coordinates.</sub>
    </td>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/signature_kernels.html">Signature kernels</a></strong><br>
      <sub>Kernels and metrics for sequential data.</sub>
    </td>
  </tr>   <tr>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/branched_signatures.html">Branched signatures</a></strong><br>
      <sub>Branched signatures, branched log signatures and branched signature kernels.</sub>
    </td>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/streams.html">Signature streams</a></strong><br>
      <sub>Online updates and constant-time interval queries.</sub>
    </td>
    <td width="33%" valign="top">
      <strong><a href="https://pysiglib.readthedocs.io/en/stable/pages/backprop.html">Backpropagation</a></strong><br>
      <sub>Manual and automatic backpropagation with PyTorch and JAX support.</sub>
    </td>
  </tr>
</table>

## Framework integrations

### PyTorch autograd

Signatures compose directly with the rest of a PyTorch model:

```python
import torch
from pysiglib.torch_api import sig

path = torch.randn(32, 1000, 10, device="cuda", requires_grad=True)
sig(path, degree=5).sum().backward()
```

### JAX transforms

The JAX API supports `jit`, `vmap`, and `grad`:

```python
import jax
import jax.numpy as jnp
from pysiglib.jax_api import sig

@jax.jit
def signature_norm(path):
    return jnp.sum(sig(path, degree=5) ** 2)

path = jax.random.normal(jax.random.key(0), (1000, 10))
gradient = jax.grad(signature_norm)(path)
```

See the [documentation](https://pysiglib.readthedocs.io) for complete examples and the full API reference.

## Citation

If the library supports your research, please consider citing the paper:

```bibtex
@article{shmelev2025pysiglib,
  title={pySigLib -- Fast Signature-Based Computations on CPU and GPU},
  author={Shmelev, Daniil and Salvi, Cristopher},
  journal={arXiv preprint arXiv:2509.10613},
  year={2025}
}
```

## Contributing

Contributions are welcome. Please open an issue first to discuss a change, then submit a pull request.

## Sponsors

If you'd like to support development, please consider [sponsoring the project](https://github.com/sponsors/daniil-shmelev).
