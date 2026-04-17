<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.png" width="350" alt="Logo">
  </picture>
</p>


<h2 align='center'>A high-performance library for path signatures and rough path methods on CPU and GPU</h2>

![PyPI - Version](https://img.shields.io/pypi/v/pysiglib)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pysiglib)
![Python Versions](https://img.shields.io/pypi/pyversions/pysiglib)
![CI - Test](https://github.com/daniil-shmelev/pySigLib/actions/workflows/unit_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/daniil-shmelev/pySigLib/graph/badge.svg?token=8W0JXOSIC7)](https://codecov.io/gh/daniil-shmelev/pySigLib)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/daniil-shmelev/pySigLib)
![Read the Docs](https://img.shields.io/readthedocs/pySigLib)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

pySigLib computes path signatures and related rough path objects on CPU and
GPU, with native PyTorch and JAX integrations so every operation is
differentiable, jittable, and runs on the device your data already lives on.

## Features

<div align="center">
<table>
  <tr>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/signatures.html"><b>Signatures</b></a></td>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/log_signatures.html"><b>Log&#8209;signatures</b></a></td>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/signature_kernels.html"><b>Signature&nbsp;kernels</b></a></td>
  </tr>
  <tr>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/branched_signatures.html"><b>Branched&nbsp;signatures</b></a></td>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/streams.html"><b>Signature&nbsp;streams</b></a></td>
    <td align="center" width="220"><a href="https://pysiglib.readthedocs.io/en/stable/pages/signature_coefficients.html"><b>Signature&nbsp;coefficients</b></a></td>
  </tr>
</table>
</div>

Every operation is available from **NumPy**, **PyTorch** (with full autograd),
and **JAX** (with `jit`, `vmap`, and `grad`), running on CPU via a
multi-threaded C++ backend or on GPU via CUDA. Additional utilities cover
[path transforms](https://pysiglib.readthedocs.io/en/stable/pages/path_transformations.html)
(time augmentation, lead-lag) and
[words / Lyndon words](https://pysiglib.readthedocs.io/en/stable/pages/words.html).

## Installation

```
pip install pysiglib              # CPU only
pip install pysiglib[cuda]        # with CUDA GPU support
```

The JAX integration is built into the wheel — install JAX separately
(`pip install jax`) if you want to use it.

For detailed and up-to-date installation instructions, including how to build
from source, see the
[installation guide](https://pysiglib.readthedocs.io/en/stable/pages/installation.html).

## Quick start

```python
import numpy as np
import pysiglib

path = np.random.randn(32, 1000, 10)  # (batch, length, dimension)
sig  = pysiglib.sig(path, degree=5)
```

## Documentation

Full documentation is available at [https://pysiglib.readthedocs.io](https://pysiglib.readthedocs.io)

## Examples

Throughout the examples below, paths are arrays of shape
`(path length, dimension)` or `(batch size, path length, dimension)`. Inputs can
be NumPy arrays, PyTorch tensors, or JAX arrays; the computation runs on
whichever device the input lives on.

### Signatures

```python
import numpy as np
import pysiglib

X = np.random.uniform(size=(32, 1000, 10))
s = pysiglib.sig(X, degree=5)
```

### Signature coefficients

```python
path  = np.random.uniform(size=(32, 1000, 5))
words = [(0,), (1, 0), (1, 2, 4)]
coefs = pysiglib.sig_coef(path, words)
```

### Log-signatures

```python
pysiglib.prepare_log_sig(dimension=10, degree=5, method=1)

X  = np.random.uniform(size=(32, 1000, 10))
ls = pysiglib.log_sig(X, degree=5, method=1)
```

### Signature kernels

```python
X = np.random.uniform(size=(32, 1000, 10))
Y = np.random.uniform(size=(32, 1000, 10))
k = pysiglib.sig_kernel(X, Y, dyadic_order=1)

# Different dyadic refinement per input when the paths have very different lengths:
X = np.random.uniform(size=(32,  100, 10))
Y = np.random.uniform(size=(32, 5000, 10))
k = pysiglib.sig_kernel(X, Y, dyadic_order=(3, 0))
```

### Branched signatures

Branched signatures index iterated integrals by decorated rooted trees instead
of words, capturing the non-geometric rough path of Ito semimartingales and
other processes where the chain rule fails.

```python
pysiglib.prepare_branched_sig(dimension=5, degree=4)

X    = np.random.randn(32, 1000, 5)
bsig = pysiglib.branched_sig(X, degree=4)
```

### PyTorch autograd

Every forward op has a backward implementation, so signatures compose cleanly
with the rest of your PyTorch model.

```python
import torch
from pysiglib.torch_api import sig

X = torch.randn(32, 1000, 10, requires_grad=True, device="cuda")
s = sig(X, degree=5)
loss = s.sum()
loss.backward()  # X.grad populated
```

### JAX

The JAX API integrates via the XLA FFI, so every op works under `jit`, `vmap`,
and `grad`.

```python
import jax
import jax.numpy as jnp
from pysiglib.jax_api import sig

@jax.jit
def signature_norm(path):
    return jnp.sum(sig(path, degree=5) ** 2)

X    = jnp.array(np.random.randn(32, 1000, 10))
grad = jax.grad(signature_norm)(X)
```

### Online signature streams

Incrementally update a signature as new points arrive, and query any interval
in O(1) via Chen's identity — useful for real-time data or sliding-window
features.

```python
stream = pysiglib.SigStream(dimension=10, degree=5)
for point in incoming_points:
    stream.push(point)

full     = stream.sig_all()      # signature of the entire path so far
interval = stream.sig(100, 200)  # signature on [t=100, t=200]
```

## Benchmarks

Every exported C++ function is benchmarked in CI via
[CodSpeed](https://codspeed.io/daniil-shmelev/pySigLib), which runs on a virtual
CPU for hardware-independent, reproducible measurements. Every PR touching
`siglib/cpsig/**` is checked for regressions against `main`.

## Citation

If you found this library useful in your research, please consider citing the paper:
```bibtex    
@article{shmelev2025pysiglib,
  title={pySigLib-Fast Signature-Based Computations on CPU and GPU},
  author={Shmelev, Daniil and Salvi, Cristopher},
  journal={arXiv preprint arXiv:2509.10613},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a pull request.

## Sponsors

If you'd like to support development, please consider
[sponsoring](https://github.com/sponsors/daniil-shmelev) the project.
