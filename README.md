<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo_light.png" width="350" alt="Logo">
  </picture>
</p>


<h2 align='center'>Fast Signature, Log Signature and Signature Kernel Computations on CPU and GPU</h2>

![PyPI - Version](https://img.shields.io/pypi/v/pysiglib)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pysiglib)
![CI - Test](https://github.com/daniil-shmelev/pySigLib/actions/workflows/unit_tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/daniil-shmelev/pySigLib/graph/badge.svg?token=8W0JXOSIC7)](https://codecov.io/gh/daniil-shmelev/pySigLib)
![Read the Docs](https://img.shields.io/readthedocs/pySigLib)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Installation

Before installing, please ensure you have a compatible C++ compiler installed
(MSVC for Windows, GCC for Linux, or Xcode Command Line Tools for macOS), then run

```
pip install pysiglib
```

pySigLib will automatically detect CUDA, provided the `CUDA_PATH` environment variable is set correctly.
To manually disable CUDA and build pySigLib for CPU only, set the `CUSIG` environment variable to `0`:

```
CUSIG=0 pip install pysiglib
```

For detailed and up-to-date installation instructions on Windows, Linux and MacOS, see the
[installation guide](https://pysiglib.readthedocs.io/en/latest/pages/installation.html).

## Documentation

Full documentation is available at [https://pysiglib.readthedocs.io](https://pysiglib.readthedocs.io)

## Examples

### Signatures

pySigLib implements truncated signature transforms through the function `pysiglib.sig`,
which takes as input a path or batch of paths `X`, and a truncation `degree`.</p>

`X` can be a numpy array or a torch tensor. For a single path, `X` must be of shape
`(path length, path dimension)`. For a batch of paths, it must be of shape
`(batch size, path length, path dimension)`.</p>

The computation will run on whichever device `X` is on. For example, passing
`X = np.random.uniform(size=(32, 1000, 10))` will trigger the computation to run
on the CPU, whilst `X = torch.rand(32, 1000, 10, device = "cuda")` will run
on the CUDA device.

```python
import numpy as np
import pysiglib

X = np.random.uniform(size=(32, 1000, 10))
s = pysiglib.sig(X, degree = 5)
```

### Signature Coefficients

In addition to truncated signatures, pySigLib allows for the computation of individual
signature coefficients through the `pysiglib.sig_coef` function, which takes as input
a path or batch of paths `X` and a word or list of words indexing the coefficients.</p>

```python
import numpy as np
import pysiglib

path = np.random.uniform(size=(32, 1000, 5))
words = [(0,), (1,0), (1,2,4)]
coefs = pysiglib.sig_coef(path, words)
```

### Log Signatures

Similarly to signatures, pySigLib implements truncated log signature transforms through the
function `pysiglib.log_sig`. The function takes a `method` parameter, which determines
the method used to compute the log signature and the form of the output. For details,
please see the documentation.

```python
import numpy as np
import pysiglib
pysiglib.prepare_log_sig(dimension = 10, degree = 5, method = 1)

X = np.random.uniform(size=(32, 1000, 10))
ls = pysiglib.log_sig(X, degree = 5, method = 1)
```

### Signature Kernels

pySigLib implements signature kernels through the function `pysiglib.sig_kernel`,
which takes as input a pair of paths or a pair of batches of paths `X, Y`. The
`dyadic_order` parameter can be used to refine the PDE grid, giving more
accurate results. If specified as an integer, the same refinement factor
is applied to both `X` and `Y`. To apply different factors to the two paths,
`dyadic_order` can be specified as a tuple.</p>

As with signatures, `X,Y` can be numpy arrays or torch tensors, and must have the
same shapes as described above. Again, the computation will run on whichever
device `X,Y` are located on.

```python
import numpy as np
import pysiglib

X = np.random.uniform(size=(32, 1000, 10))
Y = np.random.uniform(size=(32, 1000, 10))
k = pysiglib.sig_kernel(X, Y, dyadic_order=1)

# In cases where the paths differ in length, it may
# be advantageous to refine the PDE grid by different
# amounts for X and Y:

X = np.random.uniform(size=(32, 100, 10))
Y = np.random.uniform(size=(32, 5000, 10))
k = pysiglib.sig_kernel(X, Y, dyadic_order=(3, 0))
```

## Contribution

Pull requests are not being accepted at this time, but issues and suggestions are welcome.

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
