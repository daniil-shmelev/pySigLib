"""Specialize shared mathematical documentation for a public backend."""

import re


def backend_doc(doc, backend):
    if doc is None:
        return None
    array = {"numpy": "numpy.ndarray", "torch": "torch.Tensor", "jax": "jax.Array"}[
        backend
    ]
    doc = doc.replace("numpy.ndarray | torch.tensor", array)
    doc = doc.replace("numpy.ndarray | torch.Tensor", array).replace(
        "numpy.ndarray | Array", array
    )
    doc = re.sub(r"\bArrayT?\b", array, doc)
    doc = doc.replace("`numpy.ndarray` or `torch.Tensor`", f"`{array}`")
    doc = doc.replace(
        "numpy arrays, torch tensors (with autograd via ``pysiglib.torch_api``),\n    and JAX arrays (via ``pysiglib.jax_api``)",
        f"{array} arrays",
    )
    if backend != "numpy":
        # NumPy examples are documented on the base API; each framework has its own guide.
        doc = re.split(
            r"\n[ \t]*(?:Example usage|Example:|Examples?:)", doc, maxsplit=1
        )[0]
    return doc
