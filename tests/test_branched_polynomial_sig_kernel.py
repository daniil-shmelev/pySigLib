"""Independent quadrature, reverse-mode and API checks for polynomial BCK kernels."""
import math
import ctypes as ct
import numpy as np
import pytest
import torch

import pysiglib as ps
import pysiglib.torch_api as pt
from conftest import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype, ctype, suffix", [(torch.float32, ct.c_float, "f"), (torch.float64, ct.c_double, "d")])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_native_argument_contract(device, dtype, ctype, suffix, depth):
    """C callers receive an error, not a dereference of a required null pointer."""
    from pysiglib.load_siglib import CPSIG, CUSIG
    from pysiglib.error_codes import err_msg
    lib = CPSIG if device == "cpu" else CUSIG
    backend = "_cuda" if device == "cuda" else ""
    forward = getattr(lib, "branched_sig_kernel_poly" + backend + "_" + suffix)
    backward = getattr(lib, "branched_sig_kernel_poly_backprop" + backend + "_" + suffix)
    tensor = torch.zeros(16, dtype=dtype, device=device)
    pointer = ct.cast(tensor.data_ptr(), ct.POINTER(ctype))
    tail = [1] if device == "cpu" else []
    sizes = [1, 2, 3, 3, depth, 3, False, *tail]
    for function, pointers, required in (
        (forward, [pointer, pointer, None], (0, 1)),
        (backward, [pointer, pointer, pointer, None], (0, 1, 2)),
    ):
        for index in required:
            invalid = pointers.copy()
            invalid[index] = None
            status = function(*invalid, *sizes)
            assert status == 2
            assert "must not be null" in err_msg(status, device)
        for index, value in ((1, 0), (2, 0), (3, 0), (5, 65), (2, 2**63)):
            invalid = sizes.copy()
            invalid[index] = value
            assert function(*pointers, *invalid) == 2
        empty = sizes.copy()
        empty[0] = 0
        assert function(*([None] * len(pointers)), *empty) == 0
    # There is no Gram storage for a path consisting of a single vertex.
    assert forward(None, pointer, None, 1, 2, 1, 3, depth, 3, True, *tail) == 0
    torch.testing.assert_close(tensor[:3], torch.ones(3, device=device, dtype=dtype))
    assert backward(None, None, None, None, 1, 2, 1, 3, depth, 3, True, *tail) == 0


def reference(x, y, depth, order, grid=False):
    """Dense product integration using explicitly integrated Lagrange polynomials."""
    gram = np.diff(x, axis=0) @ np.diff(y, axis=0).T
    nx, ny = gram.shape
    q = order + 1
    z, _ = np.polynomial.legendre.leggauss(q)
    z = (z + 1) / 2
    A, w = np.empty((q, q)), np.empty(q)
    for b in range(q):
        roots = np.delete(z, b)
        poly = (np.polynomial.Polynomial.fromroots(roots) if roots.size
                else np.polynomial.Polynomial([1.0]))
        poly /= poly(z[b])
        integral = poly.integ()
        A[:, b] = integral(z) - integral(0)
        w[b] = integral(1) - integral(0)
    def cumulative(n):
        return np.kron(np.eye(n), A) + np.kron(np.tril(np.ones((n, n)), -1), np.tile(w, (q, 1)))
    Qx, Qy = cumulative(nx), cumulative(ny)
    C = np.repeat(np.repeat(gram, q, axis=0), q, axis=1)
    K = np.ones_like(C)
    result = np.ones((nx+1, ny+1), dtype=gram.dtype)
    for m in range(1, depth+1):
        if m == depth:
            cells = np.einsum('a,b,iajb->ij', w, w, (C*K).reshape(nx, q, ny, q))
            result[1:, 1:] = np.exp(cells.cumsum(0).cumsum(1))
        else:
            K = np.exp(Qx @ (C*K) @ Qy.T)
    return result if grid else result[-1, -1]


def paths(device, dtype=torch.float64, batch=()):
    rng = np.random.default_rng(214)
    return tuple(torch.tensor(.2*rng.normal(size=(*batch, n, 2)), dtype=dtype, device=device) for n in (4, 5))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("depth, order", [(0, 0), (1, 0), (2, 0), (2, 1), (4, 3), (4, 7)])
@pytest.mark.parametrize("grid", [False, True])
def test_forward_reference(device, dtype, depth, order, grid):
    x, y = paths(device, dtype)
    expected = reference(x.cpu().numpy().astype(float), y.cpu().numpy().astype(float), depth, order, grid)
    actual = ps.branched_sig_kernel(x, y, depth, method="polynomial", order=order, return_grid=grid)
    assert actual.device.type == device
    np.testing.assert_allclose(actual.cpu(), expected, rtol=2e-6 if dtype == torch.float32 else 2e-10, atol=2e-6 if dtype == torch.float32 else 2e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_saved_and_recomputed_reverse(device, dtype, grid, depth):
    x, y = paths(device, dtype)
    value, state = ps.branched_sig_kernel(x, y, depth, method="polynomial", order=3, return_grid=grid, _return_state=True)
    weights = torch.linspace(.2, .8, value.numel(), dtype=x.dtype, device=device).reshape(value.shape)
    kwargs = dict(method="polynomial", order=3, return_grid=grid, right_deriv=True)
    cached = ps.branched_sig_kernel_backprop(weights, x, y, depth, _state=state, **kwargs)
    recomputed = ps.branched_sig_kernel_backprop(weights, x, y, depth, **kwargs)
    for a, b in zip(cached, recomputed):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    xn, yn = x.cpu().numpy(), y.cpu().numpy()
    wn = weights.cpu().numpy()
    for target, actual in ((xn, cached[0]), (yn, cached[1])):
        expected = np.zeros_like(target)
        for idx in np.ndindex(target.shape):
            xc, yc = xn.astype(complex), yn.astype(complex)
            (xc if target is xn else yc)[idx] += 1e-25j
            expected[idx] = np.imag((reference(xc, yc, depth, 3, grid)*wn).sum())/1e-25
        tolerance = 3e-6 if dtype == torch.float32 else 3e-11
        np.testing.assert_allclose(actual.cpu().reshape(expected.shape), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid", [False, True])
def test_autograd_transforms_and_static_kernel(device, grid):
    x, y = (v.requires_grad_() for v in paths(device))
    opts = dict(method="polynomial", order=2, return_grid=grid, static_kernel=ps.RBFKernel(sigma=.7), time_aug=True, lead_lag=True)
    assert torch.autograd.gradcheck(lambda x, y: pt.branched_sig_kernel(x, y, 3, **opts).sum(), (x, y), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("grid", [False, True])
def test_gram_autograd_and_multidimensional_batches(device, symmetric, grid):
    x, y = (v.requires_grad_() for v in paths(device, batch=(1, 2)))
    if symmetric:
        y = x
    opts = dict(method="polynomial", order=2, return_grid=grid, max_batch=1)
    result = pt.branched_sig_kernel_gram(x, y, 3, **opts)
    expected = torch.stack([torch.stack([pt.branched_sig_kernel(a, b, 3, method="polynomial", order=2, return_grid=grid) for b in y.reshape(-1, *y.shape[-2:])]) for a in x.reshape(-1, *x.shape[-2:])]).reshape(result.shape)
    torch.testing.assert_close(result, expected)
    weights = torch.linspace(.1, .9, result.numel(), device=device, dtype=x.dtype).reshape(result.shape)
    inputs = (x,) if symmetric else (x, y)
    actual_grad = torch.autograd.grad((result*weights).sum(), inputs)
    expected_grad = torch.autograd.grad((expected*weights).sum(), inputs)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
    if not grid:
        normalized = pt.branched_sig_kernel_gram(x, y, 3, method="polynomial", order=2, normalize=True)
        assert normalized.shape == result.shape
        assert torch.isfinite(normalized).all()
        if symmetric:
            torch.testing.assert_close(normalized.reshape(2, 2).diagonal(), torch.ones(2, dtype=x.dtype, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["finite_difference", "polynomial"])
def test_mmd_scores_and_gradients(device, method):
    x, y = (v.requires_grad_() for v in paths(device, batch=(3,)))
    opts = dict(method=method, **({"order": 2} if method == "polynomial" else {"dyadic_order": 1}))
    xx = pt.branched_sig_kernel_gram(x, x, 3, **opts)
    xy = pt.branched_sig_kernel_gram(x, y, 3, **opts)
    yy = pt.branched_sig_kernel_gram(y, y, 3, **opts)
    expected = (xx.sum()-xx.trace())/6 - 2*xy.mean() + (yy.sum()-yy.trace())/6
    actual = pt.branched_sig_mmd(x, y, 3, max_batch=2, **opts)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (x, y)), torch.autograd.grad(expected, (x, y))):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    score = pt.branched_sig_score(x, y, 3, **opts)
    torch.testing.assert_close(score, (xx.sum()-xx.trace())/6 - 2*xy.mean(0))
    torch.testing.assert_close(pt.expected_branched_sig_score(x, y, 3, **opts), score.mean().reshape(1))
    numpy_result = ps.branched_sig_mmd(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 3, **opts)
    np.testing.assert_allclose(numpy_result, actual.detach().cpu(), atol=1e-11)


@pytest.mark.parametrize("device", DEVICES)
def test_high_order_closed_form_and_cache(device):
    x = torch.tensor([[0.], [1.]], dtype=torch.float64, device=device)
    exact = math.exp(sum(1/(k*math.factorial(k)) for k in range(1, 100)))
    for order in (15, 64):
        value = ps.branched_sig_kernel(x, x, 2, method="polynomial", order=order)
        assert float(value) == pytest.approx(exact, rel=3e-13)
        ps.clear_cache()
        assert float(ps.branched_sig_kernel(x, x, 2, method="polynomial", order=order)) == pytest.approx(exact, rel=3e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_and_constant_paths(device):
    x = torch.zeros((2, 1, 2), dtype=torch.float64, device=device, requires_grad=True)
    y = torch.ones((2, 3, 2), dtype=torch.float64, device=device, requires_grad=True)
    value = pt.branched_sig_kernel(x, y, 3, method="polynomial", order=2)
    torch.testing.assert_close(value, torch.ones_like(value))
    for grad in torch.autograd.grad(value.sum(), (x, y)):
        assert torch.count_nonzero(grad) == 0
    empty = ps.branched_sig_kernel(x[:0], y[:0], 3, method="polynomial", order=2)
    assert empty.shape == (0,)


def test_cpu_threading_and_noncontiguous_paths():
    x, y = (v.transpose(-1, -2).requires_grad_() for v in paths("cpu", batch=(2, 3)))
    # Transposing gives unequal feature counts; retain their common subspace.
    x, y = x[..., :3], y[..., :3]
    opts = dict(depth=3, method="polynomial", order=3, return_grid=True)
    one = pt.branched_sig_kernel(x, y, n_jobs=1, **opts)
    many = pt.branched_sig_kernel(x, y, n_jobs=2, **opts)
    torch.testing.assert_close(one, many)
    for a, b in zip(torch.autograd.grad(one.sum(), (x, y)), torch.autograd.grad(many.sum(), (x, y))):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kwargs", [{"order": -1}, {"order": 65}, {"order": True}, {"order": 2, "dyadic_order": 0}, {}])
def test_invalid_options(kwargs):
    x, y = paths("cpu")
    with pytest.raises((ValueError, TypeError)):
        ps.branched_sig_kernel(x, y, 2, method="polynomial", **kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_jax_jit_vmap_and_mmd(device):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import pysiglib.jax_api as pj
    platform = "gpu" if device == "cuda" else "cpu"
    try:
        target = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip("JAX backend is unavailable: " + platform)
    x, y = paths("cpu", batch=(3,))
    with jax.default_device(target):
        xj, yj = jnp.array(x.numpy()), jnp.array(y.numpy())
        for grid in (False, True):
            fun = lambda a, b: pj.branched_sig_kernel(a, b, 3, method="polynomial", order=2, return_grid=grid)
            value, grads = jax.jit(jax.value_and_grad(lambda a, b: fun(a, b).sum(), argnums=(0, 1)))(xj, yj)
            xt, yt = x.clone().requires_grad_(), y.clone().requires_grad_()
            expected = pt.branched_sig_kernel(xt, yt, 3, method="polynomial", order=2, return_grid=grid).sum()
            np.testing.assert_allclose(np.asarray(value), expected.detach().numpy(), atol=1e-11)
            for a, b in zip(grads, torch.autograd.grad(expected, (xt, yt))):
                np.testing.assert_allclose(np.asarray(a), b.numpy(), atol=1e-11)
            np.testing.assert_allclose(np.asarray(jax.jit(jax.vmap(fun))(xj, yj)), np.asarray(fun(xj, yj)), atol=1e-11)
        mmd_fun = lambda a: pj.branched_sig_mmd(a, yj, 3, method="polynomial", order=2, max_batch=2)
        value, gradient = jax.jit(jax.value_and_grad(mmd_fun))(xj)
        xt = x.clone().requires_grad_()
        expected = pt.branched_sig_mmd(xt, y, 3, method="polynomial", order=2)
        np.testing.assert_allclose(value, expected.detach().numpy(), atol=1e-11)
        np.testing.assert_allclose(gradient, torch.autograd.grad(expected, xt)[0].numpy(), atol=1e-11)
        normalized = jax.jit(lambda a: pj.branched_sig_kernel_gram(
            a, a, 3, method="polynomial", order=2, max_batch=2, normalize=True))(xj)
        expected = pt.branched_sig_kernel_gram(x, x, 3, method="polynomial", order=2, normalize=True)
        np.testing.assert_allclose(normalized, expected.numpy(), atol=1e-11)
