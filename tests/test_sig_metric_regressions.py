"""Regression checks shared by standard and branched kernel metrics."""
import numpy as np
import pytest
import torch

import pysiglib as ps
import pysiglib.torch_api as pt
from conftest import DEVICES


@pytest.fixture(params=[("numpy", "cpu"), ("jax", "cpu")] +
                [(api, device) for api in ("base_torch", "torch") for device in DEVICES],
                ids=lambda item: "-".join(item))
def metric_api(request):
    name, device = request.param
    if name == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import pysiglib.jax_api as pj
        return pj, jnp.asarray, name
    if name == "numpy":
        return ps, np.asarray, name
    api = ps if name == "base_torch" else pt
    return api, lambda value: torch.as_tensor(value, device=device), name


@pytest.mark.parametrize("branched", [False, True])
@pytest.mark.parametrize("method", ["finite_difference", "polynomial"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_metrics_exclude_large_diagonals(metric_api, branched, method, dtype):
    api, asarray, _ = metric_api
    amplitude = 2.3 if branched else 30.0
    x = np.array([[[0., 0.], [amplitude, 0.]],
                  [[0., 0.], [0., amplitude]]], dtype=dtype)
    y = np.zeros_like(x)
    x, y = asarray(x), asarray(y)
    prefix = "branched_" if branched else ""
    options = {"method": method}
    if method == "polynomial":
        options["order"] = 15 if branched else (12 if dtype == np.float32 else 64)
    else:
        options["dyadic_order"] = 3 if branched else 4
    if branched:
        options["depth"] = 2

    xx = getattr(api, prefix + "sig_kernel_gram")(x, x, **options)
    if isinstance(xx, torch.Tensor):
        xx = xx.detach().cpu().numpy()
    xx = np.asarray(xx)
    assert np.isfinite(xx).all()
    assert (np.diag(xx) > 1e16).all()
    np.testing.assert_allclose(xx[[0, 1], [1, 0]], 1.)

    for metric, expected in ((prefix + "sig_score", np.array([-1., -1.])),
                             ("expected_" + prefix + "sig_score", np.array([-1.])),
                             (prefix + "sig_mmd", np.array(0.))):
        for left, right in ((x, y), (y, x)):
            result = getattr(api, metric)(left, right, **options)
            if isinstance(result, torch.Tensor):
                assert result.device == left.device
                result = result.detach().cpu().numpy()
            result = np.asarray(result)
            assert result.dtype == dtype
            assert result.shape == expected.shape
            np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("branched", [False, True])
@pytest.mark.parametrize("method", ["finite_difference", "polynomial"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("time_aug, lead_lag", [(False, False), (True, False),
                                               (False, True), (True, True)])
def test_metrics_accept_reversed_numpy_views(branched, method, dtype, time_aug, lead_lag):
    rng = np.random.default_rng(12)
    x = rng.normal(scale=.1, size=(2, 2, 3, 2)).astype(dtype)[..., ::-1, :]
    y = rng.normal(scale=.1, size=(2, 1, 4, 2)).astype(dtype)[..., ::-1, ::-1]
    x.setflags(write=False)
    y.setflags(write=False)
    x_copy, y_copy = x.copy(), y.copy()
    prefix = "branched_" if branched else ""
    options = dict(method=method, time_aug=time_aug, lead_lag=lead_lag, max_batch=2)
    options["order" if method == "polynomial" else "dyadic_order"] = 2
    if branched:
        options["depth"] = 2
    for metric in (prefix + "sig_score", "expected_" + prefix + "sig_score",
                   prefix + "sig_mmd"):
        function = getattr(ps, metric)
        expected = function(x_copy, y_copy, **options)
        for api in ([ps, pt] if branched else [ps]):
            actual = getattr(api, metric)(x, y, **options)
            if api is pt:
                assert isinstance(actual, torch.Tensor)
                actual = actual.detach().numpy()
            assert isinstance(actual, np.ndarray)
            assert actual.dtype == dtype
            assert actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(x, x_copy)
    np.testing.assert_array_equal(y, y_copy)


@pytest.mark.parametrize("metric_api", [("jax", "cpu")] +
                         [("torch", device) for device in DEVICES], indirect=True)
@pytest.mark.parametrize("metric", ["sig_score", "expected_sig_score", "sig_mmd"])
@pytest.mark.parametrize("branched", [False, True])
@pytest.mark.parametrize("method", ["finite_difference", "polynomial"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_metric_gradients_exclude_large_diagonals(metric_api, metric, branched, method, dtype):
    api, asarray, name = metric_api
    amplitude = 2.3 if branched else 30.0
    x = np.array([[[0., 0.], [amplitude, 0.]],
                  [[0., 0.], [0., amplitude]]], dtype=dtype)
    expected = np.array([[[0., -amplitude], [0., amplitude]],
                         [[-amplitude, 0.], [amplitude, 0.]]], dtype=dtype)
    x, y = asarray(x), asarray(np.zeros_like(x))
    prefix = "branched_" if branched else ""
    options = {"method": method}
    if method == "polynomial":
        options["order"] = 15 if branched else (12 if dtype == np.float32 else 64)
    else:
        options["dyadic_order"] = 3 if branched else 4
    if branched:
        options["depth"] = 2
    function_name = "expected_" + prefix + "sig_score" if metric == "expected_sig_score" else prefix + metric
    function = getattr(api, function_name)
    if name == "torch":
        x.requires_grad_()
        gradient = torch.autograd.grad(function(x, y, **options).mean(), x)[0]
        gradient = gradient.detach().cpu().numpy()
    else:
        import jax
        gradient = jax.jit(jax.grad(lambda value: function(value, y, **options).mean()))(x)
    np.testing.assert_allclose(gradient, expected, atol=2e-5, rtol=2e-5)
