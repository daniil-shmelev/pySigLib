# Copyright 2026 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import sys

import numpy as np
import pytest
import torch

import native_api as pysiglib
import pysiglib.torch_api as torch_api
from conftest import check_close, skip_no_cuda
from pysiglib._core.branched_log_sig_backprop import (
    _branched_log_sig_from_path_backprop,
)


def _path():
    return np.array(
        [[0.0, 0.0], [0.2, -0.3], [0.7, 0.4], [0.5, 0.8]],
        dtype=np.float64,
    )


def _planar_forest_words(dimension, degree):
    basis = pysiglib.trees(dimension, degree, planar=True)[1:]
    letters = {
        forest[0]: letter
        for letter, forest in enumerate(
            basis_element for basis_element in basis if len(basis_element) == 1
        )
    }
    words = [tuple(letters[tree] for tree in forest) for forest in basis]
    return basis, words


def _lyndon_projection(words):
    lyndon_words = [word for word in words if pysiglib.is_lyndon(word)]
    basis_index = {word: idx for idx, word in enumerate(lyndon_words)}
    expansions = []
    for word in lyndon_words:
        if len(word) == 1:
            expansions.append({word: 1})
            continue

        suffix = next(
            word[start:]
            for start in range(1, len(word))
            if word[start:] in basis_index
        )
        prefix = word[:-len(suffix)]
        expansion = {}
        for left, left_coeff in expansions[basis_index[prefix]].items():
            for right, right_coeff in expansions[basis_index[suffix]].items():
                coeff = left_coeff * right_coeff
                expansion[left + right] = expansion.get(left + right, 0) + coeff
                expansion[right + left] = expansion.get(right + left, 0) - coeff
        expansions.append(expansion)

    projection = np.zeros((len(lyndon_words), len(lyndon_words)), dtype=np.int64)
    for col, expansion in enumerate(expansions):
        for word, coeff in expansion.items():
            row = basis_index.get(word)
            if row is not None:
                projection[row, col] += coeff
    return lyndon_words, projection


def test_branched_log_sig_length_matches_output_shapes():
    dimension, degree = 2, 3
    path = _path()

    assert pysiglib.branched_log_sig_length(
        dimension, degree) == pysiglib.branched_sig_length(dimension, degree)

    expanded_len = pysiglib.branched_sig_length(
        dimension, degree, planar=True)
    compressed_len = pysiglib.branched_log_sig_length(
        dimension, degree, planar=True)
    assert 0 < compressed_len < expanded_len

    for method in (1, 2, 3):
        pysiglib.prepare_branched_log_sig(
            dimension, degree, method, planar=True, device="cpu")
        out = pysiglib.branched_log_sig(
            path, degree, planar=True, method=method)
        assert out.shape == (compressed_len,)


@pytest.mark.parametrize("dimension,degree", [(0, 3), (2, 0)])
def test_compressed_zero_length_cases(dimension, degree):
    assert pysiglib.branched_log_sig_length(
        dimension, degree, planar=True) == 0
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cpu")

    bsig = np.ones(1, dtype=np.float64)
    output = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)
    gradient = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, np.empty(0, dtype=np.float64), dimension, degree,
        planar=True, method=2)

    assert output.shape == (0,)
    np.testing.assert_array_equal(gradient, np.zeros_like(bsig))


def test_branched_log_sig_length_reports_overflow():
    with pytest.raises(ValueError, match="integer overflow"):
        pysiglib.branched_log_sig_length(255, 7, planar=True)


@pytest.mark.parametrize("planar", [False, True])
def test_branched_log_sig_length_rejects_large_dimension_at_zero_degree(planar):
    with pytest.raises(ValueError, match="dimension must be <= 255"):
        pysiglib.branched_log_sig_length(256, 0, planar=planar)


@pytest.mark.parametrize("dimension,degree", [(1, 1), (1, 3), (2, 3), (3, 2)])
def test_planar_branched_log_sig_length_matches_explicit_lyndon_count(
        dimension, degree):
    _, words = _planar_forest_words(dimension, degree)
    expected = sum(pysiglib.is_lyndon(word) for word in words)
    assert pysiglib.branched_log_sig_length(
        dimension, degree, planar=True) == expected


@pytest.mark.parametrize("scalar_term", [False, True])
def test_method_one_is_expanded_lyndon_coordinate_slice(scalar_term):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 1, planar=True, device="cpu")
    bsig = pysiglib.branched_sig(
        _path(), degree, planar=True, scalar_term=scalar_term)
    expanded = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=0)
    compressed = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=1)

    _, words = _planar_forest_words(dimension, degree)
    offset = 1 if scalar_term else 0
    expected = np.asarray([
        expanded[idx + offset]
        for idx, word in enumerate(words)
        if pysiglib.is_lyndon(word)
    ])
    np.testing.assert_allclose(compressed, expected, atol=1e-13, rtol=1e-13)


def test_method_two_uses_standard_lyndon_bracket_basis():
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cpu")
    bsig = pysiglib.branched_sig(_path(), degree, planar=True)
    method_one = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=1)
    method_two = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)

    _, words = _planar_forest_words(dimension, degree)
    _, projection = _lyndon_projection(words)
    np.testing.assert_allclose(
        projection @ method_two, method_one, atol=1e-13, rtol=1e-13)


def test_planar_default_is_method_one_and_is_scalar_free():
    dimension, degree = 2, 3
    path = _path()
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 1, planar=True, device="cpu")

    expected = pysiglib.branched_log_sig(
        path, degree, planar=True, method=1)
    default = pysiglib.branched_log_sig(path, degree, planar=True)
    with_scalar_input = pysiglib.branched_log_sig(
        path, degree, planar=True, scalar_term=True)

    np.testing.assert_allclose(default, expected, atol=1e-14)
    np.testing.assert_allclose(with_scalar_input, expected, atol=1e-14)
    assert default.shape[-1] == pysiglib.branched_log_sig_length(
        dimension, degree, planar=True)


def test_nonplanar_default_is_method_zero():
    dimension, degree = 2, 3
    path = _path()
    pysiglib.prepare_branched_log_sig(dimension, degree, 0, device="cpu")

    expected = pysiglib.branched_log_sig(path, degree, method=0)
    actual = pysiglib.branched_log_sig(path, degree)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_compressed_cache_preparation_and_clearing():
    dimension, degree = 2, 3
    bsig = pysiglib.branched_sig(
        _path(), degree, planar=True, scalar_term=True)
    pysiglib.clear_cache()

    pysiglib.prepare_branched_log_sig(
        dimension, degree, 1, planar=True, device="cpu")
    pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=1)
    with pytest.raises(Exception):
        pysiglib.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=2)

    pysiglib.clear_cache()
    with pytest.raises(Exception):
        pysiglib.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=1)

    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cpu")
    pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=1)
    pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)


def test_method_three_prepares_method_two_basis_and_clears_with_cache():
    dimension, degree = 2, 3
    path = _path()
    bsig = pysiglib.branched_sig(
        path, degree, planar=True, scalar_term=True)
    pysiglib.clear_cache()
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu")

    direct = pysiglib.branched_log_sig(
        path, degree, planar=True, method=3)
    projected = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)
    np.testing.assert_allclose(direct, projected, atol=2e-12, rtol=2e-12)

    pysiglib.clear_cache()
    with pytest.raises(Exception, match="cache"):
        pysiglib.branched_log_sig(
            path, degree, planar=True, method=3)


def test_compressed_cache_disk_round_trip(tmp_path):
    dimension, degree = 2, 3
    pysiglib.set_cache_dir(str(tmp_path))
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_log_sig(
            dimension, degree, 2, planar=True, device="cpu", use_disk=True)
        bsig = pysiglib.branched_sig(
            _path(), degree, planar=True, scalar_term=True)
        expected = [
            pysiglib.branched_sig_to_log_sig(
                bsig, dimension, degree, planar=True, method=method)
            for method in (1, 2)
        ]

        pysiglib.clear_cache(use_disk=False)
        pysiglib.prepare_branched_log_sig(
            dimension, degree, 2, planar=True, device="cpu", use_disk=True)
        actual = [
            pysiglib.branched_sig_to_log_sig(
                bsig, dimension, degree, planar=True, method=method)
            for method in (1, 2)
        ]
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference, atol=1e-14, rtol=1e-14)
    finally:
        pysiglib.clear_cache(use_disk=True)


def test_malformed_planar_forest_offsets_rebuild_disk_cache(tmp_path):
    dimension, degree = 2, 3
    pysiglib.set_cache_dir(str(tmp_path))
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_log_sig(
            dimension, degree, 0, planar=True, device="cpu", use_disk=True)
        cache_dir = tmp_path / "pysiglib_cache"
        cache_file, = cache_dir.glob("planar_branched_*.bin")

        cache_bytes = bytearray(cache_file.read_bytes())
        offset = 4 * 8
        for item_size in (8, 8, 1, 8, 8, 8, 8, 8):
            count = int.from_bytes(
                cache_bytes[offset:offset + 8], byteorder=sys.byteorder)
            offset += 8 + count * item_size
        forest_data_count = int.from_bytes(
            cache_bytes[offset:offset + 8], byteorder=sys.byteorder)
        first_forest_offset = offset + 8 + forest_data_count * 8 + 8
        cache_bytes[first_forest_offset:first_forest_offset + 8] = (
            1).to_bytes(8, byteorder=sys.byteorder)
        cache_file.write_bytes(cache_bytes)

        pysiglib.clear_cache(use_disk=False)
        pysiglib.prepare_branched_log_sig(
            dimension, degree, 2, planar=True, device="cpu", use_disk=True)

        repaired_bytes = cache_file.read_bytes()
        repaired_first_offset = int.from_bytes(
            repaired_bytes[first_forest_offset:first_forest_offset + 8],
            byteorder=sys.byteorder,
        )
        assert repaired_first_offset == 0
        output = pysiglib.branched_log_sig(
            _path(), degree, planar=True, method=2)
        assert output.shape == (
            pysiglib.branched_log_sig_length(
                dimension, degree, planar=True),)
    finally:
        pysiglib.clear_cache(use_disk=True)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize(
    "time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_compressed_augmentation_matches_conversion(method, time_aug, lead_lag):
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="cpu")
    path = _path()
    bsig = pysiglib.branched_sig(
        path, degree, planar=True, time_aug=time_aug, lead_lag=lead_lag,
        scalar_term=True)
    expected = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, method=method)
    actual = pysiglib.branched_log_sig(
        path, degree, planar=True, time_aug=time_aug, lead_lag=lead_lag,
        scalar_term=True, method=method)

    effective_dimension = (
        2 * dimension if lead_lag else dimension) + int(time_aug)
    assert actual.shape == (pysiglib.branched_log_sig_length(
        effective_dimension, degree, planar=True),)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("method", [1, 2])
def test_compressed_correction_matches_conversion(method):
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    path = _path()
    correction = np.zeros((path.shape[0] - 1, dimension * dimension))
    correction[:, 0] = 0.03
    correction[:, 3] = -0.02
    bsig = pysiglib.branched_sig(
        path, degree, planar=True, correction=correction, scalar_term=True)
    expected = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    actual = pysiglib.branched_log_sig(
        path, degree, planar=True, correction=correction,
        scalar_term=True, method=method)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compressed_batch_dtype_and_empty_batch(method, dtype):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    paths = np.stack([_path(), 0.7 * _path()]).astype(dtype)

    batch = pysiglib.branched_log_sig(
        paths, degree, planar=True, method=method)
    expected = np.stack([
        pysiglib.branched_log_sig(
            path.copy(), degree, planar=True, method=method)
        for path in paths
    ])
    np.testing.assert_allclose(
        batch, expected, atol=1e-5 if dtype == np.float32 else 1e-13)
    assert batch.dtype == dtype

    input_len = pysiglib.branched_sig_length(
        dimension, degree, planar=True)
    empty = np.empty((0, input_len), dtype=dtype)
    empty_output = pysiglib.branched_sig_to_log_sig(
        empty, dimension, degree, planar=True, method=method)
    assert empty_output.shape == (
        0, pysiglib.branched_log_sig_length(dimension, degree, planar=True))
    assert empty_output.dtype == dtype


@pytest.mark.parametrize("method", [1, 2])
def test_compressed_backprop_shape_and_cotangent_unchanged(method):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    bsig = pysiglib.branched_sig(
        _path(), degree, planar=True, scalar_term=True)
    cotangent = np.linspace(
        -0.4,
        0.7,
        pysiglib.branched_log_sig_length(dimension, degree, planar=True),
    ).copy()
    saved_cotangent = cotangent.copy()

    grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, cotangent, dimension, degree, planar=True, method=method)

    assert grad.shape == bsig.shape
    np.testing.assert_array_equal(cotangent, saved_cotangent)


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("scalar_term", [False, True])
def test_compressed_backprop_matches_directional_finite_difference(
        method, scalar_term):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    bsig = pysiglib.branched_sig(
        _path(), degree, planar=True, scalar_term=scalar_term)
    rng = np.random.default_rng(7391 + method + int(scalar_term))
    cotangent = rng.normal(size=pysiglib.branched_log_sig_length(
        dimension, degree, planar=True))
    direction = rng.normal(size=bsig.shape)
    grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, cotangent, dimension, degree, planar=True, method=method)

    def objective(value):
        output = pysiglib.branched_sig_to_log_sig(
            value, dimension, degree, planar=True, method=method)
        return np.dot(output, cotangent)

    eps = 1e-6
    finite_difference = (
        objective(bsig + eps * direction) - objective(bsig - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(
        np.dot(grad, direction), finite_difference, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("method", [1, 2])
def test_torch_compressed_backward_matches_explicit_backprop(method):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    bsig_np = pysiglib.branched_sig(
        _path(), degree, planar=True, scalar_term=True)
    weights_np = np.linspace(
        -0.4,
        0.7,
        pysiglib.branched_log_sig_length(dimension, degree, planar=True),
    ).copy()

    bsig = torch.tensor(bsig_np, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    out = torch_api.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    (out * weights).sum().backward()

    expected = pysiglib.branched_sig_to_log_sig_backprop(
        bsig_np, weights_np, dimension, degree, planar=True, method=method)
    torch.testing.assert_close(
        bsig.grad, torch.tensor(expected, dtype=torch.float64),
        atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", [1, 2])
def test_torch_direct_compressed_backward_runs(method):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="cpu")
    path = torch.tensor(_path(), dtype=torch.float64, requires_grad=True)

    out = torch_api.branched_log_sig(
        path, degree, planar=True, method=method, scalar_term=True)
    out.square().sum().backward()

    assert out.shape == (
        pysiglib.branched_log_sig_length(dimension, degree, planar=True),)
    assert path.grad is not None
    assert torch.all(torch.isfinite(path.grad))


def test_branched_log_sig_method_validation():
    path = _path()

    with pytest.raises(TypeError, match="required positional argument"):
        pysiglib.prepare_branched_log_sig(2, 2)
    with pytest.raises(TypeError, match="method must be of type int"):
        pysiglib.prepare_branched_log_sig(2, 2, None)
    with pytest.raises(ValueError, match="not supported"):
        pysiglib.branched_sig_to_log_sig(
            np.ones(pysiglib.branched_sig_length(
                2, 2, planar=True, scalar_term=True)),
            2, 2, planar=True, method=3)
    with pytest.raises(ValueError, match="require planar=True"):
        pysiglib.branched_log_sig(path, 2, method=1)
    with pytest.raises(ValueError, match="require planar=True"):
        pysiglib.prepare_branched_log_sig(2, 2, 1)
    with pytest.raises(ValueError, match="one of 0, 1, 2 or 3"):
        pysiglib.branched_log_sig(path, 2, planar=True, method=4)


def test_prepare_compressed_device_both_prepares_cpu():
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 1, planar=True, device="both")
    output = pysiglib.branched_log_sig(
        _path(), degree, planar=True, method=1)
    assert output.shape == (
        pysiglib.branched_log_sig_length(dimension, degree, planar=True),)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [False, True])
def test_method_three_matches_method_two(dtype, batch):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cpu")
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu")
    path = _path().astype(dtype)
    if batch:
        path = np.stack([path, 0.7 * path])

    expected = pysiglib.branched_log_sig(
        path, degree, planar=True, method=2)
    actual = pysiglib.branched_log_sig(
        path, degree, planar=True, method=3, scalar_term=True)
    tolerance = 2e-5 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(
        actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_method_three_augmentation_matches_method_two(time_aug, lead_lag):
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="cpu")
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="cpu")
    expected = pysiglib.branched_log_sig(
        _path(), degree, planar=True, method=2,
        time_aug=time_aug, lead_lag=lead_lag)
    actual = pysiglib.branched_log_sig(
        _path(), degree, planar=True, method=3,
        time_aug=time_aug, lead_lag=lead_lag)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_method_three_backward_matches_finite_difference_and_preserves_cotangent():
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu")
    path = _path()
    rng = np.random.default_rng(9345)
    cotangent = rng.normal(size=pysiglib.branched_log_sig_length(
        dimension, degree, planar=True))
    saved_cotangent = cotangent.copy()
    direction = rng.normal(size=path.shape)
    gradient = _branched_log_sig_from_path_backprop(
        cotangent, path, degree)

    def objective(value):
        result = pysiglib.branched_log_sig(
            value, degree, planar=True, method=3)
        return np.dot(result, cotangent)

    eps = 1e-6
    finite_difference = (
        objective(path + eps * direction)
        - objective(path - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(
        np.sum(gradient * direction), finite_difference,
        atol=2e-8, rtol=2e-8)
    np.testing.assert_array_equal(cotangent, saved_cotangent)


def test_torch_method_three_backward_matches_explicit_backprop():
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cpu")
    path_np = _path()
    weights_np = np.linspace(
        -0.3, 0.8, pysiglib.branched_log_sig_length(
            dimension, degree, planar=True))
    path = torch.tensor(path_np, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(weights_np, dtype=torch.float64)

    output = torch_api.branched_log_sig(
        path, degree, planar=True, method=3)
    (output * weights).sum().backward()
    expected = _branched_log_sig_from_path_backprop(
        weights_np, path_np, degree)
    torch.testing.assert_close(
        path.grad, torch.tensor(expected), atol=1e-10, rtol=1e-10)


def test_method_three_rejects_correction():
    path = _path()
    correction = np.zeros((path.shape[0] - 1, path.shape[-1] ** 2))
    with pytest.raises(ValueError, match="correction is not supported"):
        pysiglib.branched_log_sig(
            path, 2, planar=True, method=3, correction=correction)


@pytest.mark.parametrize(
    "time_aug,lead_lag",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("batch_shape", [(), (0,)])
def test_method_three_rejects_empty_paths(
        batch_shape, time_aug, lead_lag):
    path = np.empty(batch_shape + (0, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="empty path"):
        pysiglib.branched_log_sig(
            path, 2, planar=True, method=3,
            time_aug=time_aug, lead_lag=lead_lag)

    cotangent = np.empty(
        batch_shape + (
            pysiglib.branched_log_sig_length(2, 2, planar=True),),
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="empty path"):
        _branched_log_sig_from_path_backprop(cotangent, path, 2)


@skip_no_cuda
@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("scalar_term", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_compressed_forward_and_backward_match_cpu(
        method, scalar_term, dtype):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="both")
    path = np.stack([_path(), 0.6 * _path()]).astype(dtype)
    bsig = pysiglib.branched_sig(
        path, degree, planar=True, scalar_term=scalar_term)
    expected = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=method)
    weights = np.random.default_rng(8162).normal(
        size=expected.shape).astype(dtype)
    expected_grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, weights, dimension, degree, planar=True, method=method)

    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
    bsig_cuda = torch.tensor(
        bsig, dtype=torch_dtype, device="cuda", requires_grad=True)
    weights_cuda = torch.tensor(weights, dtype=torch_dtype, device="cuda")
    saved_weights = weights_cuda.clone()
    actual = torch_api.branched_sig_to_log_sig(
        bsig_cuda, dimension, degree, planar=True, method=method)
    explicit_grad = pysiglib.branched_sig_to_log_sig_backprop(
        bsig_cuda.detach(), weights_cuda, dimension, degree,
        planar=True, method=method)
    (actual * weights_cuda).sum().backward()

    check_close(actual, expected, single_atol=2e-5, double_atol=2e-10)
    check_close(
        explicit_grad, expected_grad,
        single_atol=5e-5, double_atol=2e-9)
    check_close(
        bsig_cuda.grad, expected_grad,
        single_atol=5e-5, double_atol=2e-9)
    torch.testing.assert_close(weights_cuda, saved_weights)


@skip_no_cuda
@pytest.mark.parametrize("scalar_term", [False, True])
def test_cuda_compressed_empty_batch(scalar_term):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cuda")
    input_width = pysiglib.branched_sig_length(
        dimension, degree, planar=True, scalar_term=scalar_term)
    compact_width = pysiglib.branched_log_sig_length(
        dimension, degree, planar=True)
    bsig = torch.empty(
        (0, input_width), dtype=torch.float64, device="cuda")
    result = torch_api.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)
    cotangent = torch.empty(
        (0, compact_width), dtype=torch.float64, device="cuda")
    gradient = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, cotangent, dimension, degree, planar=True, method=2)

    assert result.shape == (0, compact_width)
    assert gradient.shape == bsig.shape


@skip_no_cuda
def test_cuda_compressed_cache_upgrade_clear_and_disk_round_trip(tmp_path):
    dimension, degree = 2, 3
    input_width = pysiglib.branched_sig_length(
        dimension, degree, planar=True, scalar_term=True)
    bsig = torch.cat((
        torch.ones(1, dtype=torch.float64, device="cuda"),
        torch.linspace(
            -0.2, 0.3, input_width - 1,
            dtype=torch.float64, device="cuda"),
    ))
    pysiglib.set_cache_dir(str(tmp_path))
    pysiglib.clear_cache(use_disk=True)
    try:
        pysiglib.prepare_branched_log_sig(
            dimension, degree, 1, planar=True, device="cuda")
        torch_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=1)
        with pytest.raises(Exception, match="cache"):
            torch_api.branched_sig_to_log_sig(
                bsig, dimension, degree, planar=True, method=2)

        pysiglib.prepare_branched_log_sig(
            dimension, degree, 2, planar=True, device="cuda",
            use_disk=True)
        expected = torch_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=2)

        pysiglib.clear_cache(use_disk=False, device="cuda")
        with pytest.raises(Exception, match="cache"):
            torch_api.branched_sig_to_log_sig(
                bsig, dimension, degree, planar=True, method=1)

        pysiglib.prepare_branched_log_sig(
            dimension, degree, 1, planar=True, device="cuda",
            use_disk=True)
        actual = torch_api.branched_sig_to_log_sig(
            bsig, dimension, degree, planar=True, method=2)
        torch.testing.assert_close(actual, expected)
    finally:
        pysiglib.clear_cache(use_disk=True)


@skip_no_cuda
@pytest.mark.parametrize("method", [0, 1, 2])
def test_cuda_direct_branched_log_sig_with_correction_matches_cpu(method):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, device="both")
    path = torch.tensor(
        _path(), dtype=torch.float64, requires_grad=True)
    correction = torch.linspace(
        -0.02, 0.03,
        (path.shape[0] - 1) * dimension * dimension,
        dtype=torch.float64,
    ).reshape(path.shape[0] - 1, dimension * dimension)
    expected = torch_api.branched_log_sig(
        path, degree, planar=True, method=method,
        correction=correction)
    weights = torch.linspace(
        -0.4, 0.6, expected.numel(), dtype=torch.float64)
    (expected * weights).sum().backward()

    path_cuda = path.detach().cuda().requires_grad_(True)
    actual = torch_api.branched_log_sig(
        path_cuda,
        degree, planar=True, method=method,
        correction=correction.cuda(),
    )
    (actual * weights.cuda()).sum().backward()

    check_close(actual, expected, double_atol=2e-9)
    check_close(path_cuda.grad, path.grad, double_atol=2e-8)


@skip_no_cuda
@pytest.mark.parametrize("dimension,degree", [(0, 3), (2, 0)])
def test_cuda_compressed_zero_width(dimension, degree):
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 2, planar=True, device="cuda")
    bsig = torch.ones(1, dtype=torch.float64, device="cuda")
    cotangent = torch.empty(0, dtype=torch.float64, device="cuda")

    output = pysiglib.branched_sig_to_log_sig(
        bsig, dimension, degree, planar=True, method=2)
    gradient = pysiglib.branched_sig_to_log_sig_backprop(
        bsig, cotangent, dimension, degree, planar=True, method=2)

    assert output.shape == (0,)
    torch.testing.assert_close(gradient, torch.zeros_like(bsig))


@skip_no_cuda
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [False, True])
def test_cuda_method_three_forward_and_backward_match_cpu(dtype, batch):
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="both")
    path = np.insert(_path().astype(dtype), 2, _path()[1], axis=0)
    if batch:
        path = np.stack([path, 0.7 * path])
    expected = pysiglib.branched_log_sig(
        path, degree, planar=True, method=3)
    weights = np.random.default_rng(9163).normal(
        size=expected.shape).astype(dtype)
    expected_grad = _branched_log_sig_from_path_backprop(
        weights, path, degree)

    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64
    path_cuda = torch.tensor(
        path, dtype=torch_dtype, device="cuda", requires_grad=True)
    weights_cuda = torch.tensor(weights, dtype=torch_dtype, device="cuda")
    saved_weights = weights_cuda.clone()
    actual = torch_api.branched_log_sig(
        path_cuda, degree, planar=True, method=3)
    explicit_grad = _branched_log_sig_from_path_backprop(
        weights_cuda, path_cuda.detach(), degree)
    (actual * weights_cuda).sum().backward()

    tolerance = 5e-4 if dtype == np.float32 else 2e-9
    check_close(actual, expected, atol=tolerance)
    check_close(explicit_grad, expected_grad, atol=tolerance)
    check_close(path_cuda.grad, expected_grad, atol=tolerance)
    torch.testing.assert_close(weights_cuda, saved_weights)


@skip_no_cuda
@pytest.mark.parametrize("method", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "time_aug,lead_lag", [(True, False), (False, True), (True, True)])
def test_cuda_augmentation_matches_cpu(method, time_aug, lead_lag):
    dimension, degree = 2, 2
    pysiglib.prepare_branched_log_sig(
        dimension, degree, method, planar=True, time_aug=time_aug,
        lead_lag=lead_lag, device="both")
    path_cpu = torch.tensor(
        _path(), dtype=torch.float64, requires_grad=True)
    expected = torch_api.branched_log_sig(
        path_cpu, degree, planar=True, method=method, time_aug=time_aug,
        lead_lag=lead_lag)
    weights = torch.linspace(
        -0.3, 0.7, expected.numel(), dtype=torch.float64)
    (expected * weights).sum().backward()

    path_cuda = torch.tensor(
        _path(), dtype=torch.float64, device="cuda", requires_grad=True)
    actual = torch_api.branched_log_sig(
        path_cuda, degree, planar=True, method=method, time_aug=time_aug,
        lead_lag=lead_lag)
    (actual * weights.cuda()).sum().backward()

    check_close(actual, expected, double_atol=2e-9)
    check_close(path_cuda.grad, path_cpu.grad, double_atol=2e-8)


@skip_no_cuda
def test_cuda_method_three_length_one_returns_zero_gradient():
    dimension, degree = 2, 3
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cuda")
    path = torch.tensor(
        [[0.2, -0.3]], dtype=torch.float64, device="cuda",
        requires_grad=True)
    output = torch_api.branched_log_sig(
        path, degree, planar=True, method=3)
    output.sum().backward()

    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(path.grad) == 0


@skip_no_cuda
def test_cuda_method_three_degree_zero_returns_empty_output_and_zero_gradient():
    dimension, degree = 2, 0
    pysiglib.prepare_branched_log_sig(
        dimension, degree, 3, planar=True, device="cuda")
    path = torch.tensor(
        [[0.0, 0.1], [0.3, -0.2], [0.7, 0.4]],
        dtype=torch.float64, device="cuda", requires_grad=True)
    output = torch_api.branched_log_sig(
        path, degree, planar=True, method=3)
    output.sum().backward()

    assert output.shape == (0,)
    assert torch.count_nonzero(path.grad) == 0


@skip_no_cuda
def test_cuda_method_three_rejects_degree_above_twenty():
    with pytest.raises(
        NotImplementedError,
        match="CUDA MKW BCH method supports degree at most 20",
    ):
        pysiglib.prepare_branched_log_sig(
            1, 21, 3, planar=True, device="cuda")
