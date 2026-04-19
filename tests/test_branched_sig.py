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

import pytest
import numpy as np
import torch
import itertools

import kauri
import kauri.bck
import kauri.mkw

import pysiglib
from conftest import DEVICES, check_close, skip_no_cuda


# ---------------------------------------------------------------------------
# Helpers: kauri-based reference implementation
# ---------------------------------------------------------------------------

def _color_tree(tree, d):
    """Generate all coloured versions of an undecorated kauri tree."""
    if tree.nodes() == 1:
        return [kauri.Tree([c]) for c in range(d)]
    children_forest = tree.unjoin()
    child_trees = list(children_forest.tree_list)
    child_colorings = [_color_tree(child, d) for child in child_trees]
    results = []
    for root_color in range(d):
        for combo in itertools.product(*child_colorings):
            new_tree = kauri.Forest(list(combo)).join(root_color=root_color)
            results.append(new_tree)
    return results


def enumerate_decorated_trees(d, N):
    """Enumerate all canonical decorated rooted trees up to order N with d labels."""
    all_trees = []
    seen = set()
    for order in range(1, N + 1):
        for shape in kauri.trees_of_order(order):
            for ct in _color_tree(shape, d):
                canon = ct.sorted_list_repr()
                if canon not in seen:
                    seen.add(canon)
                    all_trees.append(ct)
    return all_trees


def get_node_decorations(tree):
    """Extract all node decorations from a kauri coloured tree (pre-order)."""
    decs = []
    _extract_decs(tree.sorted_list_repr(), decs)
    return decs


def _extract_decs(slr, decs):
    if isinstance(slr, int):
        decs.append(slr)
        return
    root_color = slr[-1]
    decs.append(root_color)
    for child in slr[:-1]:
        _extract_decs(child, decs)


def linear_branched_sig_ref(z, trees):
    """Reference linear branched sig: X(tau) = prod z[dec(v)] / gamma(tau)."""
    coeffs = {}
    for t in trees:
        decs = get_node_decorations(t)
        gamma = t.factorial()
        num = 1.0
        for dec in decs:
            num *= z[dec]
        coeffs[t.sorted_list_repr()] = num / gamma
    return coeffs


def branched_sig_reference(path, d, N):
    """Compute branched sig using kauri Map Butcher product as ground truth."""
    trees = enumerate_decorated_trees(d, N)

    # Identity character
    X = kauri.Map(lambda t: 1.0 if t.nodes() == 0 else 0.0)

    for n in range(len(path) - 1):
        z = path[n + 1] - path[n]
        coeffs = linear_branched_sig_ref(z, trees)

        def make_char(c):
            def char_func(t):
                if t.nodes() == 0:
                    return 1.0
                canon = t.sorted_list_repr()
                return c.get(canon, 0.0)
            return char_func

        Y = kauri.Map(make_char(coeffs))
        X = X * Y  # Butcher product in BCK

    return np.array([X(t) for t in trees])


def compute_kauri_to_pysiglib_permutation(d, N):
    """Determine the permutation mapping kauri tree order to pysiglib tree order.

    Uses a multi-segment path with carefully chosen increments to break
    symmetries (e.g., chain(root=0,child=1) vs chain(root=1,child=0)
    have identical linear sigs but differ after Butcher product).
    """
    pysiglib.prepare_branched_sig(d, N)
    trees = enumerate_decorated_trees(d, N)
    num_trees = len(trees)

    # Use a 3-point path with asymmetric increments to break all symmetries
    path = np.zeros((3, d))
    for i in range(d):
        path[1, i] = np.pi * (i + 1) + np.e * (i + 1)**2
        path[2, i] = path[1, i] + np.sqrt(2) * (i + 1) + np.log(i + 2)

    # pysiglib coefficients (in pysiglib tree order)
    pysig_coeffs = np.array(pysiglib.branched_sig(path, N), dtype=np.float64)

    # kauri reference (in kauri tree order)
    kauri_arr = branched_sig_reference(path, d, N)

    # Match: for each kauri index, find the pysiglib index with matching coefficient
    perm = np.zeros(num_trees, dtype=int)
    used = set()
    for ki in range(num_trees):
        best_idx = -1
        best_diff = float('inf')
        for pi in range(num_trees):
            if pi in used:
                continue
            diff = abs(kauri_arr[ki] - pysig_coeffs[pi])
            if diff < best_diff:
                best_diff = diff
                best_idx = pi
        assert best_diff < 1e-8, f"No matching coefficient found for kauri tree {ki}: best_diff={best_diff}"
        perm[ki] = best_idx
        used.add(best_idx)

    return perm


def reorder_kauri_to_pysiglib(kauri_arr, perm):
    """Reorder a kauri-ordered array to match pysiglib ordering."""
    result = np.empty_like(kauri_arr)
    for ki in range(len(perm)):
        result[perm[ki]] = kauri_arr[ki]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,N,expected", [
    (1, 1, 2),   # 1 + 1*1
    (1, 2, 3),   # 2 + 1*1
    (1, 3, 5),   # 3 + 2*1
    (2, 1, 3),   # 1 + 1*2
    (2, 2, 7),   # 3 + 1*4
    (2, 3, 21),  # 7 + 2*{8+6}=14
    (3, 1, 4),   # 1 + 1*3
    (3, 2, 13),  # 4 + 1*9
])
def test_branched_sig_length(d, N, expected):
    pysiglib.prepare_branched_sig(d, N)
    assert pysiglib.branched_sig_length(d, N, scalar_term=True) == expected


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_length_vs_kauri(d, N):
    trees = enumerate_decorated_trees(d, N)
    expected = len(trees)
    pysiglib.prepare_branched_sig(d, N)
    assert pysiglib.branched_sig_length(d, N) == expected


def test_branched_sig_trivial_path():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    path = np.array([[1.0, 2.0], [1.0, 2.0]])  # constant path
    bsig = pysiglib.branched_sig(path, N)
    assert np.allclose(bsig, 0.0, atol=1e-14)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_single_segment(d, N):
    pysiglib.prepare_branched_sig(d, N)
    perm = compute_kauri_to_pysiglib_permutation(d, N)

    np.random.seed(123)
    z = np.random.randn(d)
    path = np.zeros((2, d))
    path[1] = z

    bsig = pysiglib.branched_sig(path, N)

    trees = enumerate_decorated_trees(d, N)
    ref_coeffs = linear_branched_sig_ref(z, trees)
    ref = np.array([ref_coeffs[t.sorted_list_repr()] for t in trees])
    ref_reordered = reorder_kauri_to_pysiglib(ref, perm)

    np.testing.assert_allclose(bsig, ref_reordered, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2), (2, 4)])
def test_branched_sig_vs_kauri(d, N):
    """Primary correctness test: compare against kauri Butcher product."""
    pysiglib.prepare_branched_sig(d, N)
    perm = compute_kauri_to_pysiglib_permutation(d, N)

    np.random.seed(42)
    L = 10
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N)
    ref = branched_sig_reference(path, d, N)
    ref_reordered = reorder_kauri_to_pysiglib(ref, perm)

    np.testing.assert_allclose(bsig, ref_reordered, atol=1e-10)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_batch(d, N):
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(55)
    B, L = 5, 8
    paths = np.random.randn(B, L, d)

    batch_result = pysiglib.branched_sig(paths, N)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N)
        np.testing.assert_allclose(batch_result[i], single, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_batch_parallel(d, N):
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(77)
    B, L = 10, 8
    paths = np.random.randn(B, L, d)

    serial = pysiglib.branched_sig(paths, N, n_jobs=1)
    parallel = pysiglib.branched_sig(paths, N, n_jobs=-1)
    np.testing.assert_allclose(serial, parallel, atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_branched_sig_dtypes(dtype):
    d, N = 2, 2
    pysiglib.prepare_branched_sig(d, N)
    path = np.random.randn(5, d).astype(dtype)
    bsig = pysiglib.branched_sig(path, N)
    assert len(bsig) == pysiglib.branched_sig_length(d, N)


def test_branched_sig_degree_1_matches_standard():
    """At degree 1 (single-vertex trees only), branched sig == standard sig level 1."""
    d = 3
    pysiglib.prepare_branched_sig(d, 1)
    np.random.seed(11)
    path = np.cumsum(np.random.randn(20, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, 1)
    total_incr = path[-1] - path[0]
    np.testing.assert_allclose(bsig, total_incr, atol=1e-12)


# ---------------------------------------------------------------------------
# time_aug and lead_lag tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_time_aug_matches_manual(d, N):
    """time_aug flag should match manually prepending a time channel."""
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(200)
    L = 10
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, time_aug=True)

    t = np.linspace(0, 1, L)[:, np.newaxis]
    path_aug = np.concatenate([path, t], axis=1)
    bsig_manual = pysiglib.branched_sig(path_aug, N)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_lead_lag_matches_manual(d, N):
    """lead_lag flag should match manually applying the lead-lag transform."""
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(201)
    L = 8
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, lead_lag=True)

    path_ll = np.array(pysiglib.transform_path(path, lead_lag=True))
    bsig_manual = pysiglib.branched_sig(path_ll, N)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_time_aug_lead_lag_matches_manual(d, N):
    """Combined time_aug + lead_lag should match manual transform."""
    aug_dim = 2 * d + 1
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(202)
    L = 8
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, time_aug=True, lead_lag=True)

    path_t = np.array(pysiglib.transform_path(path, time_aug=True, lead_lag=True))
    bsig_manual = pysiglib.branched_sig(path_t, N)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_time_aug_batch(d, N):
    """time_aug should work correctly with batched input."""
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(203)
    B, L = 4, 10
    paths = np.random.randn(B, L, d)

    batch_result = pysiglib.branched_sig(paths, N, time_aug=True)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N, time_aug=True)
        np.testing.assert_allclose(batch_result[i], single, atol=1e-12)


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2), (2, 4)])
def test_branched_sig_cuda_matches_cpu(d, N):
    """CUDA output must exactly match CPU for the same input."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(300)
    path = np.cumsum(np.random.randn(10, d) * 0.1, axis=0)
    path_t = torch.tensor(path)

    cpu = pysiglib.branched_sig(path_t, N)
    cuda = pysiglib.branched_sig(path_t.cuda(), N)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_cuda_batch(d, N):
    """Batched CUDA output matches per-element CUDA."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(301)
    B, L = 8, 12
    paths = torch.tensor(np.random.randn(B, L, d)).cuda()

    batch_result = pysiglib.branched_sig(paths, N)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N)
        check_close(batch_result[i], single, double_atol=1e-12)


@skip_no_cuda
def test_branched_sig_cuda_trivial():
    """Constant path on CUDA should give identity signature."""
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    path = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64).cuda()
    bsig = pysiglib.branched_sig(path, N)
    assert torch.allclose(bsig, torch.zeros(bsig.shape[0], device="cuda", dtype=torch.float64), atol=1e-14)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_cuda_time_aug(d, N):
    """time_aug on CUDA matches CPU."""
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(302)
    path = torch.tensor(np.cumsum(np.random.randn(10, d) * 0.1, axis=0))

    cpu = pysiglib.branched_sig(path, N, time_aug=True)
    cuda = pysiglib.branched_sig(path.cuda(), N, time_aug=True)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_cuda_lead_lag(d, N):
    """lead_lag on CUDA matches CPU."""
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N)
    np.random.seed(303)
    path = torch.tensor(np.cumsum(np.random.randn(8, d) * 0.1, axis=0))

    cpu = pysiglib.branched_sig(path, N, lead_lag=True)
    cuda = pysiglib.branched_sig(path.cuda(), N, lead_lag=True)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
def test_branched_sig_cuda_float32():
    """float32 on CUDA matches CPU."""
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(304)
    path = torch.tensor(np.random.randn(8, d).astype(np.float32)).cuda()

    cuda = pysiglib.branched_sig(path, N)
    cpu = pysiglib.branched_sig(path.cpu(), N)
    check_close(cpu, cuda, single_atol=1e-5)


# ---------------------------------------------------------------------------
# Backpropagation tests
# ---------------------------------------------------------------------------

def _finite_diff_branched_sig(path_np, d, N, eps=1e-8):
    """Compute dF/dpath via finite differences where F = sum(branched_sig)."""
    pysiglib.prepare_branched_sig(d, N)
    grad = np.zeros_like(path_np)
    f0 = np.array(pysiglib.branched_sig(path_np, N)).sum()
    for i in range(path_np.shape[0]):
        for j in range(path_np.shape[1]):
            path_p = path_np.copy()
            path_p[i, j] += eps
            f1 = np.array(pysiglib.branched_sig(path_p, N)).sum()
            grad[i, j] = (f1 - f0) / eps
    return grad


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_backprop_finite_diff(d, N):
    """Backprop matches finite differences with F = sum(bsig)."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(400)
    L = 5
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N)
    bsig_len = len(bsig)
    derivs = np.ones(bsig_len)

    grad_bp = np.array(pysiglib.branched_sig_backprop(path, bsig, derivs, N))
    grad_fd = _finite_diff_branched_sig(path, d, N)

    np.testing.assert_allclose(grad_bp, grad_fd, atol=1e-4)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_backprop_random_derivs(d, N):
    """Backprop with random upstream derivatives matches finite differences."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(401)
    L = 5
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N)
    bsig_len = len(bsig)
    derivs = np.random.randn(bsig_len)

    grad_bp = np.array(pysiglib.branched_sig_backprop(path, bsig, derivs, N))

    eps = 1e-8
    grad_fd = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_p = path.copy()
            path_p[i, j] += eps
            bsig_p = np.array(pysiglib.branched_sig(path_p, N))
            grad_fd[i, j] = np.dot(derivs, bsig_p - np.array(bsig)) / eps

    np.testing.assert_allclose(grad_bp, grad_fd, atol=1e-4)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_backprop_torch_api(d, N):
    """torch_api branched_sig backward produces correct gradients."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(402)
    path = torch.tensor(np.cumsum(np.random.randn(5, d) * 0.1, axis=0),
                        dtype=torch.float64, requires_grad=True)

    bsig = pysiglib.torch_api.branched_sig(path, N)
    loss = bsig.sum()
    loss.backward()
    grad_torch = path.grad.clone()

    grad_manual = pysiglib.branched_sig_backprop(
        path.detach(), bsig.detach(),
        torch.ones_like(bsig), N)

    check_close(grad_torch, grad_manual, double_atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_backprop_torch_api(d, N):
    """torch_api planar branched_sig backward produces correct gradients."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(4021)
    path = torch.tensor(np.cumsum(np.random.randn(5, d) * 0.1, axis=0),
                        dtype=torch.float64, requires_grad=True)

    bsig = pysiglib.torch_api.branched_sig(path, N, planar=True)
    loss = bsig.sum()
    loss.backward()
    grad_torch = path.grad.clone()

    grad_manual = pysiglib.branched_sig_backprop(
        path.detach(), bsig.detach(),
        torch.ones_like(bsig), N, planar=True)

    check_close(grad_torch, grad_manual, double_atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_backprop_batch(d, N):
    """Batched backprop matches single-path backprop."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(403)
    B, L = 4, 6
    paths = np.random.randn(B, L, d)

    bsigs = pysiglib.branched_sig(paths, N)
    bsig_len = bsigs.shape[1]
    derivs = np.random.randn(B, bsig_len)

    batch_grad = np.array(pysiglib.branched_sig_backprop(paths, bsigs, derivs, N))

    for i in range(B):
        single_grad = np.array(pysiglib.branched_sig_backprop(
            paths[i], bsigs[i], derivs[i], N))
        np.testing.assert_allclose(batch_grad[i], single_grad, atol=1e-12)


# ---------------------------------------------------------------------------
# CUDA backprop tests
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_branched_sig_backprop_cuda_matches_cpu(d, N):
    """CUDA backprop must match CPU backprop."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(500)
    path = torch.tensor(np.cumsum(np.random.randn(8, d) * 0.1, axis=0), dtype=torch.float64)
    bsig = pysiglib.branched_sig(path, N)
    derivs = torch.rand(bsig.shape, dtype=torch.float64)

    cpu_grad = pysiglib.branched_sig_backprop(path, bsig, derivs, N)
    cuda_grad = pysiglib.branched_sig_backprop(path.cuda(), bsig.cuda(), derivs.cuda(), N)
    check_close(cpu_grad, cuda_grad, double_atol=1e-10)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_branched_sig_backprop_cuda_batch(d, N):
    """Batched CUDA backprop matches single-path CUDA backprop."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(501)
    B, L = 4, 6
    paths = torch.tensor(np.random.randn(B, L, d), dtype=torch.float64, device='cuda')
    bsigs = pysiglib.branched_sig(paths, N)
    derivs = torch.rand(bsigs.shape, dtype=torch.float64, device='cuda')

    batch_grad = pysiglib.branched_sig_backprop(paths, bsigs, derivs, N)
    for i in range(B):
        single_grad = pysiglib.branched_sig_backprop(paths[i], bsigs[i], derivs[i], N)
        check_close(batch_grad[i], single_grad, double_atol=1e-10)


@skip_no_cuda
def test_branched_sig_backprop_cuda_torch_api(d=2, N=3):
    """torch_api branched_sig backward works on CUDA."""
    pysiglib.prepare_branched_sig(d, N)
    np.random.seed(502)
    path = torch.tensor(np.cumsum(np.random.randn(5, d) * 0.1, axis=0),
                        dtype=torch.float64, device='cuda', requires_grad=True)

    bsig = pysiglib.torch_api.branched_sig(path, N)
    loss = bsig.sum()
    loss.backward()
    cuda_grad = path.grad.clone()

    # Compare against CPU
    path_cpu = path.detach().cpu().requires_grad_(True)
    bsig_cpu = pysiglib.torch_api.branched_sig(path_cpu, N)
    bsig_cpu.sum().backward()
    cpu_grad = path_cpu.grad

    check_close(cpu_grad, cuda_grad, double_atol=1e-10)


# ===========================================================================
# Planar branched signature tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers: kauri MKW-based reference implementation for planar branched sigs
# ---------------------------------------------------------------------------

def enumerate_decorated_planar_trees(d, N):
    """Enumerate all decorated planar trees up to order N with d labels."""
    all_trees = []
    for order in range(1, N + 1):
        for t in kauri.colored_planar_trees_of_order(order, d):
            all_trees.append(t)
    return all_trees


def linear_planar_branched_sig_ref(z, trees):
    """Reference linear planar branched sig: X(tau) = prod z[dec(v)] / gamma(tau)."""
    coeffs = {}
    for t in trees:
        # PlanarTree.sorted_list_repr() returns self.list_repr (ordered, not sorted)
        # so _extract_decs works identically: slr[-1] is root color, slr[:-1] are children
        decs = []
        _extract_decs(t.sorted_list_repr(), decs)
        gamma = t.factorial()
        num = 1.0
        for dec in decs:
            num *= z[dec]
        coeffs[t.sorted_list_repr()] = num / gamma
    return coeffs


def planar_branched_sig_reference(path, d, N):
    """Compute planar branched sig using kauri.mkw.map_product as ground truth.

    kauri.mkw.map_product uses the NCK coproduct on PlanarTree objects, which
    computes the correct convolution product for scalar-valued MKW characters
    (shuffle coefficients cancel with the 1/k! factors).
    """
    trees = enumerate_decorated_planar_trees(d, N)

    X = kauri.Map(lambda t: 1.0 if t.nodes() == 0 else 0.0)

    for n in range(len(path) - 1):
        z = path[n + 1] - path[n]
        coeffs = linear_planar_branched_sig_ref(z, trees)

        def make_char(c):
            def char_func(t):
                if t.nodes() == 0:
                    return 1.0
                return c.get(t.sorted_list_repr(), 0.0)
            return char_func

        Y = kauri.Map(make_char(coeffs))
        X = kauri.mkw.map_product(X, Y)

    return np.array([X(t) for t in trees])


def compute_kauri_to_pysiglib_planar_permutation(d, N):
    """Find permutation mapping kauri planar tree order to pysiglib planar tree order.

    Uses a multi-segment path with irrational increments to break all symmetries.
    """
    pysiglib.prepare_branched_sig(d, N, planar=True)
    trees = enumerate_decorated_planar_trees(d, N)
    num_trees = len(trees)

    path = np.zeros((3, d))
    for i in range(d):
        path[1, i] = np.pi * (i + 1) + np.e * (i + 1)**2
        path[2, i] = path[1, i] + np.sqrt(2) * (i + 1) + np.log(i + 2)

    pysig_coeffs = np.array(pysiglib.branched_sig(path, N, planar=True), dtype=np.float64)
    kauri_arr = planar_branched_sig_reference(path, d, N)

    perm = np.zeros(num_trees, dtype=int)
    used = set()
    for ki in range(num_trees):
        best_idx = -1
        best_diff = float('inf')
        for pi in range(num_trees):
            if pi in used:
                continue
            diff = abs(kauri_arr[ki] - pysig_coeffs[pi])
            if diff < best_diff:
                best_diff = diff
                best_idx = pi
        assert best_diff < 1e-8, f"No match for planar kauri tree {ki}: best_diff={best_diff}"
        perm[ki] = best_idx
        used.add(best_idx)

    return perm


def reorder_kauri_to_pysiglib_planar(kauri_arr, perm):
    """Reorder a kauri-ordered planar array to match pysiglib ordering."""
    result = np.empty_like(kauri_arr)
    for ki in range(len(perm)):
        result[perm[ki]] = kauri_arr[ki]
    return result


# ---------------------------------------------------------------------------
# Planar length tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,N,expected", [
    (1, 1, 2),    # 1 + 1
    (1, 2, 3),    # 2 + 1
    (1, 3, 5),    # 3 + 2  (same as non-planar for d=1)
    (2, 1, 3),    # 1 + 2
    (2, 2, 7),    # 3 + 4  (same as non-planar: single child => no ordering difference)
    (2, 3, 23),   # 7 + 16 (non-planar: 21; +2 from (*_0,*_1) vs (*_1,*_0) orderings)
    (3, 1, 4),    # 1 + 3
    (3, 2, 13),   # 4 + 9
])
def test_planar_branched_sig_length(d, N, expected):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    assert pysiglib.branched_sig_length(d, N, planar=True, scalar_term=True) == expected


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_length_vs_kauri(d, N):
    # colored_planar_trees_up_to_order includes the empty tree (order 0).
    expected = len(list(kauri.colored_planar_trees_up_to_order(N, d)))
    pysiglib.prepare_branched_sig(d, N, planar=True)
    assert pysiglib.branched_sig_length(d, N, planar=True, scalar_term=True) == expected


@pytest.mark.parametrize("d,N", [(2, 3), (3, 3), (2, 4)])
def test_planar_branched_sig_length_geq_nonplanar(d, N):
    """Planar length is at least non-planar length (more trees due to ordering)."""
    pysiglib.prepare_branched_sig(d, N)
    pysiglib.prepare_branched_sig(d, N, planar=True)
    assert pysiglib.branched_sig_length(d, N, planar=True) >= pysiglib.branched_sig_length(d, N)


# ---------------------------------------------------------------------------
# Planar correctness tests
# ---------------------------------------------------------------------------

def test_planar_branched_sig_trivial_path():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N, planar=True)
    path = np.array([[1.0, 2.0], [1.0, 2.0]])
    bsig = pysiglib.branched_sig(path, N, planar=True)
    assert np.allclose(bsig, 0.0, atol=1e-14)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_single_segment(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    perm = compute_kauri_to_pysiglib_planar_permutation(d, N)

    np.random.seed(1000)
    z = np.random.randn(d)
    path = np.zeros((2, d))
    path[1] = z

    bsig = pysiglib.branched_sig(path, N, planar=True)

    trees = enumerate_decorated_planar_trees(d, N)
    ref_coeffs = linear_planar_branched_sig_ref(z, trees)
    ref = np.array([ref_coeffs[t.sorted_list_repr()] for t in trees])
    ref_reordered = reorder_kauri_to_pysiglib_planar(ref, perm)

    np.testing.assert_allclose(bsig, ref_reordered, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2), (2, 4)])
def test_planar_branched_sig_vs_kauri(d, N):
    """Primary correctness test: compare against kauri MKW map product."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    perm = compute_kauri_to_pysiglib_planar_permutation(d, N)

    np.random.seed(1001)
    L = 10
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N, planar=True)
    ref = planar_branched_sig_reference(path, d, N)
    ref_reordered = reorder_kauri_to_pysiglib_planar(ref, perm)

    np.testing.assert_allclose(bsig, ref_reordered, atol=1e-10)


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_batch(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1003)
    B, L = 5, 8
    paths = np.random.randn(B, L, d)

    batch_result = pysiglib.branched_sig(paths, N, planar=True)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N, planar=True)
        np.testing.assert_allclose(batch_result[i], single, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_batch_parallel(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1004)
    B, L = 10, 8
    paths = np.random.randn(B, L, d)

    serial = pysiglib.branched_sig(paths, N, n_jobs=1, planar=True)
    parallel = pysiglib.branched_sig(paths, N, n_jobs=-1, planar=True)
    np.testing.assert_allclose(serial, parallel, atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_planar_branched_sig_dtypes(dtype):
    d, N = 2, 2
    pysiglib.prepare_branched_sig(d, N, planar=True)
    path = np.random.randn(5, d).astype(dtype)
    bsig = pysiglib.branched_sig(path, N, planar=True)
    assert len(bsig) == pysiglib.branched_sig_length(d, N, planar=True)


def test_planar_branched_sig_degree_1_matches_standard():
    """At degree 1, planar branched sig == standard sig level 1 (same as non-planar)."""
    d = 3
    pysiglib.prepare_branched_sig(d, 1, planar=True)
    np.random.seed(1005)
    path = np.cumsum(np.random.randn(20, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, 1, planar=True)
    total_incr = path[-1] - path[0]
    np.testing.assert_allclose(bsig, total_incr, atol=1e-12)


def test_planar_branched_sig_degree_1_matches_nonplanar():
    """At degree 1, planar and non-planar branched sigs are identical."""
    d, N = 3, 1
    pysiglib.prepare_branched_sig(d, N)
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1006)
    path = np.cumsum(np.random.randn(15, d) * 0.1, axis=0)

    bsig_np = pysiglib.branched_sig(path, N)
    bsig_pl = pysiglib.branched_sig(path, N, planar=True)
    np.testing.assert_allclose(bsig_pl, bsig_np, atol=1e-14)


# ---------------------------------------------------------------------------
# Planar time_aug and lead_lag tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_time_aug_matches_manual(d, N):
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1100)
    L = 10
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, time_aug=True, planar=True)

    t = np.linspace(0, 1, L)[:, np.newaxis]
    path_aug = np.concatenate([path, t], axis=1)
    bsig_manual = pysiglib.branched_sig(path_aug, N, planar=True)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_lead_lag_matches_manual(d, N):
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1101)
    L = 8
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, lead_lag=True, planar=True)

    path_ll = np.array(pysiglib.transform_path(path, lead_lag=True))
    bsig_manual = pysiglib.branched_sig(path_ll, N, planar=True)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_time_aug_lead_lag_matches_manual(d, N):
    aug_dim = 2 * d + 1
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1102)
    L = 8
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig_flag = pysiglib.branched_sig(path, N, time_aug=True, lead_lag=True, planar=True)

    path_t = np.array(pysiglib.transform_path(path, time_aug=True, lead_lag=True))
    bsig_manual = pysiglib.branched_sig(path_t, N, planar=True)

    np.testing.assert_allclose(bsig_flag, bsig_manual, atol=1e-12)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_time_aug_batch(d, N):
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1103)
    B, L = 4, 10
    paths = np.random.randn(B, L, d)

    batch_result = pysiglib.branched_sig(paths, N, time_aug=True, planar=True)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N, time_aug=True, planar=True)
        np.testing.assert_allclose(batch_result[i], single, atol=1e-12)


# ---------------------------------------------------------------------------
# Planar CUDA tests
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2), (2, 4)])
def test_planar_branched_sig_cuda_matches_cpu(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1200)
    path = np.cumsum(np.random.randn(10, d) * 0.1, axis=0)
    path_t = torch.tensor(path)

    cpu = pysiglib.branched_sig(path_t, N, planar=True)
    cuda = pysiglib.branched_sig(path_t.cuda(), N, planar=True)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_cuda_batch(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1201)
    B, L = 8, 12
    paths = torch.tensor(np.random.randn(B, L, d)).cuda()

    batch_result = pysiglib.branched_sig(paths, N, planar=True)
    for i in range(B):
        single = pysiglib.branched_sig(paths[i], N, planar=True)
        check_close(batch_result[i], single, double_atol=1e-12)


@skip_no_cuda
def test_planar_branched_sig_cuda_trivial():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N, planar=True)
    path = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64).cuda()
    bsig = pysiglib.branched_sig(path, N, planar=True)
    assert torch.allclose(bsig, torch.zeros(bsig.shape[0], device="cuda", dtype=torch.float64), atol=1e-14)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_cuda_time_aug(d, N):
    aug_dim = d + 1
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1202)
    path = torch.tensor(np.cumsum(np.random.randn(10, d) * 0.1, axis=0))

    cpu = pysiglib.branched_sig(path, N, time_aug=True, planar=True)
    cuda = pysiglib.branched_sig(path.cuda(), N, time_aug=True, planar=True)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_cuda_lead_lag(d, N):
    aug_dim = 2 * d
    pysiglib.prepare_branched_sig(aug_dim, N, planar=True)
    np.random.seed(1203)
    path = torch.tensor(np.cumsum(np.random.randn(8, d) * 0.1, axis=0))

    cpu = pysiglib.branched_sig(path, N, lead_lag=True, planar=True)
    cuda = pysiglib.branched_sig(path.cuda(), N, lead_lag=True, planar=True)
    check_close(cpu, cuda, double_atol=1e-12)


@skip_no_cuda
def test_planar_branched_sig_cuda_float32():
    d, N = 2, 3
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1204)
    path = torch.tensor(np.random.randn(8, d).astype(np.float32)).cuda()

    cuda = pysiglib.branched_sig(path, N, planar=True)
    cpu = pysiglib.branched_sig(path.cpu(), N, planar=True)
    check_close(cpu, cuda, single_atol=1e-5)


# ---------------------------------------------------------------------------
# Planar backpropagation tests
# ---------------------------------------------------------------------------

def _finite_diff_planar_branched_sig(path_np, d, N, eps=1e-8):
    """dF/dpath via finite differences where F = sum(planar_branched_sig)."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    grad = np.zeros_like(path_np)
    f0 = np.array(pysiglib.branched_sig(path_np, N, planar=True)).sum()
    for i in range(path_np.shape[0]):
        for j in range(path_np.shape[1]):
            path_p = path_np.copy()
            path_p[i, j] += eps
            f1 = np.array(pysiglib.branched_sig(path_p, N, planar=True)).sum()
            grad[i, j] = (f1 - f0) / eps
    return grad


@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_backprop_finite_diff(d, N):
    """Planar backprop matches finite differences with F = sum(bsig)."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1300)
    L = 5
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N, planar=True)
    derivs = np.ones(len(bsig))

    grad_bp = np.array(pysiglib.branched_sig_backprop(path, bsig, derivs, N, planar=True))
    grad_fd = _finite_diff_planar_branched_sig(path, d, N)

    np.testing.assert_allclose(grad_bp, grad_fd, atol=1e-4)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_backprop_random_derivs(d, N):
    """Planar backprop with random upstream derivatives matches finite differences."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1301)
    L = 5
    path = np.cumsum(np.random.randn(L, d) * 0.1, axis=0)

    bsig = pysiglib.branched_sig(path, N, planar=True)
    derivs = np.random.randn(len(bsig))

    grad_bp = np.array(pysiglib.branched_sig_backprop(path, bsig, derivs, N, planar=True))

    eps = 1e-8
    grad_fd = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_p = path.copy()
            path_p[i, j] += eps
            bsig_p = np.array(pysiglib.branched_sig(path_p, N, planar=True))
            grad_fd[i, j] = np.dot(derivs, bsig_p - np.array(bsig)) / eps

    np.testing.assert_allclose(grad_bp, grad_fd, atol=1e-4)


@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_backprop_batch(d, N):
    """Batched planar backprop matches single-path backprop."""
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1302)
    B, L = 4, 6
    paths = np.random.randn(B, L, d)

    bsigs = pysiglib.branched_sig(paths, N, planar=True)
    derivs = np.random.randn(B, bsigs.shape[1])

    batch_grad = np.array(pysiglib.branched_sig_backprop(paths, bsigs, derivs, N, planar=True))
    for i in range(B):
        single_grad = np.array(pysiglib.branched_sig_backprop(
            paths[i], bsigs[i], derivs[i], N, planar=True))
        np.testing.assert_allclose(batch_grad[i], single_grad, atol=1e-12)


# ---------------------------------------------------------------------------
# Planar CUDA backprop tests
# ---------------------------------------------------------------------------

@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3), (3, 2)])
def test_planar_branched_sig_backprop_cuda_matches_cpu(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1400)
    path = torch.tensor(np.cumsum(np.random.randn(8, d) * 0.1, axis=0), dtype=torch.float64)
    bsig = pysiglib.branched_sig(path, N, planar=True)
    derivs = torch.rand(bsig.shape, dtype=torch.float64)

    cpu_grad = pysiglib.branched_sig_backprop(path, bsig, derivs, N, planar=True)
    cuda_grad = pysiglib.branched_sig_backprop(path.cuda(), bsig.cuda(), derivs.cuda(), N, planar=True)
    check_close(cpu_grad, cuda_grad, double_atol=1e-10)


@skip_no_cuda
@pytest.mark.parametrize("d,N", [(2, 3)])
def test_planar_branched_sig_backprop_cuda_batch(d, N):
    pysiglib.prepare_branched_sig(d, N, planar=True)
    np.random.seed(1401)
    B, L = 4, 6
    paths = torch.tensor(np.random.randn(B, L, d), dtype=torch.float64, device='cuda')
    bsigs = pysiglib.branched_sig(paths, N, planar=True)
    derivs = torch.rand(bsigs.shape, dtype=torch.float64, device='cuda')

    batch_grad = pysiglib.branched_sig_backprop(paths, bsigs, derivs, N, planar=True)
    for i in range(B):
        single_grad = pysiglib.branched_sig_backprop(paths[i], bsigs[i], derivs[i], N, planar=True)
        check_close(batch_grad[i], single_grad, double_atol=1e-10)

