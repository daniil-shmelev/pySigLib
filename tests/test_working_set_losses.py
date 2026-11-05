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

import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.special import expit

import pysiglib
from benchmarks.lasso_feature_reference import dictionary


def loss_gradient(y, eta, loss):
    r = eta - y
    if loss == "logistic":
        return np.mean(np.logaddexp(0, eta) - y*eta), (expit(eta)-y)/len(y)
    if loss == "poisson":
        return np.mean(np.exp(eta)-y*eta), (np.exp(eta)-y)/len(y)
    if loss == "huber":
        s = np.clip(r, -1, 1)
        return np.mean(s*(r-.5*s)), s/len(y)
    return np.mean(r*r)/2, r/len(y)


@pytest.mark.parametrize("mode", ["sig", "log", "hall"])
@pytest.mark.parametrize("loss", ["squared", "logistic", "huber", "poisson"])
@pytest.mark.parametrize("intercept", [True, False])
def test_dense_optimum(mode, loss, intercept):
    rng = np.random.default_rng(1103)
    paths = rng.normal(size=(63, 8, 3)).cumsum(1)
    words, X = dictionary(paths, 3, mode)
    signal = 4*X[:, 0] - 7*X[:, 3]
    y = signal + .2*rng.normal(size=len(paths))
    if loss == "logistic":
        y = rng.binomial(1, expit(signal)).astype(float)
    if loss == "poisson":
        y = rng.poisson(np.exp(.5*signal)).astype(float)
    alpha = .003
    model = pysiglib.sig_lasso(paths, y, 3, alpha, feature_type=mode, loss=loss,
                               fit_intercept=intercept, initial_words=[(0,)], tol=1e-8)
    assert model["converged"] and model["loss"] == loss
    assert model["stats"]["engine"] == "tree" and model["stats"]["block_queries"] > 0
    indices = [words.index(w) for w in model["words"]]
    p = len(words)
    def objective(z):
        beta = z[1:1+p]-z[1+p:]
        val, grad = loss_gradient(y, z[0]+X@beta, loss)
        c = X.T@grad
        return val+alpha*z[1:].sum(), np.r_[grad.sum(), c+alpha, -c+alpha]

    ref = minimize(objective, np.zeros(1+2*p), method="L-BFGS-B", jac=True,
                   bounds=[(None,None) if intercept else (0,0)]+[(0,None)]*(2*p),
                   options=dict(ftol=1e-15, gtol=1e-11, maxiter=5000))
    assert ref.success
    eta = model["intercept"] + X[:, indices]@model["coef"]
    np.testing.assert_allclose(eta, ref.x[0]+X@(ref.x[1:1+p]-ref.x[1+p:]), atol=3e-6)
    value, grad = loss_gradient(y, eta, loss)
    primal = value + alpha*abs(model["coef"]).sum()
    np.testing.assert_allclose(primal, ref.fun, atol=2e-10)
    np.testing.assert_allclose(primal, model["objective"], atol=2e-12)
    corr = X.T@grad
    errors = np.maximum(abs(corr)-alpha, 0)
    errors[indices] = abs(corr[indices]+alpha*np.sign(model["coef"]))
    error = max(errors.max(), abs(grad.sum()) if intercept else 0)
    assert error <= model["kkt_violation"] + 2e-12
    if loss == "squared":
        theta = -grad.copy()
        if intercept:
            theta -= theta.mean()
        theta *= min(1., alpha / max(abs(X.T@theta).max(), 1e-300))
        dual = y@theta - len(y)*(theta@theta)/2
        assert model["dual_gap"] + 2e-12 >= primal-dual
    else:
        assert model["dual_gap"] is None


@pytest.mark.parametrize("loss", ["logistic", "poisson", "huber"])
def test_float32_independent_responses(loss):
    rng = np.random.default_rng(430)
    paths = rng.normal(size=(2, 40, 7, 2)).astype(np.float32)
    targets = rng.normal(size=(2, 40, 2))
    if loss == "logistic":
        targets = targets > 0
    elif loss == "poisson":
        targets = rng.poisson(1.5, size=targets.shape)
    targets = targets.astype(np.float32)
    penalties = [.01, .02]
    options = dict(feature_type="hall", loss=loss, time_aug=True)
    model = pysiglib.sig_lasso(paths, targets, 2, penalties, **options)
    assert model["converged"].all()
    prediction = pysiglib.sig_lasso_predict(paths, model)
    np.testing.assert_allclose(prediction, model["intercept"] + model["signatures"] @ model["coef"], atol=1e-6)
    for j, alpha in enumerate(penalties):
        single = pysiglib.sig_lasso(paths, targets[..., j].copy(), 2, alpha, **options)
        np.testing.assert_allclose(prediction[..., j], pysiglib.sig_lasso_predict(paths, single), atol=1e-6)
        np.testing.assert_allclose(model["objective"][j], single["objective"], atol=1e-7)


@pytest.mark.parametrize("key", ["method","bound_method","search_method","batch_size","cache_bytes"])
def test_removed_flags(key):
    with pytest.raises(TypeError, match=key):
        pysiglib.sig_lasso(np.zeros((2,2,1)), np.zeros(2), 1, .1, **{key:1})


@pytest.mark.parametrize("loss,targets", [("logistic",[0.,2.]), ("poisson",[.3,1.]), ("poisson",[-1.,2.]),
                                           ("logistic",[1.,1.]), ("poisson",[0.,0.]), ("unknown",[0.,1.])])
def test_invalid_losses(loss, targets):
    with pytest.raises(ValueError):
        pysiglib.sig_lasso(np.zeros((2,2,1)),np.array(targets),1,.01,loss=loss)


def test_ties_inactive_limit_and_output_retry(monkeypatch):
    paths=np.zeros((4,2,20))
    paths[:,1]=np.array([-1,-1,1,1])[:,None]
    y=np.array([-2.,-1.,1.,2.])
    old_empty=np.empty
    def fail(shape,*args,**kwargs):
        if shape == (1,1):
            raise MemoryError("allocation failure")
        return old_empty(shape,*args,**kwargs)
    with monkeypatch.context() as ctx:
        ctx.setattr(np,"empty",fail)
        with pytest.raises(MemoryError,match="allocation failure"):
            pysiglib.sig_lasso(paths,y,1,.1)
    model=pysiglib.sig_lasso(paths,y,1,.1)
    assert model["words"] == [(0,)]
    assert model["n_iter"] == 1
    np.testing.assert_allclose(model["coef"],[1.4],atol=1e-10)
    model=pysiglib.sig_lasso(paths,y,1,2.)
    assert model["n_iter"]==0 and model["converged"]


def test_weak_penalty_and_iteration_limit():
    rng=np.random.default_rng(409)
    paths=rng.normal(size=(80,5,26))
    y=rng.normal(size=80)
    with pytest.warns(RuntimeWarning,match="KKT"):
        m=pysiglib.sig_lasso(paths,y,1,1e-5,max_iter=1)
    assert not m["converged"] and m["n_iter"]==1
    assert m["dual_gap"]>=0 and m["kkt_violation"]>1e-8


def test_cpu_abi():
    from pysiglib.load_siglib import CPSIG
    for kind in ("f","d"):
        assert hasattr(CPSIG,"sig_working_set_"+kind)
        for name in ("sig_lasso_","sig_lasso_features_","sig_lasso_predict_"):
            assert not hasattr(CPSIG,name+kind)


def test_small_tree_with_memory_limit():
    paths = np.random.default_rng(20).normal(size=(256, 2, 3))
    model = pysiglib.sig_lasso(paths, np.ones(256), 3, .01, memory_limit=110 << 10)
    assert model["converged"] and not model["words"]
    assert model["stats"]["engine"] == "tree"
    assert model["stats"]["full_transform_calls"] == 0
    assert model["stats"]["pruned"] > 0
    assert model["stats"]["peak_bytes"] <= 110 << 10


@pytest.mark.parametrize("mode", ["sig", "log", "hall"])
@pytest.mark.parametrize("n, d, degree", [(24, 3, 4), (270, 8, 3)])
def test_tree_against_full_dictionary(mode, n, d, degree):
    rng = np.random.default_rng(311)
    paths = rng.normal(size=(n, 5, d)).cumsum(1)
    words, design = dictionary(paths, degree, mode)
    y = 8*design[:, 0] - 3*design[:, 1] + rng.normal(size=n)*.1
    alpha = .2*np.max(np.abs((y-y.mean())@design))/n
    model = pysiglib.sig_lasso(paths, y, degree, alpha, feature_type=mode, initial_words=[(0,)])
    assert model["stats"]["engine"] == "tree"
    assert model["stats"]["full_transform_calls"] == 0
    assert model["stats"]["search_passes"] > 0
    assert model["converged"]
    support = [words.index(w) for w in model["words"]]
    gradient = (model["intercept"] + design[:, support]@model["coef"]-y)/n
    corr = design.T@gradient
    errors = np.maximum(abs(corr)-alpha, 0)
    errors[support] = abs(corr[support]+alpha*np.sign(model["coef"]))
    assert max(errors.max(), abs(gradient.sum())) <= model["kkt_violation"]+2e-12


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
@pytest.mark.parametrize("mode",["sig","log","hall"])
def test_cuda_rejected_before_copy(mode):
    x=torch.zeros((3,2,2))
    for paths,y,alpha in [(x.cuda(),torch.zeros(3),.1),(x,torch.zeros(3).cuda(),.1),(x,torch.zeros(3),torch.tensor(.1).cuda())]:
        with pytest.raises(ValueError,match="CPU"):
            pysiglib.sig_lasso(paths,y,2,alpha,feature_type=mode)
