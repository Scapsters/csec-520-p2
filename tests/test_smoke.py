"""Smoke tests: the pipeline runs and the pieces behave.

Tests that exercise YOUR from-scratch code skip themselves until you implement
it, then start enforcing correctness. Run them often: `make test`.
"""
import numpy as np
import pytest

from src.data import load_data
from src.distances import euclidean_sqdist, mahalanobis_sqdist
from src.kmeans import KMeansReference, KMeansScratch
from src.evaluate import map_clusters_to_classes, classification_metrics

CFG = {"seed": 0, "data": {"source": "iris", "standardize": True}}


# ---------------------------------------------------------------- provided --
def test_load_iris_shapes():
    X, y, feats, classes = load_data(CFG)
    assert X.shape == (150, 4)
    assert len(y) == 150 and set(np.unique(y)) == {0, 1, 2}
    assert len(feats) == 4 and len(classes) == 3
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9)   # standardized


def test_euclidean_sqdist_matches_manual():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    C = np.array([[0.0, 0.0]])
    assert np.allclose(euclidean_sqdist(X, C).ravel(), [0.0, 25.0])


def test_reference_kmeans_recovers_iris_structure():
    X, y, _, _ = load_data(CFG)
    est = KMeansReference(k=3, seed=0).fit(X)
    assert est.centroids_.shape == (3, 4)
    _, y_pred = map_clusters_to_classes(y, est.labels_)
    # Iris is easy: majority-vote accuracy should be well above chance (0.33).
    assert classification_metrics(y, y_pred)["accuracy"] > 0.7


# ------------------------------------------------------- your implementation --
def _scratch_or_skip(**kw):
    X, y, _, _ = load_data(CFG)
    est = KMeansScratch(k=3, seed=0, n_init=3, **kw)
    try:
        est.fit(X)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
    return X, y, est


def test_scratch_fit_sets_attributes():
    X, y, est = _scratch_or_skip()
    assert est.centroids_.shape == (3, X.shape[1])
    assert est.labels_.shape == (X.shape[0],)
    assert set(np.unique(est.labels_)) <= {0, 1, 2}
    assert est.inertia_ is not None and est.inertia_ > 0


def test_scratch_matches_reference_quality():
    """Your inertia should be within a few percent of scikit-learn's."""
    X, y, est = _scratch_or_skip()
    ref = KMeansReference(k=3, seed=0).fit(X)
    assert est.inertia_ <= ref.inertia_ * 1.10, (
        f"scratch inertia {est.inertia_:.3f} much worse than reference {ref.inertia_:.3f}"
    )


def test_scratch_is_deterministic_given_seed():
    X, _, est1 = _scratch_or_skip()
    est2 = KMeansScratch(k=3, seed=0, n_init=3).fit(X)
    assert np.array_equal(est1.labels_, est2.labels_)


def test_mahalanobis_reduces_to_euclidean_when_weights_are_one():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, -2.0]])
    C = np.array([[0.0, 0.0], [1.0, 1.0]])
    try:
        got = mahalanobis_sqdist(X, C, np.ones(2))
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
    assert np.allclose(got, euclidean_sqdist(X, C))


def test_mahalanobis_weights_scale_features():
    X = np.array([[2.0, 0.0]])
    C = np.array([[0.0, 0.0]])
    try:
        got = mahalanobis_sqdist(X, C, np.array([3.0, 1.0]))
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
    assert np.allclose(got.ravel(), [12.0])   # 3 * 2^2
