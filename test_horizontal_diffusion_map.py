import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from horizontal_diffusion_map import HorizontalDiffusionMap

DATA = [
    np.array([[1, 0, 1, 0, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0]]),
    np.array([[0, 1, 1, 0, 1], [0, 1, 1, 1, 0]]),
    np.array([[1, 1, 0, 0, 1], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1]]),
]
GROUP = np.array([0, 0, 0, 1, 1, 2, 2, 2])


@pytest.fixture
def hdm():
    return HorizontalDiffusionMap(
        DATA, alpha=0.5, bandwidth=1.0, name="toy", num_evecs=4, n_neighbors=2
    )


def test_transition_matrix_is_row_stochastic(hdm):
    row_sums = np.asarray(hdm.transition_matrix.sum(axis=1)).ravel()
    assert row_sums == pytest.approx(1.0)


def test_affinity_is_symmetric(hdm):
    assert abs(hdm.W - hdm.W.T).max() < 1e-12


def test_edges_stay_within_a_precinct(hdm):
    rows, cols = hdm.W.nonzero()
    assert np.all(GROUP[rows] == GROUP[cols])


def test_kernel_matches_hand_computation():
    # Two districts sharing 2 of 4 precincts: d = 1/2, so W = exp(-1/4).
    data = [np.array([[1, 1, 1, 0], [1, 1, 0, 1]])]
    hdm = HorizontalDiffusionMap(
        data, alpha=0.0, bandwidth=1.0, name="pair", num_evecs=1, n_neighbors=1
    )
    assert hdm.W[0, 1] == pytest.approx(np.exp(-0.25))


def test_eigenpairs_of_transition_matrix(hdm):
    residual = hdm.transition_matrix @ hdm.evecs - hdm.evecs * hdm.evals
    assert np.abs(residual).max() < 1e-10


def test_eigenvalue_one_once_per_component(hdm):
    n_components, _ = connected_components(hdm.W, directed=False)
    assert np.sum(np.isclose(hdm.evals, 1.0)) == n_components


def test_walker_never_leaves_its_precinct(hdm):
    walker = hdm.random_walk(0)
    visited = {next(walker) for _ in range(200)}
    assert visited <= {0, 1, 2}
