import numpy as np
import pytest

from rsklpr.kernels import laplacian_normalized, tricube_normalized


def test_laplacian_normalized_expected_output() -> None:
    """
    Tests the expected outputs are returned.
    """
    u: np.ndarray = np.linspace(start=0, stop=5)
    actual: np.ndarray = laplacian_normalized(_=np.empty(()), u=u)
    u /= 5.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=3, stop=4)
    actual = laplacian_normalized(_=np.empty((100)), u=u)
    u /= 4.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=-2, stop=8)
    actual = laplacian_normalized(_=np.ones(()), u=u)
    u /= 8.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=0, stop=5).reshape(1, 50)
    actual = laplacian_normalized(_=np.ones((30)), u=u)
    u /= 5.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=3, stop=4).reshape(1, 50)
    actual = laplacian_normalized(_=np.empty((12, 100)), u=u)
    u /= 4.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=-2, stop=8).reshape(1, 50)
    actual = laplacian_normalized(_=np.zeros((12, 45)), u=u)
    u /= 8.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))


def test_tricube_normalized_expected_output() -> None:
    """
    Tests the expected outputs are returned.
    """
    u: np.ndarray = np.linspace(start=0, stop=5)
    actual: np.ndarray = tricube_normalized(_=np.empty(()), u=u)
    assert actual.min() == 0.0
    assert actual.max() == 1.0
    u /= 5.0
    desired: np.ndarray = (1.0 - u**3) ** 3
    np.testing.assert_allclose(actual=actual, desired=np.atleast_2d(desired))

    u = np.linspace(start=0, stop=3).reshape(1, -1)
    actual = tricube_normalized(_=np.zeros((30)), u=u)
    assert actual.min() == 0.0
    assert actual.max() == 1.0
    u /= 3.0
    desired = (1.0 - u**3) ** 3
    np.testing.assert_allclose(actual=actual, desired=np.atleast_2d(desired))


def test_tricube_normalized_raises_when_inputs_negative() -> None:
    """
    Tests an assertion error is raised if the inputs have negative values.
    """
    u: np.ndarray = np.linspace(start=-0.1, stop=5)
    with pytest.raises(AssertionError):
        tricube_normalized(_=np.empty(()), u=u)
