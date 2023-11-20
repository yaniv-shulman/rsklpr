import numpy as np

from rsklpr.rsklpr import Rsklpr


def test_rsklpr_basic_regression_1d_with_joint_expected_output() -> None:
    """
    Smoke tests that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    x: np.ndarray = np.linspace(start=-3.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=0.0, stop=5, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="joint")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_1d_with_conden_expected_output() -> None:
    """
    Smoke tests that reasonable values are returned for a linear 1D input using the conden kernel.
    """
    x: np.ndarray = np.linspace(start=1.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=-2.0, stop=-1.0, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="conden")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_2d_with_joint_expected_output() -> None:
    """
    Smoke tests that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    rng: np.random.Generator = np.random.default_rng(seed=12)

    x: np.ndarray = np.concatenate(
        [
            np.linspace(start=-10.0, stop=10, num=50).reshape((-1, 1)),
            (np.linspace(start=-5.0, stop=7, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
        ],
        axis=1,
    )

    y: np.ndarray = np.linspace(start=-2.0, stop=0.0, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="joint")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_2d_with_conden_expected_output() -> None:
    """
    Smoke tests that reasonable values are returned for a linear 1D input using the conden kernel.
    """
    rng: np.random.Generator = np.random.default_rng(seed=12)

    x: np.ndarray = np.concatenate(
        [
            np.linspace(start=-5.0, stop=-3.0, num=50).reshape((-1, 1)),
            (np.linspace(start=0.0, stop=7, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
        ],
        axis=1,
    )

    y: np.ndarray = np.linspace(start=-1.0, stop=5, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="conden")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_1d_with_joint_increasing_windows_expected_output() -> None:
    """
    Smoke tests that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    x: np.ndarray = np.linspace(start=-3.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=0.0, stop=5, num=50)
    size_neighborhood: int
    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="joint")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)
