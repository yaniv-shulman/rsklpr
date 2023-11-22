from typing import List

import numpy as np
import pytest
from pytest_mock import MockerFixture

import rsklpr
from rsklpr.rsklpr import Rsklpr, _laplacian_normalized, _tricube_normalized


def test_rsklpr_basic_regression_1d_with_joint_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    x: np.ndarray = np.linspace(start=-3.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=0.0, stop=0.2, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="joint")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_1d_with_conden_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the conden kernel.
    """
    x: np.ndarray = np.linspace(start=1.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=0, stop=-0.5, num=50)
    target: Rsklpr = Rsklpr(size_neighborhood=20, k2="conden")

    actual: np.ndarray = target(
        x=x,
        y=y,
    )

    np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_2d_with_joint_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 2D input using the joint kernel.
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
    Smoke test that reasonable values are returned for a linear 2D input using the conden kernel.
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
    Smoke test that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    x: np.ndarray = np.linspace(start=-3.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=10.0, stop=-15, num=50)
    size_neighborhood: int
    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="joint")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_1d_with_conden_increasing_windows_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the conden kernel.
    """
    x: np.ndarray = np.linspace(start=-3.0, stop=10, num=50)
    y: np.ndarray = np.linspace(start=0.0, stop=-5, num=50)
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="conden")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_2d_with_joint_increasing_windows_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 2D input using the joint kernel.
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
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="joint")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_2d_with_conden_increasing_windows_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 2D input using the conden kernel.
    """
    rng: np.random.Generator = np.random.default_rng(seed=12)

    x: np.ndarray = np.concatenate(
        [
            np.linspace(start=-10.0, stop=10, num=50).reshape((-1, 1)),
            (np.linspace(start=-5.0, stop=7, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
        ],
        axis=1,
    )

    y: np.ndarray = np.linspace(start=10, stop=100, num=50)
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=4, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="conden")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=1e-7)


def test_rsklpr_basic_regression_5d_with_joint_increasing_windows_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 5D input using the joint kernel.
    """
    rng: np.random.Generator = np.random.default_rng(seed=12)

    x: np.ndarray = np.concatenate(
        [
            np.linspace(start=-10.0, stop=10, num=50).reshape((-1, 1)),
            (np.linspace(start=-5.0, stop=7, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=-15.0, stop=3, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=0, stop=8, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=0, stop=-3, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
        ],
        axis=1,
    )

    y: np.ndarray = np.linspace(start=-2.0, stop=5.0, num=50)
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=6, stop=50, num=15, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="joint")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=7e-5)


def test_rsklpr_basic_regression_5d_with_conden_increasing_windows_expected_output() -> None:
    """
    Smoke test that reasonable values are returned for a linear 5D input using the conden kernel.
    """
    rng: np.random.Generator = np.random.default_rng(seed=12)

    x: np.ndarray = np.concatenate(
        [
            np.linspace(start=-10.0, stop=10, num=50).reshape((-1, 1)),
            (np.linspace(start=-5.0, stop=7, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=-15.0, stop=3, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=0, stop=8, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
            (np.linspace(start=0, stop=-3, num=50) + rng.uniform(low=-0.01, high=0.01, size=50)).reshape((-1, 1)),
        ],
        axis=1,
    )

    y: np.ndarray = np.linspace(start=-2.0, stop=0.0, num=50)
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=7, stop=50, num=15, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k2="conden")

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=5e-6)


def test_rsklpr_init_expected_values(mocker: MockerFixture) -> None:
    """
    Test that correct values are assigned during initialization.
    """
    mocker.patch("rsklpr.rsklpr.np_defualt_rng")
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    assert target._size_neighborhood == 15
    assert target._degree == 1
    assert target._metric_x == "mahalanobis"
    assert target._metric_x_params is None
    assert target._k1 == _laplacian_normalized
    assert target._k2 == "joint"
    assert target._bw1 == "normal_reference"
    assert target._bw2 == "normal_reference"
    assert target._bw_global_subsample_size is None
    rsklpr.rsklpr.np_defualt_rng.assert_called_once_with(seed=888)  # type: ignore [attr-defined]

    mocker.patch("rsklpr.rsklpr.np_defualt_rng")
    target = Rsklpr(
        size_neighborhood=25,
        degree=0,
        metric_x="Minkowski",
        metric_x_params={"p": 3},
        k1="Tricube",
        k2="CondeN",
        bw1="cv_ls_glObal",
        bw2="Scott",
        bw_global_subsample_size=50,
        seed=45,
    )

    assert target._size_neighborhood == 25
    assert target._degree == 0
    assert target._metric_x == "minkowski"
    assert target._metric_x_params == {"p": 3}
    assert target._k1 == _tricube_normalized
    assert target._k2 == "conden"
    assert target._bw1 == "cv_ls_global"
    assert target._bw2 == "scott"
    assert target._bw_global_subsample_size == 50
    rsklpr.rsklpr.np_defualt_rng.assert_called_once_with(seed=45)  # type: ignore [attr-defined]

    target = Rsklpr(
        size_neighborhood=12,
        bw1=0.3,
        bw2=0.6,
    )

    assert target._bw1 == [0.3]
    assert target._bw2 == [0.6]

    target = Rsklpr(
        size_neighborhood=12,
        bw1=[0.3, 0.8],
        bw2=[0.6, 0.9, 8.0],
    )

    assert target._bw1 == [0.3, 0.8]
    assert target._bw2 == [0.6, 0.9, 8.0]

    def bw1(_: np.ndarray) -> List[float]:
        return [0.3, 0.8]

    def bw2(_: np.ndarray) -> List[float]:
        return [0.6, 0.9, 8.0]

    target = Rsklpr(
        size_neighborhood=12,
        bw1=bw1,
        bw2=bw2,
    )

    assert target._bw1 == bw1
    assert target._bw2 == bw2


def test_rsklpr_init_raises_for_incorrect_inputs() -> None:
    """
    Test that incorrect values provided to init raise exceptions.
    """
    # Too small neighborhoods
    with pytest.raises(expected_exception=ValueError, match="size_neighborhood must be at least 3"):
        Rsklpr(size_neighborhood=2)

    with pytest.raises(expected_exception=ValueError, match="degree must be one of 0 or 1"):
        Rsklpr(size_neighborhood=3, degree=2)

    with pytest.raises(
        expected_exception=ValueError, match="k1 bla is unsupported and must be one of 'laplacian' or 'tricube'"
    ):
        Rsklpr(size_neighborhood=3, k1="bla")

    with pytest.raises(
        expected_exception=ValueError, match="k2 alb is unsupported and must be one of 'conden' or 'joint'"
    ):
        Rsklpr(size_neighborhood=3, k2="alb")

    with pytest.raises(
        expected_exception=ValueError,
        match="When bandwidth is a string it must be one of 'normal_reference', "
        "'cv_ml', 'cv_ls', 'scott' or 'cv_ls_global'",
    ):
        Rsklpr(size_neighborhood=3, bw1="bla2")

    with pytest.raises(
        expected_exception=ValueError,
        match="When bandwidth is a string it must be one of 'normal_reference', "
        "'cv_ml', 'cv_ls', 'scott' or 'cv_ls_global'",
    ):
        Rsklpr(size_neighborhood=3, bw2="bla2")
