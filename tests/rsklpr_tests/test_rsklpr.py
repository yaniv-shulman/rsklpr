from typing import List

import numpy as np
import pytest
from pytest_mock import MockerFixture

import rsklpr
from rsklpr.rsklpr import Rsklpr, _laplacian_normalized, _tricube_normalized, _dim_data

_rng: np.random.Generator = np.random.default_rng(seed=12)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
        for _ in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
        for i in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[
        "tricube",
        "laplacian",
    ],
)
@pytest.mark.parametrize(
    argnames="k2",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
def test_rsklpr_smoke_test_1d_regression_increasing_windows_expected_output(
    x: np.ndarray, y: np.ndarray, k1: str, k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k1=k1, k2=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=5e-3)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.concatenate(
            [
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
            ],
            axis=1,
        )
        for _ in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
        for i in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[
        "tricube",
        "laplacian",
    ],
)
@pytest.mark.parametrize(
    argnames="k2",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
def test_rsklpr_smoke_test_2d_regression_increasing_windows_expected_output(
    x: np.ndarray, y: np.ndarray, k1: str, k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=4, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k1=k1, k2=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=6e-3)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.concatenate(
            [
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
                np.linspace(
                    start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50
                ).reshape((-1, 1))
                + _rng.uniform(low=-0.001, high=0.001, size=50).reshape((-1, 1)),
            ],
            axis=1,
        )
    ],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
    ],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[
        "tricube",
        "laplacian",
    ],
)
@pytest.mark.parametrize(
    argnames="k2",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
def test_rsklpr_smoke_test_5d_regression_increasing_windows_expected_output(
    x: np.ndarray, y: np.ndarray, k1: str, k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=7, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, k1=k1, k2=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=4e-2)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
        for _ in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[
        np.linspace(start=_rng.integers(low=-10, high=10), stop=_rng.integers(low=-10, high=10), num=50)
        + _rng.uniform(low=-0.001, high=0.001, size=50)
        for i in range(3)
    ],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[
        "tricube",
        "laplacian",
    ],
)
@pytest.mark.parametrize(
    argnames="k2",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
def test_rsklpr_smoke_test_1d_estimate_bootstrap_expected_output(
    x: np.ndarray, y: np.ndarray, k1: str, k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    target: Rsklpr = Rsklpr(size_neighborhood=20, k1=k1, k2=k2)
    target.fit(x=x, y=y)
    actual: np.ndarray
    conf_low_actual: np.ndarray
    conf_upper_actual: np.ndarray
    actual, conf_low_actual, conf_upper_actual = target.predict_bootstrap(x=x)
    np.testing.assert_allclose(actual=actual, desired=y, atol=7e-3)
    np.testing.assert_allclose(actual=conf_low_actual, desired=y, atol=7e-3)
    np.testing.assert_allclose(actual=conf_upper_actual, desired=y, atol=7e-3)


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
    assert target._k1 == "laplacian"
    assert target._k2 == "joint"
    assert target._bw1 == "normal_reference"
    assert target._bw2 == "normal_reference"
    assert target._bw_global_subsample_size is None
    rsklpr.rsklpr.np_defualt_rng.assert_called_once_with(seed=888)  # type: ignore [attr-defined]
    assert target._k1_func == _laplacian_normalized

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
    assert target._k1 == "tricube"
    assert target._k2 == "conden"
    assert target._bw1 == "cv_ls_global"
    assert target._bw2 == "scott"
    assert target._bw_global_subsample_size == 50
    rsklpr.rsklpr.np_defualt_rng.assert_called_once_with(seed=45)  # type: ignore [attr-defined]
    assert target._k1_func == _tricube_normalized

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


def test_dim_data_expected_output() -> None:
    """
    Tests the expected outputs are returned.
    """
    assert _dim_data(data=np.array([0, 1, 2])) == 1
    assert _dim_data(data=np.array([[0], [1], [2]])) == 1
    assert _dim_data(data=np.array([[0, 4], [1, 5], [2, 6]])) == 2
    assert _dim_data(data=np.ones((100, 5))) == 5


def test_laplacian_normalized_expected_output() -> None:
    """
    Tests the expected outputs are returned.
    """
    u: np.ndarray = np.linspace(start=0, stop=5)
    actual: np.ndarray = _laplacian_normalized(u=u)
    u /= 5.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=3, stop=4)
    actual = _laplacian_normalized(u=u)
    u /= 4.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=-2, stop=8)
    actual = _laplacian_normalized(u=u)
    u /= 8.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=0, stop=5).reshape(1, 50)
    actual = _laplacian_normalized(u=u)
    u /= 5.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=3, stop=4).reshape(1, 50)
    actual = _laplacian_normalized(u=u)
    u /= 4.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))

    u = np.linspace(start=-2, stop=8).reshape(1, 50)
    actual = _laplacian_normalized(u=u)
    u /= 8.0
    np.testing.assert_allclose(actual=actual, desired=np.exp(-u).reshape((1, 50)))


def test_tricube_normalized_expected_output() -> None:
    """
    Tests the expected outputs are returned.
    """
    u: np.ndarray = np.linspace(start=0, stop=5)
    actual: np.ndarray = _tricube_normalized(u=u)
    assert actual.min() == 0.0
    assert actual.max() == 1.0
    u /= 5.0
    desired: np.ndarray = (1.0 - u**3) ** 3
    np.testing.assert_allclose(actual=actual, desired=np.atleast_2d(desired))

    u = np.linspace(start=0, stop=3).reshape(1, -1)
    actual = _tricube_normalized(u=u)
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
        _tricube_normalized(u=u)
