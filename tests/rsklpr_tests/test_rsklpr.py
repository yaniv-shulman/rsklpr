from typing import List, Optional, Union, Dict, Any, Type, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import statsmodels.api as sm
from pytest_mock import MockerFixture
from statsmodels.regression.linear_model import RegressionResults

import rsklpr
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from rsklpr.rsklpr import (
    Rsklpr,
    _dim_data,
    _weighted_local_regression,
    _r_squared,
    all_metrics,
)
from tests.rsklpr_tests.utils import generate_linear_1d, generate_linear_nd, rng, generate_quad_1d, generate_sin_1d


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d() for i in range(3)],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[
        tricube_normalized_metric,
        laplacian_normalized_metric,
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
    x: np.ndarray, y: np.ndarray, k1: Callable[[np.ndarray, np.ndarray], np.ndarray], k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=3, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=k1, kr=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=8.5e-3)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_nd(dim=2) for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[tricube_normalized_metric, laplacian_normalized_metric],
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
    x: np.ndarray, y: np.ndarray, k1: Callable[[np.ndarray, np.ndarray], np.ndarray], k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=4, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=k1, kr=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=6e-3)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_nd(dim=5)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d()],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[tricube_normalized_metric, laplacian_normalized_metric],
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
    x: np.ndarray, y: np.ndarray, k1: Callable[[np.ndarray, np.ndarray], np.ndarray], k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=7, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=k1, kr=k2)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=5e-2)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="k1",
    argvalues=[laplacian_normalized_metric, tricube_normalized_metric],
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
    x: np.ndarray, y: np.ndarray, k1: Callable[[np.ndarray, np.ndarray], np.ndarray], k2: str
) -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    target: Rsklpr = Rsklpr(size_neighborhood=20, kp=k1, kr=k2)
    target.fit(x=x, y=y)
    actual: np.ndarray
    conf_low_actual: np.ndarray
    conf_upper_actual: np.ndarray
    actual, conf_low_actual, conf_upper_actual, _ = target.predict_bootstrap(x=x)
    np.testing.assert_allclose(actual=actual, desired=y, atol=7e-3)
    np.testing.assert_allclose(actual=conf_low_actual, desired=y, atol=7e-3)
    np.testing.assert_allclose(actual=conf_upper_actual, desired=y, atol=7e-3)


def test_rsklpr_init_expected_values(mocker: MockerFixture) -> None:
    """
    Test that correct values are assigned during initialization.
    """
    mocker.patch("rsklpr.rsklpr.np_default_rng")
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    assert target._size_neighborhood == 15
    assert target._degree == 1
    assert target._metric_x == "mahalanobis"
    assert target._metric_x_params is None
    assert target._kr == "joint"
    assert target._bw1 == "normal_reference"
    assert target._bw2 == "normal_reference"
    assert target._bw_global_subsample_size is None
    assert target._seed == 888
    rsklpr.rsklpr.np_default_rng.assert_called_once_with(seed=888)  # type: ignore [attr-defined]
    assert target._kp == [laplacian_normalized_metric]
    assert target._x.shape == (0,)
    assert target._y.shape == (0,)
    assert target._residuals.shape == (0,)
    assert target._mean_square_error is None
    assert target._root_mean_square_error is None
    assert target._mean_abs_error is None
    assert target._bias_error is None
    assert target._std_error is None
    assert target._r_squared.shape == (0,)
    assert target._mean_r_squared is None
    assert not target._fit

    mocker.patch("rsklpr.rsklpr.np_default_rng")
    target = Rsklpr(
        size_neighborhood=25,
        degree=0,
        metric_x="Minkowski",
        metric_x_params={"p": 3},
        kp=tricube_normalized_metric,
        kr="CondeN",
        bw1="cv_ls_glObal",
        bw2="Scott",
        bw_global_subsample_size=50,
        seed=45,
    )

    assert target._size_neighborhood == 25
    assert target._degree == 0
    assert target._metric_x == "minkowski"
    assert target._metric_x_params == {"p": 3}
    assert target._kr == "conden"
    assert target._bw1 == "cv_ls_global"
    assert target._bw2 == "scott"
    assert target._bw_global_subsample_size == 50
    assert target._seed == 45
    rsklpr.rsklpr.np_default_rng.assert_called_once_with(seed=45)  # type: ignore [attr-defined]
    assert target._kp == [tricube_normalized_metric]
    assert target._x.shape == (0,)
    assert target._y.shape == (0,)
    assert target._residuals.shape == (0,)
    assert target._mean_square_error is None
    assert target._root_mean_square_error is None
    assert target._mean_abs_error is None
    assert target._bias_error is None
    assert target._std_error is None
    assert target._r_squared.shape == (0,)
    assert target._mean_r_squared is None
    assert not target._fit

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
        expected_exception=ValueError, match="k2 alb is unsupported and must be one of 'conden' or 'joint'"
    ):
        Rsklpr(size_neighborhood=3, kr="alb")

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


@pytest.mark.parametrize(
    argnames="x_0",
    argvalues=[rng.uniform(low=-10, high=10, size=1) for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d().reshape((-1, 1)) for _ in range(2)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d() for _ in range(2)]
    + [
        generate_quad_1d(),
        generate_sin_1d(),
    ],
)
@pytest.mark.parametrize(
    argnames="weights",
    argvalues=[
        rng.uniform(low=0.01, high=1, size=50),
        np.ones(shape=50),
        np.clip(rng.normal(loc=0.5, scale=0.2, size=50), a_min=0, a_max=None),
        np.linspace(start=0.01, stop=3, num=50),
    ],
)
def test_weighted_local_regression_1d_expected_values(
    x_0: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> None:
    """test the weighted linear regression implementation gives the expected results for the 1D case"""
    actual: float
    r_squared: Optional[float]

    actual, r_squared = _weighted_local_regression(
        x_0=x_0, x=x, y=y, weights=weights, degree=1, calculate_r_squared=True
    )

    x_sm: np.ndarray = sm.add_constant(x)
    model_sm = sm.WLS(endog=y, exog=x_sm, weights=weights)
    results_sm: RegressionResults = model_sm.fit()
    assert float(actual) == pytest.approx(float((results_sm.params[0] + x_0 * results_sm.params[1]).item()))
    assert r_squared == pytest.approx(float(results_sm.rsquared))


@pytest.mark.parametrize(
    argnames="x_0",
    argvalues=[rng.uniform(low=-10, high=10, size=2) for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_nd(dim=2).reshape((-1, 2)) for _ in range(2)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d() for _ in range(2)]
    + [
        generate_quad_1d(),
        generate_sin_1d(),
    ],
)
@pytest.mark.parametrize(
    argnames="weights",
    argvalues=[
        rng.uniform(low=0.01, high=1, size=50),
        np.ones(shape=50),
        np.clip(rng.normal(loc=0.5, scale=0.2, size=50), a_min=0, a_max=None),
        np.linspace(start=0.01, stop=3, num=50),
    ],
)
def test_weighted_local_regression_2d_expected_values(
    x_0: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> None:
    """test the weighted linear regression implementation gives the expected results for the 2D case"""
    actual: float
    r_squared: Optional[float]

    actual, r_squared = _weighted_local_regression(
        x_0=x_0, x=x, y=y, weights=weights, degree=1, calculate_r_squared=True
    )

    x_sm: np.ndarray = sm.add_constant(x)
    model_sm = sm.WLS(endog=y, exog=x_sm, weights=weights)
    results_sm: RegressionResults = model_sm.fit()
    assert float(actual) == pytest.approx(
        float(results_sm.params[0] + x_0[0] * results_sm.params[1] + x_0[1] * results_sm.params[2]), rel=1e-4
    )
    assert r_squared == pytest.approx(float(results_sm.rsquared), rel=1e-5)


def test_predict_no_error_metrics_are_calculated_when_metrics_none() -> None:
    """Tests no error metrics are calculated when no metrics are provided to predict"""
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    target: Rsklpr = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)
    target.predict(x=x, metrics=None)
    assert target.residuals.shape == (0,)
    assert target.mean_square_error is None
    assert target.root_mean_square_error is None
    assert target.mean_abs_error is None
    assert target.bias_error is None
    assert target.std_error is None
    assert target.r_squared.shape == (0,)
    assert target.mean_square_error is None


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d(), generate_quad_1d(), generate_sin_1d()],
)
@pytest.mark.parametrize(
    argnames="metrics",
    argvalues=[all_metrics, "all"],
)
def test_predict_error_metrics_expected_values(x: np.ndarray, y: np.ndarray, metrics: Union[str, List[str]]) -> None:
    """Tests all error metrics are calculated correctly."""
    size_neighborhood: int = 10
    target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood)
    target.fit(x=x, y=y)
    y_hat: np.ndarray = target.predict(x=x, metrics=metrics)
    assert target.residuals.shape == y_hat.shape
    np.testing.assert_allclose(target.residuals, y_hat - y)
    assert target.mean_square_error == pytest.approx(sm.tools.eval_measures.mse(y_hat, y))
    assert target.root_mean_square_error == pytest.approx(sm.tools.eval_measures.rmse(y_hat, y))
    assert target.mean_abs_error == pytest.approx(sm.tools.eval_measures.meanabs(y_hat, y))
    assert target.bias_error == pytest.approx(sm.tools.eval_measures.bias(y_hat, y))
    assert target.std_error == pytest.approx(sm.tools.eval_measures.stde(y_hat, y))

    i: int

    for i in range(x.shape[0]):
        weights: np.ndarray
        n_x_neighbors: np.ndarray
        indices: np.ndarray
        weights, indices, n_x_neighbors = target._calculate_weights(x[i], bw1_global=None, bw2_global=None)
        x_sm: np.ndarray = np.squeeze(x[indices])
        x_sm = sm.add_constant(x_sm)
        model_sm = sm.WLS(endog=np.squeeze(y[indices]), exog=x_sm, weights=np.squeeze(weights))
        results_sm: RegressionResults = model_sm.fit()
        assert target.r_squared[i] == pytest.approx(float(results_sm.rsquared))

    assert target.mean_r_squared is not None
    assert target.mean_r_squared == pytest.approx(float(target.r_squared.mean()))


def test_predict_error_metrics_expected_metrics_are_calculated() -> None:
    """
    Tests only the expected error metrics are calculated. When residuals is specified all global metrics expected to be
    lazily evaluated. When 'root_mean_square' is specified also 'mean_square' should become available due to it's use in
    calculation
    """
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    size_neighborhood: int = 10
    asserted_metric: str
    metric: str

    for asserted_metric in all_metrics:
        if asserted_metric in ("all", "residuals", "root_mean_square"):
            continue

        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood)
        target.fit(x=x, y=y)
        target.predict(x=x, metrics=asserted_metric)

        if asserted_metric in ("mean_square", "mean_abs", "bias", "std"):
            asserted_metric += "_error"

        for metric in all_metrics:
            if metric in ("all", "root_mean_square"):
                continue

            if metric in ("mean_square", "mean_abs", "bias", "std"):
                metric += "_error"

            actual: Optional[Union[float, np.ndarray]] = getattr(target, metric)

            if metric == asserted_metric:
                if metric in ("residuals", "r_squared"):
                    assert actual.shape[0] == y.shape[0]  # type: ignore[union-attr]
                else:
                    assert isinstance(actual, float)
            else:
                if metric in ("residuals", "r_squared"):
                    assert actual.shape[0] == 0  # type: ignore[union-attr]
                else:
                    assert actual is None

    target = Rsklpr(size_neighborhood=size_neighborhood)
    target.fit(x=x, y=y)
    target.predict(x=x, metrics="residuals")
    assert target.residuals.shape == y.shape
    assert isinstance(target.mean_square_error, float)
    assert isinstance(target.root_mean_square_error, float)
    assert isinstance(target.mean_abs_error, float)
    assert isinstance(target.bias_error, float)
    assert isinstance(target.std_error, float)
    assert target.r_squared.shape[0] == 0
    assert target.mean_r_squared is None

    target = Rsklpr(size_neighborhood=size_neighborhood)
    target.fit(x=x, y=y)
    target.predict(x=x, metrics="root_mean_square")
    assert target.residuals.shape[0] == 0
    assert isinstance(target.mean_square_error, float)
    assert isinstance(target.root_mean_square_error, float)
    assert target.mean_abs_error is None
    assert target.bias_error is None
    assert target.std_error is None
    assert target.r_squared.shape[0] == 0
    assert target.mean_r_squared is None


def test_weighted_local_regression_r_squared_is_none(mocker: MockerFixture) -> None:
    """Tests that r_squared is not calculated nor returned when calculate_r_squared is False"""
    x: np.ndarray = generate_linear_1d().reshape((-1, 1))
    y: np.ndarray = generate_linear_1d()
    weights: np.ndarray = rng.uniform(low=0.01, high=1, size=50).reshape((-1, 1))
    r_squared_mock: MagicMock = mocker.patch("rsklpr.rsklpr._r_squared", return_value=1.0)
    actual: float
    r_squared: Optional[float]

    actual, r_squared = _weighted_local_regression(
        x_0=x.mean(), x=x, y=y, weights=weights, degree=1, calculate_r_squared=False
    )

    assert r_squared is None
    assert r_squared_mock.call_count == 0


def test_r_squared_raises_when_y_and_weights_different_shapes() -> None:
    """Tests an error is raised when y and weights are of different shapes"""
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    weights: np.ndarray = rng.uniform(low=0.01, high=1, size=50)

    with pytest.raises(ValueError) as exc_info:
        _r_squared(beta=np.asarray([12.3, 100.0]), x_w=x, y_w=y, y=y, weights=weights)
        assert "y and weights must have the same shape" in str(exc_info.value)


def test_estimate_bootstrap_expected_number_of_bootstrap_steps(mocker: MockerFixture) -> None:
    """
    Tests the expected nuber of estimate calls are performed and the arguments provided are correct.
    """
    x: np.ndarray = generate_linear_1d().reshape((-1, 1))
    y: np.ndarray = generate_linear_1d()
    num_bootstrap_resamples: int = 5
    side_effect: List[np.ndarray] = [i * np.ones(shape=y.shape[0]) for i in range(num_bootstrap_resamples)]

    estimate_mock: MagicMock = mocker.patch(
        target="rsklpr.rsklpr.Rsklpr._estimate",
        side_effect=side_effect,
    )

    target = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)
    target.predict_bootstrap(x=x, num_bootstrap_resamples=num_bootstrap_resamples)
    assert estimate_mock.call_count == 5

    for call in estimate_mock.call_args_list:
        np.testing.assert_array_equal(x, call.kwargs["x"])


def test_estimate_bootstrap_expected_values_are_returned(mocker: MockerFixture) -> None:
    """
    Tests the expected values are returned.
    """
    x: np.ndarray = generate_linear_1d().reshape((-1, 1))
    y: np.ndarray = generate_linear_1d()
    num_bootstrap_resamples: int = 5

    side_effect: List[np.ndarray] = [
        np.linspace(start=i * 2, stop=(i + 1) * 2, num=y.shape[0]) for i in range(num_bootstrap_resamples)
    ]

    estimate_mock: MagicMock = mocker.patch(
        target="rsklpr.rsklpr.Rsklpr._estimate",
        side_effect=side_effect,
    )

    target = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)
    actual_mean: np.ndarray
    actual_conf_low: np.ndarray
    actual_conf_high: np.ndarray
    actual_results: Optional[np.ndarray]

    actual_mean, actual_conf_low, actual_conf_high, actual_results = target.predict_bootstrap(
        x=x, num_bootstrap_resamples=num_bootstrap_resamples
    )

    expected: np.ndarray = np.asarray(side_effect).T
    np.testing.assert_allclose(actual_mean, expected.mean(axis=1))
    np.testing.assert_allclose(actual_conf_low, np.quantile(a=expected, q=0.025, axis=1))
    np.testing.assert_allclose(actual_conf_high, np.quantile(a=expected, q=0.975, axis=1))
    assert actual_results is None
    estimate_mock.reset_mock(side_effect=True)
    estimate_mock.side_effect = side_effect

    actual_mean, actual_conf_low, actual_conf_high, actual_results = target.predict_bootstrap(
        x=x, q_low=0.1, q_high=0.9, num_bootstrap_resamples=num_bootstrap_resamples, return_all_bootstraps=True
    )

    np.testing.assert_allclose(actual_mean, expected.mean(axis=1))
    np.testing.assert_allclose(actual_conf_low, np.quantile(a=expected, q=0.1, axis=1))
    np.testing.assert_allclose(actual_conf_high, np.quantile(a=expected, q=0.9, axis=1))
    np.testing.assert_allclose(actual_results, expected)  # type: ignore[arg-type]


def test_fit_raises_if_called_multiple_times() -> None:
    """
    Tests fit raises when called multiple times.
    """
    x: np.ndarray = generate_linear_1d().reshape((-1, 1))
    y: np.ndarray = generate_linear_1d()
    target = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)

    with pytest.raises(
        expected_exception=ValueError, match="Fit already called, use a new instance if you need to fit new data."
    ):
        target.fit(x=x, y=y)


@pytest.mark.parametrize(
    argnames="x, y, metric_x, metric_x_params, expected_metric_params, expected_p",
    argvalues=[
        (generate_linear_1d(), generate_linear_1d(), "euclidean", None, {}, 2.0),
        (generate_linear_1d(), generate_linear_1d(), "minkowski", {}, {}, 2.0),
        (generate_linear_1d(), generate_linear_1d(), "minkowski", {"p": 3}, {}, 3.0),
        # VI not provided
        (
            generate_linear_nd(dim=3),
            generate_quad_1d(),
            "mahalanobis",
            {},
            {},  # The VI value is added dynamically in the test
            2.0,
        ),
        # VI provided
        (
            generate_linear_nd(dim=3),
            generate_quad_1d(),
            "mahalanobis",
            {"VI": np.eye(3)},
            {"VI": np.eye(3)},
            2.0,
        ),
    ],
)
def test_get_metric_params_expected_values(
    x: np.ndarray,
    y: np.ndarray,
    metric_x: str,
    metric_x_params: Optional[Dict[str, Any]],
    expected_metric_params: Dict[str, Any],
    expected_p: float,
) -> None:
    """Tests get_metric_params returns the expected values"""
    target: Rsklpr = Rsklpr(size_neighborhood=10, metric_x=metric_x, metric_x_params=metric_x_params)
    target.fit(x=x, y=y)
    actual_metric_params: Dict[str, Any]
    actual_p: float
    actual_metric_params, actual_p = target._get_metric_params()
    assert "p" not in actual_metric_params
    assert actual_p == expected_p

    if metric_x != "mahalanobis":
        assert actual_metric_params == expected_metric_params
    else:
        if "VI" not in expected_metric_params:
            expected_metric_params["VI"] = np.linalg.inv(np.cov(m=x, rowvar=False))

        np.testing.assert_allclose(actual_metric_params["VI"], expected_metric_params["VI"])

    assert actual_p == expected_p


@pytest.mark.parametrize(
    argnames=["x", "y", "metric_x", "metric_x_params", "expected_exception"],
    argvalues=[
        # Invalid metric_x parameter
        (generate_linear_1d(), generate_linear_1d(), "invalid_metric", None, ValueError),
        # Invalid metric_param
        (generate_linear_1d(), generate_linear_1d(), "invalid_metric", {"bla": 5}, ValueError),
        # Invalid type for metric_x_params
        (generate_linear_1d(), generate_linear_1d(), "minkowski", "invalid_params", AttributeError),
    ],
)
def test_get_metric_params_error_cases(
    x: np.ndarray,
    y: np.ndarray,
    metric_x: str,
    metric_x_params: Union[str, Optional[Dict[str, Any]]],
    expected_exception: Type[Exception],
) -> None:
    """Tests the expected exception raised when incorrect values are provided"""
    target: Rsklpr = Rsklpr(
        size_neighborhood=10, metric_x=metric_x, metric_x_params=metric_x_params  # type: ignore[arg-type]
    )

    with pytest.raises(expected_exception):
        target.fit(x=x, y=y)
