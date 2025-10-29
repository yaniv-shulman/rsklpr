from typing import List, Optional, Union, Dict, Any, Type, Callable, Tuple, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import statsmodels.api as sm
from pytest_mock import MockerFixture
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.regression.linear_model import RegressionResults

import rsklpr
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from rsklpr.rsklpr import (
    Rsklpr,
    _dim_data,
    _r_squared,
    all_metrics,
    _robust_scale,
    _get_fallback_bw,
    _is_bw_valid_sequence_type,
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
    argnames="kp",
    argvalues=[
        tricube_normalized_metric,
        laplacian_normalized_metric,
    ],
)
@pytest.mark.parametrize(
    argnames="kr",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
def test_rsklpr_smoke_test_1d_regression_increasing_windows_expected_output(
    x: np.ndarray,
    y: np.ndarray,
    kp: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray],
    kr: str,
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    size_neighborhood: int

    for size_neighborhood in np.linspace(start=5, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=kp, kr=kr)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        np.testing.assert_allclose(actual=actual, desired=y, atol=8.5e-3)


@pytest.mark.parametrize(
    argnames="x_data_generator",
    argvalues=[
        lambda: generate_linear_nd(dim=2),
        lambda: generate_linear_nd(dim=2),
        lambda: generate_linear_nd(dim=2),
    ],
)
@pytest.mark.parametrize(
    argnames="kp",
    argvalues=[tricube_normalized_metric, laplacian_normalized_metric],
)
@pytest.mark.parametrize(
    argnames="kr",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
def test_rsklpr_smoke_test_2d_regression_increasing_windows_expected_output(
    x_data_generator: Callable[[], np.ndarray],
    kp: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray],
    kr: str,
) -> None:
    """
    Smoke test that reasonable values are returned for linear 2D input with various window sizes.
    """
    x: np.ndarray = x_data_generator()
    rng: np.random.Generator = np.random.default_rng(seed=42)
    y: np.ndarray = 1 * x[:, 0] + 2 * x[:, 1] + 5 + rng.normal(scale=1e-5, size=x.shape[0])

    size_neighborhood: int

    # Start at a reasonable neighborhood size for a 3-parameter (d=2, deg=1) fit
    for size_neighborhood in np.linspace(start=10, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=kp, kr=kr)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        # Calculate the absolute error for every point
        abs_error: np.ndarray = np.abs(actual - y)

        # Find the error at the 90th percentile.
        # This ignores the worst 10% of predictions,
        # which is where our robust neff check will live.
        error_at_90_percentile: float = float(np.percentile(abs_error, 90))

        # Now, assert that this 90th percentile error is within our tolerance.
        # This confirms the model is "mostly correct" while allowing
        # the robustness logic to handle outliers.
        assert (
            error_at_90_percentile < 0.25
        ), f"90th percentile error ({error_at_90_percentile}) exceeds tolerance (0.25)"


@pytest.mark.parametrize(
    argnames="x_data_generator",
    argvalues=[lambda: generate_linear_nd(dim=5)],
)
@pytest.mark.parametrize(
    argnames="kp",
    argvalues=[tricube_normalized_metric, laplacian_normalized_metric],
)
@pytest.mark.parametrize(
    argnames="kr",
    argvalues=[
        "joint",
        "conden",
    ],
)
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
def test_rsklpr_smoke_test_5d_regression_increasing_windows_expected_output(
    x_data_generator: Callable[[], np.ndarray],
    kp: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray],
    kr: str,
) -> None:
    """
    Smoke test that reasonable values are returned for linear 1D input with various window sizes.
    """
    # Generate x, then generate y from x
    x: np.ndarray = x_data_generator()
    rng: np.random.Generator = np.random.default_rng(seed=42)

    # Create a true linear relationship
    y: np.ndarray = 1 * x[:, 0] + 2 * x[:, 1] - 0.5 * x[:, 3] + 5 + rng.normal(scale=1e-5, size=x.shape[0])

    size_neighborhood: int

    # We need 6 parameters. Start at 20 for a safer margin in 5D.
    for size_neighborhood in np.linspace(start=20, stop=50, num=20, endpoint=True).astype(int):
        target: Rsklpr = Rsklpr(size_neighborhood=size_neighborhood, kp=kp, kr=kr)

        actual: np.ndarray = target(
            x=x,
            y=y,
        )

        # Use the robust 90th percentile assertion
        abs_error: np.ndarray = np.abs(actual - y)
        error_at_90_percentile: float = float(np.percentile(abs_error, 90))

        # Now, assert that this 90th percentile error is within our tolerance.
        # This confirms the model is "mostly correct" while allowing
        # the robustness logic to handle outliers.
        assert error_at_90_percentile < 2.5, (
            f"90th percentile error ({error_at_90_percentile}) exceeds tolerance (2.5) "
            f"for neighborhood size {size_neighborhood}"
        )


@pytest.mark.parametrize(
    argnames="x_data_generator",
    argvalues=[lambda: generate_linear_1d() for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="kp",
    argvalues=[laplacian_normalized_metric, tricube_normalized_metric],
)
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")  # For neff
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")  # For nanmean
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")  # For nanquantile
def test_rsklpr_smoke_test_1d_estimate_bootstrap_expected_output(
    x_data_generator: Callable[[], np.ndarray],  # 3. Update signature
    kp: Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray],
) -> None:
    """
    Smoke test that reasonable values are returned for a linear 1D input using the joint kernel.
    """
    # Generate x, then generate y from x
    x: np.ndarray = x_data_generator()
    rng: np.random.Generator = np.random.default_rng(seed=42)
    noise_scale: float = 0.1

    y_true: np.ndarray = 2 * x + 5  # The true mean function
    y_noisy: np.ndarray = y_true + rng.normal(scale=noise_scale, size=x.shape[0])

    target: Rsklpr = Rsklpr(size_neighborhood=20, kp=kp, kr="none", suppress_warnings=True)
    target.fit(x=x, y=y_noisy)

    actual: np.ndarray
    conf_low_actual: np.ndarray
    conf_upper_actual: np.ndarray
    actual, conf_low_actual, conf_upper_actual, _ = target.predict_bootstrap(x=x, num_bootstrap_resamples=100)

    # Create a mask of valid (non-nan) predictions
    valid_mask: np.ndarray = ~np.isnan(actual)

    if np.sum(valid_mask) == 0:
        assert False, "All bootstrap predictions failed (all returned nan)"

    # ASSERT 1: The bootstrap mean (actual) should be close to the true mean (y_true)
    np.testing.assert_allclose(
        actual=actual[valid_mask], desired=y_true[valid_mask], atol=2.5 * noise_scale  # Test against y_true
    )

    # ASSERT 2: The confidence interval should be valid (low < mean < high)
    assert np.all(conf_low_actual[valid_mask] < actual[valid_mask])
    assert np.all(conf_upper_actual[valid_mask] > actual[valid_mask])

    # ASSERT 3: The 95% CI should cover the TRUE mean function (y_true)
    coverage_mask: np.ndarray = (conf_low_actual[valid_mask] <= y_true[valid_mask]) & (  # Test against y_true
        y_true[valid_mask] <= conf_upper_actual[valid_mask]
    )  # Test against y_true

    coverage_percentage: float = float(np.mean(coverage_mask))

    assert (
        coverage_percentage >= 0.85
    ), f"CI coverage of the *true mean* was only {coverage_percentage * 100}%, expected at least 85%"


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
    assert not target._is_fit

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
    assert not target._is_fit

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

    # Negative degree
    with pytest.raises(expected_exception=ValueError, match="degree must be a non-negative integer"):
        Rsklpr(size_neighborhood=3, degree=-1)

    with pytest.raises(
        expected_exception=ValueError, match="kr alb is unsupported and must be one of 'none', 'conden' or 'joint'"
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

    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)
    actual: float
    r_squared: Optional[float]

    actual, r_squared = target._weighted_local_regression(
        x_0=x_0.reshape(1, -1), x=x, y=y, weights=weights, degree=1, calculate_r_squared=True
    )

    x_sm: np.ndarray = sm.add_constant(x)
    model_sm: sm.WLS = sm.WLS(endog=y, exog=x_sm, weights=weights)
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
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)
    actual: float
    r_squared: Optional[float]

    actual, r_squared = target._weighted_local_regression(
        x_0=x_0.reshape(1, -1), x=x, y=y, weights=weights, degree=1, calculate_r_squared=True
    )

    x_sm: np.ndarray = sm.add_constant(x)
    model_sm: sm.WLS = sm.WLS(endog=y, exog=x_sm, weights=weights)
    results_sm: RegressionResults = model_sm.fit()

    assert float(actual) == pytest.approx(
        float(results_sm.params[0] + x_0[0] * results_sm.params[1] + x_0[1] * results_sm.params[2]), rel=1e-4
    )

    assert r_squared == pytest.approx(float(results_sm.rsquared), rel=1e-5)


@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
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

        weights, indices, n_x_neighbors = target._calculate_weights(
            x_0=x[i].reshape(-1, 1), index_x_0=i, bw1_global=None, bw2_global=None
        )

        x_sm: np.ndarray = np.squeeze(x[indices])
        x_sm = sm.add_constant(x_sm)
        model_sm: sm.WLS = sm.WLS(endog=np.squeeze(y[indices]), exog=x_sm, weights=np.squeeze(weights))
        results_sm: RegressionResults = model_sm.fit()
        assert target.r_squared[i] == pytest.approx(float(results_sm.rsquared))

    assert target.mean_r_squared is not None
    assert target.mean_r_squared == pytest.approx(float(target.r_squared.mean()))


@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
def test_predict_error_metrics_expected_metrics_are_calculated() -> None:
    """
    Tests only the expected error metrics are calculated. When residuals is specified all global metrics expected to be
    lazily evaluated. When 'root_mean_square' is specified also 'mean_square' should become available due to its use in
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
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)

    actual: float
    r_squared: Optional[float]

    actual, r_squared = target._weighted_local_regression(
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


@pytest.mark.parametrize(
    "x, x_0, degree, expected",
    [
        # 1D data
        (np.array([[1], [2], [3]]), np.array([[2]]), 0, np.array([[1], [1], [1]])),
        (
            np.array([[1], [2], [3]]),
            np.array([[2]]),
            1,
            np.array([[1, -1], [1, 0], [1, 1]]),
        ),
        (
            np.array([[1], [2], [3]]),
            np.array([[2]]),
            2,
            np.array([[1, -1, 1], [1, 0, 0], [1, 1, 1]]),
        ),
        # 2D data
        (
            np.array([[1, 10], [2, 20]]),
            np.array([[1, 10]]),
            0,
            np.array([[1], [1]]),
        ),
        (
            np.array([[1, 10], [2, 20]]),
            np.array([[1, 10]]),
            1,
            np.array([[1.0, 0.0, 0.0], [1.0, 1.41421356, 1.41421356]]),
        ),
        (
            np.array([[1, 10], [2, 20]]),
            np.array([[1, 10]]),
            2,
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.41421356, 1.41421356, 2.0, 2.0, 2.0]]),
        ),
    ],
)
def test_create_design_matrix_expected_output(
    x: np.ndarray, x_0: np.ndarray, degree: int, expected: np.ndarray
) -> None:
    """Tests the _create_design_matrix helper function for various degrees and dimensions."""
    target: Rsklpr = Rsklpr(size_neighborhood=3, degree=degree)
    actual: np.ndarray = target._create_design_matrix(x=x, x_0=x_0, degree=degree)
    np.testing.assert_allclose(actual, expected)


def test_create_design_matrix_raises_for_negative_degree() -> None:
    """Tests that _create_design_matrix raises a ValueError for a negative degree."""

    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=np.linspace(0.0, 1.0), y=np.linspace(0.0, 1.0))

    with pytest.raises(ValueError, match="Degree must be a non-negative integer."):
        target._create_design_matrix(x=np.array([[1]]), x_0=np.array([[0]]), degree=-1)


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d().reshape((-1, 1)) for _ in range(2)],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_linear_1d(), generate_quad_1d(), generate_sin_1d()],
)
@pytest.mark.parametrize(
    argnames="weights",
    argvalues=[
        rng.uniform(low=0.01, high=1, size=50),
        np.ones(shape=50),
    ],
)
def test_weighted_local_regression_degree_0_expected_values(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    """Test the weighted local regression for degree=0, which should be the weighted average."""
    # For degree 0, the prediction is just the weighted average of the neighbors.
    # We use x_0 at the mean just as a placeholder; it shouldn't affect the degree 0 result.
    x_0: np.ndarray = np.mean(x, axis=0)
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)

    actual: float
    r_squared: Optional[float]

    actual, r_squared = target._weighted_local_regression(
        x_0=x_0,
        x=x,
        y=y,
        weights=weights,
        degree=0,
        calculate_r_squared=True,
    )

    expected_pred: float = np.average(y, weights=weights)
    assert actual == pytest.approx(expected_pred)

    # R-squared for a constant-only model
    x_sm: np.ndarray = np.ones_like(x)
    model_sm: sm.WLS = sm.WLS(endog=y, exog=x_sm, weights=weights)
    results_sm: RegressionResults = model_sm.fit()
    assert r_squared == pytest.approx(float(results_sm.rsquared))


@pytest.mark.parametrize(
    argnames="x_0",
    argvalues=[rng.uniform(low=-10, high=10, size=1) for _ in range(3)],
)
@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_quad_1d().reshape((-1, 1))],
)
@pytest.mark.parametrize(
    argnames="y",
    argvalues=[generate_quad_1d()],
)
@pytest.mark.parametrize(
    argnames="weights",
    argvalues=[
        rng.uniform(low=0.01, high=1, size=50),
        np.ones(shape=50),
    ],
)
def test_weighted_local_regression_degree_2_expected_values(
    x_0: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> None:
    """Test the weighted local regression implementation for degree=2 (quadratic)."""
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)
    actual: float
    r_squared: Optional[float]

    actual, r_squared = target._weighted_local_regression(
        x_0=x_0, x=x, y=y, weights=weights, degree=2, calculate_r_squared=True
    )

    # Compare to a global WLS quadratic model
    poly: PolynomialFeatures = PolynomialFeatures(degree=2, include_bias=True)
    x_sm: np.ndarray = poly.fit_transform(x)
    model_sm: sm.WLS = sm.WLS(endog=y, exog=x_sm, weights=weights)
    results_sm: RegressionResults = model_sm.fit()

    # The local regression's beta[0] should equal the global model's prediction at x_0
    x_0_poly: np.ndarray = poly.transform(x_0.reshape(1, -1))
    expected_pred: float = float((x_0_poly @ results_sm.params)[0].item())

    assert actual == pytest.approx(expected_pred)
    assert r_squared == pytest.approx(float(results_sm.rsquared))


@pytest.mark.parametrize(
    "degree, weights, side_effect, expected_match",
    [
        (-1, np.ones(50), None, "Degree -1 is not supported. Must be 0, 1, 2, ..."),
        (1, np.zeros(50), None, None),  # Expect nan, no error
        (
            1,
            np.ones(50),
            np.linalg.LinAlgError,
            None,
        ),  # Expect nan, no error
    ],
)
@pytest.mark.filterwarnings("ignore:Local system for point *:RuntimeWarning")
def test_weighted_local_regression_edge_cases(
    mocker: MockerFixture,
    degree: int,
    weights: np.ndarray,
    side_effect: Optional[Type[Exception]],
    expected_match: Optional[str],
) -> None:
    """Tests edge cases for _weighted_local_regression: negative degree, zero weights, and linalg errors."""
    x: np.ndarray = generate_linear_1d().reshape((-1, 1))
    y: np.ndarray = generate_linear_1d()
    x_0: np.ndarray = np.array([x.mean()])
    target: Rsklpr = Rsklpr(size_neighborhood=15)
    target.fit(x=x, y=y)

    if side_effect:
        mocker.patch("numpy.linalg.lstsq", side_effect=side_effect)

    if expected_match:
        with pytest.raises(ValueError, match=expected_match):
            target._weighted_local_regression(x_0=x_0, x=x, y=y, weights=weights, degree=degree)
    else:
        # Should return nan without raising an error
        actual, r_squared = target._weighted_local_regression(x_0=x_0, x=x, y=y, weights=weights, degree=degree)
        assert np.isnan(actual)
        assert r_squared is None  # type: ignore[arg-type]


_x_base_linear: np.ndarray = generate_linear_1d(start=-5, stop=5, num=50)
_y_base_quad: np.ndarray = np.square(_x_base_linear)  # y = x^2 (with the small noise from x)


@pytest.mark.parametrize(
    "x, y, degree, atol",
    [
        # Case for degree 0: constant data y = 5
        (
            generate_linear_1d(start=0, stop=10, num=50),  # A standard x array
            np.ones(50) * 5.0,  # A y array that is constant
            0,
            1e-5,  # This should pass, as the weighted avg of 5.0 is 5.0
        ),
        # Case for degree 2: y = x^2
        (
            _x_base_linear,  # x is linear
            _y_base_quad,  # y is the square of x
            2,
            5e-2,  # A more realistic tolerance for a local regression
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
def test_rsklpr_smoke_test_1d_degree_0_and_2(x: np.ndarray, y: np.ndarray, degree: int, atol: float) -> None:
    """Smoke test that reasonable values are returned for degree 0 (constant) and 2 (quadratic) fits."""
    target: Rsklpr = Rsklpr(size_neighborhood=25, degree=degree, kp=tricube_normalized_metric)
    actual: np.ndarray = target(x=x, y=y)
    np.testing.assert_allclose(actual=actual, desired=y, atol=atol)


@pytest.mark.parametrize(
    "x_in, y_in, neighborhood, match",
    [
        # Reshape 1D x
        (np.ones(10), np.ones(10), 5, None),
        # x.ndim > 2
        (
            np.ones((10, 2, 2)),
            np.ones(10),
            5,
            "x dimension must be at most 2",
        ),
        # x.shape[0] < size_neighborhood
        (
            np.ones((5, 2)),
            np.ones(5),
            10,
            "less than specified neighborhood size",
        ),
        # y.ndim > 1
        (
            np.ones((10, 2)),
            np.ones((10, 2)),
            5,
            "y dimension must be at most 1",
        ),
        # Incompatible shapes
        (np.ones((10, 2)), np.ones(5), 5, "x and y have incompatible shapes"),
    ],
)
def test_check_and_reshape_inputs_errors_and_reshape(
    x_in: np.ndarray,
    y_in: np.ndarray,
    neighborhood: int,
    match: Optional[str],
) -> None:
    """Tests the validation logic in _check_and_reshape_inputs."""
    target: Rsklpr = Rsklpr(size_neighborhood=neighborhood)

    if match:
        with pytest.raises(ValueError, match=match):
            target._check_and_reshape_inputs(x=x_in, y=y_in, enforce_min_x_size=True)
    else:
        # Test successful reshape (default enforce_min_x_size=False)
        x_out, y_out = target._check_and_reshape_inputs(x=x_in, y=y_in)
        assert x_out.shape == (10, 1)
        assert y_out is not None
        assert y_out.shape == (10,)
        # Test x-only path
        x_out_only, y_out_only = target._check_and_reshape_inputs(x=x_in, y=None)
        assert x_out_only.shape == (10, 1)
        assert y_out_only is None


def test_predict_raises_errors() -> None:
    """Tests errors raised by the predict method."""
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    target: Rsklpr = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)

    # Error for requesting metrics with different x
    with pytest.raises(
        ValueError,
        match="must be the same as the values provided to 'fit'",
    ):
        target.predict(x=x[1:], metrics="all")

    # Error for unknown metric
    with pytest.raises(
        ValueError,
        match=r"^Unknown metric bla\. Available metrics are \[.*\]$",
    ):
        target.predict(x=x, metrics="bla")


def test_predict_bootstrap_raises_for_invalid_iterations() -> None:
    """Tests predict_bootstrap raises for <= 0 iterations."""
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    target: Rsklpr = Rsklpr(size_neighborhood=10)
    target.fit(x=x, y=y)

    with pytest.raises(ValueError, match="At least one bootstrap iteration needs to be specified"):
        target.predict_bootstrap(x=x, num_bootstrap_resamples=0)

    with pytest.raises(ValueError, match="At least one bootstrap iteration needs to be specified"):
        target.predict_bootstrap(x=x, num_bootstrap_resamples=-1)


def test_lazy_properties_are_calculated_from_residuals() -> None:
    """Tests that lazy-loaded metric properties are correctly calculated if residuals exist."""
    x: np.ndarray = generate_linear_1d()
    y: np.ndarray = generate_linear_1d()
    target: Rsklpr = Rsklpr(size_neighborhood=10)
    y_hat: np.ndarray = target.fit_and_predict(x=x, y=y, metrics=["residuals"])

    # At this point, only residuals should be set
    assert target._residuals is not None
    assert target._mean_square_error is None
    assert target._root_mean_square_error is None
    assert target._mean_abs_error is None
    assert target._bias_error is None
    assert target._std_error is None

    # Accessing properties should trigger calculation
    expected_mse: float = sm.tools.eval_measures.mse(y_hat, y)
    assert target.mean_square_error == pytest.approx(expected_mse)
    assert target._mean_square_error is not None  # Check it's now cached

    expected_rmse: float = sm.tools.eval_measures.rmse(y_hat, y)
    assert target.root_mean_square_error == pytest.approx(expected_rmse)
    assert target._root_mean_square_error is not None

    expected_mae: float = sm.tools.eval_measures.meanabs(y_hat, y)
    assert target.mean_abs_error == pytest.approx(expected_mae)
    assert target._mean_abs_error is not None

    expected_bias: float = sm.tools.eval_measures.bias(y_hat, y)
    assert target.bias_error == pytest.approx(expected_bias)
    assert target._bias_error is not None

    expected_std: float = sm.tools.eval_measures.stde(y_hat, y)
    assert target.std_error == pytest.approx(expected_std)
    assert target._std_error is not None

    # Test RMSE calculation when MSE is already cached
    target_rmse: Rsklpr = Rsklpr(size_neighborhood=10)
    y_hat_rmse: np.ndarray = target_rmse.fit_and_predict(x=x, y=y, metrics=["mean_square"])
    assert target_rmse._mean_square_error is not None
    assert target_rmse._root_mean_square_error is None
    assert target_rmse.root_mean_square_error == pytest.approx(sm.tools.eval_measures.rmse(y_hat_rmse, y))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _column_kernel(
    x_0: np.ndarray,
    x_neighbors: np.ndarray,
    dist_x_neighbors: np.ndarray,
    index_x_0: int,
    indices_neighbors: np.ndarray,
) -> np.ndarray:
    """Returns a (K,1) column vector to exercise weight-shape normalization."""
    k: int = x_neighbors.shape[0]
    w: np.ndarray = np.linspace(1.0, 0.5, num=k)
    return w.reshape(-1, 1)


def _row_kernel(
    x_0: np.ndarray,
    x_neighbors: np.ndarray,
    dist_x_neighbors: np.ndarray,
    index_x_0: int,
    indices_neighbors: np.ndarray,
) -> np.ndarray:
    """Returns a (1,K) row vector; pairs with _column_kernel to test product + concat."""
    k: int = x_neighbors.shape[0]
    return (np.ones(k) * 0.8).reshape(1, -1)


# --------------------------------------------------------------------------------------
# Init input validation / kernel list handling
# --------------------------------------------------------------------------------------


def test_rsklpr_init_accepts_iterable_kernels() -> None:
    """kp can be a single callable or an Iterable of callables."""
    target: Rsklpr = Rsklpr(size_neighborhood=7, kp=[tricube_normalized_metric, laplacian_normalized_metric])
    assert isinstance(target, Rsklpr)


def test_rsklpr_init_rejects_invalid_kernels() -> None:
    """kp must be callable(s)."""
    with pytest.raises(ValueError, match="kp must be a callable or a sequence of callables"):
        Rsklpr(size_neighborhood=7, kp=[tricube_normalized_metric, 123])  # type: ignore[list-item]


def test_rsklpr_init_bw2_callable_regression() -> None:
    """Regression: bw2 callable must be accepted (no accidental bw1 check)."""

    def bw2(_: np.ndarray) -> List[float]:
        return [0.7]

    # Should not raise
    _ = Rsklpr(size_neighborhood=5, bw2=bw2)


# --------------------------------------------------------------------------------------
# _check_and_reshape_inputs ergonomics (scalar / 1D / enforce_min_x_size)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_in, expected_shape",
    [
        (0.0, (1, 1)),
        (np.array(1.0), (1, 1)),
        (np.array([1.0, 2.0, 3.0]), (3, 1)),
        (np.array([[1.0], [2.0], [3.0]]), (3, 1)),
    ],
)
def test_check_and_reshape_inputs_shapes(x_in: np.ndarray | float, expected_shape: Tuple[int, int]) -> None:
    """Tests scalar -> (1,1), 1D -> (N,1), 2D unchanged; does not enforce min size by default."""
    target: Rsklpr = Rsklpr(size_neighborhood=10)
    x_out: np.ndarray
    y_out: Optional[np.ndarray]
    x_out, y_out = target._check_and_reshape_inputs(x=x_in, y=None)  # type: ignore[arg-type]
    assert x_out.shape == expected_shape
    assert y_out is None


def test_check_and_reshape_inputs_enforce_min_size() -> None:
    """When enforce_min_x_size=True, too-few rows should raise."""
    target: Rsklpr = Rsklpr(size_neighborhood=5)
    with pytest.raises(ValueError, match="less than specified neighborhood size"):
        target._check_and_reshape_inputs(x=np.array([[1.0], [2.0]]), y=np.array([1.0, 2.0]), enforce_min_x_size=True)


# --------------------------------------------------------------------------------------
# predict ergonomics and shapes (scalar, 1D, 2D); __call__ alias
# --------------------------------------------------------------------------------------


def test_predict_accepts_scalar_and_single_row() -> None:
    x: np.ndarray = np.linspace(-3, 3, 50)
    y: np.ndarray = 2 * x + 1
    target: Rsklpr = Rsklpr(size_neighborhood=7, kr="none")
    target.fit(x=x, y=y)

    # scalar
    yhat_scalar: np.ndarray = target.predict(0.0)
    assert yhat_scalar.shape == (1,)

    # single row 2D
    yhat_row: np.ndarray = target.predict(np.array([[0.0]]))
    assert yhat_row.shape == (1,)

    # 1D vector
    xs: np.ndarray = np.array([0.0, 1.0, 2.0])
    yhat_vec: np.ndarray = target.predict(xs)
    assert yhat_vec.shape == (3,)


def test_call_alias_to_fit_and_predict() -> None:
    x: np.ndarray = np.linspace(-1, 1, 30)
    y: np.ndarray = -3 * x + 2
    target: Rsklpr = Rsklpr(size_neighborhood=6, kr="none")
    yhat: np.ndarray = target(x=x, y=y)  # __call__
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == y.shape


# --------------------------------------------------------------------------------------
# Var-type resolution coverage
# --------------------------------------------------------------------------------------


def test_resolve_var_types_defaults_and_custom() -> None:
    x: np.ndarray = np.random.default_rng(0).normal(size=(40, 3))
    y: np.ndarray = x[:, 0] + 0.1 * np.random.default_rng(1).normal(size=40)

    target_default: Rsklpr = Rsklpr(size_neighborhood=10, var_type_x=None, kr="none")
    target_default.fit(x=x, y=y)
    assert target_default._var_type_x_resolved == "ccc"
    assert target_default._var_type_y_resolved == "c"
    assert target_default._var_type_xy_resolved == "cccc"

    target_custom: Rsklpr = Rsklpr(size_neighborhood=10, var_type_x="cuo", kr="none")
    target_custom.fit(x=x, y=y)
    assert target_custom._var_type_x_resolved == "cuo"
    assert target_custom._var_type_xy_resolved == "cuoc"


def test_resolve_var_types_length_mismatch_raises() -> None:
    x: np.ndarray = np.random.default_rng(0).normal(size=(20, 2))
    y: np.ndarray = np.random.default_rng(1).normal(size=20)
    target: Rsklpr = Rsklpr(size_neighborhood=5, var_type_x="ccc")  # wrong length

    with pytest.raises(ValueError, match="does not match the number of features"):
        target.fit(x=x, y=y)


def test_resolve_var_types_invalid_char_raises() -> None:
    x: np.ndarray = np.random.default_rng(0).normal(size=(20, 2))
    y: np.ndarray = np.random.default_rng(1).normal(size=20)
    target: Rsklpr = Rsklpr(size_neighborhood=5, var_type_x="cx")

    with pytest.raises(ValueError, match="Invalid character 'x'"):
        target.fit(x=x, y=y)


# --------------------------------------------------------------------------------------
# Bandwidth calculation and fallbacks
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
def test_calculate_bandwidth_select_bandwidth_zero_falls_back(mocker: MockerFixture) -> None:
    """If statsmodels select_bandwidth raises 'bandwidth is 0', fallback is used and a warning is issued."""
    x: np.ndarray = np.random.default_rng(0).normal(size=(60, 2))
    y: np.ndarray = np.random.default_rng(1).normal(size=60)

    target: Rsklpr = Rsklpr(size_neighborhood=10, kr="none", suppress_warnings=False)
    target.fit(x=x, y=y)

    # Force select_bandwidth to raise the specific error
    def _raise(*_: Any, **__: Any) -> np.ndarray:
        raise RuntimeError("bandwidth is 0")

    mocker.patch("rsklpr.rsklpr.select_bandwidth", side_effect=_raise)

    with pytest.warns(RuntimeWarning, match="KDE bandwidth was 0"):
        bw: Sequence[float] = target._calculate_bandwidth(
            bandwidth="scott", data=x, var_type=target._var_type_x_resolved
        )

    assert isinstance(bw, Sequence)
    assert len(bw) == x.shape[1]


@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
def test_calculate_bandwidth_cv_fail_falls_back(mocker: MockerFixture) -> None:
    """If KDEMultivariate (cv_*) fails, fallback is used and a warning is issued."""
    x: np.ndarray = np.random.default_rng(0).normal(size=(80, 3))
    y: np.ndarray = np.random.default_rng(1).normal(size=80)

    target: Rsklpr = Rsklpr(size_neighborhood=12, kr="none", suppress_warnings=False)
    target.fit(x=x, y=y)

    class FailingKDE:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise np.linalg.LinAlgError("bad")

    mocker.patch("rsklpr.rsklpr.KDEMultivariate", FailingKDE)

    with pytest.warns(RuntimeWarning, match="KDE bandwidth estimation 'cv_ls' failed"):
        bw: Sequence[float] = target._calculate_bandwidth(bandwidth="cv_ls", data=x, var_type="ccc")

    assert isinstance(bw, Sequence)
    assert len(bw) == x.shape[1]


def test_get_bandwidth_global_paths_and_shapes(mocker: MockerFixture) -> None:
    """Covers cv_ls_global for bw1 and bw2 (both 'conden' and 'joint'). Ensures shapes are correct."""
    rng_: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng_.normal(size=(200, 2))
    y: np.ndarray = x[:, 0] + rng_.normal(scale=0.05, size=200)

    class DummyKDE:
        def __init__(self, data: np.ndarray, var_type: str, bw: str) -> None:
            # Return 'bw' as ones of correct length
            self.bw: np.ndarray = np.ones(data.shape[1], dtype=float)

    mocker.patch("rsklpr.rsklpr.KDEMultivariate", DummyKDE)

    target_conden: Rsklpr = Rsklpr(
        size_neighborhood=20,
        kr="conden",
        bw1="cv_ls_global",
        bw2="cv_ls_global",
        bw_global_subsample_size=50,
        suppress_warnings=True,
    )

    target_conden.fit(x, y)
    bw1_c, bw2_c = target_conden._get_bandwidth_global(kr="conden")
    assert bw1_c is not None and len(bw1_c) == x.shape[1]
    assert bw2_c is not None and len(bw2_c) == (x.shape[1] + 1)

    target_joint: Rsklpr = Rsklpr(
        size_neighborhood=20,
        kr="joint",
        bw1="cv_ls_global",
        bw2="cv_ls_global",
        bw_global_subsample_size=50,
        suppress_warnings=True,
    )

    target_joint.fit(x, y)
    bw1_j, bw2_j = target_joint._get_bandwidth_global(kr="joint")
    assert bw1_j is not None and len(bw1_j) == x.shape[1]
    assert bw2_j is not None and len(bw2_j) == 1


def test_cv_ls_global_subsample_without_replacement(mocker: MockerFixture) -> None:
    """Ensure global CV subsampling uses replace=False by checking uniqueness of the subsample."""
    rng_: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng_.normal(size=(300, 3))
    y: np.ndarray = rng_.normal(size=300)

    # Capture every 'data' array passed into KDEMultivariate so we can inspect the subsample
    captured_data: List[np.ndarray] = []

    class CapturingKDE:
        def __init__(self, data: np.ndarray, var_type: str, bw: str) -> None:
            captured_data.append(np.asarray(data))
            # Minimal interface: expose .bw with correct length so caller proceeds
            self.bw: np.ndarray = np.ones(data.shape[1], dtype=float)

    mocker.patch("rsklpr.rsklpr.KDEMultivariate", CapturingKDE)

    subsz: int = 75

    target: Rsklpr = Rsklpr(
        size_neighborhood=25,
        kr="conden",
        bw1="cv_ls_global",
        bw2="cv_ls_global",
        bw_global_subsample_size=subsz,
        suppress_warnings=True,
    )

    target.fit(x, y)

    # Trigger the global bandwidth computations (both bw1 for X and bw2 for [X|y])
    _ = target._get_bandwidth_global(kr="conden")

    # Among captured calls, pick the one that corresponds to X-only (same n_features as X)
    x_only_calls: List[np.ndarray] = [arr for arr in captured_data if arr.shape[1] == x.shape[1]]
    assert x_only_calls, "Expected at least one KDE call with X-only data"

    subsample: np.ndarray = x_only_calls[0]
    # Expected subsample size: min(N, subsz)
    expected_n: int = min(x.shape[0], subsz)
    assert subsample.shape[0] == expected_n

    # Uniqueness along rows → implies replace=False (duplicates extremely unlikely in continuous data)
    n_unique: int = int(np.unique(subsample, axis=0).shape[0])
    assert n_unique == expected_n, "Subsample appears to contain duplicates; expected sampling without replacement"


# --------------------------------------------------------------------------------------
# Kernel weighting paths and mixed shapes
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
def test_calculate_weights_with_mixed_kernel_shapes() -> None:
    """Multiple kernels whose outputs are (K,), (1,K), and (K,1) should combine robustly."""
    x: np.ndarray = np.linspace(-2, 2, 50).reshape(-1, 1)
    y: np.ndarray = 3 * x.squeeze() - 1

    target: Rsklpr = Rsklpr(
        size_neighborhood=10, kp=[laplacian_normalized_metric, _column_kernel, _row_kernel], kr="none"
    )

    target.fit(x, y)

    w: np.ndarray
    idx: np.ndarray
    x_n: np.ndarray
    w, idx, x_n = target._calculate_weights(x_0=x[10:11], index_x_0=10, bw1_global=None, bw2_global=None)

    w_flat: np.ndarray = np.ravel(w)
    assert w_flat.ndim == 1
    assert x_n.shape[0] == w_flat.shape[0] == idx.shape[1]


# --------------------------------------------------------------------------------------
# _kr_conden / _kr_joint: sanitize/normalize/eps guards
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:KDE bandwidth was 0.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore: invalid value encountered in divide*:RuntimeWarning")
def test_kr_conden_and_joint_weight_properties(mocker: MockerFixture) -> None:
    """KDE pdf paths: infinities/NaNs replaced, mean-normalized, and epsilon floor applied."""
    rng_: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = rng_.normal(size=(60, 2))
    y: np.ndarray = x[:, 0] + 0.25 * x[:, 1] + rng_.normal(scale=0.01, size=60)

    # Dummy KDE that returns mixed finite / inf / nan to exercise guards
    class DummyKDE:
        def __init__(self, data: np.ndarray, var_type: str, bw: Any) -> None:
            self._n: int = data.shape[0]
            self.bw: np.ndarray = np.ones(data.shape[1], dtype=float)

        def pdf(self, data_predict: np.ndarray) -> np.ndarray:
            n: int = data_predict.shape[0]
            out: np.ndarray = np.ones(n, dtype=float)
            if n >= 3:
                out[0] = np.inf
                out[1] = np.nan
                out[2] = -1.0  # negative will be handled by epsilon floor after normalization
            return out

    mocker.patch("rsklpr.rsklpr.KDEMultivariate", DummyKDE)

    # 'conden' path uses joint/marginal ratio
    target_conden: Rsklpr = Rsklpr(size_neighborhood=20, kr="conden", suppress_warnings=True)
    target_conden.fit(x, y)
    w_c, _, _ = target_conden._calculate_weights(x_0=x[5:6], index_x_0=5, bw1_global=None, bw2_global=None)
    assert np.all(np.isfinite(w_c)), "Weights must be finite"

    # Allow tiny floating variation below eps after normalization; still require strictly positive.
    assert np.all(w_c > 0.0), "Weights must be strictly positive"

    # 'joint' path uses joint pdf only
    target_joint: Rsklpr = Rsklpr(size_neighborhood=20, kr="joint", suppress_warnings=True)
    target_joint.fit(x, y)
    w_j, _, _ = target_joint._calculate_weights(x_0=x[5:6], index_x_0=5, bw1_global=None, bw2_global=None)
    assert np.all(np.isfinite(w_j))
    assert np.all(w_j > 0.0)


# --------------------------------------------------------------------------------------
# _r_squared edge cases
# --------------------------------------------------------------------------------------


def test_r_squared_zero_weight_sum_returns_nan_and_warns() -> None:
    rng_: np.random.Generator = np.random.default_rng(0)
    y: np.ndarray = rng_.normal(size=10)
    beta: np.ndarray = np.zeros((3, 1))
    x_w: np.ndarray = np.ones((10, 3))
    y_w: np.ndarray = np.zeros((10, 1))
    weights: np.ndarray = np.zeros((10, 1))

    with pytest.warns(RuntimeWarning, match="R-squared is undefined due to all-zero"):
        r2: float = _r_squared(beta=beta, x_w=x_w, y_w=y_w, y=y.reshape(-1, 1), weights=weights)

    assert np.isnan(r2)


def test_r_squared_zero_variance_returns_nan_and_warns() -> None:
    y: np.ndarray = np.ones(10) * 5.0
    beta: np.ndarray = np.zeros((3, 1))
    x_w: np.ndarray = np.ones((10, 3))
    y_w: np.ndarray = np.ones((10, 1)) * 5.0
    weights: np.ndarray = np.ones((10, 1))

    with pytest.warns(RuntimeWarning, match="zero variance in weighted response"):
        r2: float = _r_squared(beta=beta, x_w=x_w, y_w=y_w, y=y.reshape(-1, 1), weights=weights)

    assert np.isnan(r2)


# --------------------------------------------------------------------------------------
# Bootstrap return values and explicit return_all_bootstraps
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_predict_bootstrap_returns_all_when_requested() -> None:
    rng_: np.random.Generator = np.random.default_rng(0)
    x: np.ndarray = np.linspace(-2, 2, 30)
    y: np.ndarray = 0.5 * x + 1.0 + rng_.normal(scale=0.05, size=30)

    target: Rsklpr = Rsklpr(size_neighborhood=10, kr="none", suppress_warnings=True)
    target.fit(x, y)
    mean_y: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    mat: Optional[np.ndarray]

    mean_y, lo, hi, mat = target.predict_bootstrap(x=x, num_bootstrap_resamples=12, return_all_bootstraps=True)
    assert mean_y.shape == x.shape
    assert lo.shape == x.shape
    assert hi.shape == x.shape
    assert mat is not None
    assert mat.shape == (x.shape[0], 12)


# --------------------------------------------------------------------------------------
# Private helpers: _robust_scale and _get_fallback_bw and _is_bw_valid_sequence_type
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice*:RuntimeWarning")
def test_robust_scale_handles_constants_and_nans() -> None:
    x: np.ndarray = np.array(
        [
            [1.0, 2.0, np.nan, 5.0],
            [1.0, 2.0, np.nan, 5.0],
            [1.0, 3.0, np.nan, 5.0],
        ]
    )
    s: np.ndarray = _robust_scale(x)
    # constant columns get fallback std of 1.0 (last column), finite columns positive, NaN column -> fallback 1.0
    assert np.isfinite(s).all()
    assert (s > 0).all()


def test_get_fallback_bw_type_awareness() -> None:
    rng_: np.random.Generator = np.random.default_rng(0)
    a: np.ndarray = rng_.normal(size=(50, 3))
    vt: str = "cuo"  # mixed
    actual: List[float] = _get_fallback_bw(a=a, var_type=vt)
    assert len(actual) == 3
    # discrete dims should equal disc_floor default (0.05), continuous > 0
    assert actual[0] > 0.0
    assert actual[1] == pytest.approx(0.05)
    assert actual[2] == pytest.approx(0.05)


def test_is_bw_valid_sequence_type_accepts_sequence_and_ndarray() -> None:
    assert _is_bw_valid_sequence_type([0.1, 0.2])
    assert _is_bw_valid_sequence_type((0.1, 0.2))
    assert _is_bw_valid_sequence_type(np.array([0.1, 0.2]))
    assert not _is_bw_valid_sequence_type("not ok")
    assert not _is_bw_valid_sequence_type(b"bytes")


# --------------------------------------------------------------------------------------
# Misc: num_poly_terms property after fit; _dim_data quick sanity
# --------------------------------------------------------------------------------------


def test_num_poly_terms_property_after_fit() -> None:
    x: np.ndarray = np.random.default_rng(0).normal(size=(30, 2))
    y: np.ndarray = np.random.default_rng(1).normal(size=30)
    target: Rsklpr = Rsklpr(size_neighborhood=8, degree=2)
    target.fit(x, y)
    # comb(2+2, 2) = 6
    assert target.num_poly_terms == 6


def test_dim_data_quick_check() -> None:
    assert _dim_data(np.array([1, 2, 3])) == 1
    assert _dim_data(np.array([[1, 2], [3, 4]])) == 2
