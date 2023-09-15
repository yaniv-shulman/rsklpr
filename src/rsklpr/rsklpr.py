from numbers import Number
from typing import Optional, Union, Sequence

import numpy as np
import statsmodels.api as sm
from scipy.integrate import quad
from scipy.optimize import minimize, OptimizeResult
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _mean_pdf_left_out(
    h: float, sample: Sequence[float], dens: sm.nonparametric.KDEUnivariate
):
    """
    A naive implementation of the mean density obtained with leave one out cross-validation.

    Returns:
        The mean of the leave one out cross-validation pdf at the locations.
    """
    pdf_left_out_total: float = 0.0
    for i in range(len(sample)):
        subsample: Sequence[float] = sample.copy()
        del subsample[i]
        leave_one_out_kde: sm.nonparametric.KDEUnivariate = (
            sm.nonparametric.KDEUnivariate(subsample)
        )
        leave_one_out_kde.fit(kernel="gau", bw=h)
        pdf_left_out: float = dens.evaluate(sample[i])[0]
        pdf_left_out_total += pdf_left_out
    return pdf_left_out_total / len(sample)


def _lscv_squared_error(
    h: Sequence[float], sample: Sequence[float], a: float, b: float
) -> float:
    """
    Compute the LSCV squared error objective. The mean density obtained with leave one out cross-validation is
    naively implemented iteratively.

    Args:
        h: A sequence of one element, being the bandwidth used to estimate the lscv squared error.
        sample: The data used to estimate the lscv squared error.
        a: The inexact lower bound for the support of the RV.
        b: The inexact upper bound for the support of the RV.

    Returns:
        LSCV squared error loss calculated given the sample and bandwidth.
    """
    h: float = max(h[0], 1e-7)
    dens: sm.nonparametric.KDEUnivariate = sm.nonparametric.KDEUnivariate(sample)
    dens.fit(kernel="gau", bw=h)
    f_n_square = lambda x: dens.evaluate(x)[0] ** 2
    f_n_square_integral: float = quad(f_n_square, a, b)[0]
    mean_pdf_left_out: float = _mean_pdf_left_out(h=h, sample=sample, dens=dens)
    lscv_h: float = abs(2 * (f_n_square_integral - mean_pdf_left_out))
    return lscv_h


def _lscv_univarate_gaussian(sample: Sequence[float], a: float, b: float) -> float:
    """
    Estimate the optimal bandwidth minimizing LSCV criteria for a univariate gaussian KDE. Note this is done numerically
    and may result in a local minima and some executions may diverge.

    Args:
        sample: The data to optimize the bandwidth for.
        a: The inexact lower bound for the support of the RV.
        b: The inexact upper bound for the support of the RV.

    Returns:
        The optimal bandwidth
    """
    h0 = [np.array(sample).std() * (len(sample) ** (-0.2))]  # the initial value of h
    # Constraint to avoid division by zero.
    cons = {"type": "ineq", "fun": lambda x: x[0] - 10 ** (-8)}
    optimize_result: OptimizeResult = minimize(
        _lscv_squared_error, h0, args=(sample, a, b), constraints=cons
    )
    h = optimize_result.x
    return h[0]


def _normalize_array(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Offsets the array by min_val and scales by (max_val - min_val). If min_val and max_val correspond to the range of
    the values in the input then the array is normalized to [0,1].

    Args:
        array: The array to normalize.

    Returns:
        The normalized array.
    """
    return (array - min_val) / (max_val - min_val)


def _scotts_factor(n: int, d: int, weights: Optional[np.ndarray] = None) -> float:
    """
    Computes the coefficient (kde_factor) that multiplies the data variance to obtain the response kernel bandwidth
    according to Scott's Rule.

    Args:
        n: The number of points.
        d: The data dimension.

    Returns:
        The coefficient (kde_factor) that multiplies the data variance to obtain the response kernel bandwidth.
    """
    if weights is not None:
        neff = 1.0 / np.sum(weights ** 2, axis=-1)
    else:
        neff = n

    return neff ** (-1.0 / (d + 4))


def _weighted_linear_regression_with_normal_equations(
    n_x: np.ndarray,
    n_x_windowed: np.ndarray,
    n_y_windowed: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculates the closed form normal equations weighted linear regression.

    Args:
        n_x: The normalized predictors.
        n_x_windowed: The matrix containing the N nearest predictor values for all elements in n_x in each corresponding
            row. i.e. row i of n_x_windowed are the N nearest neighbours of n_x[i].
        n_y_windowed: The matrix containing the N nearest response values for all elements in n_x in each corresponding
            row. i.e. row i of n_y_windowed are the N nearest neighbours of n_y[i].
        weights: The weights to use in the weighted linear regression.

    Returns:
        The predicted y_hat at locations n_x.
    """
    sum_weights: np.ndarray = np.sum(weights, axis=1)
    weighted_sum_x: np.ndarray = np.sum(n_x_windowed * weights, axis=1)
    weighted_sum_y: np.ndarray = np.sum(n_y_windowed * weights, axis=1)
    weighted_sum_x2 = np.sum(n_x_windowed * n_x_windowed * weights, axis=1)
    weighted_sum_xy = np.sum(n_x_windowed * n_y_windowed * weights, axis=1)
    mean_x: np.ndarray = weighted_sum_x / sum_weights
    mean_y: np.ndarray = weighted_sum_y / sum_weights

    b: np.ndarray = (weighted_sum_xy - mean_x * mean_y * sum_weights) / (
        weighted_sum_x2 - mean_x * mean_x * sum_weights
    )

    a: np.ndarray = mean_y - b * mean_x
    y_hat: np.ndarray = a + b * n_x
    return y_hat


def _iterative_weighted_matrix_polynomial_regression(
    n_x: np.ndarray,
    n_y_windowed: np.ndarray,
    weights: np.ndarray,
    dist_n_x: np.ndarray,
    degree: int,
):
    """
    Calculates the closed form matrix equations weighted polynomial regression.

    Args:
        n_x: The normalized predictors.
        n_y_windowed: The matrix containing the N nearest response values for all elements in n_x in each corresponding
            row. i.e. row i of n_y_windowed are the N nearest neighbours of n_y[i].
        weights: The weights to use in the weighted linear regression.
        dist_n_x: Distances to the N nearest neighbors of each point.
        degree: The degree of the polynomial regression to compute.

    Returns:
        The predicted y_hat at locations n_x.
    """
    i: int
    x_mat: np.ndarray = np.concatenate(
        [
            np.expand_dims(np.ones_like(dist_n_x), axis=-1),
            np.expand_dims(dist_n_x, axis=-1),
        ]
        + [
            np.expand_dims(np.power(dist_n_x, i), axis=-1) for i in range(2, degree + 1)
        ],
        axis=-1,
    )
    y_hat: np.ndarray = np.empty(shape=n_x.shape)

    for i in tqdm(range(n_x.shape[0])):
        w_mat: np.ndarray = np.diag(weights[i])

        y_hat[i]: np.ndarray = (
            np.linalg.inv(x_mat[i].T @ w_mat @ x_mat[i])
            @ x_mat[i].T
            @ w_mat
            @ np.expand_dims(n_y_windowed[i], axis=-1)
        )[0]
    return y_hat


class Rsklpr1D:
    """
    Demonstrative implementation of the Robust Similarity Kernel Local Polynomial Regression for the 1D case proposed in
    ?????. Current implementation uses vectorized calculations that require relatively large amounts of memory. This
    imposes limitations on the neighbourhood/dataset size that may be used.
    """

    def __init__(
        self,
        size_neighborhood: int,
        degree: int = 1,
        predictor_bandwidth: Optional[float] = None,
        response_bandwidth: str = "lscv_global",
    ) -> None:
        """
        Args:
            size_neighborhood: The number of points in the neighborhood to consider in the local regression.
            degree: The degree of the polynomial fitted locally.
            predictor_bandwidth: The bandwidth of the predictor kernel in the factorized joint KDE. If None it will be
                calculated heuristically based on a formula derived from empirical results.
            response_bandwidth: The method used to estimate the bandwidth for the response kernel in the factorized
                joint KDE. Available options are 'scott', 'lscv_global' and 'lscv_local' which corresponds to Scott's
                Rule, LSCV estimation on the entire data and LSCV based estimation done for each neighborhood
                separately. The three methods have considerably different computational complexity with 'scott' being
                the least and 'lscv_local' being the most.
        """
        self._size_neighborhood: int = size_neighborhood
        self._degree: int = degree
        self._response_bandwidth: str = response_bandwidth

        self._predictor_bandwidth: float = (
            1 / (np.clip(a=0.1 * size_neighborhood, a_min=2.0, a_max=27.0))
            if predictor_bandwidth is None
            else predictor_bandwidth
        )

        self._n_x: np.ndarray = np.asarray(0.0)
        self._n_y: np.ndarray = np.asarray(0.0)
        self._min_x: float
        self._max_x: float
        self._min_y: float
        self._max_y: float
        self._nearest_neighbors: NearestNeighbors

    def _denormalize_y(self, n_y: np.ndarray) -> np.ndarray:
        """
        Maps the response to its original range.

        Args:
            n_y: Values that represent normalized response values.

        Returns:
            The inputs mapped to the original range of the response.
        """
        return n_y * (self._max_y - self._min_y) + self._min_y

    def _estimate(self, x: np.ndarray, iterative: bool) -> np.ndarray:
        """
        Estimates the value of m(x) at the locations.

        Args:
            x: Predictor values at locations to estimate m(x), these should be at the original range of the predictor.
            iterative: True to perform fit for each neighbourhood using matrix equations. Useful to limit the
                amount of memory used at the expense of slower computation or when fitting a 2 or higher degree local
                polynomial. False to solve vectorized using normal regression equations.

        Returns:
            The estimated values of m(x) at the locations.
        """
        n_x: np.ndarray = _normalize_array(
            np.asarray(x), min_val=self._min_x, max_val=self._max_x
        )
        dist_n_x: np.ndarray
        indices: np.ndarray
        dist_n_x, indices = self._nearest_neighbors.kneighbors(
            np.expand_dims(n_x, axis=-1)
        )
        dist_n_x = dist_n_x / np.max(dist_n_x, axis=1, keepdims=True).astype(float)
        weights: np.ndarray = np.exp(-dist_n_x)
        n_x_windowed: np.ndarray = n_x[indices]
        n_y_windowed: np.ndarray = self._n_y[indices]

        local_density = self._estimate_proportional_conditional_local_density(
            n_y_windowed=n_y_windowed, dist_n_x=dist_n_x
        )

        weights = weights * local_density
        y_hat: np.ndarray

        if not iterative:
            y_hat = _weighted_linear_regression_with_normal_equations(
                n_x=n_x,
                n_x_windowed=n_x_windowed,
                n_y_windowed=n_y_windowed,
                weights=weights,
            )
        else:
            y_hat = _iterative_weighted_matrix_polynomial_regression(
                n_x=n_x,
                n_y_windowed=n_y_windowed,
                weights=weights,
                dist_n_x=dist_n_x,
                degree=self._degree,
            )

        return self._denormalize_y(y_hat)

    def _estimate_proportional_conditional_local_density(
        self,
        n_y_windowed: np.ndarray,
        dist_n_x: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised implementation to calculate the conditional KDE of the response variable at all predictor locations.
            Note that all normalization constants that don't affect the regression are omitted to speed up calculations.

        Args:
            n_y_windowed: The matrix containing the N nearest response values for all elements in n_x in each
                corresponding row. i.e. row i of n_y_windowed are the N nearest neighbours of n_y[i].
            dist_n_x: Distances to the N nearest neighbors of each point.

        Returns:
            The proportional conditional KDE of the response in all locations.
        """
        var_n_y_windowed: np.ndarray = np.var(a=n_y_windowed, axis=1).reshape(
            (-1, 1, 1)
        )

        square_dist_n_y_windowed: np.ndarray = np.square(
            np.tile(
                np.expand_dims(n_y_windowed, axis=1),
                reps=(1, self._size_neighborhood, 1),
            )
            - np.expand_dims(n_y_windowed, axis=-1)
        )

        a: np.ndarray = (
            np.exp(-0.5 * np.power(dist_n_x / self._predictor_bandwidth, 2))
            / self._predictor_bandwidth
        )

        kde_factor: np.ndarray = self._calculate_kde_factor(n_y_windowed=n_y_windowed)

        local_density: np.ndarray = np.exp(
            -0.5 * square_dist_n_y_windowed / (var_n_y_windowed * kde_factor ** 2)
        )

        local_density = (local_density * np.expand_dims(a, axis=1)).sum(axis=-1)

        return local_density

    def _calculate_kde_factor(
        self, n_y_windowed: np.ndarray, max_sample_size: int = 75
    ):
        """
        Computes the coefficient that multiplies the data variance to obtain the response kernel bandwidth according to
        the specified method.

        Args:
            n_y_windowed: The matrix containing the N nearest response values for each element in n_x in each
            corresponding row. i.e. row i of n_y_windowed are the N nearest neighbours of n_y[i].
            max_sample_size: Denotes the maximum sample size used for LSCV estimates, ignored when using Scott's Rule.

        Returns:
            The coefficient that multiplies the data variance to obtain the response kernel bandwidth
        """
        kde_factor: np.ndarray
        sample_y: np.ndarray

        if self._response_bandwidth == "scott":
            kde_factor = np.asarray(_scotts_factor(n=self._size_neighborhood, d=1))
        elif self._response_bandwidth == "lscv_global":
            sample_y = np.random.choice(
                a=self._n_y, size=min(int(self._n_y.shape[0]), max_sample_size)
            )
            kde_factor = np.asarray(
                _lscv_univarate_gaussian(sample=sample_y.tolist(), a=0.0, b=1.0)
            )
        elif self._response_bandwidth == "lscv_local":
            kde_factor = np.empty(shape=n_y_windowed.shape[0])
            for i in range(n_y_windowed.shape[0]):
                sample_y = np.random.choice(
                    a=n_y_windowed[i],
                    size=min(int(n_y_windowed.shape[1]), max_sample_size),
                )
                kde_factor[i] = float(
                    _lscv_univarate_gaussian(
                        sample=sample_y.tolist(),
                        a=n_y_windowed[i].min(),
                        b=n_y_windowed[i].max(),
                    )
                )
            kde_factor = np.maximum(kde_factor, 1e-6)

            kde_factor[
                np.abs((kde_factor - kde_factor.mean()) / kde_factor.std()) > 2.0
            ] = np.median(kde_factor)

            kde_factor = np.reshape(a=kde_factor, newshape=(-1, 1, 1))
        else:
            raise ValueError("Unknown response_bandwidth selection method")
        return kde_factor

    def fit(
        self,
        x: Union[np.ndarray, Sequence[Number]],
        y: Union[np.ndarray, Sequence[Number]],
    ) -> None:
        """
        Fits the model to the training set.

        Args:
            x: The predictor values.
            y: The response values at the corresponding predictor locations.

        """
        x = np.asarray(x, dtype=float)
        x = np.squeeze(a=x)
        y = np.asarray(y, dtype=float)
        y = np.squeeze(a=y)
        self._min_x = x.min()
        self._max_x = x.max()
        self._n_x = _normalize_array(array=x, min_val=self._min_x, max_val=self._max_x)
        self._min_y = y.min()
        self._max_y = y.max()
        self._n_y = _normalize_array(array=y, min_val=self._min_y, max_val=self._max_y)
        self._size_neighborhood = min(self._size_neighborhood, x.shape[0])
        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self._size_neighborhood, algorithm="ball_tree"
        ).fit(np.expand_dims(self._n_x, axis=-1))

    def predict(
        self,
        x: Union[np.ndarray, Sequence, float],
    ) -> np.ndarray:
        """
        Predicts estimates of m(x) at the specified locations. Must call fit with the training data first.

        Args:
            x: The locatios to predict for.

        Returns:
            The corresponding estimated responses in the locations.
        """
        return self._estimate(x=x, iterative=self._degree > 1)

    def fit_and_predict(
        self,
        x: Union[np.ndarray, Sequence[Number]],
        y: Union[np.ndarray, Sequence[Number]],
    ) -> np.ndarray:
        """
        Fits the provided dataset and estimates the response at the locations of x.
        Args:
            x: The predictor values
            y: The response values at the corresponding predictor locations.

        Returns:
            The corresponding estimated responses in the locations.
        """
        self.fit(x=x, y=y)
        return self.predict(x=x)

    __call__ = fit_and_predict
