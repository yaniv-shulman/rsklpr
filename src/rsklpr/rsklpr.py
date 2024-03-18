import math
import warnings
from numbers import Number
from typing import Optional, Sequence, Tuple, Callable, List, Union, Any, Dict

import numpy as np
from numpy.random import default_rng as np_defualt_rng
from sklearn.neighbors import NearestNeighbors

from rsklpr.kde_statsmodels_impl.bandwidths import select_bandwidth
from rsklpr.kde_statsmodels_impl.kernel_density import KDEMultivariate


def _mean_square_error(e: np.ndarray) -> float:
    return float(np.mean(np.sum(e**2)))


def _mean_abs_error(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e)))


def _bias_error(e: np.ndarray) -> float:
    return float(np.mean(e))


def _std_error(e: np.ndarray) -> float:
    return float(np.std(e))


def _r_squared(beta: np.ndarray, x_w: np.ndarray, y_w: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if y.shape != weights.shape:
        raise ValueError("y and weights must have the same shape")

    y_hat: np.ndarray = x_w @ beta
    e: np.ndarray = y_hat - y_w
    sse: float = float(e.T @ e)
    sst_centered: float = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2))
    return 1.0 - sse / sst_centered


def _weighted_local_regression(
    x_0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    degree: int,
    calculate_r_squared: bool = False,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Calculates the closed form matrix equations weighted constant or linear local regression centered at a point.

    Args:
        x_0: The target regression point.
        x: The predictors in all observations, of shape [N, K] where N is the observations and K is the dimension.
        y: The N scalar response values corresponding to the provided predictors.
        weights: The N scalar weights associated with each observation.
        degree: The regression polynomial degree, supported values are 0 or 1.

    Returns:
        The predicted y_hat at locations x.
    """
    if degree not in (0, 1):
        raise ValueError(f"Degree {degree} is not supported. Supported values are 0 and 1.")

    if x.ndim != 2:
        raise ValueError("x must be a two dimensional array.")

    y = y.reshape((x.shape[0]), 1)
    weights = weights.reshape(y.shape)
    bias: np.ndarray = np.ones(shape=(x.shape[0], 1))

    x_mat: np.ndarray = (
        np.concatenate(
            [bias, x - x_0],
            axis=1,
        )
        if degree == 1
        else bias
    )

    w_sqrt: np.ndarray = np.sqrt(weights)
    y_w: np.ndarray = w_sqrt * y
    x_mat_w: np.ndarray = w_sqrt * x_mat
    del x_mat, bias
    beta: np.ndarray = np.linalg.inv(x_mat_w.T @ x_mat_w) @ x_mat_w.T @ y_w
    r_squared: Optional[float] = None

    if calculate_r_squared:
        r_squared = _r_squared(beta=beta, x_w=x_mat_w, y_w=y_w, y=y, weights=weights)

    return beta[0], r_squared


def _dim_data(data: np.ndarray) -> int:
    """
    Calculates the dimension of the data. The data is assumed to be at most a 2D numpy for which the first dimension
    represents the number of observations.

    Args:
        data: The data to calculates the dimensions for.

    Returns:
        The data dimensionality.
    """
    return data.shape[1] if data.ndim > 1 else 1


def _laplacian_normalized(u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Laplacian kernel. The inputs are first scaled to the range [0,1] before applying the kernel.

    Args:
        u: The kernel input.

    Returns:
        The kernel output.
    """
    return np.exp(-u / np.max(np.atleast_2d(u), axis=1, keepdims=True).astype(float))  # type: ignore [no-any-return]


def _tricube_normalized(u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Tricube kernel. The implementation assumes all inputs are non-negative with a 0
    value present. The inputs are scaled to the range [0,1] before applying the kernel.

    Args:
        u: The kernel input, note it is assumed all inputs are non-negative.

    Returns:
        The kernel output.
    """
    assert u.min() >= 0  # negative values are not expected to happen during normal execution.
    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=np.power((1 - np.power(u / np.max(np.atleast_2d(u), axis=1, keepdims=True).astype(float), 3)), 3),
        a_min=0.0,
        a_max=None,
    )


class Rsklpr:
    """
    Implementation of the Robust Similarity Kernel Local Polynomial Regression for proposed in the paper
    https://github.com/yaniv-shulman/rsklpr/blob/main/paper/rsklpr.pdf.
    """

    def __init__(
        self,
        size_neighborhood: int,
        degree: int = 1,
        metric_x: str = "mahalanobis",
        metric_x_params: Optional[Dict[str, Any]] = None,
        k1: str = "laplacian",
        k2: str = "joint",
        bw1: Union[str, float, Sequence[float], Callable[[Any], Sequence[float]]] = "normal_reference",  # type: ignore [misc]
        bw2: Union[str, float, Sequence[float], Callable[[Any], Sequence[float]]] = "normal_reference",  # type: ignore [misc]
        bw_global_subsample_size: Optional[int] = None,
        seed: int = 888,
    ) -> None:
        """
        Args:
            size_neighborhood: The number of points in the neighborhood to consider in the local regression.
            degree: The degree of the polynomial fitted locally, supported values are 0 or 1 (default) that result in
                local constant and local linear regression respectively.
            metric_x: Metric for distance computation for the predictors using sklearn.neighbors.NearestNeighbors. See
                https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html for details.
            metric_x_params: Metric parameters if required. For 'mahalanobis' (default) the inverted covariance matrix
                'VI' can, but need not, be specified since it is calculated during fitting if left unspecified. For the
                Minkowski metric 'p' can be specified here however defaults to 2 if left unspecified. See
                https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance for more
                details on the various metrics and their parameters.
            k1: The kernel that models the effect of distance on weight between the local target regression to its
                neighbours. This is similar to the kernel used in standard polynomial regression. Available options are
                'laplacian' (default) and 'tricube', of which the latter is traditionally used in LOESS.
            k2: The kernel that models the 'importance' of the response at the location. Available options are 'joint'
                (joint density) and 'conden' (conditional density). The 'joint' kernel implements a weighted KDE of the
                marginal density of the response where the weights are based on the distance of the predictor from the
                location. The 'conden' kernel calculates the KDE of (Y|X).
            bw1: The method used to estimate the bandwidth for the marginal predictor's kernel used by both methods
                implemented for k2. Supported options are 'normal_reference' and 'scott' which correspond to the local
                normal reference rule of thumb (default) and local Scott's rule, both implemented in statsmodels. See
                https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html. An additional
                provided option is 'cv_ls_global' which correspond to global least squares cross validation based
                bandwidth estimation implemented in sm.nonparametric.KDEMultivariate class, see
                https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html.
                Furthermore, the 'conden' kernel supports the 'cv_ml' and 'cv_ls' options corresponding to local cross
                validation maximum likelihood and local cross validation least squares as implemented in
                sm.nonparametric.KDEMultivariate. Finally, it is also possible to pass in a sequence of floats, one per
                dimension of the data, or a callable that takes the data as input and returns the same, i.e a sequence
                of floats have one scalar value per dimension.
            bw2: The method used to estimate the second bandwidth used by k2. The semantics depend on the kernel used
                for k2. For the 'conden' kernel, bw2 represents the bandwidth estimation method for the joint KDE of
                (X,Y). For the 'joint' kernel this is the bandwidth estimation method for marginal KDE of Y. The
                supported options are the same as bw1.
            bw_global_subsample_size: The size of subsample taken from the data for global cross validation bandwidth
                estimation. If None the entire data is used for bandwidth estimation. This could be useful to speedup
                    global cross validation based estimates.
            seed: The seed used for random sub sampling for cross validation bandwidth estimation.
        """
        if size_neighborhood < 3:
            raise ValueError("size_neighborhood must be at least 3")

        if degree not in (0, 1):
            raise ValueError("degree must be one of 0 or 1")

        k1 = k1.lower()
        if k1 not in ("laplacian", "tricube"):
            raise ValueError(f"k1 {k1} is unsupported and must be one of 'laplacian' or 'tricube'")

        k2 = k2.lower()
        if k2 not in ("conden", "joint"):
            raise ValueError(f"k2 {k2} is unsupported and must be one of 'conden' or 'joint'")

        bw_error_str: str = (
            "When bandwidth is a string it must be one of 'normal_reference', 'cv_ml', 'cv_ls', "
            "'scott' or 'cv_ls_global'"
        )

        bw_methods: Tuple[str, ...] = (
            "normal_reference",
            "cv_ml",
            "cv_ls",
            "scott",
            "cv_ls_global",
        )

        if isinstance(bw1, str):
            bw1 = bw1.lower()

            if bw1 not in bw_methods:
                raise ValueError(bw_error_str)
        elif isinstance(bw1, float):
            bw1 = [bw1]

        if isinstance(bw2, str):
            bw2 = bw2.lower()

            if bw2 not in bw_methods:
                raise ValueError(bw_error_str)
        elif isinstance(bw2, float):
            bw2 = [bw2]

        self._size_neighborhood: int = int(size_neighborhood)
        self._degree: int = int(degree)
        self._metric_x: str = metric_x.lower()
        self._metric_x_params: Optional[Dict[str, Any]] = metric_x_params
        self._k1: str = k1
        self._k2: str = k2
        self._bw1: Union[str, Sequence[float], Callable[[Any], Sequence[float]]] = bw1  # type: ignore [misc]
        self._bw2: Union[str, Sequence[float], Callable[[Any], Sequence[float]]] = bw2  # type: ignore [misc]

        self._bw_global_subsample_size: Optional[int] = (
            int(bw_global_subsample_size) if bw_global_subsample_size is not None else None
        )

        self._seed: int = seed

        self._k1_func: Callable[[np.ndarray], np.ndarray] = (
            _laplacian_normalized if k1 == "laplacian" else _tricube_normalized
        )

        self._rnd_gen: np.random.Generator = np_defualt_rng(seed=seed)
        self._x: np.ndarray = np.ndarray(shape=())
        self._y: np.ndarray = np.ndarray(shape=())
        self._mean_square_error: Optional[float] = None
        self._root_mean_square_error: Optional[float] = None
        self._mean_abs_error: Optional[float] = None
        self._bias_error: Optional[float] = None
        self._std_error: Optional[float] = None
        self._r_squared: np.ndarray = np.empty(())
        self._nearest_neighbors: NearestNeighbors

    def _calculate_bandwidth(  # type: ignore [return]
        self,
        bandwidth: Union[str, Callable[[Any], Sequence[float]]],  # type: ignore [misc]
        data: np.ndarray,
    ) -> Sequence[float]:
        """
        Estimate the bandwidth for the data.

        Args:
            bandwidth: The method used to estimate the bandwidth for the data.
            data: The data to estimate the bandwidth for.

        Returns:
            The estimated bandwidth for the data.
        """
        if isinstance(bandwidth, Callable):  # type: ignore [arg-type]
            return bandwidth(data)  # type: ignore [operator]
        elif isinstance(bandwidth, str):
            if bandwidth in ("scott", "normal_reference"):
                return select_bandwidth(x=data, bw=bandwidth, kernel=None).tolist()  # type: ignore [no-any-return]
            elif bandwidth.startswith("cv_"):
                subsample: np.ndarray = data

                if (
                    bandwidth == "cv_ls_global"
                    and self._bw_global_subsample_size is not None
                    and data.shape[0] > self._bw_global_subsample_size
                ):
                    sample_idx: np.ndarray = self._rnd_gen.choice(
                        a=data.shape[0],
                        size=min(int(data.shape[0]), self._bw_global_subsample_size),
                    )
                    subsample = data[sample_idx, :]

                return KDEMultivariate(  # type: ignore [no-any-return]
                    data=subsample,
                    var_type="c" * _dim_data(data=subsample),
                    bw="cv_ls",
                ).bw
        else:
            raise ValueError(f"Unknown bandwidth {bandwidth}")

    def _k2_conden(
        self,
        n_x_neighbors: np.ndarray,
        n_y_neighbors: np.ndarray,
        bw1_global: Optional[Sequence[float]] = None,
        bw2_global: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Calculates the conditional density similarity kernel for the observations in the neighborhood.

        Args:
            n_x_neighbors: The predictors of all neighbors.
            n_y_neighbors: The corresponding response of all neighbors.
            bw1_global: The bw1 calculated from the global data, if None a local bandwidth estimation will be used.
            bw2_global: The bw2 calculated from the global data, if None a local bandwidth estimation will be used.

        Returns:
            The kernel values to all observations.
        """
        if n_x_neighbors.ndim > 2:
            n_x_neighbors = np.squeeze(n_x_neighbors).reshape(-1, n_x_neighbors.shape[-1])

        n_xy_neighbors: np.ndarray = np.concatenate(
            [n_x_neighbors, n_y_neighbors],
            axis=-1,
        )

        var_type: str = "c" * _dim_data(data=n_x_neighbors)

        kde_marginal_x: KDEMultivariate = KDEMultivariate(
            data=n_x_neighbors,
            var_type=var_type,
            bw=self._bw1
            if (self._bw1 in ("cv_ls", "cv_ml") or isinstance(self._bw1, List))
            else self._calculate_bandwidth(bandwidth=self._bw1, data=n_x_neighbors)  # type: ignore [arg-type]
            if bw1_global is None
            else bw1_global,
        )

        kde_joint: KDEMultivariate = KDEMultivariate(
            data=n_xy_neighbors,
            var_type=var_type + "c",
            bw=self._bw2
            if (self._bw2 in ("cv_ls", "cv_ml") or isinstance(self._bw1, List))
            else self._calculate_bandwidth(bandwidth=self._bw2, data=n_xy_neighbors)  # type: ignore [arg-type]
            if bw2_global is None
            else bw2_global,
        )

        return kde_joint.pdf(data_predict=n_xy_neighbors) / kde_marginal_x.pdf(  # type: ignore [no-any-return]
            data_predict=n_x_neighbors
        )

    def _k2_joint(
        self,
        n_x_neighbors: np.ndarray,
        n_y_neighbors: np.ndarray,
        dist_n_x_neighbors: np.ndarray,
        bw1_global: Optional[Sequence[float]] = None,
        bw2_global: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Calculates the joint density similarity kernel for the observations in the neighborhood.

        Args:
            n_x_neighbors: The predictors of all neighbors.
            n_y_neighbors: The corresponding response of all neighbors.
            dist_n_x_neighbors: The distance of all neighbors to the regression target location.
            bw1_global: The bw1 calculated from the global data, if None a local bandwidth estimation will be used.
            bw2_global: The bw2 calculated from the global data, if None a local bandwidth estimation will be used.

        Returns:
            The kernel values to all observations.
        """
        square_dist_n_y_windowed: np.ndarray = np.square(n_y_neighbors - n_y_neighbors.T)

        bw_x: np.ndarray = np.asarray(
            (
                self._bw1
                if isinstance(self._bw1, List)
                else self._calculate_bandwidth(bandwidth=self._bw1, data=n_x_neighbors)  # type: ignore [arg-type]
            )
            if bw1_global is None
            else bw1_global
        )

        bw_x = bw_x.mean()
        weights: np.ndarray = np.exp(-0.5 * np.power(dist_n_x_neighbors / bw_x, 2)) / bw_x

        bw_y: np.ndarray = np.asarray(
            (
                self._bw2
                if isinstance(self._bw2, List)
                else self._calculate_bandwidth(bandwidth=self._bw2, data=n_y_neighbors)  # type: ignore [arg-type]
            )
            if bw2_global is None
            else bw2_global
        )

        if bw_y.size != 1:
            raise ValueError(f"Too many values ({bw_y.size}) specified for y bandwidth")

        local_density: np.ndarray = np.exp(-0.5 * square_dist_n_y_windowed / (bw_y**2))
        local_density = (local_density * weights).sum(axis=-1)
        return local_density

    def _estimate(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float],
        metrics: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Estimates the value of m(x) at the locations.

        Args:
            x: Predictor values at locations to estimate m(x), these should be at the original range of the predictor.
            metrics: Optional error metrics to calculate. Options are 'mean_square', 'mean_abs', 'root_mean_square',
                'bias', 'std', 'r_squared' and 'mean_r_squared'. The metrics are made available through attributes on
                the model object having similar corresponding names. Note that x must be exactly the same as the
                training data provided to 'fit' if any metrics are specified.

        Returns:
            The estimated values of m(x) at the locations.
        """
        x_arr: np.ndarray
        x_arr, _ = self._check_and_reshape_inputs(x=x)
        n: int = x_arr.shape[0]

        if metrics is not None and not np.allclose(a=x_arr, b=self._x):
            raise ValueError(
                "When specifying metrics the provided predictor values must be the same as the values "
                "provided to 'fit'."
            )

        y_hat: np.ndarray = np.empty((n))
        bw1_global: Optional[Sequence[float]]
        bw2_global: Optional[Sequence[float]]
        bw1_global, bw2_global = self._get_bandwidth_global(k2=self._k2)
        calculate_r_squared: bool = False

        if metrics is not None and ("r_squared" in metrics or "mean_r_squared" in metrics):
            self._r_squared = np.empty(shape=(n))
            calculate_r_squared = True

        i: int

        for i in range(n):
            dist_n_x_neighbors: np.ndarray
            indices: np.ndarray
            dist_n_x_neighbors, indices = self._nearest_neighbors.kneighbors(X=x_arr[i].reshape(1, -1))
            weights: np.ndarray = self._k1_func(dist_n_x_neighbors)
            n_x_neighbors: np.ndarray = self._x[indices].squeeze(axis=0)

            if self._k2 == "conden":
                weights *= self._k2_conden(
                    n_x_neighbors=n_x_neighbors,
                    n_y_neighbors=self._y[indices].T,
                    bw1_global=bw1_global,
                    bw2_global=bw2_global,
                )
            elif self._k2 == "joint":
                weights *= self._k2_joint(
                    n_x_neighbors=n_x_neighbors,
                    n_y_neighbors=self._y[indices].T,
                    dist_n_x_neighbors=dist_n_x_neighbors,
                    bw1_global=bw1_global,
                    bw2_global=bw2_global,
                )

            r_squared: Optional[float]

            y_hat[i], r_squared = _weighted_local_regression(
                x_0=x_arr[i].reshape(1, -1),
                x=n_x_neighbors,
                y=self._y[indices].T,
                weights=weights,
                degree=self._degree,
                calculate_r_squared=calculate_r_squared,
            )

            if calculate_r_squared:
                self._r_squared[i] = r_squared

        if metrics is not None:
            errors: np.ndarray = y_hat - self._y

            if "mean_square" in metrics or "root_mean_square" in metrics:
                self._mean_square_error = float(_mean_square_error(e=errors))

                if "root_mean_square" in metrics:
                    self._root_mean_square_error = math.sqrt(self._mean_square_error)

            if "mean_abs" in metrics:
                self._mean_abs_error = float(_mean_abs_error(e=errors))
            if "bias" in metrics:
                self._bias_error = float(_bias_error(e=errors))
            if "std" in metrics:
                self._std_error = float(_std_error(e=errors))

        return y_hat

    def _estimate_bootstrap(
        self, x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float], bootstrap_iterations: int
    ) -> np.ndarray:
        x_arr: np.ndarray
        x_arr, _ = self._check_and_reshape_inputs(x=x)
        y_hat: np.ndarray = np.empty((x_arr.shape[0], bootstrap_iterations + 1))
        y_hat[:, 0] = self._estimate(x=x_arr)
        i: int

        for i in range(bootstrap_iterations):
            resmaple_idx: np.ndarray = self._rnd_gen.choice(
                a=np.arange(stop=self._x.shape[0]), size=self._x.shape[0], replace=True  # type: ignore[call-overload]
            )

            x_resample: np.ndarray = self._x[resmaple_idx, :]
            y_resample: np.ndarray = self._y[resmaple_idx]

            model: Rsklpr = Rsklpr(
                size_neighborhood=self._size_neighborhood,
                degree=self._degree,
                metric_x=self._metric_x,
                metric_x_params=self._metric_x_params,
                k1=self._k1,
                k2=self._k2,
                bw1=self._bw1,
                bw2=self._bw2,
                bw_global_subsample_size=self._bw_global_subsample_size,
                seed=self._seed,
            )

            model.fit(x=x_resample, y=y_resample)
            y_hat[:, i + 1] = model._estimate(x=x_arr)

        return y_hat

    def _get_bandwidth_global(self, k2: str) -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
        """
        Calculates bandwidth estimates from the global data if configured to do so.

        Args:
            k2: The k2 used.

        Returns:
            A tuple representing the global bw1 and bw2 estimates if applicable or None if the estimator is configured
            to use local bandwidth estimates.
        """
        bw1_global: Optional[Sequence[float]] = None

        if self._bw1 == "cv_ls_global":
            bw1_global = self._calculate_bandwidth(
                bandwidth=self._bw1,  # type: ignore [arg-type]
                data=self._x,
            )

        bw2_global: Optional[Sequence[float]] = None

        if self._bw2 == "cv_ls_global":
            if k2 == "conden":
                bw2_global = self._calculate_bandwidth(
                    bandwidth=self._bw2,  # type: ignore [arg-type]
                    data=np.concatenate([self._x, np.expand_dims(a=self._y, axis=-1)], axis=1),
                )
            elif k2 == "joint":
                bw2_global = self._calculate_bandwidth(
                    bandwidth=self._bw2,  # type: ignore [arg-type]
                    data=np.expand_dims(a=self._y, axis=-1),
                )

        return bw1_global, bw2_global

    def fit(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]]],
        y: Union[np.ndarray, Sequence[Number]],
    ) -> None:
        """
        Fits the model to the training set. The number of observations must not be smaller than the neighborhood size
        specified.

        Args:
            x: The predictor values. Must be compatible with a numpy array of dimension two at most. The first axis
                denotes the observation and the second axis the vector components of each observation, i.e. the
                coordinates of data point i are given by x[i,:].
            y: The response values at the corresponding predictor locations. Must be compatible with a 1 dimensional
                numpy array.
        """
        x_arr: np.ndarray
        y_arr: np.ndarray
        x_arr, y_arr = self._check_and_reshape_inputs(x=x, y=y)  # type: ignore [assignment]
        self._x = x_arr
        self._y = y_arr
        metric_params: Dict[str, Any]
        p: float
        metric_params, p = self._get_metric_params()

        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self._size_neighborhood,
            algorithm="auto",
            metric=self._metric_x,
            p=p,
            metric_params=metric_params,
        )

        self._nearest_neighbors.fit(self._x)

    def _get_metric_params(self) -> Tuple[Dict[str, Any], float]:
        """
        Creates the metric params object required for creating the NearestNeighbors object. For 'mahalanobis' metric the
        inverse covariance matrix is calculated for the data fitted and added to the metric_params. For the 'minkowski'
        metric, 'p' provided in construction is assigned to a separate variable and removed from the metric_params. The
        default value of 'p' is 2 since this is the default used in NearestNeighbors. For all other metrics the
        metric_params provided in construction is returned.

        Returns:
            The metric params and a 'p' parameter required to construct a NearestNeighbors object.
        """
        metric_params: Dict[str, Any] = dict() if self._metric_x_params is None else self._metric_x_params.copy()
        p: float = 2.0

        if self._metric_x == "mahalanobis":
            if "VI" not in metric_params.keys():
                cov: np.ndarray = np.atleast_2d(np.cov(m=self._x, rowvar=False))
                metric_params["VI"] = np.linalg.inv(cov)

        elif self._metric_x == "minkowski":
            if "p" in metric_params.keys():
                p = metric_params["p"]
                del metric_params["p"]

        return metric_params, p

    def _check_and_reshape_inputs(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float],
        y: Optional[Union[np.ndarray, Sequence[Number]]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Checks and reshapes the input so that x is a 2D numpy array of dimensions [N,K] where N is the observations and
        K is the dimensionality of the observations. if y is provided it is reshaped into a 1D ndarray.

        Args:
           x: The predictor values. Must be compatible with a numpy array of dimension two at most. The first axis
                denotes the observation and the second axis the vector components of each observation, i.e. the
                coordinates of data point i are given by x[i,:].
           y: Optional response values at the corresponding predictor locations. Must be compatible with a 1 dimensional
                numpy array.

        Returns:
            The reshaped x array and reshaped y values if provided.

        Raises:
            ValueError: When x.ndim is larger than 2.
            ValueError: When N is smaller than the specified neighborhood size.
            ValueError: When y dimension is larger than one.
            ValueError: when x and y has incompatible shapes
        """
        if isinstance(x, np.ndarray):
            x = x.copy().astype(float)
        else:
            x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape((-1, 1))
        elif x.ndim > 2:
            raise ValueError("x dimension must be at most 2")

        if x.shape[0] < x.shape[1]:
            warnings.warn("There are less observations than the number of dimensions, is this intended?")

        if x.shape[0] < self._size_neighborhood:
            ValueError(
                f"Provided inputs have {x.shape[0]} observations which is less than specified "
                f"neighborhood size {self._size_neighborhood}"
            )

        if y is not None:
            if isinstance(y, np.ndarray):
                y = y.copy().astype(float)
            else:
                y = np.asarray(y, dtype=float)

            y = np.squeeze(a=y)

            if y.ndim > 1:
                raise ValueError("y dimension must be at most 1")

            if x.shape[0] != y.shape[0]:
                raise ValueError("x and y have incompatible shapes")

        return x, y

    def predict(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float],
        metrics: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Predicts estimates of m(x) at the specified locations. Must call fit with the training data first.

        Args:
            x: The locations to predict for.
            metrics: Optional error metrics to calculate. Options are 'mean_square', 'mean_abs', 'root_mean_square',
                'bias', 'std', 'r_squared' and 'mean_r_squared'. The metrics are made available through attributes on
                the model object having similar corresponding names. Note that x must be exactly the same as the
                training data provided to 'fut' if any metrics are specified.

        Returns:
            The estimated responses at the corresponding locations.x
        """
        return self._estimate(x=x, metrics=metrics)

    def predict_bootstrap(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float],
        q_low: float = 0.025,
        q_high: float = 0.975,
        num_bootstrap_resamples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts estimates of m(x) at the specified locations multiple times. The first estimate is done using the
        data provided when calling fit (the original sample), following additional 'num_bootstrap_resamples' estimates.
        The method then uses all estimates to calculate confidence intervals. Note that calling fit with the training
        data must be done first.

        Args:
            x: The locations to predict for.
            q_low: The lower confidence quantile estimated from the posterior of y_hat.
            q_high: The upper confidence quantile estimated from the posterior of y_hat.
            num_bootstrap_resamples: The number of bootstrap resamples to take.

        Returns:
            The estimated responses at the corresponding locations according to the original fit data and the low and
            high confidence bands.
        """
        if num_bootstrap_resamples <= 0:
            raise ValueError("At least one bootstrap iteration need to be specified")

        y_hat: np.ndarray = self._estimate_bootstrap(x=x, bootstrap_iterations=num_bootstrap_resamples)
        y_conf_low: np.ndarray = np.quantile(a=y_hat, q=q_low, axis=1)
        y_conf_high: np.ndarray = np.quantile(a=y_hat, q=q_high, axis=1)
        return y_hat[:, 0], y_conf_low, y_conf_high

    def fit_and_predict(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]]],
        y: Union[np.ndarray, Sequence[Number]],
        metrics: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fits the provided dataset and estimates the response at the locations of x.

        Args:
            x: The predictor values
            y: The response values at the corresponding predictor locations.
            metrics: Optional error metrics to calculate. Options are 'mean_square', 'mean_abs', 'root_mean_square',
                'bias', 'std', 'r_squared' and 'mean_r_squared'. The metrics are made available through attributes on
                the model object having similar corresponding names.

        Returns:
            The estimated responses at the corresponding locations.
        """
        self.fit(x=x, y=y)
        return self.predict(x=x, metrics=metrics)

    __call__ = fit_and_predict
