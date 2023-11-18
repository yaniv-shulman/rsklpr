import warnings
from numbers import Number
from typing import Optional, Sequence, Tuple, Callable, List, Union

import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.bandwidths import select_bandwidth


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


def _weighted_local_regression(
    x_0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    degree: int,
):
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
    y = y.reshape((x.shape[0]), 1)
    bias: np.ndarray = np.ones(shape=(x.shape[0], 1))
    x_mat: np.ndarray = (
        np.concatenate(
            [bias, x - x_0],
            axis=1,
        )
        if degree == 1
        else bias
    )

    w_mat: np.ndarray = np.diag(np.squeeze(weights))

    return (np.linalg.inv(x_mat.T @ w_mat @ x_mat) @ x_mat.T @ w_mat @ y)[0]


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


def _laplacian(u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Laplacian kernel assuming all inputs are non-negative.

    Args:
        u: The kernel input, note it is assumed all inputs are non-negative.

    Returns:
        The kernel output.
    """
    return np.exp(-u)  # type: ignore [no-any-return]


def _tricube(u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Tricube kernel assuming all inputs are non-negative.

    Args:
        u: The kernel input, note it is assumed all inputs are non-negative.

    Returns:
        The kernel output.
    """
    return np.power((1 - np.power(u, 3)), 3)


class Rsklpr:
    """
    Implementation of the Robust Similarity Kernel Local Polynomial Regression for proposed in the paper
    https://github.com/yaniv-shulman/rsklpr/blob/main/paper/rsklpr.pdf.
    """

    def __init__(
        self,
        size_neighborhood: int,
        degree: int = 1,
        k1: str = "laplacian",
        k2: str = "joint",
        bw1: Union[str, Sequence[float], Callable[[...], Sequence[float]]] = "normal_reference",  # type: ignore [misc]
        bw2: Union[str, Sequence[float], Callable[[...], Sequence[float]]] = "normal_reference",  # type: ignore [misc]
        bw_global_subsample_size: Optional[int] = None,
        seed: int = 888,
    ) -> None:
        """
        Args:
            size_neighborhood: The number of points in the neighborhood to consider in the local regression.
            degree: The degree of the polynomial fitted locally, supported values are 0 or 1 (default) that result in
                local constant and local linear regression respectively.
            k1: The kernel that models the effect of distance on weight between the local target regression to it's
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
                (X,Y). For the 'joint' kernel this is the bandwidth estimation method for marginal KDE of Y. The supported
                options are the same as bw1.
            bw_global_subsample_size: The size of subsample taken from the data for global cross validation bandwidth
                estimation. If None the entire data is used for bandwidth estimation. This could be useful to speedup
                    global cross validation based estimates.
            seed: The seed used for random sub sampling for cross validation bandwidth estimation.
        """
        if degree not in (0, 1):
            raise ValueError("Degree must be one of 0 or 1")

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

        if isinstance(bw2, str):
            bw2 = bw2.lower()

            if bw2 not in bw_methods:
                raise ValueError(bw_error_str)

        self._size_neighborhood: int = size_neighborhood
        self._degree: int = degree

        self._k1: Callable[[np.ndarray], np.ndarray] = _laplacian if k1 == "laplacian" else _tricube

        self._k2: str = k2
        self._bw1: Union[str, Sequence[float], Callable[[...], Sequence[float]]] = bw1  # type: ignore [misc]
        self._bw2: Union[str, Sequence[float], Callable[[...], Sequence[float]]] = bw2  # type: ignore [misc]
        self._bw_global_subsample_size: Optional[int] = bw_global_subsample_size
        self._rnd_gen: np.random.Generator = np.random.default_rng(seed=seed)
        self._n_x: np.ndarray = np.ndarray(shape=())
        self._n_y: np.ndarray = np.ndarray(shape=())
        self._min_x: float = 0.0
        self._max_x: float = 0.0
        self._min_y: float = 0.0
        self._max_y: float = 0.0

        self._nearest_neighbors: NearestNeighbors = NearestNeighbors(
            n_neighbors=self._size_neighborhood, algorithm="ball_tree"
        )

    def _denormalize_y(self, n_y: np.ndarray) -> np.ndarray:
        """
        Maps the response to its original range.

        Args:
            n_y: Values that represent normalized response values.

        Returns:
            The inputs mapped to the original range of the response.
        """
        return n_y * (self._max_y - self._min_y) + self._min_y

    def _calculate_bandwidth(  # type: ignore [return]
        self,
        bandwidth: Union[str, Callable[[...], Sequence[float]]],  # type: ignore [misc]
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

                return sm.nonparametric.KDEMultivariate(  # type: ignore [no-any-return]
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

        kde_marginal_x: sm.nonparametric.KDEMultivariate = sm.nonparametric.KDEMultivariate(
            data=n_x_neighbors,
            var_type=var_type,
            bw=self._bw1
            if (self._bw1 in ("cv_ls", "cv_ml") or isinstance(self._bw1, List))
            else self._calculate_bandwidth(bandwidth=self._bw1, data=n_x_neighbors)  # type: ignore [arg-type]
            if bw1_global is None
            else bw1_global,
        )

        kde_joint: sm.nonparametric.KDEMultivariate = sm.nonparametric.KDEMultivariate(
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

        local_density: np.ndarray = np.exp(-0.5 * square_dist_n_y_windowed / (bw_y**2))
        local_density = (local_density * weights).sum(axis=-1)
        return local_density

    def _estimate(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]], float],
    ) -> np.ndarray:
        """
        Estimates the value of m(x) at the locations.

        Args:
            x: Predictor values at locations to estimate m(x), these should be at the original range of the predictor.

        Returns:
            The estimated values of m(x) at the locations.
        """
        x_arr: np.ndarray
        x_arr, _ = self._check_and_reshape_inputs(x=x)

        n_x: np.ndarray = _normalize_array(x_arr, min_val=self._min_x, max_val=self._max_x)
        del x_arr
        y_hat: np.ndarray = np.empty((n_x.shape[0]))
        bw1_global: Optional[Sequence[float]]
        bw2_global: Optional[Sequence[float]]
        bw1_global, bw2_global = self._get_bandwidth_global(k2=self._k2)
        i: int

        for i in range(n_x.shape[0]):
            dist_n_x_neighbors: np.ndarray
            indices: np.ndarray
            dist_n_x_neighbors, indices = self._nearest_neighbors.kneighbors(X=n_x[i].reshape(1, -1))
            dist_n_x_neighbors = dist_n_x_neighbors / np.max(dist_n_x_neighbors, axis=1, keepdims=True).astype(float)
            weights: np.ndarray = self._k1(dist_n_x_neighbors)
            n_x_neighbors: np.ndarray = self._n_x[indices].squeeze(axis=0)

            if self._k2 == "conden":
                weights *= self._k2_conden(
                    n_x_neighbors=n_x_neighbors,
                    n_y_neighbors=self._n_y[indices].T,
                    bw1_global=bw1_global,
                    bw2_global=bw2_global,
                )
            elif self._k2 == "joint":
                weights *= self._k2_joint(
                    n_x_neighbors=n_x_neighbors,
                    n_y_neighbors=self._n_y[indices].T,
                    dist_n_x_neighbors=dist_n_x_neighbors,
                    bw1_global=bw1_global,
                    bw2_global=bw2_global,
                )

            y_hat[i] = _weighted_local_regression(
                x_0=n_x[i].reshape(1, -1),
                x=n_x_neighbors,
                y=self._n_y[indices].T,
                weights=weights,
                degree=self._degree,
            )

        return self._denormalize_y(y_hat)

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
                data=self._n_x,
            )

        bw2_global: Optional[Sequence[float]] = None

        if self._bw2 == "cv_ls_global":
            if k2 == "conden":
                bw2_global = self._calculate_bandwidth(
                    bandwidth=self._bw2,  # type: ignore [arg-type]
                    data=np.concatenate([self._n_x, np.expand_dims(a=self._n_y, axis=-1)], axis=1),
                )
            elif k2 == "joint":
                bw2_global = self._calculate_bandwidth(
                    bandwidth=self._bw2,  # type: ignore [arg-type]
                    data=np.expand_dims(a=self._n_y, axis=-1),
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

        self._min_x = x_arr.min(axis=0)
        self._max_x = x_arr.max(axis=0)
        self._n_x = _normalize_array(array=x_arr, min_val=self._min_x, max_val=self._max_x)
        self._min_y = y_arr.min()
        self._max_y = y_arr.max()

        self._n_y = _normalize_array(array=y_arr, min_val=self._min_y, max_val=self._max_y)

        self._nearest_neighbors.fit(self._n_x)

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
    ) -> np.ndarray:
        """
        Predicts estimates of m(x) at the specified locations. Must call fit with the training data first.

        Args:
            x: The locations to predict for.

        Returns:
            The estimated responses at the corresponding locations.x
        """
        return self._estimate(x=x)

    def fit_and_predict(
        self,
        x: Union[np.ndarray, Sequence[Number], Sequence[Sequence[Number]]],
        y: Union[np.ndarray, Sequence[Number]],
    ) -> np.ndarray:
        """
        Fits the provided dataset and estimates the response at the locations of x.

        Args:
            x: The predictor values
            y: The response values at the corresponding predictor locations.

        Returns:
            The estimated responses at the corresponding locations.
        """
        self.fit(x=x, y=y)
        return self.predict(x=x)

    __call__ = fit_and_predict
