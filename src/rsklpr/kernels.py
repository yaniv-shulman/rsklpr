import numpy as np


def _normalize_distances(u: np.ndarray) -> np.ndarray:
    """
    Normalizes distances for LOESS-style kernels by scaling by the maximum distance in the neighborhood.

    Args:
        u: The neighbors metric, 1D.

    Returns:
        The normalized distances in the range [0, 1].
    """
    u_shifted: np.ndarray = u - u.min()
    u_max: np.ndarray = np.max(np.atleast_2d(u_shifted), axis=1, keepdims=True).astype(float)

    # Use 1.0 as the denominator if u_max is 0 to avoid division by zero.
    # This correctly results in a normalized distance of 0 for all points.
    u_max = np.where(u_max > np.finfo(u_max.dtype).eps, u_max, 1.0)

    return u_shifted / u_max  # type: ignore [no-any-return]


def laplacian_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Laplacian kernel on the neighbours metric. The inputs are first scaled to the range [0,1]
    before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)
    return np.exp(-u_normalized)  # type: ignore [no-any-return]


def tricube_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Tricube kernel on the neighbours metric. The inputs are scaled to the range [0,1]
    before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=np.power(1.0 - np.power(u_normalized, 3), 3),
        a_min=0.0,
        a_max=None,
    )


def gaussian_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Gaussian kernel on the neighbours metric. The inputs are scaled to the range [0,1]
    before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)
    return np.exp(-0.5 * np.square(u_normalized))  # type: ignore [no-any-return]


def epanechnikov_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Epanechnikov kernel on the neighbours metric. The inputs are scaled to the range
    [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=1.0 - np.square(u_normalized),
        a_min=0.0,
        a_max=None,
    )


def triangular_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Triangular (or "cone") kernel on the neighbours metric.
    The inputs are scaled to the range [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=1.0 - u_normalized,
        a_min=0.0,
        a_max=None,
    )


def biweight_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Biweight (or "quartic") kernel on the neighbours metric.
    The inputs are scaled to the range [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=np.square(1.0 - np.square(u_normalized)),
        a_min=0.0,
        a_max=None,
    )


def triweight_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Triweight kernel on the neighbours metric.
    The inputs are scaled to the range [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_normalized: np.ndarray = _normalize_distances(u)

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=np.power(1.0 - np.square(u_normalized), 3),
        a_min=0.0,
        a_max=None,
    )


def uniform_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Uniform (or "boxcar") kernel.
    All neighbors in the window are weighted equally.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output (an array of ones).
    """
    u_normalized: np.ndarray = _normalize_distances(u)
    # All normalized distances are in [0, 1], so the kernel is 1 for all.
    return np.ones_like(u_normalized)  # type: ignore [no-any-return]
