import numpy as np


def laplacian_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Laplacian kernel on the neighbours metric. The implementation requires all inputs are
    non-negative. The inputs are first scaled to the range [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric.

    Returns:
        The kernel output.
    """
    u_shifted: np.ndarray = u - u.min()
    u_max: np.ndarray = np.max(np.atleast_2d(u_shifted), axis=1, keepdims=True).astype(float)

    # Use 1.0 as the denominator if u_max is 0 to avoid division by zero.
    # This correctly results in a normalized distance of 0 for all points.
    u_max = np.where(u_max > np.finfo(u_max.dtype).eps, u_max, 1.0)
    u_normalized: np.ndarray = u_shifted / u_max
    return np.exp(-u_normalized)  # type: ignore [no-any-return]


def tricube_normalized_metric(_: np.ndarray, __: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the normalized Tricube kernel on the neighbours metric. The implementation requires all inputs are
    non-negative. The inputs are scaled to the range [0,1] before applying the kernel.

    Args:
        _: ignored.
        __: ignored.
        u: The neighbours metric, requires all non-negative values.

    Returns:
        The kernel output.
    """
    u_shifted: np.ndarray = u - u.min()

    # Find the max, ensuring it's 2D for broadcasting
    u_max: np.ndarray = np.max(np.atleast_2d(u_shifted), axis=1, keepdims=True).astype(float)

    # Use 1.0 as the denominator if u_max is 0 to avoid division by zero.
    # This correctly results in a normalized distance of 0 for all points.
    u_max = np.where(u_max > np.finfo(u_max.dtype).eps, u_max, 1.0)
    u_normalized: np.ndarray = u_shifted / u_max

    return np.clip(  # type: ignore [no-any-return, call-overload]
        a=np.power(1.0 - np.power(u_normalized, 3), 3),
        a_min=0.0,
        a_max=None,
    )
