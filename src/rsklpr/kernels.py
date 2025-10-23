import numpy as np


def laplacian_normalized(_: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Implementation of the Laplacian kernel. The inputs are first scaled to the range [0,1] before applying the kernel.

    Args:
        u: The kernel input.

    Returns:
        The kernel output.
    """
    return np.exp(-u / np.max(np.atleast_2d(u), axis=1, keepdims=True).astype(float))  # type: ignore [no-any-return]


def tricube_normalized(_: np.ndarray, u: np.ndarray) -> np.ndarray:
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
