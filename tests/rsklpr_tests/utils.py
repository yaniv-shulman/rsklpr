from typing import Optional

import numpy as np

rng: np.random.Generator = np.random.default_rng(seed=12)


def generate_linear_1d(start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    """
    Generates a roughly linearly spaced 1D array.

    Args:
        start: If provided this will be the initial value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        stop: If provided this will be the last value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        num: The number of points to generate.

    Returns:
        The created linearly spaced 1D array.
    """
    if start is None:
        start = rng.integers(low=-10, high=10)

    if stop is None:
        stop = rng.integers(low=-10, high=10)

    x: np.ndarray = np.linspace(start=start, stop=stop, num=num) + rng.uniform(low=-0.001, high=0.001, size=num)
    return x


def generate_linear_nd(dim: int, start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    """
    Generates a roughly linearly spaced ND array where the dimension are independent of each other.

    Args:
        dim: The dimension of the array.
        start: If provided this will be the initial value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        stop: If provided this will be the last value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        num: The number of points to generate.

    Returns:
        The created linearly spaced 1D array.
    """

    return np.concatenate(
        [generate_linear_1d(start=start, stop=stop, num=num).reshape((-1, 1)) for _ in range(dim)],
        axis=1,
    )


def generate_quad_1d(start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    """
    Generates a roughly quadratic valued 1D array.

    Args:
        start: If provided this will be the initial value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        stop: If provided this will be the last value for the array. If not provided, the method will sample an
            integer from -10,...,10.
        num: The number of points to generate.

    Returns:
        The created 1D array.
    """

    return np.square(generate_linear_1d(start=start, stop=stop, num=num))  # type: ignore[no-any-return]


def generate_sin_1d(start: Optional[int] = 0, stop: Optional[int] = 1, num: int = 50) -> np.ndarray:
    """
    Generates a roughly sinusoidal 1D array.

    Args:
        start: If provided this will be the initial value for the array.
        stop: If provided this times 2 pi will be the last value for the array.
        num: The number of points to generate.

    Returns:
        The created 1D array.
    """

    return np.sin(generate_linear_1d(start=start, stop=stop, num=num) * 2 * np.pi)  # type: ignore[no-any-return]
