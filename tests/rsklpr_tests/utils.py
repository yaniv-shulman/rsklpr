from typing import Optional

import numpy as np

rng: np.random.Generator = np.random.default_rng(seed=12)


def generate_linear_1d(start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    if start is None:
        start = rng.integers(low=-10, high=10)

    if stop is None:
        stop = rng.integers(low=-10, high=10)

    x: np.ndarray = np.linspace(start=start, stop=stop, num=num) + rng.uniform(low=-0.001, high=0.001, size=num)
    return x


def generate_linear_nd(dim: int, start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    return np.concatenate(
        [generate_linear_1d(start=start, stop=stop, num=num).reshape((-1, 1)) for _ in range(dim)],
        axis=1,
    )


def generate_quad_1d(start: Optional[int] = None, stop: Optional[int] = None, num: int = 50) -> np.ndarray:
    return np.square(generate_linear_1d(start=start, stop=stop, num=num))  # type: ignore[no-any-return]


def generate_sin_1d(start: Optional[int] = 0, stop: Optional[int] = 1, num: int = 50) -> np.ndarray:
    return np.sin(generate_linear_1d(start=start, stop=stop, num=num) * 2 * np.pi)  # type: ignore[no-any-return]
