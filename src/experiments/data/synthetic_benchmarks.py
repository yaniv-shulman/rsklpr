from typing import Tuple

import numpy as np


def benchmark_curve_1(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth curve with added noise. In the case of heteroscedastic noise the
    noise scale is sampled from a uniform distribution up to noise_ratio of the response range.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)
    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = np.sqrt(np.abs(np.power(x, 3) - 4 * np.power(x, 4) / 3)) + (
        0.1 * x / np.max(x) * np.sin(x * 3 * np.pi) * np.sin(x * 3 * np.pi)
    )

    scale: np.ndarray

    if hetero:
        scale = generator.uniform(low=0.001, high=noise_ratio * (y_true.max() - y_true.min()), size=x.shape[0])
    else:
        scale = np.full(shape=x.shape[0], fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_2(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth curve with added noise. In the case of heteroscedastic noise the
    noise scale is sampled from a uniform distribution up to noise_ratio of the response range.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=28)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)
    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = (
        -3 * x
        + np.square(x)
        - np.square(np.square(np.power(x, 2)))
        + (1.3 * x / np.max(x) * np.sin(x * -5 * np.pi) * np.sin(x * 0.5 * np.pi))
    )

    if hetero:
        scale = generator.uniform(low=0.001, high=noise_ratio * (y_true.max() - y_true.min()), size=x.shape[0])
    else:
        scale = np.full(shape=x.shape[0], fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_3(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth curve with added noise. In the case of heteroscedastic noise the
    error variance is proportional to the noise_scale parameter and to the predictor variable value at the location.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=661)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)

    x += generator.normal(scale=0.01, size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = (
        5 * x + np.square(x) + np.square(np.square(np.power(x, 3))) + (1.3 * x / np.max(x) * np.sin(x * -2.5 * np.pi))
    )

    if hetero:
        scale = noise_ratio * 3 * np.abs(x)
    else:
        scale = np.full(shape=x.shape[0], fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale, size=x.shape[0])

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_4(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth curve with added noise. In the case of heteroscedastic noise the
    noise scale is sampled from a uniform distribution up to noise_ratio of the response range.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=99)
    x: np.ndarray = np.linspace(start=-5.0, stop=5.0, num=num_points)

    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = np.power(x, 2) * np.cos(x)

    if hetero:
        scale = generator.uniform(low=0.001, high=noise_ratio * (y_true.max() - y_true.min()), size=x.shape[0])
    else:
        scale = np.full(shape=x.shape[0], fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_5(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth curve with added noise. In the case of heteroscedastic noise the
    error scale is proportional to a nonlinear function of the predictor and the response.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=122)
    x: np.ndarray = np.linspace(start=-2.0, stop=7.0, num=num_points)

    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = x / 5 + np.flip(x * np.cos(x))

    if hetero:
        scale = noise_ratio * np.sqrt(np.abs(x * y_true**3) + 1e-4)
    else:
        scale = np.full(shape=x.shape[0], fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_plane_1(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth function with added noise. In the case of heteroscedastic noise
    the noise scale is sampled from a uniform distribution up to noise_ratio of the response range.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the function.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)

    x: np.ndarray = generator.multivariate_normal(mean=(0, 0), cov=((0.5, 0.1), (0.1, 0.2)), size=num_points)

    y_true: np.ndarray = np.clip(
        a=5 * x[:, 0] - 0.5 * np.square(x[:, 1]) + (5.0 * x / np.max(x) * np.sin(x * -2.5 * np.pi)).sum(axis=1),
        a_min=-10,
        a_max=10,
    )

    scale: np.ndarray

    if hetero:
        scale = generator.uniform(
            low=0.001,
            high=noise_ratio * (y_true.max() - y_true.min()),
            size=num_points,
        )
    else:
        scale = np.full(shape=num_points, fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_plane_2(noise_ratio: float, hetero: bool, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from smooth function with added noise. In the case of heteroscedastic noise
    the noise scale is sampled from a uniform distribution up to noise_ratio of the response range.

    Args:
        noise_ratio: The ratio of noise of the response range used to generate noise.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the function.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)

    x: np.ndarray = np.concatenate(
        [
            generator.uniform(low=0.0, high=0.7, size=(num_points, 1)),
            generator.uniform(low=-0.5, high=0.8, size=(num_points, 1)),
        ],
        axis=1,
    )

    y_true: np.ndarray = np.power(x[:, 0], 2) * np.cos(0.5 * np.pi * x[:, 1]) + np.sin(np.square(x.sum(axis=1)))
    scale: np.ndarray

    if hetero:
        scale = generator.uniform(
            low=0.001,
            high=noise_ratio * (y_true.max() - y_true.min()),
            size=num_points,
        )
    else:
        scale = np.full(shape=num_points, fill_value=noise_ratio * (y_true.max() - y_true.min()))

    y: np.ndarray = y_true + generator.normal(scale=scale)

    return (
        x,
        y,
        y_true,
    )
