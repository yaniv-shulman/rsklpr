from typing import Tuple

import numpy as np
from scipy.special import gamma as gamma_function


def benchmark_curve_exponential(num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from a exponential distribution where the mean is a smooth curve.

    Args:
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

    y_true = y_true - y_true.min() + 0.1

    y: np.ndarray = generator.exponential(scale=y_true)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_log_normal(num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from a log-normal distribution where the mean matches a smooth curve.

    Args:
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)
    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    x = np.maximum(x, 0)  # Ensure x is non-negative
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = np.abs(np.sin(2 * np.pi * x) + 0.5 * np.power(x, 1.5))
    y_true = y_true - y_true.min() + 0.1

    sigma = 0.5  # Standard deviation of the log-normal distribution
    mu = np.log(y_true) - sigma**2 / 2

    y: np.ndarray = generator.lognormal(mean=mu, sigma=sigma)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_gamma(num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from a gamma distribution where the mean matches a smooth curve.

    Args:
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)
    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = np.abs(np.power(x, 2) - 2 * x + 0.5)
    y_true = y_true - y_true.min() + 0.1

    shape = 2.0  # Gamma shape parameter
    scale = y_true / shape

    y: np.ndarray = generator.gamma(shape=shape, scale=scale)

    return (
        x,
        y,
        y_true,
    )


def benchmark_curve_weibull(num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dataset of points sampled from a Weibull distribution where the mean matches a smooth curve.

    Args:
        num_points: The number of points sampled from the curve.

    Returns:
        The predictor, response and ground truth.
    """
    generator: np.random.Generator = np.random.default_rng(seed=14)
    x: np.ndarray = np.linspace(start=0.0, stop=1.0, num=num_points)
    x += generator.normal(scale=1 / np.sqrt(num_points), size=x.shape[0])
    sort_idx: np.ndarray = np.argsort(a=x)
    x = x[sort_idx]

    y_true: np.ndarray = np.abs(np.cos(np.pi * x) + x * x)
    y_true = y_true - y_true.min() + 0.1

    shape = 1.5  # Weibull shape parameter
    scale = y_true / gamma_function(1 + 1 / shape)

    y: np.ndarray = scale * np.power(-np.log(1 - generator.uniform(size=num_points)), 1 / shape)

    return (
        x,
        y,
        y_true,
    )
