from typing import Callable

import numpy as np
import pytest

from rsklpr.kernels import (
    _normalize_distances,
    laplacian_normalized_metric,
    tricube_normalized_metric,
    gaussian_normalized_metric,
    epanechnikov_normalized_metric,
    triangular_normalized_metric,
    biweight_normalized_metric,
    triweight_normalized_metric,
    uniform_normalized_metric,
)


@pytest.mark.parametrize(
    "u, expected",
    [
        # Standard case, 0 min
        (np.array([[0.0, 2.0, 5.0]]), np.array([[0.0, 0.4, 1.0]])),
        # Non-zero min
        (np.array([[5.0, 6.0, 9.0]]), np.array([[0.0, 0.25, 1.0]])),
        # All equal (edge case for division by zero)
        (np.array([[2.0, 2.0, 2.0]]), np.array([[0.0, 0.0, 0.0]])),
        # Single value
        (np.array([[10.0]]), np.array([[0.0]])),
        # Negative values
        (np.array([[-5.0, -2.0, 0.0]]), np.array([[0.0, 0.6, 1.0]])),
    ],
)
def test_normalize_distances(u: np.ndarray, expected: np.ndarray) -> None:
    """Tests the _normalize_distances helper function directly."""
    actual: np.ndarray = _normalize_distances(u=u)
    np.testing.assert_allclose(actual=actual, desired=expected)


# --- Test Data for All Kernels ---

# Dummy arguments for the kernel functions, as they are ignored
DUMMY_ARG: np.ndarray = np.empty(1)

# u1: Standard case, 0 min
u1: np.ndarray = np.array([[0.0, 2.0, 5.0]])
u1_norm: np.ndarray = np.array([[0.0, 0.4, 1.0]])

# u2: Non-zero min
u2: np.ndarray = np.array([[3.0, 4.0, 7.0]])
u2_norm: np.ndarray = np.array([[0.0, 0.25, 1.0]])

# u3: All equal (edge case)
u3: np.ndarray = np.array([[5.0, 5.0, 5.0]])
u3_norm: np.ndarray = np.array([[0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "kernel_func, u, u_norm, expected",
    [
        # --- Laplacian ---
        (laplacian_normalized_metric, u1, u1_norm, np.exp(-u1_norm)),
        (laplacian_normalized_metric, u2, u2_norm, np.exp(-u2_norm)),
        (laplacian_normalized_metric, u3, u3_norm, np.exp(-u3_norm)),
        # --- Tricube ---
        (
            tricube_normalized_metric,
            u1,
            u1_norm,
            np.power(1.0 - np.power(u1_norm, 3), 3),
        ),
        (
            tricube_normalized_metric,
            u2,
            u2_norm,
            np.power(1.0 - np.power(u2_norm, 3), 3),
        ),
        (
            tricube_normalized_metric,
            u3,
            u3_norm,
            np.power(1.0 - np.power(u3_norm, 3), 3),
        ),
        # --- Gaussian ---
        (
            gaussian_normalized_metric,
            u1,
            u1_norm,
            np.exp(-0.5 * np.square(u1_norm)),
        ),
        (
            gaussian_normalized_metric,
            u2,
            u2_norm,
            np.exp(-0.5 * np.square(u2_norm)),
        ),
        (
            gaussian_normalized_metric,
            u3,
            u3_norm,
            np.exp(-0.5 * np.square(u3_norm)),
        ),
        # --- Epanechnikov ---
        (
            epanechnikov_normalized_metric,
            u1,
            u1_norm,
            np.clip(1.0 - np.square(u1_norm), 0.0, None),
        ),
        (
            epanechnikov_normalized_metric,
            u2,
            u2_norm,
            np.clip(1.0 - np.square(u2_norm), 0.0, None),
        ),
        (
            epanechnikov_normalized_metric,
            u3,
            u3_norm,
            np.clip(1.0 - np.square(u3_norm), 0.0, None),
        ),
        # --- Triangular ---
        (
            triangular_normalized_metric,
            u1,
            u1_norm,
            np.clip(1.0 - u1_norm, 0.0, None),
        ),
        (
            triangular_normalized_metric,
            u2,
            u2_norm,
            np.clip(1.0 - u2_norm, 0.0, None),
        ),
        (
            triangular_normalized_metric,
            u3,
            u3_norm,
            np.clip(1.0 - u3_norm, 0.0, None),
        ),
        # --- Biweight ---
        (
            biweight_normalized_metric,
            u1,
            u1_norm,
            np.clip(np.square(1.0 - np.square(u1_norm)), 0.0, None),
        ),
        (
            biweight_normalized_metric,
            u2,
            u2_norm,
            np.clip(np.square(1.0 - np.square(u2_norm)), 0.0, None),
        ),
        (
            biweight_normalized_metric,
            u3,
            u3_norm,
            np.clip(np.square(1.0 - np.square(u3_norm)), 0.0, None),
        ),
        # --- Triweight ---
        (
            triweight_normalized_metric,
            u1,
            u1_norm,
            np.clip(np.power(1.0 - np.square(u1_norm), 3), 0.0, None),
        ),
        (
            triweight_normalized_metric,
            u2,
            u2_norm,
            np.clip(np.power(1.0 - np.square(u2_norm), 3), 0.0, None),
        ),
        (
            triweight_normalized_metric,
            u3,
            u3_norm,
            np.clip(np.power(1.0 - np.square(u3_norm), 3), 0.0, None),
        ),
        # --- Uniform ---
        (uniform_normalized_metric, u1, u1_norm, np.ones_like(u1_norm)),
        (uniform_normalized_metric, u2, u2_norm, np.ones_like(u2_norm)),
        (uniform_normalized_metric, u3, u3_norm, np.ones_like(u3_norm)),
    ],
)
def test_all_normalized_kernels_expected_output(
    kernel_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    u: np.ndarray,
    u_norm: np.ndarray,  # (unused, for clarity)
    expected: np.ndarray,
) -> None:
    """
    Tests that all normalized kernels return the expected output for various inputs.
    """
    actual: np.ndarray = kernel_func(DUMMY_ARG, DUMMY_ARG, u)
    np.testing.assert_allclose(actual=actual, desired=expected)
