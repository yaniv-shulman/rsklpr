"""
Tests that the local copy of KDEMultivariate and select_bandwidth that are used in rsklpr are consistent with the
implementation in statsmodels
"""
from typing import Union, Sequence, Callable

import numpy as np
import pytest
from statsmodels.nonparametric.bandwidths import select_bandwidth as sm_select_bandwidth
from statsmodels.nonparametric.kernel_density import KDEMultivariate as SM_KDEMultivariate

from rsklpr.kde_statsmodels_impl.bandwidths import select_bandwidth
from rsklpr.kde_statsmodels_impl.kernel_density import KDEMultivariate
from tests.rsklpr_tests.utils import generate_linear_1d, generate_linear_nd


def test_kdemultivariate_public_attributes_are_the_same() -> None:
    """
    Tests the public attributes for KDEMultivariate are consistent with statsmodels.
    """
    data: np.ndarray = generate_linear_1d(start=-8, stop=3)
    target: KDEMultivariate = KDEMultivariate(data=data, var_type="c")
    sm_ref: SM_KDEMultivariate = SM_KDEMultivariate(data=data, var_type="c")
    assert sorted(dir(target)) == sorted(dir(sm_ref))

    attr: str

    for attr in dir(target):
        if not attr.startswith("_") and not isinstance(getattr(sm_ref, attr), Callable):  # type: ignore [arg-type]
            correct: Union[bool, Sequence[bool], np.ndarray] = getattr(target, attr) == getattr(sm_ref, attr)
            if isinstance(correct, Sequence) or isinstance(correct, np.ndarray):
                assert all(getattr(target, attr) == getattr(sm_ref, attr))
            else:
                assert correct


@pytest.mark.parametrize(
    argnames="data",
    argvalues=[generate_linear_1d() for _ in range(3)]
    + [generate_linear_nd(dim=2) for _ in range(3)]
    + [generate_linear_nd(dim=5)],
)
@pytest.mark.parametrize(
    argnames="bw",
    argvalues=[[0.1], "normal_reference", "cv_ml", "cv_ls"],
)
def test_kdemultivariate_density_is_equal(data: np.ndarray, bw: Union[str, Sequence[float]]) -> None:
    """
    Tests the density estimates for KDEMultivariate are consistent with statsmodels.
    """
    var_type: str = "c"

    if data.ndim > 1:
        var_type = "c" * data.shape[-1]  # type: ignore [operator, assignment]
        bw = bw * data.shape[-1]  # type: ignore [operator, assignment]

    target: KDEMultivariate = KDEMultivariate(data=data, var_type=var_type, bw=bw)
    sm_ref: SM_KDEMultivariate = SM_KDEMultivariate(data=data, var_type=var_type, bw=bw)
    np.testing.assert_allclose(actual=target.cdf(), desired=sm_ref.cdf())
    np.testing.assert_allclose(actual=target.pdf(), desired=sm_ref.pdf())


@pytest.mark.parametrize(
    argnames="x",
    argvalues=[generate_linear_1d() for _ in range(3)]
    + [generate_linear_nd(dim=2) for _ in range(3)]
    + [generate_linear_nd(dim=5)],
)
@pytest.mark.parametrize(
    argnames="bw",
    argvalues=["scott", "normal_reference"],
)
def test_select_bandwidth_expected_values(x: np.ndarray, bw: str) -> None:
    """
    Tests select_bandwidth is consistent with statsmodels.
    """
    actual: Union[float, np.ndarray] = select_bandwidth(x=x, bw=bw, kernel=None)
    expected: Union[float, np.ndarray] = sm_select_bandwidth(x=x, bw=bw, kernel=None)
    np.testing.assert_allclose(actual=actual, desired=expected)
