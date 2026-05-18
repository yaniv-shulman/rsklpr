# rsklpr: Robust and Generalized Local Polynomial Regression in Python #

[![Downloads](https://static.pepy.tech/badge/rsklpr)](https://pepy.tech/project/rsklpr) ![Tests](https://github.com/yaniv-shulman/rsklpr/actions/workflows/linting_and_tests.yml/badge.svg?branch=main) [![Pyversions](https://img.shields.io/pypi/pyversions/rsklpr.svg?style=flat-square)](https://pypi.python.org/pypi/rsklpr) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Robust local polynomial regression, arbitrary-kernel local polynomial regression, and compound-kernel support for context-aware smoothing.

## TL;DR ##

`rsklpr` provides a practical Python implementation of robust and generalized local polynomial regression. It can be used as a robust alternative to [LOESS](https://en.wikipedia.org/wiki/Local_regression)-style smoothing when observations may contain outliers, high-leverage points, heteroscedastic noise, or sparse regions. Starting with version 2.0, the package also supports arbitrary kernels through the `kp` parameter, including compound / product kernels, making it useful for generalized local polynomial regression where the fitting coordinates and weighting context need not be identical.

The package supports the robust similarity-kernel method from ["Robust Local Polynomial Regression with Similarity Kernels"](https://arxiv.org/abs/2501.10729) and the generalized kernel capabilities used by the GC-LPR framework described in ["Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels"](https://arxiv.org/abs/2604.25237).

This library may be useful when:

1. You want a flexible nonparametric regression method and do not want to specify a global parametric model.
1. The underlying regression function is expected to be reasonably smooth locally, but may be nonlinear globally.
1. The data may contain outliers, high-leverage points, heteroscedasticity, or other non-ideal noise patterns.
1. You want a robust local polynomial smoother based on similarity-kernel weighting.
1. You want a standard local polynomial regression implementation with off-the-shelf kernels.
1. You want to define your own kernel or kernel parameters through the generalized `kp` interface.
1. You want to use compound / product kernels to combine several sources of similarity.
1. You want to fit locally in one Euclidean feature space while weighting observations using additional context variables.
1. You want to predict at new locations that are not too far from the observed data support.
1. Your independent inputs are univariate or multivariate.
1. Your dependent variable is univariate.
1. You want a straightforward way to tune smoothness, polynomial degree, and kernel behavior.
1. You want bootstrap-based confidence intervals.
1. You want to denoise, impute, resample, or smooth irregular data.

For very dense, clean data with approximately Gaussian noise and no substantial outliers, classical LOESS or standard local polynomial regression may be sufficient. The robust similarity-kernel mode is most useful when local neighborhoods may be contaminated or sparse. The generalized kernel interface is most useful when Euclidean distance in the fitting variables alone is not the right notion of locality.

## Installation ##

Install from [PyPI](https://pypi.org/project/rsklpr/) using pip:

```bash
pip install rsklpr
```

## Quick start ##

```python
from rsklpr.rsklpr import Rsklpr

model = Rsklpr(size_neighborhood=30)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
```

To use standard local polynomial regression without robust KDE-based response weighting, set `kr="none"` and choose a kernel through `kp`:

```python
from rsklpr.kernels import tricube_normalized_metric
from rsklpr.rsklpr import Rsklpr

model = Rsklpr(
    size_neighborhood=30,
    degree=1,
    kp=tricube_normalized_metric,
    kr="none",
)
```

To use compound / product kernels, pass an iterable of kernel callables. Their weights are multiplied inside each local neighborhood:

```python
from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from rsklpr.rsklpr import Rsklpr

model = Rsklpr(
    size_neighborhood=30,
    kp=[tricube_normalized_metric, laplacian_normalized_metric],
    kr="none",
)
```

See the usage notebook for a fuller example:
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/docs/usage.ipynb

## Conceptual overview ##

In `rsklpr`, kernels are the main mechanism for defining locality. The same local polynomial fitting engine can therefore be used for several purposes: standard distance-based smoothing, robust weighting, custom similarity weighting, and compound context-aware weighting. The `kp` parameter controls the primary neighborhood kernel, while `kr` controls whether an additional robust KDE-based response-similarity kernel is used.

The original RSKLPR method uses this kernel structure for robustness. A geometric neighborhood is first selected around each prediction point, and the robust similarity-kernel mode then uses localized density information to reduce the influence of observations that are atypical within that neighborhood. This is useful when local fits may be affected by outliers, high-leverage observations, heteroscedastic noise, or sparse regions.

The generalized interface added in version 2.0 makes the primary kernel arbitrary. Users can supply built-in kernels, custom kernel callables, or an iterable of kernels through `kp`. When multiple kernels are supplied, their weights are multiplied, producing a compound / product kernel.

Compound kernels are the bridge to GC-LPR-style context-aware smoothing. Classical local polynomial regression usually uses the same Euclidean variables to define both the local neighborhood and the polynomial fit. GC-LPR decouples these roles: the polynomial is fitted in a primary Euclidean fitting space, while observation weights may also be computed from separate context variables such as graph distance, network structure, manifold coordinates, categories, spatial context, time, or mixed data.

## What's new? ##

- Version 2.0.0:
  - Improved numerical stability.
  - Generalized the API for arbitrary kernels through the `kp` parameter.
  - Added support for disabling the robust KDE-based kernel with `kr="none"`, allowing the package to be used as a standard local polynomial regression implementation.
  - Added support for compound / product kernels by passing an iterable of kernel callables to `kp`, enabling context-aware weighting schemes.
  - Added additional off-the-shelf kernels.
  - Added support for arbitrary polynomial degree in the local fit.
  - These changes make the package suitable both for the original RSKLPR method and for generalized context-aware LPR workflows such as GC-LPR.
- Version 1.0.0 - Dropped support for Python 3.8 and added support for Python 3.12.
- Version 0.7.0 - Metrics including local R-squared and more efficient computation of WLS.
- Version 0.6.0 - Bootstrap inference and confidence intervals.

## When should I use which mode? ##

`rsklpr` can be used in several modes:

- Robust similarity-kernel LPR, following the original RSKLPR paper.
- Standard local polynomial regression with conventional kernels.
- Custom-kernel LPR using user-specified kernel behavior through `kp`.
- Compound-kernel LPR for context-aware or decomposed weighting schemes.

Use robust similarity-kernel LPR when your data may contain outliers or high-leverage observations, local neighborhoods may be contaminated, or you want the method described in the RSKLPR paper.

Use standard or arbitrary-kernel LPR when you want local polynomial regression with a specific kernel, you want to compare kernels experimentally, or you want to disable robust KDE-based weighting.

Use compound / context-aware kernels when the variables used for the polynomial fit are not sufficient to define locality; observations have graph, network, spatial, categorical, temporal, manifold, or mixed-data context; or you want the GC-LPR-style separation between fitting space and weighting context.

## Details ##

### Local polynomial regression ###

Local polynomial regression generalizes moving-average and polynomial-regression ideas by fitting a low-degree polynomial in a local neighborhood around the prediction point. Nearby observations receive larger weights, and more distant observations receive smaller weights. This gives a flexible nonparametric estimator that can model nonlinear relationships without specifying a global parametric form.

### Robust similarity-kernel LPR ###

The original `rsklpr` method is described in ["Robust Local Polynomial Regression with Similarity Kernels"](https://arxiv.org/abs/2501.10729). That method modifies the local weighting mechanism to improve robustness to outliers and high-leverage observations. Instead of relying only on geometric proximity in the predictor space, the robust similarity-kernel approach uses localized density information to reduce the influence of observations that are atypical within the local neighborhood.

Use this mode when the main concern is robustness: noisy data, outliers, sparse samples, heteroscedasticity, or high-leverage observations.

### Generalized arbitrary-kernel LPR ###

Starting with version 2.0, `rsklpr` generalizes the kernel interface through the `kp` parameter. This allows users to disable the robust KDE-based kernel when desired and use the package as a standard or generalized local polynomial regression engine with arbitrary kernels.

This makes `rsklpr` useful not only for the original robust estimator, but also for experiments where locality is defined by custom similarity functions.

### Compound and context-aware kernels ###

The generalized kernel interface also supports compound / product kernels. In `rsklpr`, pass an iterable of kernel callables to `kp`; the resulting neighborhood weights are multiplied together.

Compound kernels are useful when the regression should be fitted in one set of variables but weighted using another source of contextual similarity. This is the idea used in ["Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels"](https://arxiv.org/abs/2604.25237) and the experimental [`gclpr` code](https://github.com/yaniv-shulman/gclpr). In that framework, the local polynomial fit is performed in a primary Euclidean fitting space, while the local neighborhood weights are computed using a context space that may represent graph distance, network structure, manifold coordinates, categories, spatial context, time, or mixed data.

This decoupling is useful when "nearby" should not mean only "nearby in the variables used for the polynomial fit."

## Experimental results ##

Experiments and demonstrations for `rsklpr` are available as interactive Jupyter notebooks:
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/src/experiments/

Experiments for the GC-LPR paper are maintained separately in the `gclpr` repository:
https://github.com/yaniv-shulman/gclpr

## Citation ##

### Which paper should I cite? ###

If you use `rsklpr` in academic work, please cite the software package and the relevant method paper(s):

- Cite the software package whenever results depend on the `rsklpr` implementation.
- Cite "Robust Local Polynomial Regression with Similarity Kernels" when using the robust similarity-kernel / KDE-based weighting method.
- Cite "Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels" when using arbitrary, decomposed, context-aware, or compound / product kernels in the GC-LPR sense.
- If your work uses `rsklpr` only as a standard local polynomial regression implementation with a conventional kernel, cite the software package. Cite the method papers only if their specific methods are relevant to the analysis.

### Software package ###

```bibtex
@software{shulman_rsklpr_2026,
  author = {Shulman, Yaniv},
  title = {{rsklpr: Robust and Generalized Local Polynomial Regression in Python}},
  year = {2026},
  version = {2.2.0},
  url = {https://github.com/yaniv-shulman/rsklpr},
  note = {Python package}
}
```

### Robust similarity-kernel LPR ###

```bibtex
@misc{shulman2025robustlocalpolynomialregression,
  title = {Robust Local Polynomial Regression with Similarity Kernels},
  author = {Yaniv Shulman},
  year = {2025},
  eprint = {2501.10729},
  archivePrefix = {arXiv},
  primaryClass = {stat.ME},
  url = {https://arxiv.org/abs/2501.10729}
}
```

### Generalized context-aware LPR ###

```bibtex
@misc{shulman2026generalizedlocalpolynomialregression,
  title = {Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels},
  author = {Yaniv Shulman},
  year = {2026},
  eprint = {2604.25237},
  archivePrefix = {arXiv},
  primaryClass = {stat.ME},
  doi = {10.48550/arXiv.2604.25237},
  url = {https://arxiv.org/abs/2604.25237}
}
```

## KDE implementation note ##

The KDE implementation used by the robust similarity-kernel mode is adapted from [statsmodels](https://www.statsmodels.org/stable/index.html). The relevant code is included directly to avoid adding statsmodels as a required dependency, since statsmodels is comparatively heavy and pulls in additional dependencies.

## Contribution and feedback ##

Contributions and feedback are most welcome for both the papers and the code. Please see [CONTRIBUTING.md](https://github.com/yaniv-shulman/rsklpr/tree/main/CONTRIBUTING.md) for further details.
