# Robust Local Polynomial Regression with Similarity Kernels #

[![Downloads](https://static.pepy.tech/badge/rsklpr)](https://pepy.tech/project/rsklpr) ![Tests](https://github.com/yaniv-shulman/rsklpr/actions/workflows/linting_and_tests.yml/badge.svg?branch=main) [![phorm.ai](https://img.shields.io/badge/ask%20phorm.ai-8A2BE2)](https://www.phorm.ai/query?projectId=04cda149-26bc-4050-bfa6-5b28ee16ceda) [![Pyversions](https://img.shields.io/pypi/pyversions/rsklpr.svg?style=flat-square)](https://pypi.python.org/pypi/rsklpr)

## TL;DR ##
This library is useful to perform robust locally weighted regression (similar to [LOESS](https://en.wikipedia.org/wiki/Local_regression)) in Python when:
1. There are no particular assumptions on the underlying function except that it is "reasonably smooth". In particular,
you don't know which parametric model to specify or if an appropriate model exists. 
1. There are no particular assumptions on the type and intensity of noise present.
1. There are no particular assumptions on the presence of outliers and their extent.
1. You may want to predict in locations not explicitly present in the dataset but also not too far from existing
observations or far outside the areas where observations exist. 
1. The independent inputs are univariate or multivariate.
1. The dependent variable is univariate.
1. You want a straightforward hassle-free way to tune the model and the smoothness of fit.
1. You may want to calculate confidence intervals.
1. You may want to filter noise to recover the original underlying process.
1. You may want to impute or resample the data. 

If the above use cases hold then this library could be useful for you. Have a look at this notebook
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/docs/usage.ipynb for an example of how to use
this library to perform regression easily.

## Installation ##
Install from [PyPI](https://pypi.org/project/rsklpr/) using pip (preferred method):
```bash
pip install rsklpr
```

### What's new? ###
- Version 1.0.0 - Dropped support for Python 3.8 and added support for Python 3.12.
- Version 0.7.0 - Metrics including local R-Squared and more efficient computation of WLS. 
- Version 0.6.0 - Bootstrap inference and confidence intervals


## Details ##
Local polynomial regression (LPR) is a powerful and flexible statistical technique that has gained increasing popularity
in recent years due to its ability to model complex relationships between variables. Local polynomial regression
generalizes the polynomial regression and moving average methods by fitting a low-degree polynomial to a nearest
neighbors subset of the data at the location. The polynomial is fitted using weighted ordinary least squares, giving
more weight to nearby points and less weight to points further away. Local polynomial regression is however susceptible
to outliers and high leverage points which may cause an adverse impact on the estimation accuracy. This library 
implements a variant of LPR presented in the 
[Robust Local Polynomial Regression with Similarity Kernels draft paper](https://github.com/yaniv-shulman/rsklpr/tree/main/paper/rsklpr.pdf) which uses a generalized similarity kernel
that assign robust weights to mitigate the adverse effect of outliers in the local neighborhood by estimating and
utilizing the density at the local locations. 


### Experimental results ###
The experimental results and demonstration of the library for various experimental settings are available as interactive
Jupyter notebooks at https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/src/experiments/

### KDE Implementation ###
KDE implementation is a copy of the code from statsmodels https://www.statsmodels.org/stable/index.html. The copy is done to
remove statsmodels as a dependency of this package since statsmodels is quite heavy and pulls a lot of additional
packages.

## Contribution and feedback ##
The paper is work in progress and the library in early stages of development but both are in a useful state.
Contributions and feedback are most welcome both to the paper and the code. Please see
[CONTRIBUTING.md](https://github.com/yaniv-shulman/rsklpr/tree/main/CONTRIBUTING.md) for further details.
