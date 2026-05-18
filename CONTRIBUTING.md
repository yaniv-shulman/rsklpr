# Contributing to rsklpr #

`rsklpr` is a Python package for robust and generalized local polynomial regression. Contributions are welcome for the package implementation, documentation, examples, tests, and related research material.

The repository now supports both:

- the robust similarity-kernel method described in ["Robust Local Polynomial Regression with Similarity Kernels"](https://arxiv.org/abs/2501.10729);
- the generalized arbitrary-kernel and compound-kernel capabilities used by the GC-LPR framework described in ["Generalized Local Polynomial Regression with Decomposed Context-Aware Kernels"](https://arxiv.org/abs/2604.25237).

## Contribution and feedback ##

Useful contribution areas include:

- Improving the local polynomial regression implementation, including numerical stability, input validation, performance, and multiprocessing.
- Adding or improving kernels, including standard kernels, custom-kernel examples, and compound / product kernel workflows.
- Improving robust similarity-kernel behavior, including robust bandwidth estimators, KDE choices, and robustness experiments.
- Adding tests for edge cases, metrics, bootstrap inference, multivariate inputs, arbitrary polynomial degrees, and custom kernels.
- Improving documentation, examples, notebooks, and comparison experiments.
- Fixing issues in the paper source, references, figures, or explanations.
- Porting ideas to other languages or providing interoperability examples.

To contribute, please open a pull request, create an issue, or get in touch by email using the address specified in the paper.

## Development setup ##

Install the package dependencies with Poetry:

```bash
poetry install
```

For development tools such as Black, Ruff, mypy, pytest, and coverage:

```bash
poetry install --with dev
```

For normal package development, do not use `--no-root`: installing the package itself helps catch packaging and import issues. Use `--no-root` only when you intentionally want dependencies without installing `rsklpr`, for example in a notebook environment where `PYTHONPATH` is configured separately.

For experiment notebooks and plotting dependencies:

```bash
poetry install --with experiments
```

The helper script `configure.sh` installs both the `dev` and `experiments` groups and starts a Poetry shell:

```bash
./configure.sh
```

## Checks before submitting ##

Please run the same checks used by CI before opening a pull request:

```bash
poetry run tests/lint.sh
poetry run pytest -n auto
```

To apply fixable Black/Ruff changes, run:

```bash
poetry run tests/lint.sh -f
```

If you changed documentation or citation metadata, also check Markdown/CFF formatting where possible:

```bash
git diff --check
cffconvert --validate --infile CITATION.cff
```

`cffconvert` is optional and can be installed in a separate virtual environment if it conflicts with local development dependencies.

## Experiments ##

Experiments and demonstrations for `rsklpr` are available as interactive Jupyter notebooks:
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/src/experiments/

Experiments for the GC-LPR paper are maintained separately in the `gclpr` repository:
https://github.com/yaniv-shulman/gclpr

## Example usage for developers ##

```python
import numpy as np

from rsklpr.rsklpr import Rsklpr

x: np.ndarray = np.linspace(0, 1, 100)
y: np.ndarray = np.sin(2 * np.pi * x)

model = Rsklpr(size_neighborhood=20)
y_hat: np.ndarray = model.fit_and_predict(x=x, y=y)
```

For more complete examples, see the usage notebook:
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/docs/usage.ipynb
