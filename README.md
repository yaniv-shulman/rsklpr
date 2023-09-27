# Robust Local Polynomial Regression with Similarity Kernels #

This repository is intended to share and facilitate community contribution for completing the research and implementation 
presented in the [Robust Local Polynomial Regression with Similarity Kernels draft paper](https://github.com/yaniv-shulman/rsklpr/tree/main/paper/rsklpr.pdf). The repository contains
the source for the paper and a demonstrative implementation of the proposed method including several experimental results.
Note the paper is a draft and the code is for demonstrative purposes still so both may contain issues.

### Contribution and feedback ###

Contributions and feedback are most welcome to the paper and code in any area related to:
- Further development of the method and completing the paper:
  - Asymptotic analysis of the estimator
  - Improving related work coverage
  - Improving or adding experiments and the presentation of experiments including comparison to other robust LPR methods
  - Experimenting with robust estimators e.g. robust losses, robust bandwidth estimators and robust KDEs
  - Proposing and experimenting with different similarity kernels
  - Fixing issues if found
- Adding and improving functions in the implementation:
  - Proposing and experimenting with additional kernels
  - Improving numerical stability
  - Confidence intervals
  - Implementing in other languages
- Productionzing the code:
  - improving input checks and error handling
  - Tests
  - Packaging

- And more...

To contribute please submit a pull request, create an issue or get in touch by email to the address specified in the
paper.

### How do I get set up? ###
The easiest way to setup for development or explore the code is to create or activate a Poetry virtual environment by
executing configure.sh. The included development environment uses Python 3.10 and Poetry 1.6.1 or higher is recommended.
If you require any help getting setup please get in touch by email to the address specified in the paper.

### Example usage ###

```python
import numpy as np
import pandas as pd

from experiments.common import plot_results, ExperimentConfig
from experiments.data.synthetic_benchmarks import benchmark_curve_1
from rsklpr.rsklpr import Rsklpr

experiment_config: ExperimentConfig = ExperimentConfig(
    data_provider=benchmark_curve_1,
    size_neighborhood=20,
    noise_ratio=0.3,
    hetero=True,
    num_points=150,
    bw1=[0.4],
    bw2="normal_reference",
    k2="joint",
)

x: np.ndarray
y: np.ndarray
y_true: np.ndarray

x, y, y_true = experiment_config.data_provider(
    experiment_config.noise_ratio,
    experiment_config.hetero,
    experiment_config.num_points,
)

rsklpr: Rsklpr = Rsklpr(
    size_neighborhood=experiment_config.size_neighborhood,
    bw1=experiment_config.bw1,
    bw2=experiment_config.bw2,
)

y_hat: np.ndarray = rsklpr(
    x=x,
    y=y,
)

estimates: pd.DataFrame = pd.DataFrame(data=y_hat, columns=["y_hat"])

plot_results(
    x=x,
    y=y,
    y_true=y_true,
    estimates=estimates,
    title="Example usage",
)
```
![Example usage curve_plot](./example_usage_curve.png)


```python
import numpy as np
import pandas as pd

from experiments.common import plot_results, ExperimentConfig
from experiments.data.synthetic_benchmarks import benchmark_plane_2
from rsklpr.rsklpr import Rsklpr

experiment_config: ExperimentConfig = ExperimentConfig(
    data_provider=benchmark_plane_2,
    size_neighborhood=20,
    noise_ratio=0.1,
    hetero=True,
    num_points=100,
    bw1=[0.4],
    bw2="normal_reference",
    k2="joint",
)

x: np.ndarray
y: np.ndarray
y_true: np.ndarray

x, y, y_true = experiment_config.data_provider(
    experiment_config.noise_ratio,
    experiment_config.hetero,
    experiment_config.num_points,
)

rsklpr: Rsklpr = Rsklpr(
    size_neighborhood=experiment_config.size_neighborhood,
    bw1=experiment_config.bw1,
    bw2=experiment_config.bw2,
)

y_hat: np.ndarray = rsklpr(
    x=x,
    y=y,
)

estimates: pd.DataFrame = pd.DataFrame(data=y_hat, columns=["y_hat"])

plot_results(
    x=x,
    y=y,
    y_true=y_true,
    estimates=estimates,
    title="Example usage",
)
```
![Example usage plane_plot](./example_usage_plane.png)
### Experimental results ###
The experimental results are available as interactive Jupyter notebooks at 
https://nbviewer.org/github/yaniv-shulman/rsklpr/tree/main/src/experiments/
