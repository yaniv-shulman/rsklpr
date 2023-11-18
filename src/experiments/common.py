import dataclasses
import time
from typing import List, Tuple, Callable, Optional, Dict, Union, Sequence, Any

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from IPython.core.display_functions import clear_output
from localreg import localreg, rbf, RBFnet
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess

from rsklpr.rsklpr import Rsklpr


@dataclasses.dataclass
class ExperimentConfig:
    """
    Contains configuration values for an experiment.
    """

    data_provider: Callable[[float, bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    size_neighborhood: int
    noise_ratio: float
    hetero: bool
    num_points: int
    bw1: Union[str, Sequence[float], Callable[[Any], List[float]], float]  # type: ignore [misc]
    bw2: Union[str, Sequence[float], Callable[[Any], List[float]], float]  # type: ignore [misc]
    k2: str = "joint"
    degree: int = 1


def benchmark_data(
    x: np.ndarray,
    y: np.ndarray,
    y_true: Optional[np.ndarray],
    k2: str,
    size_neighborhood: Union[int, Dict[str, int]],
    num_points: int,
    bw1: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    bw2: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    methods: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs benchmark of the data across a number of methods and compiles the results.

    Args:
         x: The predictors X for all observations with shape [N,K] where N is the number of observations and K is the
            dimension of X.
        y: The corresponding response for all observations.
        y_true: The ground truth response at the locations x.
        k2: The kernel that models the conditional 'importance' of the response at the location.
        size_neighborhood: The number of neighbors to use for local fitting. If an int the same value is used for all
            relevant methods, if a dict then an explicit value need to be provided for each method. Relevant only for
            rsklpr linear and local quadratic methods i.e 'rsklpr', 'lowess', 'robust_lowess' and 'local_quad'.
        num_points: The number of points sampled from the ground truth function.
        bw1: The response bandwidth selection method to use in the marginal predictor KDE.
        bw2: The response bandwidth selection method to use in the joint KDE.
        methods: The list of methods to estimate on.

    Returns:
        The predictor, a dataframe containing the dataset response data, the ground truth and predictions
        from all estimators and a dataframe of calculated predictions RMSE and execution times.

    """
    use_neighbors: int = -1

    if isinstance(size_neighborhood, int):
        use_neighbors = min(num_points, size_neighborhood)

    def rmse(s: pd.Series) -> float:
        return np.sqrt((s - y_true).pow(2).mean()) if y_true is not None else 0.0  # type: ignore [no-any-return]

    result: pd.DataFrame = pd.DataFrame()
    stats: pd.DataFrame = pd.DataFrame(index=methods, columns=["rmse", "time"])

    if "rsklpr" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["rsklpr"]

        rsklpr: Rsklpr = Rsklpr(
            size_neighborhood=use_neighbors,
            degree=1,
            k2=k2,
            bw1=bw1,
            bw2=bw2,
        )
        start: float = time.time()

        result["rsklpr"] = rsklpr(
            x=x,
            y=y,
        )
        compute_time: float = time.time() - start
        stats.loc["rsklpr", "rmse"] = rmse(s=result["rsklpr"])
        stats.loc["rsklpr", "time"] = compute_time

    if "lowess" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["lowess"]

        start = time.time()
        result["lowess"] = lowess(y, x, frac=use_neighbors / x.shape[0], it=0)[:, 1]
        compute_time = time.time() - start
        stats.loc["lowess", "rmse"] = rmse(s=result["lowess"])
        stats.loc["lowess", "time"] = compute_time

    if "robust_lowess" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["robust_lowess"]

        start = time.time()
        result["robust_lowess"] = lowess(y, x, frac=use_neighbors / x.shape[0], it=5)[:, 1]
        compute_time = time.time() - start
        stats.loc["robust_lowess", "rmse"] = rmse(s=result["robust_lowess"])
        stats.loc["robust_lowess", "time"] = compute_time

    if "kernel_reg_ll" in methods:
        start = time.time()

        result["kernel_reg_ll"] = KernelReg(
            y, x, var_type="c" * x.shape[1] if x.ndim > 1 else "c", reg_type="ll"
        ).fit()[0]

        compute_time = time.time() - start
        stats.loc["kernel_reg_ll", "rmse"] = rmse(s=result["kernel_reg_ll"])
        stats.loc["kernel_reg_ll", "time"] = compute_time

    if "kernel_reg_lc" in methods:
        start = time.time()

        result["kernel_reg_lc"] = KernelReg(
            y, x, var_type="c" * x.shape[1] if x.ndim > 1 else "c", reg_type="lc"
        ).fit()[0]

        compute_time = time.time() - start
        stats.loc["kernel_reg_lc", "rmse"] = rmse(s=result["kernel_reg_lc"])
        stats.loc["kernel_reg_lc", "time"] = compute_time

    if "local_quad" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["local_quad"]

        start = time.time()

        result["local_quad"] = localreg(x, y, degree=2, kernel=rbf.epanechnikov, frac=use_neighbors / x.shape[0])

        compute_time = time.time() - start
        stats.loc["local_quad", "rmse"] = rmse(s=result["local_quad"])
        stats.loc["local_quad", "time"] = compute_time

    if "rbfnet" in methods:
        start = time.time()
        net: RBFnet = RBFnet()
        net.train(input=x, output=y)
        result["rbfnet"] = net.predict(input=x)
        compute_time = time.time() - start
        stats.loc["rbfnet", "rmse"] = rmse(s=result["rbfnet"])
        stats.loc["rbfnet", "time"] = compute_time

    return result, stats


def plot_results(
    x: np.ndarray,
    y: np.ndarray,
    y_true: Optional[np.ndarray],
    estimates: pd.DataFrame,
    title: str,
) -> None:
    """
    Plots a scatter plot of all columns in the provided dataframe where each column is treated as a separate variable.
    It is assumed all plots share the same x coordinates.

    Args:
        x: The predictors X for all observations with shape [N,K] where N is the number of observations and K is the
            dimension of X.
        y: The corresponding response for all observations.
        y_true: The ground truth response at the locations x.
        estimates: The estimates y_hat given x. it is assumed each column in the dataframe is the results of a
            different estimator.
        title: The plot title.
    """
    fig = go.Figure()

    if x.ndim > 1 and x.shape[1] == 2:
        fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=y, mode="markers", name="data"))

        if y_true is not None:
            fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=y_true, mode="markers", name="y_true"))

        for c in estimates.columns:
            fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=estimates[c], mode="markers", name=c))

        x_lines: List[Optional[float]] = []
        y_lines: List[Optional[float]] = []
        z_lines: List[Optional[float]] = []
        estimates["y"] = y

        if y_true is not None:
            estimates["y_true"] = y_true

        min_z: np.ndarray = estimates.values.min(axis=1)
        max_z: np.ndarray = estimates.values.max(axis=1)
        estimates.drop(columns=["y", "y_true"], inplace=True)

        for i in range(x.shape[0]):
            x_lines.append(float(x[i, 0]))
            y_lines.append(float(x[i, 1]))
            z_lines.append(float(min_z[i]))

            x_lines.append(float(x[i, 0]))
            y_lines.append(float(x[i, 1]))
            z_lines.append(float(max_z[i]))

            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name="lines"))

        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            go.Layout(
                title=title,
                autosize=False,
                width=1000,
                height=1000,
            )
        )
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))

        if y_true is not None:
            fig.add_trace(go.Scatter(x=x, y=y_true, mode="markers", name="y_true"))

        for c in estimates.columns:
            fig.add_trace(go.Scatter(x=x, y=estimates[c], mode="lines+markers", name=c))

        fig.update_layout(
            go.Layout(
                title=title,
                autosize=False,
                width=1400,
                height=500,
            )
        )

    plotly.offline.iplot(fig)


def plot_stats(stats: pd.DataFrame, title: str) -> None:
    """
    Pretty plot of tabular stats.

    Args:
        stats: The stats to present
        title: The plot title.
    """
    values = [list(stats.index)]
    for c in stats.columns:
        values.append(stats[c])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["method"] + list(stats.columns), align="left"),
                cells=dict(values=values, align="left"),
            )
        ]
    )

    fig.update_layout(
        go.Layout(
            title=f"Performance stats for {title}",
            autosize=False,
            width=1400,
            height=300,
        )
    )

    plotly.offline.iplot(fig)


def _best_results_by_method(results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]]) -> pd.DataFrame:
    """
    Finds the best result in terms of RMSE of the predictions for each of the methods in the results data.

    Args:
        results: List containing the results of all experiments conducted.

    Returns:
        The best result having the lowest prediction RMSE obtained for each method.
    """
    best_nn_for_method: pd.Series = (
        pd.concat(
            [r["stat"]["rmse"].rename(f"{r['experiment'].size_neighborhood}") for r in results],  # type: ignore [union-attr]
            axis=1,
        )
        .astype(float)
        .idxmin(axis=1)
    )
    best_list: List[pd.DataFrame] = []
    result: Dict[str, Union[np.ndarray, pd.DataFrame]]
    for result in results:
        best_results_for_nn: pd.DataFrame = best_nn_for_method[
            best_nn_for_method == str(result["experiment"].size_neighborhood)  # type: ignore [union-attr]
        ]
        if len(best_results_for_nn) > 0:
            best_list.append(result["result"][best_results_for_nn.index])

    return pd.concat(best_list, axis=1).sort_index(axis=1)


def _best_worst_results_by_size_neighborhood(
    rmse_stats: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str, str]:
    """
    Finds the best and worst performing method for each neighbourhood size.

    Args:
        rmse_stats: The dataframe of RMSE by method and num neighbors.

    Returns:
        A tuple consisting of: best method for each neighborhood size, worst method for each neighborhood size, best
        method name, best neighborhood size, worst method name and worst neighborhood size.
    """
    best_for_nn: pd.DataFrame = pd.concat(
        [
            rmse_stats.idxmin().to_frame().rename(columns={0: "method"}),
            rmse_stats.min().to_frame().rename(columns={0: "rmse"}),
        ],
        axis=1,
    )
    worst_for_nn: pd.DataFrame = pd.concat(
        [
            rmse_stats.idxmax().to_frame().rename(columns={0: "method"}),
            rmse_stats.max().to_frame().rename(columns={0: "rmse"}),
        ],
        axis=1,
    )
    best_nn: str = best_for_nn["rmse"].idxmin()
    best_method: str = best_for_nn.loc[best_nn, "method"]
    worst_nn: str = worst_for_nn["rmse"].idxmax()
    worst_method: str = worst_for_nn.loc[worst_nn, "method"]

    return (
        best_for_nn,
        worst_for_nn,
        best_method,
        best_nn,
        worst_method,
        worst_nn,
    )


def _benchmark_increasing_neighborhoods(
    x: np.ndarray,
    y: np.ndarray,
    y_true: Optional[np.ndarray],
    data_provider: Callable[[float, bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    bw1: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    bw2: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    k2: str,
    size_neighborhoods: List[int],
    methods: List[str],
) -> List[Dict[str, Union[np.ndarray, pd.DataFrame, ExperimentConfig]]]:
    """
    Runs a series of experiments on the provided data with various neighborhoods sizes.

    Args:
        x: The predictors X for all observations with shape [N,K] where N is the number of observations and K is the
            dimension of X.
        y: The corresponding response for all observations.
        y_true: The ground truth response at the locations x.
        data_provider: The callable that provides the data to benchmark performance on.
        noise_ratio: The noise ratio used in the experiments.
        hetero: True to use heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve in the experiments.
        bw1: Used for Rsklpr bw1.
        bw2: Used for Rsklpr bw2.
        k2: Used for Rsklpr, the kernel that models the conditional 'importance' of the response at the location.
        size_neighborhoods: The sizes of neighborhoods to use in the experiments.
        methods: The list of methods to estimate on.

    Returns:
        The experiments results.
    """
    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]] = []
    size_neighborhood: int

    for size_neighborhood in size_neighborhoods:
        experiment = ExperimentConfig(
            data_provider=data_provider,
            size_neighborhood=size_neighborhood,
            noise_ratio=noise_ratio,
            hetero=hetero,
            num_points=num_points,
            bw1=bw1,
            bw2=bw2,
            k2=k2,
        )

        result: pd.DataFrame
        stat: pd.DataFrame

        result, stat = benchmark_data(
            x=x,
            y=y,
            y_true=y_true,
            k2=k2,
            size_neighborhood=size_neighborhood,
            num_points=num_points,
            bw1=bw1,
            bw2=bw2,
            methods=methods,
        )
        results.append(
            {
                "result": result,
                "stat": stat,
                "experiment": experiment,
            }
        )

    return results


def run_increasing_size_neighborhoods_experiments(
    data_provider: Callable[[float, bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    size_neighborhoods: List[int],
    bw1: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    bw2: Union[str, Sequence[float], Callable[[Any], List[float]]],  # type: ignore [misc]
    k2: str,
    methods: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a series of experiments on a single curve with various neighborhoods sizes.

    Args:
        data_provider: The callable that provides the data to benchmark performance on.
        noise_ratio: The noise ratio used in the experiments.
        hetero: True to use heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve in the experiments.
        size_neighborhoods: The sizes of neighborhoods to use in the experiments.
        bw1: The bandwidth selection method to used in the marginal predictor's kernel.
        bw2: The joint or response bandwidth selection method to use in the k2.
        k2: The kernel that models the conditional 'importance' of the response at the location.
        methods: The list of methods to estimate on.

    Returns:
        The experiments results.
    """
    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]]

    x: np.ndarray
    y: np.ndarray
    y_true: Optional[np.ndarray]

    x, y, y_true = data_provider(noise_ratio, hetero, num_points)

    if len(methods) > 0:
        results = _benchmark_increasing_neighborhoods(
            x=x,
            y=y,
            y_true=y_true,
            data_provider=data_provider,
            noise_ratio=noise_ratio,
            hetero=hetero,
            num_points=num_points,
            bw1=bw1,
            bw2=bw2,
            k2=k2,
            size_neighborhoods=size_neighborhoods,
            methods=methods,
        )
        clear_output()
        results_stats: List[pd.DataFrame] = [r["stat"] for r in results]
        experiments: List[ExperimentConfig] = [r["experiment"] for r in results]  # type: ignore [misc]
        rmse_stats: pd.DataFrame = pd.concat(
            [i["rmse"].rename(f"w_{c.size_neighborhood}") for i, c in zip(results_stats, experiments)],
            axis=1,
        ).astype(float)
        rmse_stats["mean"] = rmse_stats.mean(axis=1)
        rmse_stats["std"] = rmse_stats.std(axis=1)
        time_stats: pd.DataFrame = pd.concat(
            [i["time"].rename(f"w_{c.size_neighborhood}") for i, c in zip(results_stats, experiments)],
            axis=1,
        ).astype(float)
        plot_stats(stats=rmse_stats, title=f"- Curve {data_provider} RMSE all experiments")
        rmse_stats.T.plot.bar(title="RMSE all methods").legend(
            loc="center right",
            bbox_to_anchor=(1.35, 0.25),
            ncol=1,
            fancybox=True,
            shadow=True,
        )
        time_stats.T.plot.bar(title="Time all methods").legend(
            loc="center right",
            bbox_to_anchor=(1.35, 0.25),
            ncol=1,
            fancybox=True,
            shadow=True,
        )
        rmse_stats = rmse_stats.drop(columns=["mean", "std"])
        (
            best_for_size_neighborhood,
            worst_for_size_neighborhood,
            best_method,
            best_size_neighborhood,
            worst_method,
            worst_size_neighborhood,
        ) = _best_worst_results_by_size_neighborhood(rmse_stats=rmse_stats)

        print(f"Best performing method is {best_method} for size_neighborhood {best_size_neighborhood}")
        best_results_for_methods: pd.DataFrame = _best_results_by_method(results=results)

    else:
        best_results_for_methods = pd.DataFrame()
        best_for_size_neighborhood = pd.DataFrame()
        worst_for_size_neighborhood = pd.DataFrame()

    plot_results(
        x=x,
        y=y,
        y_true=y_true,
        estimates=best_results_for_methods,
        title="Best obtained results for all methods",
    )
    return best_for_size_neighborhood, worst_for_size_neighborhood
