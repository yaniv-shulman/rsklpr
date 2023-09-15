import dataclasses
import time
from typing import List, Tuple, Callable, Optional, Dict, Union

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from IPython.core.display_functions import clear_output
from localreg import localreg, rbf, RBFnet
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess

from rsklpr.rsklpr import (
    Rsklpr1D,
)


@dataclasses.dataclass
class ExperimentConfig:
    """
    Contains configuration values for an experiment.
    """

    curve: Callable[[float, bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    size_neighborhood: int
    noise_ratio: float
    hetero: bool
    num_points: int
    response_bandwidth: str


def plot_results(x, y, estimates: pd.DataFrame, title: str) -> None:
    """
    Plots a scatter plot of all columns in the provided dataframe where each column is treated as a separate variable.
    It is assumed all plots share the same x coordinates.

    Args:
        x: The predictor values in the dataset. These are assumed common all estimates.
        y: The response values in the dataset.
        estimates: The estimates y_hat given x. it is assumed each column in the dataframe is the results of a
            different estimator.
        title: The plot title.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
    for c in estimates.columns:
        if c == "y":
            continue
        fig.add_trace(go.Scatter(x=x, y=estimates[c], mode="lines+markers", name=c))

    fig.update_layout(title=title)
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

    fig.update_layout(title=f"Performance stats for {title}", height=300)
    plotly.offline.iplot(fig)


def best_results_by_method(
    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]]
) -> pd.DataFrame:
    """
    Finds the best result in terms of RMSE of the predictions for each of the methods in the results data.

    Args:
        results: List containing the results of all experiments conducted.

    Returns:
        The best result having the lowest prediction RMSE obtained for each method.
    """
    best_nn_for_method: pd.Series = (
        pd.concat(
            [
                r["stat"]["rmse"].rename(f"{r['experiment'].size_neighborhood}")
                for r in results
            ],
            axis=1,
        )
        .astype(float)
        .idxmin(axis=1)
    )
    best_list: List[pd.DataFrame] = [results[0]["result"][["y", "y_true"]]]
    result: Dict[str, Union[np.ndarray, pd.DataFrame]]
    for result in results:
        best_results_for_nn: pd.DataFrame = best_nn_for_method[
            best_nn_for_method == str(result["experiment"].size_neighborhood)
        ]
        if len(best_results_for_nn) > 0:
            best_list.append(result["result"][best_results_for_nn.index])

    return pd.concat(best_list, axis=1).sort_index(axis=1)


def best_worst_results_by_size_neighborhood(
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


def benchmark_curve(
    curve: Callable[
        [float, bool, int], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
    ],
    size_neighborhood: Union[int, Dict[str, int]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    response_bandwidth: str,
    predictor_bandwidth: Optional[float] = None,
    methods: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Runs benchmark of a curve across a number of methods and compiles the results.

    Args:
        curve: The curve to benchmark performance on.
        size_neighborhood: The number of neighbors to use for local fitting. If an int the same value is used for all
            relevant methods, if a dict then an explicit value need to be provided for each method. Relevant only for
            rsklpr linear and quadratic, lowess, robust lowess and local quadratic methods.
        noise_ratio: The intensity of noise to apply to the ground truth curve.
        hetero: True to generate heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve.
        response_bandwidth: The response bandwidth selection method to use in the joint KDE.
        predictor_bandwidth: The bandwidth of the predictor kernel in the factorized joint KDE. If None it will be
            calculated heuristically based on a formula derived from empirical results.
        methods: The list of methods to estimate on. If None the default methods are run.

    Returns:
        The predictor, response, a dataframe containing the dataset response data, the ground truth and predictions
        from all estimators and a dataframe of calculated predictions RMSE and execution times.

    """
    if methods is None:
        methods = [
            "rsklpr",
            "rsklpr_quad",
            "lowess",
            "robust_lowess",
            "kernel_reg_ll",
            "kernel_reg_lc",
            "local_quad",
            "rbfnet",
        ]

    use_neighbors: int = -1

    if isinstance(size_neighborhood, int):
        use_neighbors = min(num_points, size_neighborhood)

    x: np.ndarray
    y: np.ndarray
    y_true: Optional[np.ndarray]

    x, y, y_true = curve(noise_ratio, hetero, num_points)

    def rmse(s: pd.Series) -> float:
        return np.sqrt((s - y_true).pow(2).mean()) if y_true is not None else 0.0

    result: pd.DataFrame = pd.DataFrame(data=y, index=x, columns=["y"])
    stats: pd.DataFrame = pd.DataFrame(index=methods, columns=["rmse", "time"])

    if "rsklpr" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["rsklpr"]

        rsklpr: Rsklpr1D = (
            Rsklpr1D(
                size_neighborhood=use_neighbors,
                predictor_bandwidth=predictor_bandwidth,
                response_bandwidth=response_bandwidth,
            )
        )
        start: float = time.time()

        result["rsklpr"] = rsklpr(
            x=x,
            y=y,
        )
        compute_time: float = time.time() - start
        stats.loc["rsklpr", "rmse"] = rmse(s=result["rsklpr"])
        stats.loc["rsklpr", "time"] = compute_time

    if "rsklpr_quad" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["rsklpr_quad"]

        rsklpr_quad: Rsklpr1D = (
            Rsklpr1D(
                size_neighborhood=use_neighbors,
                degree=2,
                predictor_bandwidth=predictor_bandwidth,
            )
        )
        start: float = time.time()

        result["rsklpr_quad"] = rsklpr_quad(
            x=x,
            y=y,
        )
        compute_time: float = time.time() - start
        stats.loc["rsklpr_quad", "rmse"] = rmse(s=result["rsklpr_quad"])
        stats.loc["rsklpr_quad", "time"] = compute_time

    if "lowess" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["lowess"]

        start = time.time()
        result[f"lowess"] = lowess(y, x, frac=use_neighbors / x.shape[0], it=0)[:, 1]
        compute_time = time.time() - start
        stats.loc["lowess", "rmse"] = rmse(s=result["lowess"])
        stats.loc["lowess", "time"] = compute_time

    if "robust_lowess" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["robust_lowess"]

        start = time.time()
        result[f"robust_lowess"] = lowess(y, x, frac=use_neighbors / x.shape[0], it=5)[
            :, 1
        ]
        compute_time = time.time() - start
        stats.loc["robust_lowess", "rmse"] = rmse(s=result["robust_lowess"])
        stats.loc["robust_lowess", "time"] = compute_time

    if "kernel_reg_ll" in methods:
        start = time.time()
        result[f"kernel_reg_ll"] = KernelReg(y, x, var_type="c", reg_type="ll").fit()[0]
        compute_time = time.time() - start
        stats.loc["kernel_reg_ll", "rmse"] = rmse(s=result["kernel_reg_ll"])
        stats.loc["kernel_reg_ll", "time"] = compute_time

    if "kernel_reg_lc" in methods:
        start = time.time()
        result[f"kernel_reg_lc"] = KernelReg(y, x, var_type="c", reg_type="lc").fit()[0]
        compute_time = time.time() - start
        stats.loc["kernel_reg_lc", "rmse"] = rmse(s=result["kernel_reg_lc"])
        stats.loc["kernel_reg_lc", "time"] = compute_time

    if "local_quad" in methods:
        if isinstance(size_neighborhood, Dict):
            use_neighbors = size_neighborhood["local_quad"]

        start = time.time()
        result[f"local_quad"] = localreg(
            x, y, degree=2, kernel=rbf.epanechnikov, frac=use_neighbors / x.shape[0]
        )
        compute_time = time.time() - start
        stats.loc["local_quad", "rmse"] = rmse(s=result["local_quad"])
        stats.loc["local_quad", "time"] = compute_time

    if "rbfnet" in methods:
        start = time.time()
        net: RBFnet = RBFnet()
        net.train(input=x, output=y)
        result[f"rbfnet"] = net.predict(input=x)
        compute_time = time.time() - start
        stats.loc["rbfnet", "rmse"] = rmse(s=result["rbfnet"])
        stats.loc["rbfnet", "time"] = compute_time

    result["y_true"] = y_true
    return x, y, result.sort_index(axis=1), stats


def _benchmark_curves_increasing_neighborhoods(
    curve: Callable[[float, bool], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    response_bandwidth: str,
    size_neighborhoods: Optional[List[int]] = None,
    predictor_bandwidth: Optional[float] = None,
) -> List[Dict[str, Union[np.ndarray, pd.DataFrame, ExperimentConfig]]]:
    """
    Runs a series of experiments on a single curve with various neighborhoods sizes.

    Args:
        curve: The curve to use in the experiments.
        noise_ratio: The noise ratio used in the experiments.
        hetero: True to use heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve in the experiments.
        response_bandwidth: The method used to estimate the bandwidth for the response kernel in the factorized
            joint KDE.
        size_neighborhoods: The sizes of neighborhoods to use in the experiments.
        predictor_bandwidth: The bandwidth of the predictor kernel in the factorized joint KDE.

    Returns:
        The experiments results.
    """
    if size_neighborhoods is None:
        size_neighborhoods = [15, 20, 30, 40, 50, 60, 70, 80]

    experiments: List[ExperimentConfig] = [
        ExperimentConfig(
            curve=curve,
            size_neighborhood=size_neighborhood,
            noise_ratio=noise_ratio,
            hetero=hetero,
            num_points=num_points,
            response_bandwidth=response_bandwidth,
        )
        for size_neighborhood in size_neighborhoods
    ]

    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]] = []

    curve: ExperimentConfig

    for curve in experiments:
        x: np.ndarray
        y: np.ndarray
        result: pd.DataFrame
        stat: pd.DataFrame
        x, y, result, stat = benchmark_curve(
            curve=curve.curve,
            size_neighborhood=curve.size_neighborhood,
            noise_ratio=curve.noise_ratio,
            hetero=curve.hetero,
            num_points=num_points,
            response_bandwidth=curve.response_bandwidth,
            predictor_bandwidth=predictor_bandwidth,
        )
        results.append(
            {"x": x, "y": y, "result": result, "stat": stat, "experiment": curve}
        )

    return results


def run_increasing_bandwidth_experiments(
    curve: Callable[[float, bool], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    size_neighborhood: int,
    predictor_bandwidths: Optional[List[float]] = None,
) -> List[Dict[str, Union[np.ndarray, pd.DataFrame, ExperimentConfig]]]:
    """
    Runs a series of experiments on a single curve with various predictor bandwidths.

    Args:
        curve: The curve to use in the experiments.
        noise_ratio: The noise ratio used in the experiments.
        hetero: True to use heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve in the experiments.
        predictor_bandwidths: The bandwidth of the predictor kernel in the factorized joint KDE.
        size_neighborhood: The number of points to use for local fitting. Relevant only for rsklpr linear and
            rsklpr quadratic, lowess and local quadratic methods.
        predictor_bandwidths: The bandwidths of the predictor kernel in the factorized joint KDE to run experiments
            with. If None some default values are used.

    Returns:
        The experiments results.
    """
    if predictor_bandwidths is None:
        predictor_bandwidths = np.linspace(0.001, 0.8, num=45).tolist()

    experiment: ExperimentConfig = ExperimentConfig(
        curve=curve,
        size_neighborhood=size_neighborhood,
        noise_ratio=noise_ratio,
        hetero=hetero,
        num_points=num_points,
        response_bandwidth="lscv_global"
    )

    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]] = []
    predictor_bandwidth: float

    for predictor_bandwidth in predictor_bandwidths:
        x: np.ndarray
        y: np.ndarray
        result: pd.DataFrame
        stat: pd.DataFrame
        x, y, result, stat = benchmark_curve(
            curve=experiment.curve,
            size_neighborhood=experiment.size_neighborhood,
            noise_ratio=experiment.noise_ratio,
            hetero=experiment.hetero,
            num_points=num_points,
            response_bandwidth=experiment.response_bandwidth,
            predictor_bandwidth=predictor_bandwidth,
            methods=["rsklpr"],
        )
        results.append(
            {
                "x": x,
                "y": y,
                "result": result,
                "stat": stat,
                "curve": curve,
                "lam": predictor_bandwidth,
            }
        )

    return results


def run_increasing_size_neighborhoods_experiments(
    curve: Callable[[float, bool], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    noise_ratio: float,
    hetero: bool,
    num_points: int,
    size_neighborhoods: List[int],
    response_bandwidth: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a series of experiments on a single curve with various neighborhoods sizes.

    Args:
        curve: The curve to use in the experiments.
        noise_ratio: The noise ratio used in the experiments.
        hetero: True to use heteroscedastic noise, False for homoscedastic noise.
        num_points: The number of points sampled from the curve in the experiments.
        response_bandwidth: The method used to estimate the bandwidth for the response kernel in the factorized
            joint KDE.
        size_neighborhoods: The sizes of neighborhoods to use in the experiments.

    Returns:
        The experiments results.
    """
    results: List[Dict[str, Union[np.ndarray, pd.DataFrame]]]
    results = _benchmark_curves_increasing_neighborhoods(
        curve=curve,
        noise_ratio=noise_ratio,
        hetero=hetero,
        num_points=num_points,
        response_bandwidth=response_bandwidth,
        size_neighborhoods=size_neighborhoods,
    )
    clear_output()
    results_stats: List[pd.DataFrame] = [r["stat"] for r in results]
    experiments: List[ExperimentConfig] = [r["experiment"] for r in results]
    rmse_stats: pd.DataFrame = pd.concat(
        [
            i["rmse"].rename(f"w_{c.size_neighborhood}")
            for i, c in zip(results_stats, experiments)
        ],
        axis=1,
    ).astype(float)
    rmse_stats["mean"] = rmse_stats.mean(axis=1)
    rmse_stats["std"] = rmse_stats.std(axis=1)
    time_stats: pd.DataFrame = pd.concat(
        [
            i["time"].rename(f"w_{c.size_neighborhood}")
            for i, c in zip(results_stats, experiments)
        ],
        axis=1,
    ).astype(float)
    plot_stats(stats=rmse_stats, title=f"- Curve {curve} RMSE all experiments")
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
    ) = best_worst_results_by_size_neighborhood(rmse_stats=rmse_stats)
    best_result: Dict[str, Union[np.ndarray, pd.DataFrame]] = [
        r for r in results if f"w_{r['experiment'].size_neighborhood}" == best_size_neighborhood
    ][0]
    print(
        f"Best performing method is {best_method} for size_neighborhood {best_size_neighborhood}"
    )
    best_results_for_methods: pd.DataFrame = best_results_by_method(results=results)
    plot_results(
        x=best_result["x"],
        y=best_result["y"],
        estimates=best_results_for_methods,
        title="Best obtained results for all methods",
    )
    return best_for_size_neighborhood, worst_for_size_neighborhood
