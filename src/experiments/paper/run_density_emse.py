#!/usr/bin/env python
"""Monte Carlo density experiment for the RSKLPR paper.

This script replaces the single-run "RMSE as a function of data density" plots
with repeated trials, empirical mean squared error (EMSE) on a fixed clean
evaluation grid, summary statistics, and publication-ready figures.
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_function
from statsmodels.nonparametric.smoothers_lowess import lowess

from rsklpr.rsklpr import Rsklpr

repo_dir_env = os.environ.get("REPO_DIR")
if not repo_dir_env:
    raise RuntimeError("REPO_DIR must be defined; experiment outputs are written under $REPO_DIR/out.")

repo_root = Path(repo_dir_env).resolve()
out_root = repo_root / "out"
out_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(out_root / ".matplotlib"))

DistributionGenerator = Callable[[np.ndarray, np.ndarray, np.random.Generator, float], np.ndarray]
CsvValue = TypeVar("CsvValue")


@dataclass(frozen=True)
class MethodSpec:
    """
    Display metadata for one compared method.

    Attributes:
        name: Display metadata for one compared method.
        label: Display metadata for one compared method.
    """

    name: str
    label: str


methods: Dict[str, MethodSpec] = {
    "rsklpr_conden": MethodSpec("rsklpr_conden", "RSKLPR-cond."),
    "rsklpr_joint": MethodSpec("rsklpr_joint", "RSKLPR-joint"),
    "lpr": MethodSpec("lpr", "LPR"),
    "lowess": MethodSpec("lowess", "LOWESS"),
    "robust_lowess": MethodSpec("robust_lowess", "Robust LOWESS"),
}


colors: Dict[str, str] = {
    "rsklpr_conden": "#1f77b4",
    "rsklpr_joint": "#9467bd",
    "lpr": "#2ca02c",
    "lowess": "#ff7f0e",
    "robust_lowess": "#d62728",
}


def parse_csv_arg(value: str, cast: Callable[[str], CsvValue]) -> List[CsvValue]:
    """
    Parse a comma-separated command-line argument.

    Args:
        value: The comma-separated value to parse.
        cast: The callable used to cast each parsed value.

    Returns:
        The computed result list.
    """
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def regression_mean(x: np.ndarray) -> np.ndarray:
    """
    Smooth positive target curve used for all density experiments.

    Args:
        x: The predictor values.

    Returns:
        The resulting array.
    """
    curve = np.sqrt(np.abs(np.power(x, 3) - 4.0 * np.power(x, 4) / 3.0))
    curve += 0.1 * x * np.square(np.sin(3.0 * np.pi * x))
    return np.asarray(curve - float(np.min(curve)) + 0.1, dtype=float)


def gaussian_response(mean: np.ndarray, x: np.ndarray, rng: np.random.Generator, noise_ratio: float) -> np.ndarray:
    """
    Generate heteroscedastic Gaussian responses with the requested mean.

    Args:
        mean: The desired conditional mean values.
        x: The predictor values.
        rng: The random number generator.
        noise_ratio: The noise scale relative to the response range.

    Returns:
        The resulting array.
    """
    scale = noise_ratio * (float(np.max(mean)) - float(np.min(mean)))
    hetero_scale = scale * (0.35 + 1.3 * x)
    return np.asarray(mean + rng.normal(loc=0.0, scale=hetero_scale, size=mean.shape[0]), dtype=float)


def bimodal_response(mean: np.ndarray, x: np.ndarray, rng: np.random.Generator, noise_ratio: float) -> np.ndarray:
    """
    Generate symmetric bimodal responses around the requested mean.

    Args:
        mean: The desired conditional mean values.
        x: The predictor values.
        rng: The random number generator.
        noise_ratio: The noise scale relative to the response range.

    Returns:
        The resulting array.
    """
    response_range = float(np.max(mean) - np.min(mean))
    scale = noise_ratio * response_range * (0.35 + 0.9 * x)
    mode_offset = 0.32 * response_range * (0.65 + 0.35 * np.sin(2.0 * np.pi * x) ** 2)
    signs = rng.choice(np.array([-1.0, 1.0]), size=mean.shape[0])
    return np.asarray(mean + signs * mode_offset + rng.normal(loc=0.0, scale=scale, size=mean.shape[0]), dtype=float)


def exponential_response(mean: np.ndarray, _: np.ndarray, rng: np.random.Generator, __: float) -> np.ndarray:
    """
    Generate exponential responses with the requested mean.

    Args:
        mean: The desired conditional mean values.
        _: The _ argument.
        rng: The random number generator.
        __: The __ argument.

    Returns:
        The resulting array.
    """
    return np.asarray(rng.exponential(scale=mean), dtype=float)


def gamma_response(mean: np.ndarray, _: np.ndarray, rng: np.random.Generator, __: float) -> np.ndarray:
    """
    Generate gamma responses with the requested mean.

    Args:
        mean: The desired conditional mean values.
        _: The _ argument.
        rng: The random number generator.
        __: The __ argument.

    Returns:
        The resulting array.
    """
    shape = 2.0
    return np.asarray(rng.gamma(shape=shape, scale=mean / shape), dtype=float)


def lognormal_response(mean: np.ndarray, _: np.ndarray, rng: np.random.Generator, __: float) -> np.ndarray:
    """
    Generate log-normal responses with the requested mean.

    Args:
        mean: The desired conditional mean values.
        _: The _ argument.
        rng: The random number generator.
        __: The __ argument.

    Returns:
        The resulting array.
    """
    sigma = 0.5
    mu = np.log(mean) - sigma**2 / 2.0
    return np.asarray(rng.lognormal(mean=mu, sigma=sigma), dtype=float)


def weibull_response(mean: np.ndarray, _: np.ndarray, rng: np.random.Generator, __: float) -> np.ndarray:
    """
    Generate Weibull responses with the requested mean.

    Args:
        mean: The desired conditional mean values.
        _: The _ argument.
        rng: The random number generator.
        __: The __ argument.

    Returns:
        The resulting array.
    """
    shape = 1.5
    scale = mean / gamma_function(1.0 + 1.0 / shape)
    return np.asarray(scale * rng.weibull(a=shape, size=mean.shape[0]), dtype=float)


distributions: Dict[str, DistributionGenerator] = {
    "gaussian": gaussian_response,
    "bimodal": bimodal_response,
    "exponential": exponential_response,
    "gamma": gamma_response,
    "lognormal": lognormal_response,
    "weibull": weibull_response,
}


def squared_density_mean_ratio(distribution: str) -> float:
    """
    Return E_{f^2}[Y] / E_f[Y] for the synthetic response law.

    Args:
        distribution: The synthetic response distribution key.

    Returns:
        The computed scalar value.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    if distribution in {"gaussian", "bimodal"}:
        return 1.0
    if distribution == "exponential":
        return 0.5
    if distribution == "gamma":
        shape = 2.0
        return (2.0 * shape - 1.0) / (2.0 * shape)
    if distribution == "lognormal":
        sigma = 0.5
        return float(np.exp(-0.75 * sigma**2))
    if distribution == "weibull":
        shape = 1.5
        return float(2.0 ** (-1.0 / shape) / (gamma_function(2.0 - 1.0 / shape) * gamma_function(1.0 + 1.0 / shape)))
    raise ValueError(f"Unknown distribution: {distribution}")


def theoretical_bias_rmse(distribution: str, x_eval: np.ndarray) -> float:
    """
    Compute the theoretical density-tilted RMSE reference.

    Args:
        distribution: The synthetic response distribution key.
        x_eval: The clean evaluation-grid predictor values.

    Returns:
        The computed scalar value.
    """
    ratio = squared_density_mean_ratio(distribution)
    y_true = regression_mean(x_eval)
    return float(abs(ratio - 1.0) * np.sqrt(np.mean(np.square(y_true))))


def distribution_label(distribution: str) -> str:
    """
    Return a display label for a synthetic response distribution.

    Args:
        distribution: The synthetic response distribution key.

    Returns:
        The display string.
    """
    labels = {
        "bimodal": "Bimodal",
        "exponential": "Exponential",
        "gamma": "Gamma",
        "gaussian": "Gaussian",
        "lognormal": "Log-normal",
        "weibull": "Weibull",
    }
    return labels.get(distribution, distribution.capitalize())


def make_training_data(
    n_train: int,
    distribution: str,
    rng: np.random.Generator,
    noise_ratio: float,
    outlier_frac: float,
    outlier_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one synthetic training sample and optional outliers.

    Args:
        n_train: The number of training observations.
        distribution: The synthetic response distribution key.
        rng: The random number generator.
        noise_ratio: The noise scale relative to the response range.
        outlier_frac: The requested fraction of response outliers.
        outlier_scale: The outlier magnitude multiplier.

    Returns:
        The resulting array.
    """
    x = np.linspace(0.0, 1.0, num=n_train)
    x += rng.normal(loc=0.0, scale=0.15 / max(n_train, 1), size=n_train)
    x = np.clip(x, 0.0, 1.0)
    x.sort()

    y_true = regression_mean(x)
    y = distributions[distribution](y_true, x, rng, noise_ratio)
    outlier_mask = contaminate_response(
        y=y,
        y_true=y_true,
        rng=rng,
        outlier_frac=outlier_frac,
        outlier_scale=outlier_scale,
    )
    return x, y, y_true, outlier_mask


def contaminate_response(
    y: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
    outlier_frac: float,
    outlier_scale: float,
) -> np.ndarray:
    """
    Inject large additive response outliers into a training sample.

    Args:
        y: The response values.
        y_true: The clean target values.
        rng: The random number generator.
        outlier_frac: The requested fraction of response outliers.
        outlier_scale: The outlier magnitude multiplier.

    Returns:
        The resulting array.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    if outlier_frac <= 0.0:
        return np.zeros(y.shape[0], dtype=bool)
    if outlier_frac >= 1.0:
        raise ValueError("--outlier-frac must be less than 1")
    if outlier_scale <= 0.0:
        raise ValueError("--outlier-scale must be positive")

    outlier_mask = np.asarray(rng.random(y.shape[0]) < outlier_frac, dtype=bool)
    n_outliers = int(np.sum(outlier_mask))
    if n_outliers == 0:
        return outlier_mask

    response_range = float(np.max(y_true) - np.min(y_true))
    if not np.isfinite(response_range) or response_range <= np.finfo(float).eps:
        response_range = float(np.nanstd(y_true))
    if not np.isfinite(response_range) or response_range <= np.finfo(float).eps:
        response_range = 1.0

    signs = rng.choice(np.array([-1.0, 1.0]), size=n_outliers)
    magnitudes = outlier_scale * response_range * (0.5 + rng.random(n_outliers))
    y[outlier_mask] += signs * magnitudes
    return outlier_mask


def make_model(method: str, n_neighbors: int, seed: int) -> Rsklpr:
    """
    Construct an RSKLPR-family model for one method key.

    Args:
        method: The method key.
        n_neighbors: The local-neighborhood size.
        seed: The random seed.

    Returns:
        The configured RSKLPR model.
    """
    kr = {
        "rsklpr_conden": "conden",
        "rsklpr_joint": "joint",
        "lpr": "none",
    }[method]
    return Rsklpr(
        size_neighborhood=n_neighbors,
        degree=1,
        metric_x="euclidean",
        kr=kr,
        bw1="normal_reference",
        bw2="normal_reference",
        seed=seed,
        suppress_warnings=True,
    )


def predict_method(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    n_neighbors: int,
    seed: int,
) -> np.ndarray:
    """
    Fit one method and predict on the clean evaluation grid.

    Args:
        method: The method key.
        x_train: The training predictor values.
        y_train: The training response values.
        x_eval: The clean evaluation-grid predictor values.
        n_neighbors: The local-neighborhood size.
        seed: The random seed.

    Returns:
        The resulting array.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    if method in {"rsklpr_conden", "rsklpr_joint", "lpr"}:
        model = make_model(method=method, n_neighbors=n_neighbors, seed=seed)
        model.fit(x=x_train, y=y_train)
        return model.predict(x=x_eval)

    frac = min(1.0, max(n_neighbors / x_train.shape[0], 3.0 / x_train.shape[0]))
    if method == "lowess":
        return np.asarray(
            lowess(y_train, x_train, frac=frac, it=0, xvals=x_eval, return_sorted=False),
            dtype=float,
        )
    if method == "robust_lowess":
        return np.asarray(
            lowess(y_train, x_train, frac=frac, it=5, xvals=x_eval, return_sorted=False),
            dtype=float,
        )
    raise ValueError(f"Unknown method: {method}")


def metric_row(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    runtime_s: float,
) -> Dict[str, float]:
    """
    Compute clean-grid error metrics for one fitted method.

    Args:
        y_hat: The predicted response values.
        y_true: The clean target values.
        runtime_s: The elapsed runtime in seconds.

    Returns:
        The computed scalar value.
    """
    residual = y_hat - y_true
    return {
        "emse": float(np.nanmean(np.square(residual))),
        "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
        "mae": float(np.nanmean(np.abs(residual))),
        "bias": float(np.nanmean(residual)),
        "error_std": float(np.nanstd(residual)),
        "runtime_s": float(runtime_s),
        "nan_fraction": float(np.mean(~np.isfinite(y_hat))),
    }


def neighbor_count(n_train: int, neighbor_frac: float, min_neighbors: int, max_neighbors: int) -> int:
    """
    Compute the local neighborhood size for a training sample size.

    Args:
        n_train: The number of training observations.
        neighbor_frac: The requested neighborhood fraction.
        min_neighbors: The minimum allowed neighborhood size.
        max_neighbors: The maximum allowed neighborhood size.

    Returns:
        The computed integer value.
    """
    return min(n_train, max(min_neighbors, min(max_neighbors, int(round(neighbor_frac * n_train)))))


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    """
    Run the Monte Carlo density experiment and return trial rows.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The resulting data frame.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    apply_experiment_defaults(args)
    densities = parse_csv_arg(args.densities, int)
    distributions = parse_csv_arg(args.distributions, str)
    selected_methods = parse_csv_arg(args.methods, str)

    for distribution in distributions:
        if distribution not in distributions:
            raise ValueError(f"Unknown distribution {distribution}. Available: {sorted(distributions)}")
    for method in selected_methods:
        if method not in methods:
            raise ValueError(f"Unknown method {method}. Available: {sorted(methods)}")

    x_eval = np.linspace(args.eval_min, args.eval_max, num=args.n_eval)
    y_eval_true = regression_mean(x_eval)

    rows: List[Dict[str, object]] = []
    total = len(distributions) * len(densities) * args.trials
    completed = 0

    for distribution in distributions:
        for n_train in densities:
            n_neighbors = neighbor_count(
                n_train=n_train,
                neighbor_frac=args.neighbor_frac,
                min_neighbors=args.min_neighbors,
                max_neighbors=args.max_neighbors,
            )
            for trial in range(args.trials):
                seed = args.seed + 100000 * trial + 1000 * n_train + 17 * (1 + list(distributions).index(distribution))
                rng = np.random.default_rng(seed=seed)
                x_train, y_train, _, outlier_mask = make_training_data(
                    n_train=n_train,
                    distribution=distribution,
                    rng=rng,
                    noise_ratio=args.noise_ratio,
                    outlier_frac=args.outlier_frac,
                    outlier_scale=args.outlier_scale,
                )

                for method in selected_methods:
                    start = time.perf_counter()
                    y_hat = predict_method(
                        method=method,
                        x_train=x_train,
                        y_train=y_train,
                        x_eval=x_eval,
                        n_neighbors=n_neighbors,
                        seed=seed,
                    )
                    runtime_s = time.perf_counter() - start
                    row = {
                        "distribution": distribution,
                        "n_train": n_train,
                        "n_eval": args.n_eval,
                        "trial": trial,
                        "seed": seed,
                        "method": method,
                        "method_label": methods[method].label,
                        "experiment": args.experiment,
                        "n_neighbors": n_neighbors,
                        "neighbor_frac": n_neighbors / n_train,
                        "outlier_frac_requested": args.outlier_frac,
                        "outlier_frac_realized": float(np.mean(outlier_mask)),
                        "outlier_scale": args.outlier_scale,
                    }
                    row.update(metric_row(y_hat=y_hat, y_true=y_eval_true, runtime_s=runtime_s))
                    rows.append(row)

                completed += 1
                if args.verbose and (completed == total or completed % args.progress_every == 0):
                    print(f"completed {completed}/{total} distribution-density-trial jobs", flush=True)

    return pd.DataFrame(rows)


def resolve_output_dir(output_dir_arg: str) -> Path:
    """
    Resolve an output subdirectory under the repository output root.

    Args:
        output_dir_arg: The requested output directory argument.

    Returns:
        The resolved filesystem path.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    output_dir = Path(output_dir_arg)
    if output_dir.is_absolute():
        raise ValueError("--output-dir must be a relative subdirectory under $REPO_DIR/out")
    return out_root / output_dir


def summarize_trials(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Monte Carlo trial metrics by distribution, size, and method.

    Args:
        results: The trial-level results table.

    Returns:
        The resulting data frame.
    """
    keys = ["distribution", "n_train", "method", "method_label"]
    stats = results.groupby(keys, as_index=False).agg(
        emse_mean=("emse", "mean"),
        emse_median=("emse", "median"),
        emse_std=("emse", "std"),
        emse_q025=("emse", lambda s: s.quantile(0.025)),
        emse_q25=("emse", lambda s: s.quantile(0.25)),
        emse_q75=("emse", lambda s: s.quantile(0.75)),
        emse_q975=("emse", lambda s: s.quantile(0.975)),
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        rmse_std=("rmse", "std"),
        rmse_q025=("rmse", lambda s: s.quantile(0.025)),
        rmse_q25=("rmse", lambda s: s.quantile(0.25)),
        rmse_q75=("rmse", lambda s: s.quantile(0.75)),
        rmse_q975=("rmse", lambda s: s.quantile(0.975)),
        mae_mean=("mae", "mean"),
        bias_mean=("bias", "mean"),
        error_std_mean=("error_std", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        runtime_s_median=("runtime_s", "median"),
        nan_fraction_mean=("nan_fraction", "mean"),
        outlier_frac_realized_mean=("outlier_frac_realized", "mean"),
        trials=("trial", "nunique"),
    )

    winners = results.copy()
    winners["is_winner"] = winners["rmse"] == winners.groupby(["distribution", "n_train", "trial"])["rmse"].transform(
        "min"
    )
    win_rates = winners.groupby(keys, as_index=False).agg(win_rate=("is_winner", "mean"))
    return stats.merge(win_rates, on=keys, how="left")


def make_latex_table(summary: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """
    Create a compact LaTeX-ready summary table.

    Args:
        summary: The summary results table.
        metric: The metric name to summarize.

    Returns:
        The resulting data frame.
    """
    value = f"{metric}_mean"
    spread = f"{metric}_std"
    table = summary.copy()
    table["entry"] = table.apply(lambda r: f"{r[value]:.4g} $\\pm$ {r[spread]:.2g}", axis=1)
    return table.pivot_table(
        index=["distribution", "n_train"],
        columns="method_label",
        values="entry",
        aggfunc="first",
    ).reset_index()


def plot_density(
    summary: pd.DataFrame,
    output_dir: Path,
    band: str,
    xscale: str,
    show_theoretical_bias: bool,
    n_eval: int,
    eval_min: float,
    eval_max: float,
) -> None:
    """
    Write RMSE curves for the density experiment.

    Args:
        summary: The summary results table.
        output_dir: The output directory.
        band: The uncertainty-band type to plot.
        xscale: The x-axis scale.
        show_theoretical_bias: Whether to draw the theoretical bias reference.
        n_eval: The number of evaluation-grid points.
        eval_min: The lower evaluation-grid endpoint.
        eval_max: The upper evaluation-grid endpoint.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    distributions = list(summary["distribution"].drop_duplicates())
    method_keys = list(summary["method"].drop_duplicates())
    densities = sorted(summary["n_train"].drop_duplicates())
    x_eval = np.linspace(eval_min, eval_max, num=n_eval)
    ncols = min(3, len(distributions))
    nrows = int(np.ceil(len(distributions) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)

    for ax, distribution in zip(axes.ravel(), distributions):
        subset = summary[summary["distribution"] == distribution]
        for method in method_keys:
            method_subset = subset[subset["method"] == method].sort_values("n_train")
            if method_subset.empty:
                continue

            x = method_subset["n_train"].to_numpy(dtype=float)
            y = method_subset["rmse_mean"].to_numpy(dtype=float)
            if band == "iqr":
                y_low = method_subset["rmse_q25"].to_numpy(dtype=float)
                y_high = method_subset["rmse_q75"].to_numpy(dtype=float)
            else:
                y_low = method_subset["rmse_q025"].to_numpy(dtype=float)
                y_high = method_subset["rmse_q975"].to_numpy(dtype=float)

            ax.plot(x, y, marker="o", linewidth=1.8, label=methods[method].label, color=colors.get(method))
            ax.fill_between(x, y_low, y_high, color=colors.get(method), alpha=0.16, linewidth=0)

        if show_theoretical_bias:
            bias_rmse = theoretical_bias_rmse(distribution=distribution, x_eval=x_eval)
            ax.axhline(
                bias_rmse,
                color="#333333",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label="Theoretical bias RMSE",
            )

        ax.set_title(distribution.capitalize())
        ax.set_xlabel("Training points")
        ax.set_ylabel("RMSE against clean mean")
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.set_xticks(densities)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(distributions) :]:
        ax.axis("off")

    handles: List[Any] = []
    labels: List[str] = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / "density_emse_rmse.png", dpi=300)
    fig.savefig(output_dir / "density_emse_rmse.pdf")
    if xscale == "log":
        fig.savefig(output_dir / "density_emse_rmse_logx.png", dpi=300)
        fig.savefig(output_dir / "density_emse_rmse_logx.pdf")
    plt.close(fig)


def plot_runtime(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Write runtime curves for the density experiment.

    Args:
        summary: The summary results table.
        output_dir: The output directory.
    """
    import matplotlib.pyplot as plt

    runtime = (
        summary.groupby(["n_train", "method", "method_label"], as_index=False)
        .agg(runtime_s_mean=("runtime_s_mean", "mean"))
        .sort_values("n_train")
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in runtime["method"].drop_duplicates():
        subset = runtime[runtime["method"] == method]
        ax.plot(
            subset["n_train"],
            subset["runtime_s_mean"],
            marker="o",
            linewidth=1.8,
            label=methods[method].label,
            color=colors.get(method),
        )
    ax.set_xlabel("Training points")
    ax.set_ylabel("Mean runtime per fit/predict run (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "density_emse_runtime.png", dpi=300)
    fig.savefig(output_dir / "density_emse_runtime.pdf")
    plt.close(fig)


def plot_example_samples(args: argparse.Namespace, output_dir: Path) -> None:
    """
    Write appendix sample-data plots for selected distributions.

    Args:
        args: Parsed command-line arguments.
        output_dir: The output directory.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    import matplotlib.pyplot as plt

    distributions = parse_csv_arg(args.example_distributions, str)
    densities = parse_csv_arg(args.example_densities, int)
    if len(densities) != 2:
        raise ValueError("--example-densities must contain exactly two sample sizes")
    for distribution in distributions:
        if distribution not in distributions:
            raise ValueError(f"Unknown distribution {distribution}. Available: {sorted(distributions)}")

    x_eval = np.linspace(args.eval_min, args.eval_max, num=args.n_eval)
    y_eval = regression_mean(x_eval)

    samples: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for distribution in distributions:
        distribution_index = list(distributions).index(distribution)
        for n_train in densities:
            seed = args.seed + 505050 + 1000 * n_train + 17 * (1 + distribution_index)
            rng = np.random.default_rng(seed=seed)
            x_train, y_train, _, outlier_mask = make_training_data(
                n_train=n_train,
                distribution=distribution,
                rng=rng,
                noise_ratio=args.noise_ratio,
                outlier_frac=args.outlier_frac,
                outlier_scale=args.outlier_scale,
            )
            samples[(distribution, n_train)] = (x_train, y_train, outlier_mask)

    fig, axes = plt.subplots(
        nrows=len(distributions),
        ncols=2,
        figsize=(8.4, 2.65 * len(distributions)),
        sharex=True,
        squeeze=False,
    )
    for row, distribution in enumerate(distributions):
        row_values = [y_eval]
        for n_train in densities:
            row_values.append(samples[(distribution, n_train)][1])
        y_min = min(float(np.nanmin(values)) for values in row_values)
        y_max = max(float(np.nanmax(values)) for values in row_values)
        y_pad = 0.06 * max(y_max - y_min, 1e-6)

        for col, n_train in enumerate(densities):
            ax = axes[row, col]
            x_train, y_train, outlier_mask = samples[(distribution, n_train)]
            marker_size = 18 if n_train <= 200 else 5
            marker_alpha = 0.7 if n_train <= 200 else 0.26

            ax.scatter(
                x_train[~outlier_mask],
                y_train[~outlier_mask],
                s=marker_size,
                alpha=marker_alpha,
                color="#4c78a8",
                linewidths=0,
                label="Sampled response",
            )
            if np.any(outlier_mask):
                ax.scatter(
                    x_train[outlier_mask],
                    y_train[outlier_mask],
                    s=max(marker_size, 14),
                    alpha=0.75,
                    color="#d62728",
                    linewidths=0,
                    label="Injected outlier",
                )
            ax.plot(x_eval, y_eval, color="#111111", linewidth=2.0, label="True mean")
            ax.set_title(f"{distribution_label(distribution)}, n={n_train}")
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.22)
            if row == len(distributions) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")

    handles: List[Any] = []
    labels: List[str] = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / "density_emse_example_samples.png", dpi=300)
    fig.savefig(output_dir / "density_emse_example_samples.pdf")
    plt.close(fig)


def write_outputs(results: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Write density experiment tables, summaries, and figures.

    Args:
        results: The trial-level results table.
        args: Parsed command-line arguments.
    """
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_trials(results)
    table = make_latex_table(summary, metric="rmse")

    results.to_csv(output_dir / "density_emse_trials.csv", index=False)
    summary.to_csv(output_dir / "density_emse_summary.csv", index=False)
    table.to_csv(output_dir / "density_emse_rmse_table.csv", index=False)
    with (output_dir / "density_emse_rmse_table.tex").open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, escape=False))

    plot_density(
        summary=summary,
        output_dir=output_dir,
        band=args.band,
        xscale=args.xscale,
        show_theoretical_bias=not args.hide_theoretical_bias and args.experiment == "clean-target-bias",
        n_eval=args.n_eval,
        eval_min=args.eval_min,
        eval_max=args.eval_max,
    )
    plot_runtime(summary=summary, output_dir=output_dir)
    plot_example_samples(args=args, output_dir=output_dir)

    best = summary.sort_values(["distribution", "n_train", "rmse_mean"]).groupby(["distribution", "n_train"]).head(1)
    print("\nBest method by distribution and density:")
    print(best[["distribution", "n_train", "method_label", "rmse_mean", "rmse_std", "win_rate"]].to_string(index=False))
    print(f"\nWrote results and figures to: {output_dir}")


def write_plots_from_summary(args: argparse.Namespace) -> None:
    """
    Regenerate density figures from existing summary CSV files.

    Args:
        args: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If a required cached input file is missing.
    """
    apply_experiment_defaults(args)
    output_dir = resolve_output_dir(args.output_dir)
    summary_path = output_dir / "density_emse_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot plot without an existing summary file: {summary_path}")

    summary = pd.read_csv(summary_path)
    n_eval = args.n_eval
    trials_path = output_dir / "density_emse_trials.csv"
    if trials_path.exists():
        trial_metadata = pd.read_csv(trials_path, usecols=["n_eval"], nrows=1)
        if not trial_metadata.empty:
            n_eval = int(trial_metadata.loc[0, "n_eval"])

    plot_density(
        summary=summary,
        output_dir=output_dir,
        band=args.band,
        xscale=args.xscale,
        show_theoretical_bias=not args.hide_theoretical_bias and args.experiment == "clean-target-bias",
        n_eval=n_eval,
        eval_min=args.eval_min,
        eval_max=args.eval_max,
    )
    plot_runtime(summary=summary, output_dir=output_dir)
    plot_example_samples(args=args, output_dir=output_dir)
    print(f"Wrote figures from existing summary to: {output_dir}")


def apply_experiment_defaults(args: argparse.Namespace) -> None:
    """
    Apply preset-specific defaults to parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    if args.experiment == "clean-target-bias":
        if args.outlier_frac is None:
            args.outlier_frac = 0.0
        if args.output_dir == "density_emse":
            args.output_dir = "density_emse_clean_target_bias"
    elif args.experiment == "outlier-robustness":
        if args.outlier_frac is None:
            args.outlier_frac = 0.1
        if args.output_dir == "density_emse":
            args.output_dir = "density_emse_outlier_robustness"
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the density experiment.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("clean-target-bias", "outlier-robustness"),
        default="clean-target-bias",
        help="Named experiment preset. Sets default output directory and outlier fraction unless overridden.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Monte Carlo trials per distribution/density.")
    parser.add_argument("--densities", default="50,100,200,400", help="Comma-separated training sample sizes.")
    parser.add_argument(
        "--distributions",
        default="gaussian,bimodal,exponential,gamma,lognormal,weibull",
        help=f"Comma-separated distributions from {sorted(distributions)}.",
    )
    parser.add_argument(
        "--methods",
        default="rsklpr_conden,rsklpr_joint,lowess,robust_lowess",
        help=f"Comma-separated methods from {sorted(methods)}.",
    )
    parser.add_argument("--n-eval", type=int, default=200, help="Number of clean evaluation grid points.")
    parser.add_argument("--eval-min", type=float, default=0.0, help="Lower end of the evaluation grid.")
    parser.add_argument("--eval-max", type=float, default=1.0, help="Upper end of the evaluation grid.")
    parser.add_argument(
        "--noise-ratio", type=float, default=0.18, help="Gaussian noise level relative to response range."
    )
    parser.add_argument(
        "--outlier-frac",
        type=float,
        default=None,
        help="Fraction of training responses contaminated with large additive outliers. Defaults depend on --experiment.",
    )
    parser.add_argument(
        "--outlier-scale",
        type=float,
        default=5.0,
        help="Outlier magnitude multiplier relative to the clean response range.",
    )
    parser.add_argument("--neighbor-frac", type=float, default=0.25, help="Neighborhood size as a fraction of n.")
    parser.add_argument("--min-neighbors", type=int, default=15, help="Minimum neighborhood size.")
    parser.add_argument("--max-neighbors", type=int, default=200, help="Maximum neighborhood size.")
    parser.add_argument("--seed", type=int, default=20260518, help="Base random seed.")
    parser.add_argument("--band", choices=("iqr", "q95"), default="iqr", help="Uncertainty band shown on RMSE plot.")
    parser.add_argument(
        "--xscale",
        choices=("linear", "log"),
        default="log",
        help="Scale for the training-point density axis in the RMSE plot.",
    )
    parser.add_argument(
        "--output-dir",
        default="density_emse",
        help="Relative subdirectory under $REPO_DIR/out for CSV, LaTeX, and figure outputs.",
    )
    parser.add_argument("--progress-every", type=int, default=5, help="Progress print frequency.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from an existing density_emse_summary.csv without rerunning experiments.",
    )
    parser.add_argument(
        "--hide-theoretical-bias",
        action="store_true",
        help="Do not draw the theoretical asymptotic bias RMSE on clean target-validation plots.",
    )
    parser.add_argument(
        "--example-distributions",
        default="gaussian,exponential,lognormal",
        help="Comma-separated distributions for the appendix sample-data figure.",
    )
    parser.add_argument(
        "--example-densities",
        default="50,2000",
        help="Two comma-separated sample sizes for the appendix sample-data figure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Parse command-line arguments and run the density experiment.

    Args:
        argv: Optional command-line arguments. If None, arguments are read from sys.argv.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.verbose = not args.quiet

    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.n_eval <= 1:
        raise ValueError("--n-eval must be greater than one")
    if args.outlier_scale <= 0.0:
        raise ValueError("--outlier-scale must be positive")

    if args.plot_only:
        write_plots_from_summary(args=args)
    else:
        apply_experiment_defaults(args)
        if args.outlier_frac < 0.0 or args.outlier_frac >= 1.0:
            raise ValueError("--outlier-frac must be in [0, 1)")
        results = run_experiment(args)
        write_outputs(results=results, args=args)


if __name__ == "__main__":
    main()
