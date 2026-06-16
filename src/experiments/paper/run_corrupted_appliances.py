#!/usr/bin/env python
"""Controlled target-corruption benchmark on UCI Appliances Energy.

Models are fitted on corrupted training targets, while inner-CV model
selection and outer test evaluation are scored against the original clean
target. This produces a controlled recovery benchmark: the public dataset
provides the clean signal, and the script adds known noise distributions at
known amplitudes.

Data are cached under $REPO_DIR/data and all artifacts are written under
$REPO_DIR/out. The script refuses to run if REPO_DIR is undefined.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.paper.run_public_appliances import (
    colors,
    data_file,
    data_urls,
    model_labels,
    model_order,
    RobustLowessRegressor,
    RsklprRegressor,
    download_dataset,
    make_features,
    params_to_text,
    select_features,
)
from rsklpr.kernels import tricube_normalized_metric

repo_dir_env = os.environ.get("REPO_DIR")
if not repo_dir_env:
    raise RuntimeError(
        "REPO_DIR must be defined; data are cached under $REPO_DIR/data and outputs under $REPO_DIR/out."
    )

repo_root = Path(repo_dir_env).resolve()
out_root = repo_root / "out"
out_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(out_root / ".matplotlib"))

dataset_key = "appliances_energy_corrupted"
noise_labels = {
    "gaussian": "Gaussian",
    "student_t3": "Student-t(3)",
    "exponential": "Centered exponential",
    "lognormal": "Centered log-normal",
    "contamination": "One-sided contamination",
    "heteroscedastic_gaussian": "Heteroscedastic Gaussian",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for the controlled target-corruption benchmark.

    Attributes:
        grid_profile: The named search-grid profile.
        inner_folds: Configuration values used to run the experiment.
        holdout_fraction: Configuration values used to run the experiment.
        holdout_repeats: Configuration values used to run the experiment.
        random_state: Configuration values used to run the experiment.
        n_jobs: Configuration values used to run the experiment.
        output_dir: The output directory.
        max_samples: Configuration values used to run the experiment.
        max_features: The maximum number of selected features.
        add_time_features: Whether to add cyclic timestamp features.
        keep_random_features: Configuration values used to run the experiment.
        force_download: Configuration values used to run the experiment.
        no_download: Configuration values used to run the experiment.
        noise_types: The requested corruption-noise types.
        noise_levels: The requested corruption-noise levels.
        contamination_rate: The probability of contamination for contamination noise.
        contamination_multiplier: The contamination magnitude multiplier.
    """

    grid_profile: str = "coarse"
    inner_folds: int = 4
    holdout_fraction: float = 0.20
    holdout_repeats: int = 5
    random_state: int = 42
    n_jobs: int = -1
    output_dir: str = "corrupted_appliances"
    max_samples: Optional[int] = None
    max_features: int = 8
    add_time_features: bool = True
    keep_random_features: bool = False
    force_download: bool = False
    no_download: bool = False
    noise_types: Tuple[str, ...] = ("gaussian", "student_t3", "exponential", "lognormal", "contamination")
    noise_levels: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
    contamination_rate: float = 0.05
    contamination_multiplier: float = 8.0


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


def build_search_grids(grid_profile: str) -> Dict[str, List[Dict[str, List[Any]]]]:
    """
    Build estimator search grids for the corruption benchmark.

    Args:
        grid_profile: The named search-grid profile.

    Returns:
        The display string.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    lowess_metric_blocks: List[Dict[str, List[Any]]]
    robust_metric_blocks: List[Dict[str, List[Any]]]
    rsklpr_metric_blocks: List[Dict[str, List[Any]]]
    if grid_profile == "smoke":
        lowess_neighborhoods = [31]
        robust_neighborhoods = [31]
        rsklpr_neighborhoods = [31]
        lowess_metric_blocks = [{"metric_x": ["minkowski"], "metric_x_params": [{"p": 1}]}]
        robust_metric_blocks = [{"metric_x": ["mahalanobis"], "metric_x_params": [None]}]
        rsklpr_metric_blocks = lowess_metric_blocks
        rsklpr_kr = ["conden"]
    elif grid_profile == "coarse":
        lowess_neighborhoods = [31, 63, 95, 127]
        robust_neighborhoods = [31, 63, 95, 127]
        rsklpr_neighborhoods = [31, 63, 95, 127]
        lowess_metric_blocks = [
            {"metric_x": ["minkowski"], "metric_x_params": [{"p": 1}, {"p": 2}]},
            {"metric_x": ["mahalanobis"], "metric_x_params": [None]},
        ]
        robust_metric_blocks = lowess_metric_blocks
        rsklpr_metric_blocks = lowess_metric_blocks
        rsklpr_kr = ["conden", "joint"]
    else:
        raise ValueError("grid_profile must be one of {'smoke', 'coarse'}")

    lowess_base: Dict[str, List[Any]] = {
        "size_neighborhood": lowess_neighborhoods,
        "degree": [1],
        "kp": [tricube_normalized_metric],
        "kr": ["none"],
    }
    robust_base: Dict[str, List[Any]] = {
        "size_neighborhood": robust_neighborhoods,
        "degree": [1],
        "kp": [tricube_normalized_metric],
        "robust_iters": [3],
    }
    rsklpr_base: Dict[str, List[Any]] = {
        "size_neighborhood": rsklpr_neighborhoods,
        "degree": [1],
        "kp": [tricube_normalized_metric],
        "kr": rsklpr_kr,
    }

    return {
        "lowess": [lowess_base | block for block in lowess_metric_blocks],
        "robust_lowess": [robust_base | block for block in robust_metric_blocks],
        "rsklpr": [rsklpr_base | block for block in rsklpr_metric_blocks],
    }


def filter_grid_for_cv_train_size(
    grid: List[Dict[str, List[Any]]], cv: KFold, n_outer_train: int
) -> List[Dict[str, List[Any]]]:
    """
    Drop neighborhood sizes that cannot fit inside inner-CV training folds.

    Args:
        grid: The parameter grid blocks.
        cv: The cross-validation splitter.
        n_outer_train: The outer-training sample size.

    Returns:
        The display string.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    min_inner_train = min(len(inner_train) for inner_train, _ in cv.split(np.arange(n_outer_train)))
    filtered: List[Dict[str, List[Any]]] = []
    for block in grid:
        block_copy = {key: list(value) for key, value in block.items()}
        if "size_neighborhood" in block_copy:
            block_copy["size_neighborhood"] = [
                value for value in block_copy["size_neighborhood"] if int(value) <= min_inner_train
            ]
        if block_copy.get("size_neighborhood"):
            filtered.append(block_copy)
    if not filtered:
        raise ValueError("All neighborhood sizes exceed the inner-CV training size.")
    return filtered


def make_unit_noise(
    noise_type: str,
    size: int,
    rng: np.random.Generator,
    y_reference: np.ndarray,
    contamination_rate: float,
    contamination_multiplier: float,
) -> np.ndarray:
    """
    Generate centered unit-scale corruption noise.

    Args:
        noise_type: The corruption-noise type.
        size: The number of random values to generate.
        rng: The random number generator.
        y_reference: The reference response values used to scale noise.
        contamination_rate: The probability of contamination for contamination noise.
        contamination_multiplier: The contamination magnitude multiplier.

    Returns:
        The resulting array.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    if noise_type == "gaussian":
        noise = rng.normal(size=size)
    elif noise_type == "student_t3":
        noise = rng.standard_t(df=3, size=size)
    elif noise_type == "exponential":
        noise = rng.exponential(scale=1.0, size=size)
    elif noise_type == "lognormal":
        noise = rng.lognormal(mean=0.0, sigma=1.0, size=size)
    elif noise_type == "contamination":
        noise = rng.normal(scale=0.1, size=size)
        contaminated = rng.random(size=size) < contamination_rate
        noise[contaminated] += rng.exponential(scale=contamination_multiplier, size=int(np.sum(contaminated)))
    elif noise_type == "heteroscedastic_gaussian":
        y_scaled = (y_reference - np.nanmin(y_reference)) / (np.nanmax(y_reference) - np.nanmin(y_reference))
        noise = rng.normal(size=size) * (0.25 + y_scaled)
    else:
        raise ValueError(f"Unknown noise type {noise_type!r}; choose from {list(noise_labels)}")

    noise = noise - float(np.nanmean(noise))
    scale = float(np.nanstd(noise))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros(size, dtype=float)
    return noise / scale


def corrupt_target(
    y_clean: np.ndarray,
    noise_type: str,
    noise_level: float,
    rng: np.random.Generator,
    y_scale: float,
    contamination_rate: float,
    contamination_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add scaled corruption noise to clean training targets.

    Args:
        y_clean: The clean response values.
        noise_type: The corruption-noise type.
        noise_level: The noise level relative to the clean target scale.
        rng: The random number generator.
        y_scale: The scale used to size additive corruption.
        contamination_rate: The probability of contamination for contamination noise.
        contamination_multiplier: The contamination magnitude multiplier.

    Returns:
        The resulting array.
    """
    noise_unit = make_unit_noise(
        noise_type=noise_type,
        size=len(y_clean),
        rng=rng,
        y_reference=y_clean,
        contamination_rate=contamination_rate,
        contamination_multiplier=contamination_multiplier,
    )
    additive_noise = noise_level * y_scale * noise_unit
    return y_clean + additive_noise, additive_noise


def stable_noise_offset(noise_type: str) -> int:
    """
    Compute a deterministic seed offset for one noise type.

    Args:
        noise_type: The corruption-noise type.

    Returns:
        The computed integer value.
    """
    return sum((idx + 1) * ord(char) for idx, char in enumerate(noise_type))


def clean_metric_rows(y_true_clean: np.ndarray, y_pred: np.ndarray, runtime_s: float) -> Dict[str, Union[float, bool]]:
    """
    Compute clean-target metrics for one prediction vector.

    Args:
        y_true_clean: The clean target values.
        y_pred: The predicted response values.
        runtime_s: The elapsed runtime in seconds.

    Returns:
        The computed scalar value.
    """
    finite = np.isfinite(y_pred)
    valid = bool(np.all(finite))
    if not valid:
        return {
            "valid": False,
            "non_finite_pct": float(100.0 * np.mean(~finite)),
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_ae": float("nan"),
            "bias": float("nan"),
            "error_std": float("nan"),
            "r2": float("nan"),
            "runtime_s": runtime_s,
        }

    residual = y_pred - y_true_clean
    return {
        "valid": True,
        "non_finite_pct": 0.0,
        "rmse": float(root_mean_squared_error(y_true_clean, y_pred)),
        "mae": float(mean_absolute_error(y_true_clean, y_pred)),
        "median_ae": float(median_absolute_error(y_true_clean, y_pred)),
        "bias": float(np.nanmean(residual)),
        "error_std": float(np.nanstd(residual)),
        "r2": float(r2_score(y_true_clean, y_pred)),
        "runtime_s": runtime_s,
    }


def candidate_rows(grid: List[Dict[str, List[Any]]], cv: KFold, n_outer_train: int) -> List[Dict[str, Any]]:
    """
    Expand a filtered parameter grid into candidate dictionaries.

    Args:
        grid: The parameter grid blocks.
        cv: The cross-validation splitter.
        n_outer_train: The outer-training sample size.

    Returns:
        The display string.
    """
    filtered = filter_grid_for_cv_train_size(grid=grid, cv=cv, n_outer_train=n_outer_train)
    return list(ParameterGrid(filtered))


def fit_manual_search(
    model_key: str,
    x_train: np.ndarray,
    y_clean_train: np.ndarray,
    y_corrupt_train: np.ndarray,
    grid: List[Dict[str, List[Any]]],
    config: ExperimentConfig,
    repeat: int,
) -> Tuple[BaseEstimator, Dict[str, Any], float, pd.DataFrame]:
    """
    Run manual inner-CV scored against clean targets.

    Args:
        model_key: The model key.
        x_train: The training predictor values.
        y_clean_train: The clean training response values.
        y_corrupt_train: The corrupted training response values.
        grid: The parameter grid blocks.
        config: The experiment configuration.
        repeat: The repeated holdout index.

    Returns:
        The resulting data frame.
    """
    estimator: BaseEstimator
    if model_key == "robust_lowess":
        estimator = RobustLowessRegressor()
    else:
        estimator = RsklprRegressor(seed=config.random_state + repeat)

    cv = KFold(n_splits=config.inner_folds, shuffle=True, random_state=config.random_state + repeat)
    params_list = candidate_rows(grid=grid, cv=cv, n_outer_train=len(y_clean_train))
    split_list = list(cv.split(x_train))

    def evaluate_fold(
        params: Dict[str, Any], fold: int, inner_train: np.ndarray, inner_val: np.ndarray
    ) -> Dict[str, Any]:
        model = clone(estimator).set_params(**params)
        try:
            model.fit(x_train[inner_train], y_corrupt_train[inner_train])
            y_pred = np.asarray(model.predict(x_train[inner_val]), dtype=float).ravel()
            if y_pred.shape[0] != len(inner_val) or not np.all(np.isfinite(y_pred)):
                score = float("inf")
            else:
                score = float(root_mean_squared_error(y_clean_train[inner_val], y_pred))
        except Exception:
            score = float("inf")
        return {
            "fold": fold,
            "params": params_to_text(params),
            "clean_cv_rmse": score,
        }

    tasks = [
        delayed(evaluate_fold)(params, fold, inner_train, inner_val)
        for params in params_list
        for fold, (inner_train, inner_val) in enumerate(split_list)
    ]
    rows: List[Dict[str, Any]] = Parallel(n_jobs=config.n_jobs)(tasks)

    scores_by_param: Dict[str, List[float]] = {}
    params_by_text = {params_to_text(params): params for params in params_list}
    for row in rows:
        param_text = str(row["params"])
        scores_by_param.setdefault(param_text, []).append(float(row["clean_cv_rmse"]))

    best_score = float("inf")
    best_params: Dict[str, Any] = params_list[0]
    for param_text, scores in scores_by_param.items():
        mean_score = float(np.mean(scores)) if np.all(np.isfinite(scores)) else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_params = params_by_text[param_text]

    best_estimator = clone(estimator).set_params(**best_params)
    best_estimator.fit(x_train, y_corrupt_train)
    cv_results = pd.DataFrame(rows)
    cv_results["mean_clean_cv_rmse"] = cv_results.groupby("params")["clean_cv_rmse"].transform("mean")
    cv_results["rank_clean_cv_rmse"] = cv_results["mean_clean_cv_rmse"].rank(method="dense")
    return best_estimator, best_params, best_score, cv_results


def run_benchmark(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all repeated controlled target-corruption benchmark splits.

    Args:
        config: The experiment configuration.

    Returns:
        The resulting data frame.

    Raises:
        FileNotFoundError: If a required cached input file is missing.
    """
    if not config.no_download:
        download_dataset(force=config.force_download)
    elif not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}. Run without --no-download first.")

    df = pd.read_csv(data_file)
    if config.max_samples is not None and config.max_samples < len(df):
        df = df.sample(n=config.max_samples, random_state=config.random_state).reset_index(drop=True)
    x, y = make_features(
        df=df,
        add_time_features=config.add_time_features,
        drop_random_features=not config.keep_random_features,
    )
    grids = build_search_grids(config.grid_profile)

    cv_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for repeat in range(config.holdout_repeats):
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=config.holdout_fraction,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        x_train_all = x.iloc[train_idx].reset_index(drop=True)
        y_train_clean = y.iloc[train_idx].reset_index(drop=True)
        x_test_all = x.iloc[test_idx].reset_index(drop=True)
        y_test_clean = y.iloc[test_idx].reset_index(drop=True)
        selected_features = select_features(x_train_all, y_train_clean, config.max_features)

        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train_all[selected_features])
        x_test = x_scaler.transform(x_test_all[selected_features])
        y_train_clean_arr = y_train_clean.to_numpy(dtype=float)
        y_test_clean_arr = y_test_clean.to_numpy(dtype=float)
        y_scale = float(np.nanstd(y_train_clean_arr))

        print(
            f"\n--- Holdout {repeat + 1}/{config.holdout_repeats} "
            f"(train={len(train_idx)}, test={len(test_idx)}) ---",
            flush=True,
        )
        print(f"selected features={selected_features}", flush=True)

        for noise_type in config.noise_types:
            for noise_level in config.noise_levels:
                rng = np.random.default_rng(config.random_state + 100_000 * repeat + stable_noise_offset(noise_type))
                y_train_corrupt, train_noise = corrupt_target(
                    y_clean=y_train_clean_arr,
                    noise_type=noise_type,
                    noise_level=noise_level,
                    rng=rng,
                    y_scale=y_scale,
                    contamination_rate=config.contamination_rate,
                    contamination_multiplier=config.contamination_multiplier,
                )
                print(
                    f"\n  noise={noise_type}, level={noise_level:g}, " f"noise_std={np.nanstd(train_noise):.3f}",
                    flush=True,
                )

                for model_key in model_order:
                    print(f"    searching {model_labels[model_key]}...", flush=True)
                    print(
                        f"      manual CV requested n_jobs={config.n_jobs}; effective n_jobs={effective_n_jobs(config.n_jobs)}",
                        flush=True,
                    )
                    start = time.perf_counter()
                    best_estimator, best_params, best_cv_rmse, cv_results = fit_manual_search(
                        model_key=model_key,
                        x_train=x_train,
                        y_clean_train=y_train_clean_arr,
                        y_corrupt_train=y_train_corrupt,
                        grid=grids[model_key],
                        config=config,
                        repeat=repeat,
                    )
                    search_s = time.perf_counter() - start

                    for _, row in cv_results.iterrows():
                        cv_rows.append(
                            {
                                "dataset": dataset_key,
                                "repeat": repeat,
                                "noise_type": noise_type,
                                "noise_label": noise_labels[noise_type],
                                "noise_level": noise_level,
                                "model": model_key,
                                "model_label": model_labels[model_key],
                                "fold": row["fold"],
                                "clean_cv_rmse": row["clean_cv_rmse"],
                                "mean_clean_cv_rmse": row["mean_clean_cv_rmse"],
                                "rank_clean_cv_rmse": row["rank_clean_cv_rmse"],
                                "params": row["params"],
                                "selected_features": ",".join(selected_features),
                                "n_features": len(selected_features),
                            }
                        )

                    selection_rows.append(
                        {
                            "dataset": dataset_key,
                            "repeat": repeat,
                            "noise_type": noise_type,
                            "noise_label": noise_labels[noise_type],
                            "noise_level": noise_level,
                            "model": model_key,
                            "model_label": model_labels[model_key],
                            "best_clean_cv_rmse": best_cv_rmse,
                            "selected_params": params_to_text(best_params),
                            "selected_features": ",".join(selected_features),
                            "n_features": len(selected_features),
                            "search_runtime_s": search_s,
                        }
                    )

                    pred_start = time.perf_counter()
                    y_pred = np.asarray(best_estimator.predict(x_test), dtype=float).ravel()
                    pred_s = time.perf_counter() - pred_start
                    row = {
                        "dataset": dataset_key,
                        "repeat": repeat,
                        "noise_type": noise_type,
                        "noise_label": noise_labels[noise_type],
                        "noise_level": noise_level,
                        "model": model_key,
                        "model_label": model_labels[model_key],
                        "best_clean_cv_rmse": best_cv_rmse,
                        "selected_params": params_to_text(best_params),
                        "selected_features": ",".join(selected_features),
                        "n_features": len(selected_features),
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                        "train_noise_std": float(np.nanstd(train_noise)),
                        "train_noise_mean": float(np.nanmean(train_noise)),
                        "search_runtime_s": search_s,
                    }
                    row.update(clean_metric_rows(y_true_clean=y_test_clean_arr, y_pred=y_pred, runtime_s=pred_s))
                    test_rows.append(row)
                    print(
                        f"    > {model_labels[model_key]}: clean test RMSE={row['rmse']:.3f}, "
                        f"MAE={row['mae']:.3f}, params={row['selected_params']}",
                        flush=True,
                    )

    return pd.DataFrame(cv_rows), pd.DataFrame(selection_rows), pd.DataFrame(test_rows)


def summarize(test_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate corruption benchmark test metrics by noise and model.

    Args:
        test_results: The held-out test results table.

    Returns:
        The resulting data frame.
    """
    keys = ["dataset", "noise_type", "noise_label", "noise_level", "model", "model_label"]
    valid_results = test_results[test_results["valid"]].copy()
    summary = valid_results.groupby(keys, as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_median=("mae", "median"),
        median_ae_mean=("median_ae", "mean"),
        bias_mean=("bias", "mean"),
        error_std_mean=("error_std", "mean"),
        r2_mean=("r2", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        search_runtime_s_mean=("search_runtime_s", "mean"),
        repeats=("repeat", "nunique"),
    )
    winners = valid_results.copy()
    winners["is_winner"] = winners["rmse"] == winners.groupby(["dataset", "noise_type", "noise_level", "repeat"])[
        "rmse"
    ].transform("min")
    win_rates = winners.groupby(keys, as_index=False).agg(win_rate=("is_winner", "mean"))
    return summary.merge(win_rates, on=keys, how="left").sort_values(["noise_type", "noise_level", "rmse_mean"])


def summarize_relative(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model metrics relative to standard LOWESS.

    Args:
        summary: The summary results table.

    Returns:
        The resulting data frame.
    """
    baseline = summary[summary["model"] == "lowess"][["noise_type", "noise_level", "rmse_mean", "mae_mean"]].rename(
        columns={"rmse_mean": "lowess_rmse_mean", "mae_mean": "lowess_mae_mean"}
    )
    relative = summary.merge(baseline, on=["noise_type", "noise_level"], how="left")
    relative["rmse_vs_lowess"] = relative["rmse_mean"] / relative["lowess_rmse_mean"]
    relative["mae_vs_lowess"] = relative["mae_mean"] / relative["lowess_mae_mean"]
    return relative


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Write corruption benchmark RMSE and relative-RMSE figures.

    Args:
        summary: The summary results table.
        output_dir: The output directory.
    """
    import matplotlib.pyplot as plt

    noise_types = list(dict.fromkeys(summary["noise_type"]))
    fig, axes = plt.subplots(len(noise_types), 1, figsize=(7.5, max(3.0, 2.4 * len(noise_types))), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, noise_type in zip(axes_arr, noise_types):
        subset = summary[summary["noise_type"] == noise_type]
        for model_key in model_order:
            model_subset = subset[subset["model"] == model_key].sort_values("noise_level")
            ax.errorbar(
                model_subset["noise_level"],
                model_subset["rmse_mean"],
                yerr=model_subset["rmse_std"].fillna(0.0),
                marker="o",
                color=colors[model_key],
                label=model_labels[model_key],
                capsize=3,
            )
        ax.set_title(noise_labels[noise_type])
        ax.set_ylabel("Clean-target RMSE")
        ax.grid(alpha=0.25)
    axes_arr[-1].set_xlabel("Noise level (fraction of clean training-target std)")
    axes_arr[0].legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output_dir / "corrupted_appliances_rmse_by_noise.png", dpi=300)
    fig.savefig(output_dir / "corrupted_appliances_rmse_by_noise.pdf")
    plt.close(fig)

    relative = summarize_relative(summary)
    fig, axes = plt.subplots(len(noise_types), 1, figsize=(7.5, max(3.0, 2.4 * len(noise_types))), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, noise_type in zip(axes_arr, noise_types):
        subset = relative[relative["noise_type"] == noise_type]
        for model_key in model_order:
            model_subset = subset[subset["model"] == model_key].sort_values("noise_level")
            ax.plot(
                model_subset["noise_level"],
                model_subset["rmse_vs_lowess"],
                marker="o",
                color=colors[model_key],
                label=model_labels[model_key],
            )
        ax.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_title(noise_labels[noise_type])
        ax.set_ylabel("RMSE / LOWESS")
        ax.grid(alpha=0.25)
    axes_arr[-1].set_xlabel("Noise level (fraction of clean training-target std)")
    axes_arr[0].legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(output_dir / "corrupted_appliances_relative_rmse.png", dpi=300)
    fig.savefig(output_dir / "corrupted_appliances_relative_rmse.pdf")
    plt.close(fig)


def write_outputs(
    cv_results: pd.DataFrame,
    selection: pd.DataFrame,
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """
    Write corruption benchmark tables, metadata, and figures.

    Args:
        cv_results: The cross-validation results table.
        selection: The selected-hyperparameter results table.
        test: The held-out test results table.
        config: The experiment configuration.
    """
    output_dir = resolve_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(test)
    relative = summarize_relative(summary)

    cv_results.to_csv(output_dir / "corrupted_appliances_cv_results.csv", index=False)
    selection.to_csv(output_dir / "corrupted_appliances_selected_hyperparameters.csv", index=False)
    test.to_csv(output_dir / "corrupted_appliances_test_results.csv", index=False)
    summary.to_csv(output_dir / "corrupted_appliances_summary.csv", index=False)
    relative.to_csv(output_dir / "corrupted_appliances_relative_summary.csv", index=False)
    with (output_dir / "corrupted_appliances_summary.tex").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, float_format="%.3f"))

    metadata = {
        "dataset": "UCI Appliances Energy Prediction with controlled target corruption",
        "source_urls": data_urls,
        "data_file": str(data_file),
        "clean_target": "Appliances",
        "target_protocol": "fit on corrupted training targets; select and evaluate against clean targets",
        "noise_labels": noise_labels,
        "config": asdict(config),
        "search_grids": {
            model: [params_to_text({key: value for key, value in block.items()}) for block in blocks]
            for model, blocks in build_search_grids(config.grid_profile).items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    plot_results(summary=summary, output_dir=output_dir)

    print("\nClean-target test summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote corrupted benchmark artifacts to: {output_dir}")


def parse_csv_tuple(raw: str, cast: Any = str) -> Tuple[Any, ...]:
    """
    Parse a comma-separated command-line argument as a tuple.

    Args:
        raw: The raw comma-separated value.
        cast: The callable used to cast each parsed value.

    Returns:
        The computed result tuple.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    parsed = tuple(cast(item.strip()) for item in raw.split(",") if item.strip())
    if not parsed:
        raise ValueError("Expected at least one comma-separated value.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the corruption benchmark.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-profile", choices=["smoke", "coarse"], default="coarse")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-dir", default="corrupted_appliances", help="Relative subdirectory under $REPO_DIR/out."
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--noise-types", default="gaussian,student_t3,exponential,lognormal,contamination")
    parser.add_argument("--noise-levels", default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--contamination-rate", type=float, default=0.05)
    parser.add_argument("--contamination-multiplier", type=float, default=8.0)
    parser.add_argument("--force-download", action="store_true", help="Download the dataset even if cached.")
    parser.add_argument("--no-download", action="store_true", help="Require the cached dataset and skip download.")
    parser.add_argument(
        "--no-time-features", action="store_true", help="Do not add cyclic hour/day timestamp features."
    )
    parser.add_argument(
        "--keep-random-features", action="store_true", help="Keep rv1/rv2 random variables as features."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Parse command-line arguments and run the corruption benchmark.

    Args:
        argv: Optional command-line arguments. If None, arguments are read from sys.argv.

    Raises:
        ValueError: If an argument value is invalid for the experiment.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.inner_folds <= 1:
        raise ValueError("--inner-folds must be greater than 1")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be in (0, 1)")
    if args.holdout_repeats <= 0:
        raise ValueError("--holdout-repeats must be positive")
    if args.max_features < 0:
        raise ValueError("--max-features must be non-negative")
    if args.contamination_rate < 0.0 or args.contamination_rate > 1.0:
        raise ValueError("--contamination-rate must be in [0, 1]")

    noise_types = parse_csv_tuple(args.noise_types, str)
    invalid_noise = [noise_type for noise_type in noise_types if noise_type not in noise_labels]
    if invalid_noise:
        raise ValueError(f"Unknown noise types {invalid_noise}; choose from {list(noise_labels)}")
    noise_levels = parse_csv_tuple(args.noise_levels, float)
    if any(level < 0.0 for level in noise_levels):
        raise ValueError("--noise-levels must be non-negative")

    config = ExperimentConfig(
        grid_profile=args.grid_profile,
        inner_folds=args.inner_folds,
        holdout_fraction=args.holdout_fraction,
        holdout_repeats=args.holdout_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_features=args.max_features,
        add_time_features=not args.no_time_features,
        keep_random_features=args.keep_random_features,
        force_download=args.force_download,
        no_download=args.no_download,
        noise_types=noise_types,
        noise_levels=noise_levels,
        contamination_rate=args.contamination_rate,
        contamination_multiplier=args.contamination_multiplier,
    )
    cv_results, selection, test = run_benchmark(config)
    write_outputs(cv_results=cv_results, selection=selection, test=test, config=config)


if __name__ == "__main__":
    main()
