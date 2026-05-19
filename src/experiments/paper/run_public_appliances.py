#!/usr/bin/env python
"""Public benchmark on the UCI Appliances Energy Prediction dataset.

The protocol mirrors the holdout-CV structure used in the GCLPR experiments:
for each repeated train/test split, each model family is tuned with inner
GridSearchCV on the training portion and evaluated once on the held-out test
portion. The coarse grid is deliberately broad; after inspecting the selected
neighborhoods, narrow the explicit grid below and rerun the whole experiment.

Data are cached under $REPO_DIR/data and all artifacts are written under
$REPO_DIR/out. The script refuses to run if REPO_DIR is undefined.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

sys.dont_write_bytecode = True

REPO_DIR_ENV = os.environ.get("REPO_DIR")
if not REPO_DIR_ENV:
    raise RuntimeError(
        "REPO_DIR must be defined; data are cached under $REPO_DIR/data and outputs under $REPO_DIR/out."
    )

REPO_ROOT = Path(REPO_DIR_ENV).resolve()
DATA_ROOT = REPO_ROOT / "data" / "appliances_energy"
OUT_ROOT = REPO_ROOT / "out"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rsklpr.kernels import laplacian_normalized_metric, tricube_normalized_metric
from rsklpr.rsklpr import Rsklpr


KernelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]

DATA_URLS = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",)
DATA_FILE = DATA_ROOT / "energydata_complete.csv"
TARGET = "Appliances"

MODEL_ORDER = ["lowess", "robust_lowess", "rsklpr"]
MODEL_LABELS = {
    "lowess": "LOWESS",
    "robust_lowess": "Robust LOWESS",
    "rsklpr": "RSKLPR",
}
COLORS = {
    "lowess": "#2ca02c",
    "robust_lowess": "#ff7f0e",
    "rsklpr": "#1f77b4",
}
KERNEL_LABELS = {
    laplacian_normalized_metric: "laplacian",
    tricube_normalized_metric: "tricube",
}


@dataclass(frozen=True)
class ExperimentConfig:
    grid_profile: str = "coarse"
    inner_folds: int = 4
    holdout_fraction: float = 0.20
    holdout_repeats: int = 5
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    output_dir: str = "public_appliances"
    max_samples: int | None = None
    max_features: int = 8
    add_time_features: bool = True
    keep_random_features: bool = False
    force_download: bool = False
    no_download: bool = False


class RsklprRegressor(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper around Rsklpr."""

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: KernelFn = laplacian_normalized_metric,
        kr: str = "none",
        metric_x: str = "minkowski",
        metric_x_params: dict[str, Any] | None = None,
        bw1: str = "normal_reference",
        bw2: str = "normal_reference",
        seed: int = 42,
    ) -> None:
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.kp = kp
        self.kr = kr
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params
        self.bw1 = bw1
        self.bw2 = bw2
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RsklprRegressor":
        self.model_ = Rsklpr(
            size_neighborhood=self.size_neighborhood,
            degree=self.degree,
            kp=self.kp,
            kr=self.kr,
            metric_x=self.metric_x,
            metric_x_params=self.metric_x_params,
            bw1=self.bw1,
            bw2=self.bw2,
            seed=self.seed,
            suppress_warnings=True,
        )
        self.model_.fit(x=np.asarray(X, dtype=float), y=np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(x=np.asarray(X, dtype=float)), dtype=float).ravel()


class RobustLowessRegressor(BaseEstimator, RegressorMixin):
    """Multivariate robust LOWESS using iterative bisquare residual weights."""

    def __init__(
        self,
        size_neighborhood: int = 50,
        degree: int = 1,
        kp: KernelFn = tricube_normalized_metric,
        metric_x: str = "minkowski",
        metric_x_params: dict[str, Any] | None = None,
        robust_iters: int = 3,
    ) -> None:
        self.size_neighborhood = size_neighborhood
        self.degree = degree
        self.kp = kp
        self.metric_x = metric_x
        self.metric_x_params = metric_x_params
        self.robust_iters = robust_iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustLowessRegressor":
        self.x_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float).ravel()
        if self.x_.shape[0] < self.size_neighborhood:
            raise ValueError(
                f"Provided inputs have {self.x_.shape[0]} observations, less than "
                f"size_neighborhood={self.size_neighborhood}"
            )
        if self.degree not in (0, 1):
            raise ValueError("RobustLowessRegressor currently supports degree 0 or 1")

        metric_params, p = self._metric_params()
        self.neighbors_ = NearestNeighbors(
            n_neighbors=self.size_neighborhood,
            algorithm="auto",
            metric=self.metric_x,
            p=p,
            metric_params=metric_params,
        )
        self.neighbors_.fit(self.x_)
        self.robust_weights_ = np.ones(self.x_.shape[0], dtype=float)

        for _ in range(self.robust_iters):
            fitted = self._predict_with_current_weights(self.x_)
            residual = self.y_ - fitted
            scale = np.nanmedian(np.abs(residual))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                break
            u = residual / (6.0 * scale)
            new_weights = np.square(1.0 - np.square(u))
            new_weights[np.abs(u) >= 1.0] = 0.0
            new_weights[~np.isfinite(new_weights)] = 0.0
            self.robust_weights_ = new_weights

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_with_current_weights(np.asarray(X, dtype=float))

    def _metric_params(self) -> tuple[dict[str, Any], float]:
        metric_params: dict[str, Any] = {} if self.metric_x_params is None else self.metric_x_params.copy()
        p = float(metric_params.pop("p", 2.0))
        if self.metric_x == "mahalanobis" and "VI" not in metric_params:
            metric_params["VI"] = np.linalg.pinv(np.atleast_2d(np.cov(self.x_, rowvar=False)))
        return metric_params, p

    def _predict_with_current_weights(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self.neighbors_.kneighbors(X)
        predictions = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            neighbor_idx = indices[i]
            x_neighbors = self.x_[neighbor_idx]
            y_neighbors = self.y_[neighbor_idx]
            weights = np.asarray(
                self.kp(X[i : i + 1], x_neighbors, distances[i], i, neighbor_idx),
                dtype=float,
            ).ravel()
            weights = weights * self.robust_weights_[neighbor_idx]
            predictions[i] = self._local_prediction(X[i], x_neighbors, y_neighbors, weights)
        return predictions

    def _local_prediction(
        self, x0: np.ndarray, x_neighbors: np.ndarray, y_neighbors: np.ndarray, weights: np.ndarray
    ) -> float:
        if np.sum(weights) <= np.finfo(float).eps:
            return float(np.nanmean(y_neighbors))

        if self.degree == 0:
            return float(np.average(y_neighbors, weights=weights))

        design = np.column_stack([np.ones(x_neighbors.shape[0]), x_neighbors - x0])
        sqrt_w = np.sqrt(np.clip(weights, a_min=0.0, a_max=None))
        try:
            beta, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], y_neighbors * sqrt_w, rcond=None)
        except np.linalg.LinAlgError:
            return float(np.average(y_neighbors, weights=weights))
        return float(beta[0])


def resolve_output_dir(output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    if output_dir.is_absolute():
        raise ValueError("--output-dir must be a relative subdirectory under $REPO_DIR/out")
    return OUT_ROOT / output_dir


def download_dataset(force: bool = False) -> Path:
    if DATA_FILE.exists() and not force:
        return DATA_FILE

    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            print(f"Downloading {url} -> {DATA_FILE}", flush=True)
            urllib.request.urlretrieve(url, DATA_FILE)
            return DATA_FILE
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to download Appliances Energy dataset: {last_error}") from last_error


def load_dataset(config: ExperimentConfig) -> pd.DataFrame:
    if not config.no_download:
        download_dataset(force=config.force_download)
    elif not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Run without --no-download first.")

    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r}; found columns {list(df.columns)}")

    if config.max_samples is not None and config.max_samples < len(df):
        df = df.sample(n=config.max_samples, random_state=config.random_state).reset_index(drop=True)
    return df


def make_features(
    df: pd.DataFrame, add_time_features: bool, drop_random_features: bool
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].astype(float)
    x = df.drop(columns=[TARGET]).copy()

    if add_time_features:
        dt = pd.to_datetime(x["date"])
        hour = dt.dt.hour + dt.dt.minute / 60.0
        day = dt.dt.dayofweek.astype(float)
        x["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        x["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        x["day_sin"] = np.sin(2.0 * np.pi * day / 7.0)
        x["day_cos"] = np.cos(2.0 * np.pi * day / 7.0)

    x = x.drop(columns=["date"])
    if drop_random_features:
        x = x.drop(columns=[c for c in ("rv1", "rv2") if c in x.columns])

    x = x.apply(pd.to_numeric, errors="raise")
    return x, y


def select_features(x_train: pd.DataFrame, y_train: pd.Series, max_features: int) -> list[str]:
    if max_features <= 0 or max_features >= x_train.shape[1]:
        return list(x_train.columns)

    scores = {}
    y_arr = y_train.to_numpy(dtype=float)
    for column in x_train.columns:
        x_arr = x_train[column].to_numpy(dtype=float)
        if np.nanstd(x_arr) <= np.finfo(float).eps:
            scores[column] = 0.0
        else:
            corr = np.corrcoef(x_arr, y_arr)[0, 1]
            scores[column] = abs(float(corr)) if np.isfinite(corr) else 0.0

    return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_features]]


def build_search_grids(grid_profile: str) -> dict[str, list[dict[str, list[Any]]]]:
    """Build explicit grids. Use coarse first, then focused after inspecting winners."""
    if grid_profile == "smoke":
        neighborhoods = [31]
        kernels = [laplacian_normalized_metric]
        metric_blocks = [
            {"metric_x": ["minkowski"], "metric_x_params": [{"p": 2}]},
        ]
        robust_metric_blocks = metric_blocks
        rsklpr_kr = ["joint"]
    elif grid_profile == "coarse":
        neighborhoods = [15, 31, 63, 127, 255]
        kernels = [laplacian_normalized_metric, tricube_normalized_metric]
        metric_blocks = [
            {"metric_x": ["minkowski"], "metric_x_params": [{"p": 1}, {"p": 2}]},
            {"metric_x": ["mahalanobis"], "metric_x_params": [None]},
        ]
        robust_metric_blocks = metric_blocks
        rsklpr_kr = ["conden", "joint"]
    elif grid_profile == "focused":
        neighborhoods = list(range(35, 122, 6))
        kernels = [tricube_normalized_metric]
        metric_blocks = [
            {"metric_x": ["minkowski"], "metric_x_params": [{"p": 1}]},
        ]
        robust_metric_blocks = [
            {"metric_x": ["mahalanobis"], "metric_x_params": [None]},
        ]
        rsklpr_kr = ["conden"]
    else:
        raise ValueError("grid_profile must be one of {'smoke', 'coarse', 'focused'}")

    lowess_base = {
        "size_neighborhood": neighborhoods,
        "degree": [1],
        "kp": kernels,
        "kr": ["none"],
    }
    robust_lowess_base = {
        "size_neighborhood": neighborhoods,
        "degree": [1],
        "kp": kernels,
        "robust_iters": [3],
    }
    rsklpr_base = {
        "size_neighborhood": neighborhoods,
        "degree": [1],
        "kp": kernels,
        "kr": rsklpr_kr,
    }

    return {
        "lowess": [lowess_base | block for block in metric_blocks],
        "robust_lowess": [robust_lowess_base | block for block in robust_metric_blocks],
        "rsklpr": [rsklpr_base | block for block in metric_blocks],
    }


def finite_neg_rmse(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = np.asarray(estimator.predict(X), dtype=float).ravel()
    if y_pred.shape[0] != len(y) or not np.all(np.isfinite(y_pred)):
        return -np.inf
    return -float(root_mean_squared_error(y, y_pred))


def filter_grid_for_cv_train_size(
    grid: list[dict[str, list[Any]]], cv: KFold, n_outer_train: int
) -> list[dict[str, list[Any]]]:
    min_inner_train = min(len(inner_train) for inner_train, _ in cv.split(np.arange(n_outer_train)))
    filtered: list[dict[str, list[Any]]] = []
    for block in grid:
        block_copy = {key: list(value) for key, value in block.items()}
        if "size_neighborhood" in block_copy:
            block_copy["size_neighborhood"] = [
                value for value in block_copy["size_neighborhood"] if int(value) <= min_inner_train
            ]
        if block_copy.get("size_neighborhood"):
            filtered.append(block_copy)

    if not filtered:
        raise ValueError(
            "All neighborhood sizes exceed the inner-CV training size. "
            "Increase --max-samples or reduce the neighborhood grid."
        )
    return filtered


def serialize_param_value(value: Any) -> Any:
    if callable(value):
        return KERNEL_LABELS.get(value, getattr(value, "__name__", str(value)))
    if isinstance(value, dict):
        return {key: serialize_param_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_param_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(serialize_param_value(item) for item in value)
    return value


def params_to_text(params: dict[str, Any]) -> str:
    clean = {key: serialize_param_value(value) for key, value in params.items()}
    return json.dumps(clean, sort_keys=True)


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, runtime_s: float) -> dict[str, float | bool]:
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

    residual = y_pred - y_true
    return {
        "valid": True,
        "non_finite_pct": 0.0,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "bias": float(np.nanmean(residual)),
        "error_std": float(np.nanstd(residual)),
        "r2": float(r2_score(y_true, y_pred)),
        "runtime_s": runtime_s,
    }


def cv_results_to_rows(
    search: GridSearchCV, repeat: int, model_key: str, selected_features: list[str]
) -> list[dict[str, Any]]:
    cv_results = pd.DataFrame(search.cv_results_)
    rows: list[dict[str, Any]] = []
    for _, row in cv_results.iterrows():
        rows.append(
            {
                "dataset": "appliances_energy",
                "repeat": repeat,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "rank_test_rmse": row["rank_test_score"],
                "mean_cv_rmse": -row["mean_test_score"],
                "std_cv_rmse": row["std_test_score"],
                "params": params_to_text(row["params"]),
                "selected_features": ",".join(selected_features),
                "n_features": len(selected_features),
            }
        )
    return rows


def fit_search(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    grid: list[dict[str, list[Any]]],
    config: ExperimentConfig,
    repeat: int,
) -> GridSearchCV:
    if model_key == "robust_lowess":
        estimator: BaseEstimator = RobustLowessRegressor()
    else:
        estimator = RsklprRegressor(seed=config.random_state + repeat)
    cv = KFold(n_splits=config.inner_folds, shuffle=True, random_state=config.random_state + repeat)
    filtered_grid = filter_grid_for_cv_train_size(grid=grid, cv=cv, n_outer_train=len(y_train))
    print(
        f"    GridSearchCV requested n_jobs={config.n_jobs}; effective n_jobs={effective_n_jobs(config.n_jobs)}",
        flush=True,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=filtered_grid,
        scoring=finite_neg_rmse,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
        verbose=config.verbose,
        error_score=np.nan,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    return search


def run_benchmark(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(config)
    x, y = make_features(
        df=df,
        add_time_features=config.add_time_features,
        drop_random_features=not config.keep_random_features,
    )
    grids = build_search_grids(config.grid_profile)

    cv_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for repeat in range(config.holdout_repeats):
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=config.holdout_fraction,
            shuffle=True,
            random_state=config.random_state + repeat,
        )
        x_train_all = x.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_test_all = x.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        selected_features = select_features(x_train_all, y_train, config.max_features)

        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train_all[selected_features])
        x_test = x_scaler.transform(x_test_all[selected_features])
        y_train_arr = y_train.to_numpy(dtype=float)
        y_test_arr = y_test.to_numpy(dtype=float)

        print(
            f"\n--- Holdout {repeat + 1}/{config.holdout_repeats} "
            f"(train={len(train_idx)}, test={len(test_idx)}) ---",
            flush=True,
        )
        print(f"selected features={selected_features}", flush=True)

        for model_key in MODEL_ORDER:
            print(f"  searching {MODEL_LABELS[model_key]}...", flush=True)
            start = time.perf_counter()
            search = fit_search(
                model_key=model_key,
                x_train=x_train,
                y_train=y_train_arr,
                grid=grids[model_key],
                config=config,
                repeat=repeat,
            )
            search_s = time.perf_counter() - start
            best_params = search.best_params_
            cv_rows.extend(cv_results_to_rows(search, repeat, model_key, selected_features))
            selection_rows.append(
                {
                    "dataset": "appliances_energy",
                    "repeat": repeat,
                    "model": model_key,
                    "model_label": MODEL_LABELS[model_key],
                    "best_cv_rmse": -float(search.best_score_),
                    "selected_params": params_to_text(best_params),
                    "selected_features": ",".join(selected_features),
                    "n_features": len(selected_features),
                    "search_runtime_s": search_s,
                }
            )

            pred_start = time.perf_counter()
            y_pred = np.asarray(search.best_estimator_.predict(x_test), dtype=float).ravel()
            pred_s = time.perf_counter() - pred_start
            row = {
                "dataset": "appliances_energy",
                "repeat": repeat,
                "model": model_key,
                "model_label": MODEL_LABELS[model_key],
                "best_cv_rmse": -float(search.best_score_),
                "selected_params": params_to_text(best_params),
                "selected_features": ",".join(selected_features),
                "n_features": len(selected_features),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "search_runtime_s": search_s,
            }
            row.update(metric_rows(y_true=y_test_arr, y_pred=y_pred, runtime_s=pred_s))
            test_rows.append(row)
            print(
                f"  > {MODEL_LABELS[model_key]}: test RMSE={row['rmse']:.3f}, "
                f"MAE={row['mae']:.3f}, params={row['selected_params']}",
                flush=True,
            )

    return pd.DataFrame(cv_rows), pd.DataFrame(selection_rows), pd.DataFrame(test_rows)


def summarize(test_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "model_label"]
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
    winners["is_winner"] = winners["rmse"] == winners.groupby(["dataset", "repeat"])["rmse"].transform("min")
    win_rates = winners.groupby(keys, as_index=False).agg(win_rate=("is_winner", "mean"))
    return summary.merge(win_rates, on=keys, how="left").sort_values("rmse_mean")


def plot_results(summary: pd.DataFrame, test_results: pd.DataFrame, output_dir: Path) -> None:
    order = list(summary["model_label"])
    colors = [COLORS.get(m, "#333333") for m in summary["model"]]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(order, summary["rmse_mean"], yerr=summary["rmse_std"].fillna(0.0), color=colors, alpha=0.85, capsize=4)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Appliances Energy Prediction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "appliances_test_rmse.png", dpi=300)
    fig.savefig(output_dir / "appliances_test_rmse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    data = [
        test_results[test_results["model_label"] == label]["rmse"].dropna().to_numpy(dtype=float) for label in order
    ]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Holdout Test RMSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "appliances_test_rmse_boxplot.png", dpi=300)
    fig.savefig(output_dir / "appliances_test_rmse_boxplot.pdf")
    plt.close(fig)


def write_outputs(
    cv_results: pd.DataFrame,
    selection: pd.DataFrame,
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    output_dir = resolve_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(test)

    cv_results.to_csv(output_dir / "appliances_cv_results.csv", index=False)
    selection.to_csv(output_dir / "appliances_selected_hyperparameters.csv", index=False)
    test.to_csv(output_dir / "appliances_test_results.csv", index=False)
    summary.to_csv(output_dir / "appliances_summary.csv", index=False)
    with (output_dir / "appliances_summary.tex").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, float_format="%.3f"))

    metadata = {
        "dataset": "UCI Appliances Energy Prediction",
        "source_urls": DATA_URLS,
        "data_file": str(DATA_FILE),
        "config": asdict(config),
        "search_grids": {
            model: [params_to_text({key: value for key, value in block.items()}) for block in blocks]
            for model, blocks in build_search_grids(config.grid_profile).items()
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    plot_results(summary=summary, test_results=test, output_dir=output_dir)

    print("\nTest summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote benchmark artifacts to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-profile", choices=["smoke", "coarse", "focused"], default="coarse")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default="public_appliances", help="Relative subdirectory under $REPO_DIR/out.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--force-download", action="store_true", help="Download the dataset even if cached.")
    parser.add_argument("--no-download", action="store_true", help="Require the cached dataset and skip download.")
    parser.add_argument(
        "--no-time-features", action="store_true", help="Do not add cyclic hour/day timestamp features."
    )
    parser.add_argument(
        "--keep-random-features", action="store_true", help="Keep rv1/rv2 random variables as features."
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
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

    config = ExperimentConfig(
        grid_profile=args.grid_profile,
        inner_folds=args.inner_folds,
        holdout_fraction=args.holdout_fraction,
        holdout_repeats=args.holdout_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_features=args.max_features,
        add_time_features=not args.no_time_features,
        keep_random_features=args.keep_random_features,
        force_download=args.force_download,
        no_download=args.no_download,
    )
    cv_results, selection, test = run_benchmark(config)
    write_outputs(cv_results=cv_results, selection=selection, test=test, config=config)


if __name__ == "__main__":
    main()
