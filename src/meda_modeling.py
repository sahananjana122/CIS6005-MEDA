"""Resource-aware modelling utilities used by the MEDA rebuild script."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .meda_pipeline import TARGET


def streaming_sol_stratified_sample(
    path: str | Path,
    rows_per_sol: int = 3_000,
    batch_size: int = 100_000,
    seed: int = 6005,
) -> pd.DataFrame:
    """Priority-sample every sol while streaming a one-row-group Parquet file.

    The old notebook sampled only the beginning of the single row group. Here a
    deterministic hash priority retains up to ``rows_per_sol`` observations for
    every sol encountered across the entire file.
    """
    parquet = pq.ParquetFile(path)
    counts: dict[int, int] = {}
    for batch in parquet.iter_batches(columns=["sol", TARGET], batch_size=batch_size):
        sol_values = batch.column(0).to_numpy(zero_copy_only=False)
        targets = batch.column(1).to_numpy(zero_copy_only=False)
        valid = np.isfinite(sol_values) & np.isfinite(targets)
        unique, frequency = np.unique(sol_values[valid].astype("int64"), return_counts=True)
        for sol_key, amount in zip(unique, frequency):
            counts[int(sol_key)] = counts.get(int(sol_key), 0) + int(amount)
    if not counts:
        raise ValueError("No valid target rows were found in the Parquet file.")

    # One global seeded offset per sol and an integer stride select rows spread
    # across the complete sol without retaining or sorting all candidates.
    rng = np.random.default_rng(seed)
    stride = {key: max(1, counts[key] // rows_per_sol) for key in counts}
    offset = {key: int(rng.integers(0, stride[key])) for key in counts}
    seen = {key: 0 for key in counts}
    retained: list[pd.DataFrame] = []
    retained_count = {key: 0 for key in counts}
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        valid = frame[TARGET].notna() & frame["sol"].notna()
        frame = frame.loc[valid].copy()
        selections: list[np.ndarray] = []
        for sol_value, positions in frame.groupby("sol", sort=False).indices.items():
            key = int(sol_value)
            local = np.arange(len(positions), dtype="int64") + seen[key]
            choose = ((local - offset[key]) % stride[key] == 0)
            remaining = rows_per_sol - retained_count[key]
            chosen_positions = np.asarray(positions)[np.flatnonzero(choose)[:remaining]]
            if len(chosen_positions):
                selections.append(chosen_positions)
                retained_count[key] += len(chosen_positions)
            seen[key] += len(positions)
        if selections:
            retained.append(frame.iloc[np.concatenate(selections)])
    sampled = pd.concat(retained, ignore_index=True)
    return sampled.sort_values(["sol", "SCLK"]).reset_index(drop=True)


def temporal_split_indices(sol: pd.Series) -> dict[str, np.ndarray]:
    ordered = np.sort(pd.to_numeric(sol, errors="coerce").dropna().unique())
    if len(ordered) < 10:
        raise ValueError("At least ten unique sols are required for temporal evaluation.")
    train_end = max(1, int(np.floor(len(ordered) * 0.70)))
    valid_end = max(train_end + 1, int(np.floor(len(ordered) * 0.85)))
    train_sols, valid_sols, test_sols = ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]
    values = pd.to_numeric(sol, errors="coerce").to_numpy()
    result = {
        "train": np.flatnonzero(np.isin(values, train_sols)),
        "validation": np.flatnonzero(np.isin(values, valid_sols)),
        "test": np.flatnonzero(np.isin(values, test_sols)),
    }
    assert not set(result["train"]) & set(result["validation"])
    assert not set(result["train"]) & set(result["test"])
    assert not set(result["validation"]) & set(result["test"])
    return result


def rolling_origin_folds(sol: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = np.sort(pd.to_numeric(sol, errors="coerce").dropna().unique())
    usable = ordered[: max(3, int(np.floor(len(ordered) * 0.85)))]
    boundaries = ((0.40, 0.55), (0.55, 0.70), (0.70, 0.85))
    values = pd.to_numeric(sol, errors="coerce").to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_fraction, valid_fraction in boundaries:
        train_end = max(1, int(len(ordered) * train_fraction))
        valid_end = max(train_end + 1, int(len(ordered) * valid_fraction))
        train_sols = ordered[:train_end]
        valid_sols = ordered[train_end:valid_end]
        folds.append((np.flatnonzero(np.isin(values, train_sols)), np.flatnonzero(np.isin(values, valid_sols))))
    return folds


class PhysicsResidualRegressor(BaseEstimator, RegressorMixin):
    """Harmonic Ridge baseline plus a conservative XGBoost residual model."""

    def __init__(self, alpha: float = 10.0, residual_weight: float = 0.35, xgb_params: dict[str, Any] | None = None):
        self.alpha = alpha
        self.residual_weight = residual_weight
        self.xgb_params = xgb_params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Raw absolute mission time extrapolates linearly and produced physically
        # impossible pressure far beyond the labelled sol range. The transparent
        # baseline therefore uses cyclic time/angle terms and observed sensors,
        # while the bounded tree residual may still use the complete schema.
        excluded_linear = {"SCLK", "sol", "SOLAR_LONGITUDE_ANGLE"}
        self.base_columns_ = [
            column for column in X.columns
            if column not in excluded_linear and X[column].notna().any()
        ]
        self.base_ = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        )
        self.base_.fit(X[self.base_columns_], y)
        residual = np.asarray(y) - self.base_.predict(X[self.base_columns_])
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.04,
            "min_child_weight": 8,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "tree_method": "hist",
            "n_jobs": 3,
            "objective": "reg:squarederror",
            "random_state": 6005,
        }
        params.update(self.xgb_params or {})
        self.residual_ = XGBRegressor(**params).fit(X, residual)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_.predict(X[self.base_columns_]) + self.residual_weight * self.residual_.predict(X)


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, prediction))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def fit_and_measure(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame):
    started = time.perf_counter()
    fitted = clone(model).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(fitted.predict(X_eval))
    inference_seconds = time.perf_counter() - started
    return fitted, prediction, fit_seconds, inference_seconds
