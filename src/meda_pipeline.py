"""Reproducible preprocessing and inference for the MEDA pressure virtual sensor.

The functions in this module are deliberately shared by training, notebooks,
submission generation and Streamlit. This prevents notebook/app feature drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

TARGET = "PRESSURE"
ID_COLUMN = "row_id"
MARS_YEAR_SOLS = 668.6
CLOCK_COLUMNS = ("LMST", "LTST")
ANGLE_COLUMNS = (
    "SOLAR_LONGITUDE_ANGLE",
    "ROVER_PITCH",
    "ROVER_YAW",
    "ROVER_ROLL",
)
CORE_REQUIRED_COLUMNS = (
    "LMST",
    "LTST",
    "SOLAR_LONGITUDE_ANGLE",
    "SOLAR_ZENITHAL_ANGLE",
    "sol",
)


def parse_clock(value: Any) -> float:
    """Convert MEDA LMST/LTST strings to decimal hours."""
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"(?:M|\s)(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)", str(value))
    if not match:
        return np.nan
    hour, minute, second = map(float, match.groups())
    if hour >= 24 or minute >= 60 or second >= 60:
        return np.nan
    return hour + minute / 60.0 + second / 3600.0


def validate_raw_schema(
    frame: pd.DataFrame,
    required: Iterable[str] = CORE_REQUIRED_COLUMNS,
) -> list[str]:
    """Return human-readable schema problems; an empty list means valid."""
    problems: list[str] = []
    missing = [column for column in required if column not in frame.columns]
    if missing:
        problems.append(f"Missing required columns: {', '.join(missing)}")
    if TARGET in frame.columns:
        problems.append(f"Input contains target column {TARGET}; it will not be used as a feature.")
    for clock in CLOCK_COLUMNS:
        if clock in frame.columns:
            parsed = frame[clock].map(parse_clock)
            bad = int(parsed.isna().sum() - frame[clock].isna().sum())
            if bad:
                problems.append(f"{clock} contains {bad} malformed non-empty value(s).")
    return problems


def _float32_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.select_dtypes(include=["number", "bool"]).copy()
    for column in output.columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output


def build_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build numerical, missing-aware, cyclic features from raw MEDA records."""
    raw = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    original_missing = raw.isna().sum(axis=1).astype("float32")

    clock_hours: dict[str, pd.Series] = {}
    for clock in CLOCK_COLUMNS:
        if clock in raw.columns:
            clock_hours[clock] = raw.pop(clock).map(parse_clock).astype("float32")

    output = _float32_numeric(raw)
    output["missing_count"] = original_missing

    for clock, hours in clock_hours.items():
        for harmonic in (1, 2, 3):
            angle = 2.0 * np.pi * harmonic * hours / 24.0
            output[f"{clock}_sin_h{harmonic}"] = np.sin(angle).astype("float32")
            output[f"{clock}_cos_h{harmonic}"] = np.cos(angle).astype("float32")

    if "sol" in output.columns:
        sol_angle = 2.0 * np.pi * output["sol"] / MARS_YEAR_SOLS
        for harmonic in (1, 2):
            output[f"mars_sol_sin_h{harmonic}"] = np.sin(harmonic * sol_angle).astype("float32")
            output[f"mars_sol_cos_h{harmonic}"] = np.cos(harmonic * sol_angle).astype("float32")

    for column in ANGLE_COLUMNS:
        if column in output.columns:
            radians = np.deg2rad(output[column])
            output[f"{column}_sin"] = np.sin(radians).astype("float32")
            output[f"{column}_cos"] = np.cos(radians).astype("float32")

    if feature_columns is not None:
        output = output.reindex(columns=feature_columns)
    return output


def save_feature_contract(path: Path, feature_columns: list[str]) -> None:
    path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    required = {"model", "feature_columns", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid deployment bundle; expected keys {sorted(required)}")
    return bundle


def predict_frame(bundle: dict[str, Any], raw_frame: pd.DataFrame) -> np.ndarray:
    problems = [p for p in validate_raw_schema(raw_frame) if not p.startswith("Input contains target")]
    if problems:
        raise ValueError(" ".join(problems))
    features = build_features(raw_frame, list(bundle["feature_columns"]))
    prediction = np.asarray(bundle["model"].predict(features), dtype="float64")
    bounds = bundle.get("metadata", {}).get("prediction_bounds_pa")
    if bounds is not None:
        prediction = np.clip(prediction, float(bounds[0]), float(bounds[1]))
    if prediction.ndim != 1 or len(prediction) != len(raw_frame) or not np.isfinite(prediction).all():
        raise ValueError("Model returned invalid or non-finite predictions.")
    return prediction
