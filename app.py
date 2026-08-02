"""Streamlit application for the MEDA pressure virtual sensor."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.meda_pipeline import (
    CLOCK_COLUMNS,
    CORE_REQUIRED_COLUMNS,
    ID_COLUMN,
    TARGET,
    build_features,
    load_bundle,
    parse_clock,
    predict_frame,
    validate_raw_schema,
)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "meda_pressure_bundle_v2.joblib"

st.set_page_config(page_title="MEDA Pressure Virtual Sensor", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.main-title {font-size:2.35rem;font-weight:750;color:#173346;margin-bottom:.2rem}
.subtitle {color:#526675;margin-bottom:1.4rem}
.prediction-card {padding:1.4rem;border-radius:12px;background:linear-gradient(135deg,#173346,#2f6f8f);color:white;text-align:center}
.warning-card {padding:1rem;border-left:5px solid #b7653d;background:#fff6f0;border-radius:6px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Deployment bundle not found: {BUNDLE_PATH}")
    return load_bundle(BUNDLE_PATH)


def optional_number(label: str, value: float, help_text: str = "") -> float:
    enabled = st.checkbox(f"Provide {label}", value=True, key=f"use_{label}")
    if not enabled:
        return np.nan
    return st.number_input(label, value=float(value), help=help_text, key=f"value_{label}")


def single_record_form() -> pd.DataFrame | None:
    with st.form("single_prediction"):
        st.subheader("Mission time and solar geometry")
        c1, c2, c3 = st.columns(3)
        with c1:
            sol = st.number_input("Mission sol", min_value=0, value=50, step=1)
            lmst = st.text_input("LMST", "00050M12:00:00.000", help="Format: 00050M12:00:00.000")
            ltst = st.text_input("LTST", "0050 11:30:00", help="Format: 0050 11:30:00")
        with c2:
            solar_longitude = st.number_input("Solar longitude angle (degrees)", -180.0, 180.0, 45.0)
            solar_zenith = st.number_input("Solar zenithal angle (degrees)", 0.0, 180.0, 75.0)
            sclk = st.number_input("Spacecraft clock", min_value=0, value=670000000, step=1)
        with c3:
            transducer = st.selectbox("Pressure transducer", [1.0, 2.0])
            rover_yaw = st.number_input("Rover yaw (degrees)", -180.0, 180.0, 0.0)
            rover_pitch = st.number_input("Rover pitch (degrees)", -90.0, 90.0, 0.0)

        with st.expander("Optional rover and environmental sensor fields"):
            a, b, c = st.columns(3)
            with a:
                rover_x = optional_number("Rover position X", 0.0)
                rover_y = optional_number("Rover position Y", -30.0)
                rover_z = optional_number("Rover position Z", 1.0)
                rover_roll = optional_number("Rover roll", 0.0)
            with b:
                downward = optional_number("Downward LW irradiance", 28.0, "W/m²")
                upward = optional_number("Upward LW irradiance", 105.0, "W/m²")
                humidity_temp = optional_number("Humidity local temperature", 220.0, "K")
                rover_velocity = optional_number("Rover velocity", 0.0)
            with c:
                low_tilt = st.selectbox("Rover low tilt flag", [np.nan, 0.0, 1.0], format_func=lambda x: "Missing" if pd.isna(x) else str(int(x)))
                rover_still = st.selectbox("Rover still flag", [np.nan, 0.0, 1.0], format_func=lambda x: "Missing" if pd.isna(x) else str(int(x)))
                sun_outside = st.selectbox("Sun outside TIRS FOV", [np.nan, 0.0, 1.0], format_func=lambda x: "Missing" if pd.isna(x) else str(int(x)))
        submitted = st.form_submit_button("Predict pressure", type="primary", use_container_width=True)
    if not submitted:
        return None
    return pd.DataFrame([{
        "SCLK": sclk, "LMST": lmst, "LTST": ltst, "SOLAR_LONGITUDE_ANGLE": solar_longitude,
        "SOLAR_ZENITHAL_ANGLE": solar_zenith, "ROVER_POSITION_X": rover_x, "ROVER_POSITION_Y": rover_y,
        "ROVER_POSITION_Z": rover_z, "ROVER_VELOCITY": rover_velocity, "ROVER_PITCH": rover_pitch,
        "ROVER_YAW": rover_yaw, "ROVER_ROLL": rover_roll, "sol": sol, "TRANSDUCER": transducer,
        "HUMIDITY_LOCAL_TEMP": humidity_temp, "DOWNWARD_LW_IRRADIANCE": downward,
        "UPWARD_LW_IRRADIANCE": upward, "ROVER_LOW_TILT": low_tilt,
        "SUN_OUTSIDE_TIRS_FOV": sun_outside, "ROVER_STILL": rover_still,
    }])


def read_upload(uploaded) -> pd.DataFrame:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(uploaded.getvalue()))
    raise ValueError("Upload must be CSV or Parquet.")


def feature_range_warnings(model_bundle, frame: pd.DataFrame, limit: int = 8) -> list[str]:
    """Report finite model features outside the saved training envelope."""
    features = build_features(frame, list(model_bundle["feature_columns"]))
    ranges = model_bundle.get("feature_ranges", {})
    warnings = []
    for column in features.columns:
        limits = ranges.get(column, {})
        low, high = limits.get("min"), limits.get("max")
        values = features[column].dropna()
        if low is not None and high is not None and not values.empty:
            count = int(((values < low) | (values > high)).sum())
            if count:
                warnings.append(f"{column}: {count} value(s) outside training range [{low:.3g}, {high:.3g}]")
        if len(warnings) >= limit:
            break
    return warnings


def row_validation_errors(frame: pd.DataFrame) -> pd.Series:
    """Return row-aligned explanations for missing core values or malformed clocks."""
    errors = pd.Series("", index=frame.index, dtype="object")
    for column in CORE_REQUIRED_COLUMNS:
        if column in frame.columns:
            bad = frame[column].isna()
            errors.loc[bad] += f"missing {column}; "
    for column in CLOCK_COLUMNS:
        if column in frame.columns:
            bad = frame[column].notna() & frame[column].map(parse_clock).isna()
            errors.loc[bad] += f"malformed {column}; "
    return errors.str.rstrip("; ")


st.markdown('<div class="main-title">🛰️ MEDA Pressure Virtual Sensor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Recover atmospheric pressure from Perseverance rover mission and environmental measurements.</div>', unsafe_allow_html=True)

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(str(exc)); st.stop()

tab_single, tab_batch, tab_info = st.tabs(["Single prediction", "Batch prediction", "Model information"])
with tab_single:
    record = single_record_form()
    if record is not None:
        problems = validate_raw_schema(record)
        if problems:
            st.warning(" ".join(problems))
        try:
            range_messages = feature_range_warnings(bundle, record)
            if range_messages:
                st.warning("Outside the training envelope: " + " | ".join(range_messages))
            value = float(predict_frame(bundle, record)[0])
            st.markdown(f'<div class="prediction-card"><div>Predicted atmospheric pressure</div><h1>{value:,.2f} Pa</h1></div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Upload raw MEDA observations")
    uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet", "pq"])
    if uploaded:
        try:
            batch = read_upload(uploaded)
            st.caption(f"{len(batch):,} rows, {len(batch.columns)} columns")
            problems = validate_raw_schema(batch)
            blocking = [p for p in problems if p.startswith("Missing required columns")]
            if blocking:
                st.error(" ".join(blocking))
            else:
                row_errors = row_validation_errors(batch)
                valid = row_errors.eq("")
                if not valid.all():
                    error_table = pd.DataFrame({"row": batch.index[~valid], "error": row_errors[~valid]})
                    st.error(f"{len(error_table):,} row(s) were rejected. Correct them before operational use.")
                    st.dataframe(error_table.head(500), use_container_width=True)
                range_messages = feature_range_warnings(bundle, batch.loc[valid]) if valid.any() else []
                if range_messages:
                    st.warning("Outside the training envelope: " + " | ".join(range_messages))
                predictions = []
                valid_batch = batch.loc[valid]
                for start in range(0, len(valid_batch), 100_000):
                    predictions.append(predict_frame(bundle, valid_batch.iloc[start:start+100_000]))
                prediction = pd.Series(np.nan, index=batch.index, dtype="float64")
                if predictions:
                    prediction.loc[valid] = np.concatenate(predictions)
                output = pd.DataFrame({ID_COLUMN: batch[ID_COLUMN] if ID_COLUMN in batch else np.arange(len(batch)), TARGET: prediction})
                st.dataframe(output.head(100), use_container_width=True)
                st.line_chart(output[TARGET].head(5_000))
                st.download_button("Download predictions", output.to_csv(index=False).encode("utf-8"),
                                   file_name="meda_pressure_predictions.csv", mime="text/csv", type="primary")
        except Exception as exc:
            st.error(f"Could not process upload: {exc}")

with tab_info:
    metadata = bundle["metadata"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected model", metadata.get("selected_model", "Unknown"))
    final = metadata.get("final_test_metrics", {})
    m2.metric("Temporal-test RMSE", f"{final.get('rmse', float('nan')):.3f} Pa")
    m3.metric("Temporal-test R²", f"{final.get('r2', float('nan')):.3f}")
    st.json({"training_coverage": [metadata.get("sample_sol_min"), metadata.get("sample_sol_max")],
             "selection_rule": metadata.get("selection_rule"), "bundle_version": metadata.get("bundle_version")})
    st.markdown("#### Appropriate-use warning")
    st.markdown('<div class="warning-card">This is an educational virtual sensor, not a flight-qualified or safety-critical instrument. Predictions outside the training regime require domain review.</div>', unsafe_allow_html=True)
    st.markdown("#### Known limitations")
    for limitation in metadata.get("limitations", []):
        st.write(f"- {limitation}")
