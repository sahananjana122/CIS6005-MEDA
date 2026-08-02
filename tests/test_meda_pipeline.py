import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.meda_modeling import temporal_split_indices
from src.meda_pipeline import build_features, parse_clock, predict_frame, validate_raw_schema


def raw_frame():
    return pd.DataFrame({
        "LMST": ["00015M12:30:00.000", "00016M06:00:00.000"],
        "LTST": ["0015 12:00:00", "0016 05:45:00"],
        "SOLAR_LONGITUDE_ANGLE": [25.0, 27.0],
        "SOLAR_ZENITHAL_ANGLE": [30.0, 110.0],
        "ROVER_YAW": [10.0, -175.0],
        "sol": [15, 16],
        "DOWNWARD_LW_IRRADIANCE": [31.2, np.nan],
    })


def test_clock_and_harmonic_features_are_finite():
    assert parse_clock("0015 12:30:00") == pytest.approx(12.5)
    features = build_features(raw_frame())
    assert "LMST_sin_h3" in features
    assert "SOLAR_LONGITUDE_ANGLE_cos" in features
    assert np.isfinite(features["LMST_sin_h1"]).all()


def test_malformed_clock_is_reported():
    frame = raw_frame()
    frame.loc[0, "LMST"] = "bad-clock"
    assert any("malformed" in problem for problem in validate_raw_schema(frame))


def test_feature_contract_reorders_exactly():
    features = build_features(raw_frame())
    order = list(reversed(features.columns.tolist()))
    rebuilt = build_features(raw_frame(), order)
    assert rebuilt.columns.tolist() == order


def test_temporal_partitions_are_disjoint():
    split = temporal_split_indices(pd.Series(np.repeat(np.arange(1, 101), 2)))
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_bundle_prediction_and_missing_required_column():
    frame = raw_frame()
    features = build_features(frame)
    model = DummyRegressor(strategy="mean").fit(features, [730.0, 740.0])
    bundle = {"model": model, "feature_columns": features.columns.tolist(), "metadata": {}}
    prediction = predict_frame(bundle, frame)
    assert prediction.tolist() == pytest.approx([735.0, 735.0])
    with pytest.raises(ValueError, match="Missing required"):
        predict_frame(bundle, frame.drop(columns="LTST"))
