"""Generate a traceable Kaggle submission from the versioned MEDA bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.meda_pipeline import ID_COLUMN, TARGET, load_bundle, predict_frame  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    raw = ROOT / "data" / "raw"
    output_path = ROOT / "submissions" / "submission_04_temporal_physics_residual.csv"
    bundle = load_bundle(ROOT / "models" / "meda_pressure_bundle_v2.joblib")
    sample = pd.read_csv(raw / "sample_submission.csv")
    predictions = []
    parquet = pq.ParquetFile(raw / "test.parquet")
    for batch in parquet.iter_batches(batch_size=100_000):
        predictions.append(predict_frame(bundle, batch.to_pandas()))
    prediction = np.concatenate(predictions)
    if len(prediction) != len(sample):
        raise ValueError(f"Prediction count {len(prediction)} != sample count {len(sample)}")
    submission = pd.DataFrame({ID_COLUMN: sample[ID_COLUMN], TARGET: prediction})
    if not np.isfinite(submission[TARGET]).all():
        raise ValueError("Submission contains non-finite predictions")
    submission.to_csv(output_path, index=False)
    evidence_path = ROOT / "outputs" / "kaggle_evidence.csv"
    evidence = pd.read_csv(evidence_path)
    row = {
        "file": output_path.name,
        "model": bundle["metadata"]["selected_model"],
        "local_mse": bundle["metadata"]["final_test_metrics"]["mse"],
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "sha256": sha256(output_path),
        "kaggle_description": "Temporal physics-residual model; selected without leaderboard optimisation",
        "public_score": np.nan,
        "final_selected": False,
        "evidence": "Awaiting student upload/selection",
    }
    evidence = pd.concat([evidence[evidence.file.ne(output_path.name)], pd.DataFrame([row])], ignore_index=True)
    evidence.to_csv(evidence_path, index=False)
    print(output_path, len(submission), row["sha256"], submission[TARGET].describe().to_dict())


if __name__ == "__main__":
    main()
