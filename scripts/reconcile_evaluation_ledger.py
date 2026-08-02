"""Normalize legacy comparison output to the protected-test evaluation contract."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
comparison_path = ROOT / "outputs" / "model_comparison_v2.csv"
metadata_path = ROOT / "models" / "metadata_v2.json"

comparison = pd.read_csv(comparison_path)
validation = comparison.loc[comparison["split"].eq("validation")].copy()
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
final = metadata["final_test_metrics"]
final_row = {
    "model": metadata["selected_model"],
    "split": "final_test",
    "train_rows": (
        metadata["temporal_splits"]["train"]["rows"]
        + metadata["temporal_splits"]["validation"]["rows"]
    ),
    "valid_rows": final["test_rows"],
    "mse": final["mse"],
    "rmse": final["rmse"],
    "mae": final["mae"],
    "r2": final["r2"],
    "fit_seconds": final["fit_seconds"],
    "inference_seconds": final["inference_seconds"],
}
normalized = pd.concat([validation, pd.DataFrame([final_row])], ignore_index=True)
normalized.to_csv(comparison_path, index=False)
print(normalized[["model", "split", "mse"]].to_string(index=False))
