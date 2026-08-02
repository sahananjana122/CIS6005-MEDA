"""Run cross-artifact acceptance checks and save a machine-readable summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.meda_pipeline import build_features, predict_frame

results: dict[str, object] = {}

full_sols = set(pq.read_table(ROOT / "data/raw/train.parquet", columns=["sol"])["sol"].to_pylist())
sample = pd.read_parquet(ROOT / "data/processed/stratified_training_sample.parquet")
sample_sols = set(sample["sol"].dropna().tolist())
assert sample_sols == full_sols
results["sample_sol_coverage"] = {"available": len(full_sols), "sampled": len(sample_sols)}

indices = np.load(ROOT / "data/processed/temporal_indices_v2.npz")
sol_sets = {name: set(sample.iloc[indices[name]]["sol"].tolist()) for name in indices.files}
assert sol_sets["train"].isdisjoint(sol_sets["validation"])
assert sol_sets["train"].isdisjoint(sol_sets["test"])
assert sol_sets["validation"].isdisjoint(sol_sets["test"])
results["temporal_sol_ranges"] = {
    name: [int(min(values)), int(max(values))] for name, values in sol_sets.items()
}

ledger = pd.read_csv(ROOT / "outputs/model_comparison_v2.csv")
assert ledger["split"].eq("final_test").sum() == 1
assert ledger.loc[ledger["split"].eq("final_test"), "model"].iloc[0] == "physics_residual"
assert ledger["split"].isin(["validation", "final_test"]).all()
results["evaluation_ledger"] = ledger["split"].value_counts().to_dict()

bundle = joblib.load(ROOT / "models/meda_pressure_bundle_v2.joblib")
example = pd.read_csv(ROOT / "data/example_meda_input.csv")
single = predict_frame(bundle, example.iloc[[0]])
batch = predict_frame(bundle, example)
assert np.isfinite(batch).all() and np.isclose(single[0], batch[0])
assert build_features(example, bundle["feature_columns"]).columns.tolist() == bundle["feature_columns"]
results["prediction_parity"] = {"rows": len(batch), "finite": True, "first_pa": float(batch[0])}

submission = ROOT / "submissions/submission_04_temporal_physics_residual.csv"
digest = hashlib.sha256(submission.read_bytes()).hexdigest()
assert digest == "24aa771b8f9bfb6f77a9bed98638615dbed3ec2fc1a4c0e52f351e39669d79ad"
row_count = 0
for candidate, template in zip(
    pd.read_csv(submission, chunksize=250_000),
    pd.read_csv(ROOT / "data/raw/sample_submission.csv", usecols=["row_id"], chunksize=250_000),
):
    assert candidate.columns.tolist() == ["row_id", "PRESSURE"]
    assert candidate["row_id"].equals(template["row_id"])
    assert np.isfinite(candidate["PRESSURE"]).all()
    row_count += len(candidate)
assert row_count == 3_949_990
results["submission"] = {"rows": row_count, "sha256": digest, "finite": True, "order_matches": True}

notebook_results = {}
for text_path in [ROOT / "README.md", ROOT / "PROJECT_AUDIT.md", *sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
    if text_path.name == "acceptance_check.py":
        continue
    source = text_path.read_text(encoding="utf-8")
    assert not any(token in source for token in ("STUDENT RESPONSE", "TODO", "TBD", "complete after"))
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    assert not any(
        token in cell.source
        for cell in notebook.cells
        for token in ("STUDENT RESPONSE", "TODO", "TBD", "complete after")
    )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
    notebook_results[path.name] = len(code_cells)
results["executed_notebooks"] = notebook_results

report_count = json.loads((ROOT / "outputs/report_word_count.json").read_text(encoding="utf-8"))
assert int(report_count["counted_words_before_references"]) <= 4000
results["report"] = {"counted_words": report_count["counted_words_before_references"], "pdf_pages": 12}

out = ROOT / "outputs/acceptance_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
