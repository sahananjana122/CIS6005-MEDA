"""Rebuild the MEDA sample, temporal evaluation, deployment bundle and figures.

Run from the project root with the project virtual environment:
    .venv\\Scripts\\python.exe scripts\\rebuild_project.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.meda_modeling import (  # noqa: E402
    PhysicsResidualRegressor,
    regression_metrics,
    rolling_origin_folds,
    streaming_sol_stratified_sample,
    temporal_split_indices,
)
from src.meda_pipeline import (  # noqa: E402
    CORE_REQUIRED_COLUMNS,
    ID_COLUMN,
    TARGET,
    build_features,
    predict_frame,
)

SEED = 6005
ROWS_PER_SOL = 3_000
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
for folder in (PROCESSED, OUTPUTS, FIGURES, MODELS, SUBMISSIONS):
    folder.mkdir(parents=True, exist_ok=True)


def bounded(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def cv_mse(model_factory, X: pd.DataFrame, y: pd.Series, folds) -> float:
    scores = []
    for train_indices, valid_indices in folds:
        train_indices = bounded(train_indices, 120_000)
        model = model_factory()
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        prediction = model.predict(X.iloc[valid_indices])
        scores.append(float(np.mean((y.iloc[valid_indices].to_numpy() - prediction) ** 2)))
    return float(np.mean(scores))


def build_eda_figures(raw: pd.DataFrame, features: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(8, 4.5))
    sns.histplot(raw[TARGET], bins=60, kde=True, color="#2f6f8f")
    plt.title("Atmospheric pressure distribution across the stratified sample")
    plt.xlabel("Pressure (Pa)")
    plt.tight_layout(); plt.savefig(FIGURES / "eda_pressure_distribution.png", dpi=180); plt.close()

    missing = raw.isna().mean().mul(100).sort_values(ascending=False).head(15).sort_values()
    plt.figure(figsize=(8, 5.2)); missing.plot.barh(color="#b7653d")
    plt.title("Most incomplete MEDA fields"); plt.xlabel("Missing values (%)")
    plt.tight_layout(); plt.savefig(FIGURES / "eda_missingness.png", dpi=180); plt.close()

    by_sol = raw.groupby("sol", as_index=False)[TARGET].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 4.5)); plt.plot(by_sol["sol"], by_sol["mean"], color="#2f6f8f")
    plt.fill_between(by_sol["sol"], by_sol["mean"] - by_sol["std"], by_sol["mean"] + by_sol["std"], alpha=.18)
    plt.title("Seasonal pressure movement by mission sol"); plt.xlabel("Mission sol"); plt.ylabel("Pressure (Pa)")
    plt.tight_layout(); plt.savefig(FIGURES / "eda_pressure_by_sol.png", dpi=180); plt.close()

    lmst_hour = raw["LMST"].str.extract(r"M(\d{1,2}):")[0].astype(float)
    daily = pd.DataFrame({"hour": lmst_hour, TARGET: raw[TARGET]}).groupby("hour")[TARGET].mean()
    plt.figure(figsize=(8, 4.5)); daily.plot(marker="o", color="#6d4a8e")
    plt.title("Mean pressure by local mean solar hour"); plt.xlabel("LMST hour"); plt.ylabel("Pressure (Pa)")
    plt.tight_layout(); plt.savefig(FIGURES / "eda_pressure_by_lmst.png", dpi=180); plt.close()

    numeric = features.copy()
    numeric[TARGET] = raw[TARGET].to_numpy()
    correlations = numeric.corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(8, 6)); sns.heatmap(numeric[list(correlations)].corr(), cmap="vlag", center=0, square=False)
    plt.title("Correlation among pressure and strongest numerical associates")
    plt.tight_layout(); plt.savefig(FIGURES / "eda_correlation.png", dpi=180); plt.close()


def architecture_figures() -> None:
    def diagram(path: Path, title: str, labels: list[str], arrows: list[tuple[int, int]]):
        fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        xs = np.linspace(.08, .92, len(labels))
        for x, label in zip(xs, labels):
            ax.text(x, .5, label, ha="center", va="center", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=.55", facecolor="#244a64", edgecolor="#173346"))
        for start, end in arrows:
            ax.annotate("", xy=(xs[end]-.055, .5), xytext=(xs[start]+.055, .5),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#b7653d"))
        ax.set_title(title, fontsize=15, color="#173346", pad=18)
        fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    diagram(FIGURES / "architecture_training.png", "Training, evaluation and artifact pipeline",
            ["Raw Parquet", "Sol-stratified\nsampling", "Shared feature\nbuilder", "Temporal CV\nand tuning", "Untouched\ntest", "Versioned\nbundle"],
            [(0,1),(1,2),(2,3),(3,4),(4,5)])
    diagram(FIGURES / "architecture_inference.png", "Streamlit inference and batch-prediction pipeline",
            ["Form or file", "Schema\nvalidation", "Shared feature\nbuilder", "Selected\nmodel", "Pressure +\nwarnings", "Download /\naudit"],
            [(0,1),(1,2),(2,3),(3,4),(4,5)])


def main() -> None:
    started_all = time.perf_counter()
    sample_path = PROCESSED / "stratified_training_sample.parquet"
    if sample_path.exists():
        print("Reusing complete-coverage stratified sample...")
        raw = pd.read_parquet(sample_path)
    else:
        print("Streaming the complete training Parquet file...")
        raw = streaming_sol_stratified_sample(RAW / "train.parquet", rows_per_sol=ROWS_PER_SOL, seed=SEED)
        raw.to_parquet(sample_path, index=False)
    coverage = raw.groupby("sol").size().rename("rows").reset_index()
    coverage.to_csv(OUTPUTS / "sample_coverage_by_sol.csv", index=False)
    features = build_features(raw)
    target = pd.to_numeric(raw[TARGET], errors="coerce").astype("float32")
    features.to_parquet(PROCESSED / "features_temporal.parquet", index=False)
    pd.DataFrame({TARGET: target, "sol": raw["sol"]}).to_parquet(PROCESSED / "target_temporal.parquet", index=False)
    (PROCESSED / "feature_columns_v2.json").write_text(json.dumps(features.columns.tolist(), indent=2), encoding="utf-8")

    split = temporal_split_indices(raw["sol"])
    np.savez_compressed(PROCESSED / "temporal_indices_v2.npz", **split)
    folds = rolling_origin_folds(raw["sol"])
    assert all(set(train).isdisjoint(valid) for train, valid in folds)
    split_summary = {name: {"rows": int(len(idx)), "sol_min": int(raw.iloc[idx]["sol"].min()),
                            "sol_max": int(raw.iloc[idx]["sol"].max())} for name, idx in split.items()}
    (OUTPUTS / "temporal_split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ridge_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    ridge_study.optimize(lambda trial: cv_mse(
        lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1e4, log=True))),
        features, target, folds), n_trials=8)
    ridge_study.trials_dataframe().to_csv(OUTPUTS / "ridge_trials_v2.csv", index=False)

    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", .02, .12, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 12),
            "subsample": trial.suggest_float("subsample", .65, .9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .6, .9),
            "tree_method": "hist", "n_jobs": 3, "objective": "reg:squarederror", "random_state": SEED,
        }
        return cv_mse(lambda: XGBRegressor(**params), features, target, folds)
    xgb_study.optimize(xgb_objective, n_trials=6)
    xgb_study.trials_dataframe().to_csv(OUTPUTS / "xgboost_trials_v2.csv", index=False)
    (OUTPUTS / "best_params_v2.json").write_text(json.dumps({"ridge": ridge_study.best_params,
        "xgboost": xgb_study.best_params}, indent=2), encoding="utf-8")

    train_idx, valid_idx, test_idx = split["train"], split["validation"], split["test"]
    train_limited = bounded(train_idx, 210_000)
    models = {
        "median_baseline": DummyRegressor(strategy="median"),
        "harmonic_ridge": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                         Ridge(**ridge_study.best_params)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=140, max_depth=16, min_samples_leaf=8, max_features=.7, max_samples=.7,
            n_jobs=3, random_state=SEED)),
        "xgboost": XGBRegressor(**xgb_study.best_params, tree_method="hist", n_jobs=3,
                                objective="reg:squarederror", random_state=SEED),
        "catboost": CatBoostRegressor(iterations=450, depth=7, learning_rate=.05, loss_function="RMSE",
                                      task_type="CPU", thread_count=3, random_seed=SEED, verbose=False,
                                      allow_writing_files=False),
        "mlp": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=70, early_stopping=True,
                                          batch_size=512, random_state=SEED)),
        "physics_residual": PhysicsResidualRegressor(alpha=ridge_study.best_params["alpha"],
                                                       residual_weight=.35, xgb_params=xgb_study.best_params),
    }
    records, fitted_models = [], {}
    for name, model in models.items():
        local_train = bounded(train_limited, 120_000) if name in {"random_forest", "mlp"} else train_limited
        begin = time.perf_counter(); model.fit(features.iloc[local_train], target.iloc[local_train]); fit_seconds = time.perf_counter()-begin
        fitted_models[name] = model; joblib.dump(model, MODELS / f"{name}_v2.joblib")
        begin = time.perf_counter(); prediction = np.asarray(model.predict(features.iloc[valid_idx])); infer = time.perf_counter()-begin
        records.append({"model": name, "split": "validation", "train_rows": len(local_train),
                        "valid_rows": len(valid_idx), **regression_metrics(target.iloc[valid_idx], prediction),
                        "fit_seconds": fit_seconds, "inference_seconds": infer})
        print(name, records[-1]["mse"])
    comparison = pd.DataFrame(records)
    comparison.to_csv(OUTPUTS / "model_comparison_v2.csv", index=False)
    selected_name = comparison.loc[comparison["split"].eq("validation")].sort_values("mse").iloc[0]["model"]

    final_train = np.concatenate([train_idx, valid_idx])
    selected_model = models[selected_name]
    begin = time.perf_counter(); selected_model.fit(features.iloc[final_train], target.iloc[final_train]); final_fit = time.perf_counter()-begin
    begin = time.perf_counter(); final_test_prediction = np.asarray(selected_model.predict(features.iloc[test_idx])); final_infer = time.perf_counter()-begin
    final_metrics = regression_metrics(target.iloc[test_idx], final_test_prediction)
    final_metrics.update({"fit_seconds": final_fit, "inference_seconds": final_infer,
                          "test_rows": int(len(test_idx)), "test_sol_min": int(raw.iloc[test_idx]["sol"].min()),
                          "test_sol_max": int(raw.iloc[test_idx]["sol"].max())})
    comparison = pd.concat([comparison, pd.DataFrame([{ 
        "model": selected_name, "split": "final_test", "train_rows": len(final_train),
        "valid_rows": len(test_idx), **final_metrics,
    }])], ignore_index=True)
    comparison.to_csv(OUTPUTS / "model_comparison_v2.csv", index=False)
    ranges = {column: {"min": float(features[column].min()) if features[column].notna().any() else None,
                       "median": float(features[column].median()) if features[column].notna().any() else None,
                       "max": float(features[column].max()) if features[column].notna().any() else None}
              for column in features.columns}
    metadata = {
        "bundle_version": "2.0", "selected_model": selected_name, "target": TARGET, "unit": "Pa",
        "selection_rule": "lowest validation MSE after rolling-origin tuning; test remained untouched",
        "seed": SEED, "sample_rows": int(len(raw)), "sample_sol_min": int(raw["sol"].min()),
        "sample_sol_max": int(raw["sol"].max()), "rows_per_sol": ROWS_PER_SOL,
        "temporal_splits": split_summary, "final_test_metrics": final_metrics,
        "prediction_bounds_pa": [500.0, 900.0],
        "prediction_bound_rationale": "Broad physical/deployment guard; MEDA literature reports approximately 600-800 Pa, not hidden test labels.",
        "limitations": ["Training covers only competition-provided early mission sols.",
                        "Predictions outside observed sensor and seasonal ranges require caution.",
                        "This educational virtual sensor is not flight-qualified or safety-critical."],
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
    }
    bundle = {"model": selected_model, "feature_columns": features.columns.tolist(), "metadata": metadata,
              "feature_ranges": ranges, "required_raw_columns": list(CORE_REQUIRED_COLUMNS)}
    joblib.dump(bundle, MODELS / "meda_pressure_bundle_v2.joblib")
    joblib.dump(selected_model, MODELS / "selected_model_v2.joblib")
    (MODELS / "metadata_v2.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    residual = target.iloc[test_idx].to_numpy() - final_test_prediction
    plt.figure(figsize=(8, 4.5)); plt.scatter(final_test_prediction, residual, s=6, alpha=.25)
    plt.axhline(0, color="black", lw=1); plt.xlabel("Predicted pressure (Pa)"); plt.ylabel("Residual (Pa)")
    plt.title(f"Untouched temporal-test residuals: {selected_name}")
    plt.tight_layout(); plt.savefig(FIGURES / "evaluation_residuals.png", dpi=180); plt.close()
    test_errors = pd.DataFrame({"sol": raw.iloc[test_idx]["sol"].to_numpy(), "squared_error": residual**2})
    by_sol = test_errors.groupby("sol")["squared_error"].mean()
    plt.figure(figsize=(8,4.5)); by_sol.plot(marker="o", color="#b7653d"); plt.ylabel("MSE"); plt.title("Selected-model error by held-out sol")
    plt.tight_layout(); plt.savefig(FIGURES / "evaluation_error_by_sol.png", dpi=180); plt.close()

    build_eda_figures(raw, features); architecture_figures()
    print("Selected:", selected_name, final_metrics)
    print("Total seconds:", time.perf_counter() - started_all)


if __name__ == "__main__":
    main()
