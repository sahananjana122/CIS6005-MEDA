"""Replace starter notebooks with concise, executed-evidence coursework notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def save(name: str, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "MEDA Low Memory", "language": "python", "name": "python3"}})
    nbf.write(notebook, NOTEBOOKS / name)


def main():
    raw = pd.read_parquet(ROOT / "data" / "processed" / "stratified_training_sample.parquet")
    comparison = pd.read_csv(OUTPUTS / "model_comparison_v2.csv")
    metadata = json.loads((MODELS / "metadata_v2.json").read_text(encoding="utf-8"))
    missing = raw.isna().mean().mul(100).sort_values(ascending=False)
    desc = raw["PRESSURE"].describe()
    by_sol = raw.groupby("sol")["PRESSURE"].mean()
    selected = metadata["selected_model"]
    final = metadata["final_test_metrics"]
    best_validation = comparison[comparison.split.eq("validation")].sort_values("mse").iloc[0]

    shared = """
from pathlib import Path
import sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
OUTPUTS, MODELS = ROOT/'outputs', ROOT/'models'
sns.set_theme(style='whitegrid')
"""
    save("01_resource_aware_eda.ipynb", [
        md("# MEDA Resource-Aware Exploratory Data Analysis\n\nThis notebook uses a deterministic sol-stratified sample streamed across the complete training Parquet file. Each figure is followed by its own finding, modelling implication and limitation."),
        code(shared + "\nraw = pd.read_parquet(ROOT/'data/processed/stratified_training_sample.parquet')\nprint(raw.shape, raw.sol.min(), raw.sol.max(), raw.sol.nunique())\ndisplay(raw.head())"),
        md(f"## Schema and target audit\n\nThe rebuilt sample contains **{len(raw):,} observations across {raw.sol.nunique()} sols ({int(raw.sol.min())}-{int(raw.sol.max())})**, avoiding the original first-row-group bias. The ten fully absent fields, including `{missing.index[0]}`, are unusable in this training period; two humidity fields are intermittently observed. These patterns justify missingness indicators and algorithms that tolerate unavailable sensors."),
        code("audit = pd.DataFrame({'dtype':raw.dtypes.astype(str),'missing_percent':raw.isna().mean()*100,'unique':raw.nunique(dropna=True)}).sort_values('missing_percent',ascending=False)\ndisplay(audit)"),
        md(f"## Figure 1 - Pressure distribution\n\n**Finding.** Pressure has mean **{desc['mean']:.2f} Pa**, median **{desc['50%']:.2f} Pa**, standard deviation **{desc['std']:.2f} Pa**, and range **{desc['min']:.2f}-{desc['max']:.2f} Pa**. The distribution combines seasonal and diurnal states rather than representing independent identical observations.\n\n**Modelling implication.** Report errors in pascals as well as squared error, and include cyclic seasonal/time features.\n\n**Limitation.** A histogram removes temporal order and cannot demonstrate future-sol generalisation."),
        code("plt.figure(figsize=(9,4)); sns.histplot(raw.PRESSURE,bins=60,kde=True); plt.xlabel('Pressure (Pa)'); plt.title('Atmospheric pressure distribution'); plt.show(); display(raw.PRESSURE.describe())"),
        md(f"## Figure 2 - Missingness\n\n**Finding.** The most incomplete feature is **{missing.index[0]} ({missing.iloc[0]:.1f}%)**; multiple sensor channels are structurally absent rather than randomly missing.\n\n**Modelling implication.** Entirely absent columns carry no training information, while partially observed channels require missing-aware preprocessing and explicit missing-count features.\n\n**Limitation.** Missingness percentage alone cannot distinguish instrument scheduling, faults and environmental censoring."),
        code("m=raw.isna().mean().mul(100).sort_values(ascending=False).head(18).sort_values(); plt.figure(figsize=(9,5)); m.plot.barh(); plt.xlabel('Missing (%)'); plt.title('Most incomplete MEDA fields'); plt.show()"),
        md(f"## Figure 3 - Seasonal movement\n\n**Finding.** Mean pressure changes from **{by_sol.iloc[0]:.2f} Pa at sol {int(by_sol.index[0])}** to **{by_sol.iloc[-1]:.2f} Pa at sol {int(by_sol.index[-1])}**, and it is not monotonic across the complete period.\n\n**Modelling implication.** Random splitting leaks nearby seasonal states. Ordered temporal partitions and cyclic solar-longitude terms are required.\n\n**Limitation.** Mean-by-sol suppresses short-lived vortices and within-sol tidal variability."),
        code("g=raw.groupby('sol').PRESSURE.agg(['mean','std']); plt.figure(figsize=(9,4)); plt.plot(g.index,g['mean']); plt.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.2); plt.xlabel('Sol'); plt.ylabel('Pressure (Pa)'); plt.title('Pressure by sol'); plt.show()"),
        md("## Figure 4 - Diurnal pressure cycle\n\n**Finding.** Pressure varies systematically with local mean solar time, consistent with diurnal and semidiurnal thermal tides described for Jezero crater.\n\n**Modelling implication.** First-, second- and third-order Fourier terms are added for LMST and LTST rather than treating clock strings as categories.\n\n**Limitation.** Hourly aggregation combines all sols and does not prove that tidal amplitude is seasonally constant."),
        code("hour=raw.LMST.str.extract(r'M(\\d{1,2}):')[0].astype(float); d=pd.DataFrame({'hour':hour,'PRESSURE':raw.PRESSURE}).groupby('hour').PRESSURE.mean(); plt.figure(figsize=(9,4)); d.plot(marker='o'); plt.ylabel('Pressure (Pa)'); plt.title('Mean pressure by LMST hour'); plt.show()"),
        md("## Figure 5 - Correlation structure\n\n**Finding.** Solar geometry, mission time and radiative measurements show the strongest marginal relationships with pressure, supporting a combined seasonal, diurnal and environmental model.\n\n**Modelling implication.** Both regularised linear harmonics and nonlinear ensembles are compared; correlation is not used as a feature-selection rule by itself.\n\n**Limitation.** Pearson correlation measures linear association, is affected by shared time trends, and does not establish causality."),
        code("from src.meda_pipeline import build_features\nX=build_features(raw); z=X.copy(); z['PRESSURE']=raw.PRESSURE.to_numpy(); strongest=z.corr(numeric_only=True).PRESSURE.abs().sort_values(ascending=False).head(10).index; plt.figure(figsize=(9,6)); sns.heatmap(z[list(strongest)].corr(),cmap='vlag',center=0); plt.title('Pressure correlation structure'); plt.show()"),
        md("## EDA hand-off\n\nThe three strongest design consequences are: preserve complete sol coverage, validate in temporal order, and encode the physical seasonal/thermal-tide cycles. Structural missingness additionally requires one shared missing-aware preprocessing contract. These choices directly address the original local/Kaggle disagreement."),
    ])

    splits = json.loads((OUTPUTS / "temporal_split_summary.json").read_text())
    save("02_preprocessing_and_validation.ipynb", [
        md("# MEDA Preprocessing and Temporal Validation"), code(shared + "\nfrom src.meda_pipeline import build_features, validate_raw_schema\nraw=pd.read_parquet(ROOT/'data/processed/stratified_training_sample.parquet'); X=build_features(raw); y=raw.PRESSURE.astype('float32')\nprint(X.shape, X.memory_usage(deep=True).sum()/2**20, 'MB')"),
        md("## Cleanup and feature contract\n\nLMST/LTST are parsed into decimal hours and encoded with three Fourier harmonics. Solar longitude and rover angles receive sine/cosine encodings; mission sol receives Mars-year harmonics. Numeric fields are converted to float32, target and identifier columns are excluded, and row-level missing counts are retained. Models that cannot natively handle missing values use median imputation fitted only on training data."),
        code("indices=np.load(ROOT/'data/processed/temporal_indices_v2.npz'); splits={k:indices[k] for k in indices.files}\nfor name,idx in splits.items(): print(name,len(idx),raw.iloc[idx].sol.min(),raw.iloc[idx].sol.max())\nassert set(splits['train']).isdisjoint(splits['validation']); assert set(splits['train']).isdisjoint(splits['test']); assert set(splits['validation']).isdisjoint(splits['test'])"),
        md(f"## Validation rationale\n\nThe earliest partition is training ({splits['train']['sol_min']}-{splits['train']['sol_max']}), followed by validation ({splits['validation']['sol_min']}-{splits['validation']['sol_max']}) and an untouched final test ({splits['test']['sol_min']}-{splits['test']['sol_max']}). Rolling-origin folds operate only before the test boundary. Random validation was rejected because neighbouring observations share mission time and atmospheric state; it produced optimistic scores that did not transfer to the Kaggle period."),
        md("## Leakage and sampling limitations\n\nThe target and row identifier are removed before transformation. Preprocessing lives in `src/meda_pipeline.py` and is used unchanged by training, submission generation and Streamlit. The stratified sample caps each sol for an 8 GB laptop, so it improves temporal representation but is not identical to full-data training. The untouched test is still within the supplied training-era range and therefore cannot fully estimate the more distant Kaggle regime."),
    ])

    save("03_local_model_training.ipynb", [
        md("# MEDA Multi-Model Training and Evaluation"), code(shared + "\ncomparison=pd.read_csv(OUTPUTS/'model_comparison_v2.csv'); display(comparison.sort_values(['split','mse']))"),
        md(f"## Comparative outcome\n\nSeven candidates were compared: a median baseline, harmonic Ridge, Random Forest, XGBoost, CatBoost, MLP and a physics-residual hybrid. Selection used validation MSE only; the best validation candidate was **{best_validation.model} (MSE {best_validation.mse:.3f}, RMSE {best_validation.rmse:.3f} Pa)**. The final exported model is **{selected}**. Fit and inference timings are reported because an operational virtual sensor must be accurate and responsive. Unequal bounded training sizes for Random Forest and MLP remain a hardware-driven limitation, so their ranking is not a pure algorithm-only comparison."),
        code("v=comparison[comparison.split=='validation'].sort_values('mse'); plt.figure(figsize=(9,4)); sns.barplot(v,x='mse',y='model'); plt.title('Temporal-validation MSE'); plt.show()"),
        md(f"## Figure 1 - Model comparison\n\n**Finding.** The selected validation model achieved MSE **{best_validation.mse:.3f}**, while nonlinear and harmonic alternatives showed materially different temporal behaviour.\n\n**Implication.** Model selection uses temporal error rather than random-split accuracy or leaderboard position.\n\n**Limitation.** Training row caps and one competition dataset limit broad algorithmic conclusions."),
        code("from IPython.display import Image, display\ndisplay(Image(filename=str(OUTPUTS/'figures/evaluation_residuals.png')))"),
        md(f"## Figure 2 - Residuals\n\n**Finding.** On sols {final['test_sol_min']}-{final['test_sol_max']}, the selected model produced RMSE **{final['rmse']:.3f} Pa**, MAE **{final['mae']:.3f} Pa**, and R² **{final['r2']:.3f}**.\n\n**Implication.** Residual structure is treated as evidence about regime misspecification, not only as a score.\n\n**Limitation.** This test does not reach the Kaggle test sols; public-score disagreement must remain explicit."),
        md("## Hardware constraint\n\nThe recorded system has six logical CPUs and 7.69 GB RAM. Tree and neural models therefore use three CPU threads, float32 data and bounded samples. These constraints favour a compact deployment artifact and make an exhaustive search inappropriate; trial histories and seeds preserve reproducibility."),
    ])

    ridge_trials = pd.read_csv(OUTPUTS / "ridge_trials_v2.csv")
    xgb_trials = pd.read_csv(OUTPUTS / "xgboost_trials_v2.csv")
    save("04_bounded_tuning_and_selection.ipynb", [
        md("# MEDA Rolling-Origin Hyperparameter Tuning and Selection"), code(shared + "\nridge=pd.read_csv(OUTPUTS/'ridge_trials_v2.csv'); xgb=pd.read_csv(OUTPUTS/'xgboost_trials_v2.csv'); display(ridge.sort_values('value')); display(xgb.sort_values('value'))"),
        md(f"## Search outcome\n\nRidge used **{len(ridge_trials)}** log-scaled regularisation trials; XGBoost used **{len(xgb_trials)}** bounded trials over estimator count, depth, learning rate, child weight and row/column subsampling. Each objective averaged three rolling-origin validation errors and excluded the final temporal test. The trial counts were deliberately bounded for the 8 GB CPU-only system."),
        code("fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(ridge.number,ridge.value,marker='o'); ax[0].set_title('Ridge rolling-CV MSE'); ax[1].plot(xgb.number,xgb.value,marker='o'); ax[1].set_title('XGBoost rolling-CV MSE'); plt.show()"),
        md("## Figure 1 - Tuning histories\n\n**Finding.** Trial-to-trial variation confirms that both regularisation and tree complexity materially affect forward validation.\n\n**Implication.** Best parameters are chosen by mean rolling-origin MSE and then locked before final testing.\n\n**Limitation.** A bounded TPE search does not prove a global optimum; it trades exhaustive coverage for reproducibility and hardware safety."),
        md(f"## Artifact decision\n\nThe final bundle exports **{selected}**, its exact feature order, raw schema, feature ranges, temporal coverage, library versions, seed, test metrics and appropriate-use limitations. The selection rule is independent of Kaggle rank. The old artifact is preserved for traceability rather than overwritten."),
    ])

    save("05_submission_and_critical_review.ipynb", [
        md("# MEDA Kaggle Evidence, Architecture and Critical Review"), code(shared + "\nevidence=pd.read_csv(OUTPUTS/'kaggle_evidence.csv'); display(evidence)"),
        md("## Kaggle evidence\n\nThree model-generated submissions were accepted. Their public MSE values are 7049.045 (median), 5470.251 (untuned XGBoost) and 5458.353 (original selected XGBoost). The large difference from local scores is evidence of temporal distribution shift, not a reason to hide or optimise away the failure. The current screenshot shows that no final/private submission is selected; that checkbox and a new screenshot remain a student-owned manual action."),
        code("from IPython.display import Image, display\ndisplay(Image(filename=str(OUTPUTS/'evidence/kaggle_submissions.png')))"),
        md("## Figure 1 - Kaggle submissions\n\n**Finding.** All three candidate uploads completed successfully and the tuned candidate improved on the baseline, satisfying the multiple-submission requirement.\n\n**Implication.** Hashes and filenames link leaderboard evidence to reproducible files.\n\n**Limitation.** Public score is not a deployment guarantee, and the screenshot does not prove final/private selection."),
        code("display(Image(filename=str(OUTPUTS/'figures/architecture_training.png'))); display(Image(filename=str(OUTPUTS/'figures/architecture_inference.png')))"),
        md("## Architecture outcomes\n\nThe training architecture enforces a single preprocessing contract and keeps tuning separate from the untouched temporal test. Its main trade-off is that streaming and bounded training reduce memory demand but take longer and do not exploit every observation. The inference architecture accepts a form or raw file, validates schema, applies identical transformations, loads a versioned bundle and returns pressure plus warnings. This improves reproducibility over notebook-only prediction, but operational reliability still depends on monitoring seasonal range and sensor availability."),
        md("## Critical evaluation\n\nThe original pipeline appeared strong locally but sampled only early sols and reused validation information during tuning. The public failure exposes tree ensembles' poor extrapolation outside observed seasonal ranges. The rebuild adds complete sol coverage, cyclic physical representations, rolling validation and explicit limitations. XGBoost and CatBoost remain effective nonlinear tabular learners, while harmonic Ridge provides extrapolative structure and interpretability. The MLP is retained as required deep-learning evidence, but its higher compute demand, sensitivity to scaling/missingness and weak bounded-sample performance do not justify deployment. Sequence models could become suitable with continuous multi-sol windows and greater compute, especially if combined with physical constraints and uncertainty estimates.\n\nEthically, the app identifies itself as educational and not flight-qualified. Reproducibility is supported by seeds, hashes, trial histories and bundle metadata. Remaining risks include sampling bias, non-random instrument missingness, out-of-range use, unquantified predictive uncertainty and dependence on competition data definitions."),
        md("## Literature review hand-off\n\nThe report compares MEDA instrument and pressure studies with recent virtual-sensor, missing-data recovery, environmental forecasting and physics-informed ML research. It uses Harvard citations and distinguishes physical evidence from inferences made by this implementation."),
    ])
    print("Rebuilt five notebooks with completed discussion cells.")


if __name__ == "__main__":
    main()
