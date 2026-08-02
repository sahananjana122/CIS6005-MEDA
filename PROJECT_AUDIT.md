# MEDA CIS6005 Requirements Audit

## Current compliance after remediation

| Requirement | Status | Evidence |
|---|---|---|
| Competition eligibility | Confirmed by student | Official MEDA Kaggle competition link retained in report |
| EDA and chart discussion | Remediated | Notebook 01 and report provide a finding, design implication and limitation for every figure |
| Data cleanup and preprocessing | Remediated | Shared `src/meda_pipeline.py` contract and Notebook 02 |
| Multiple models | Met | Median, Ridge, Random Forest, XGBoost, CatBoost, MLP and physics-residual model |
| Hyperparameter tuning | Remediated | Rolling-origin Optuna histories for Ridge and XGBoost |
| Model evaluation | Remediated | Disjoint temporal validation/test metrics, timings, residuals and error-by-sol analysis |
| Model artifact | Remediated | Versioned deployment bundle with schema, ranges, metadata and limitations |
| Working application | Remediated | Streamlit single and batch prediction application |
| Multiple Kaggle submissions | Met | Three successful public submissions and hashes in `outputs/kaggle_evidence.csv` |
| Final/private Kaggle selection | Student action required | Screenshot currently shows all final-selection checkboxes empty |
| Literature review and citations | Remediated | Report uses peer-reviewed MEDA, virtual-sensor and physics-informed ML studies |
| High-level architecture | Remediated | Training and inference diagrams; no class/use-case diagrams |
| Report <= 4,000 words | Remediated | Editable DOCX and submission PDF generated with automated word-count check |

## Important correction

The original resource-aware sampler read only the beginning of a single Parquet row group. Consequently, its 750,000-row sample represented approximately sols 1-32 instead of the entire training period. The original local future MSE of 9.016 therefore did not estimate the later Kaggle regime, consistent with the much larger public score of 5458.353. The rebuilt pipeline streams the complete Parquet file and priority-samples every sol before creating ordered temporal partitions.

## Manual completion checkpoint

Before the competition closes, select the intended final/private Kaggle submission and capture a new screenshot showing its checked selection. Update `final_selected` in `outputs/kaggle_evidence.csv`, add the screenshot to the report appendix, and retain the selection timestamp.
