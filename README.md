# MEDA Pressure Virtual Sensor - CIS6005

This project recovers atmospheric pressure from Mars Environmental Dynamics Analyzer (MEDA) observations for the Kaggle MEDA Virtual Sensor Recovery competition. It is designed for an 8 GB, CPU-only laptop and now includes temporally honest evaluation, a versioned model artifact, a Streamlit application, completed notebook discussions, Kaggle evidence and a cited report.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name meda-low-memory --display-name "MEDA Low Memory"
```

Download the official Kaggle files into `data/raw/`, then run the rebuild and tests:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_project.py
.\.venv\Scripts\python.exe -m pytest -q
```

Launch the application:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The app supports a guided single MEDA observation and batch CSV/Parquet prediction. An example batch is provided at `data/example_meda_input.csv`.

## Resource policy

- Use three CPU threads so Windows remains responsive.
- Start with bounded samples; increase only after checking RAM.
- Close other applications during tree training.
- Restart the kernel between large models when necessary.
- Report all sample sizes and hardware limitations honestly.

## Reproducibility and academic integrity

All metrics and figures are generated from the supplied competition data. Kaggle scores are transcribed only from the student's screenshot and are tied to SHA-256 hashes. The report discloses the earlier biased sampling result and the disagreement between local and public evaluation. The student must still make and evidence the final/private Kaggle selection personally.
