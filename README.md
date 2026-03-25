# Chronic Disease Risk Modeling

NHANES-first research pipeline for chronic disease risk feature engineering, model training, evaluation, and later web artifact export.

## Setup

1. `python -m pip install -r requirements.txt`
2. `python -m pytest tests/config tests/features tests/data_sources tests/preprocessing tests/modeling tests/evaluation -q`

## Planned Workflow

1. `python scripts/download_nhanes.py`
2. `python scripts/build_nhanes_interim.py`
3. `python scripts/build_model_dataset.py`
4. `python scripts/train_baseline.py`
5. `python scripts/train_candidates.py`
6. `python scripts/export_artifacts.py`
7. `python scripts/run_nhanes_pipeline.py`

## Outputs

- `reports/tables/`: metrics CSV output
- `reports/figures/`: plots and explanation figures
- `artifacts/`: web-facing JSON payloads
