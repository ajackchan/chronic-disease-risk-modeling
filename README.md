# Chronic Disease Risk Modeling

NHANES-first research pipeline for chronic disease risk feature engineering, model training, evaluation, and later web artifact export.

## Setup

1. `python -m pip install -r requirements.txt`
2. `python -m pytest tests/config/test_config.py -q`

## Planned Workflow

1. Download NHANES source data
2. Build interim merged tables
3. Construct processed modeling datasets
4. Train baseline and candidate models
5. Export reports and web-facing artifacts
