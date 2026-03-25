# NHANES Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible NHANES-first chronic disease risk research pipeline that downloads source data, engineers GLM7-related features, trains baseline and machine learning models, and exports evaluation artifacts for later web integration.

**Architecture:** Use a Python `src/` package with small modules for config loading, data access, preprocessing, feature engineering, modeling, evaluation, and interpretability. Keep raw data immutable under `data/raw`, generate intermediate and processed datasets through scripts, and export model/report artifacts in stable formats for the later demo website.

**Tech Stack:** Python 3.11, pandas, numpy, PyYAML, requests, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, pytest

---

## Planned File Structure

- `pyproject.toml`: project metadata, package layout, pytest config
- `requirements.txt`: runtime and dev dependency list
- `README.md`: local setup, command flow, output directory guide
- `configs/paths.yaml`: directory and artifact paths
- `configs/nhanes.yaml`: cycle, table, and download registry settings
- `configs/modeling.yaml`: split rules, task list, metric names, model hyperparameter search space
- `configs/glm7.yaml`: GLM7 input field names, transform flags, optional score coefficients placeholder
- `configs/nhanes_variables.yaml`: per-table field mapping for each NHANES cycle
- `configs/label_rules.yaml`: initial disease and multimorbidity label rules
- `src/chronic_disease_risk/config.py`: YAML loading and repo-relative path resolution
- `src/chronic_disease_risk/data_sources/`: NHANES registry, downloader, XPT reader
- `src/chronic_disease_risk/preprocessing/`: merge, cohort filtering, label building, split logic
- `src/chronic_disease_risk/features/`: AIP, TyG, TyG-BMI, GLM7 builders
- `src/chronic_disease_risk/modeling/`: baseline models, candidate models, training orchestration
- `src/chronic_disease_risk/evaluation/`: metrics, plots, report writers
- `src/chronic_disease_risk/interpretability/`: SHAP generation and export
- `src/chronic_disease_risk/artifacts/`: website-facing JSON export
- `scripts/`: CLI entry points for download, build, train, explain, export
- `tests/`: unit, contract, and smoke tests

### Task 1: Bootstrap The Python Project And Config Loader

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `configs/paths.yaml`
- Create: `configs/nhanes.yaml`
- Create: `configs/modeling.yaml`
- Create: `src/chronic_disease_risk/__init__.py`
- Create: `src/chronic_disease_risk/config.py`
- Create: `tests/conftest.py`
- Create: `tests/config/test_config.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing config loader test**

```python
from pathlib import Path

from chronic_disease_risk.config import load_yaml_config, resolve_repo_path


def test_load_yaml_config_reads_repo_relative_file(tmp_path: Path):
    config_path = tmp_path / "paths.yaml"
    config_path.write_text("raw_dir: data/raw\n", encoding="utf-8")

    data = load_yaml_config(config_path)

    assert data["raw_dir"] == "data/raw"


def test_resolve_repo_path_returns_absolute_path(tmp_path: Path):
    resolved = resolve_repo_path(tmp_path, "data/raw")
    assert resolved == tmp_path / "data" / "raw"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/config/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing `load_yaml_config`

- [ ] **Step 3: Create the package skeleton and dependency manifest**

```toml
[project]
name = "chronic-disease-risk"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

```python
from pathlib import Path
import yaml


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_repo_path(repo_root: str | Path, relative_path: str) -> Path:
    return Path(repo_root).joinpath(relative_path).resolve()
```

- [ ] **Step 4: Add initial project configs and ignore rules**

```yaml
# configs/paths.yaml
raw_dir: data/raw
interim_dir: data/interim
processed_dir: data/processed
models_dir: models
reports_dir: reports
artifacts_dir: artifacts
```

```gitignore
artifacts/
reports/figures/
reports/tables/
.ruff_cache/
```

- [ ] **Step 5: Install dependencies and rerun the config tests**

Run: `python -m pip install -r requirements.txt`
Expected: dependency installation succeeds

Run: `python -m pytest tests/config/test_config.py -q`
Expected: PASS

- [ ] **Step 6: Commit the bootstrap task**

```bash
git add pyproject.toml requirements.txt README.md configs src tests .gitignore
git commit -m "feat: bootstrap research pipeline project"
```

### Task 2: Implement Formula Builders For AIP, TyG, TyG-BMI, And GLM7 Inputs

**Files:**
- Create: `configs/glm7.yaml`
- Create: `src/chronic_disease_risk/features/__init__.py`
- Create: `src/chronic_disease_risk/features/formulas.py`
- Create: `src/chronic_disease_risk/features/glm7.py`
- Create: `tests/features/test_formulas.py`
- Create: `tests/features/test_glm7.py`

- [ ] **Step 1: Write the failing formula tests**

```python
import math

import pytest

from chronic_disease_risk.features.formulas import compute_aip, compute_tyg, compute_tyg_bmi


def test_compute_aip_uses_log10_ratio():
    assert compute_aip(2.0, 1.0) == pytest.approx(math.log10(2.0))


def test_compute_tyg_matches_reference_formula():
    expected = math.log((150.0 * 100.0) / 2.0)
    assert compute_tyg(150.0, 100.0) == pytest.approx(expected)


def test_compute_tyg_bmi_multiplies_tyg_and_bmi():
    assert compute_tyg_bmi(8.0, 24.0) == pytest.approx(192.0)
```

- [ ] **Step 2: Run the formula tests to verify they fail**

Run: `python -m pytest tests/features/test_formulas.py -q`
Expected: FAIL with missing module or missing functions

- [ ] **Step 3: Implement the deterministic formula functions**

```python
import math


def compute_aip(triglycerides: float, hdl_c: float) -> float:
    return math.log10(triglycerides / hdl_c)


def compute_tyg(triglycerides_mg_dl: float, fasting_glucose_mg_dl: float) -> float:
    return math.log((triglycerides_mg_dl * fasting_glucose_mg_dl) / 2.0)


def compute_tyg_bmi(tyg: float, bmi: float) -> float:
    return tyg * bmi
```

- [ ] **Step 4: Write the failing GLM7 builder test with config-driven coefficients**

```python
from chronic_disease_risk.features.glm7 import GLM7Builder


def test_glm7_builder_returns_raw_and_score_columns():
    builder = GLM7Builder(weights={"age": 0.1, "insulin": 0.2}, intercept=-1.0)
    row = {"age": 50.0, "insulin": 10.0}

    result = builder.transform_row(row)

    assert "glm7_score" in result
    assert result["glm7_score"] > 0
```

- [ ] **Step 5: Implement a config-driven GLM7 builder without hardcoding literature coefficients**

```python
class GLM7Builder:
    def __init__(self, weights: dict[str, float], intercept: float = 0.0):
        self.weights = weights
        self.intercept = intercept

    def transform_row(self, row: dict[str, float]) -> dict[str, float]:
        linear = self.intercept + sum(row[name] * weight for name, weight in self.weights.items())
        return {**row, "glm7_score": linear}
```

- [ ] **Step 6: Run the feature tests and make them pass**

Run: `python -m pytest tests/features -q`
Expected: PASS

- [ ] **Step 7: Commit the feature builder task**

```bash
git add configs/glm7.yaml src/chronic_disease_risk/features tests/features
git commit -m "feat: add chronic disease feature formulas"
```

### Task 3: Build The NHANES Registry And Downloader

**Files:**
- Modify: `configs/nhanes.yaml`
- Create: `src/chronic_disease_risk/data_sources/__init__.py`
- Create: `src/chronic_disease_risk/data_sources/nhanes_registry.py`
- Create: `src/chronic_disease_risk/data_sources/nhanes_download.py`
- Create: `scripts/download_nhanes.py`
- Create: `tests/data_sources/test_nhanes_registry.py`
- Create: `tests/data_sources/test_nhanes_download.py`

- [ ] **Step 1: Write the failing registry test for cycle/table URL generation**

```python
from chronic_disease_risk.data_sources.nhanes_registry import build_xpt_url


def test_build_xpt_url_for_demographics_table():
    url = build_xpt_url(cycle_suffix="J", component="DEMO", table="P_DEMO")
    assert url.endswith("/P_DEMO_J.XPT")
```

- [ ] **Step 2: Run the registry test to verify it fails**

Run: `python -m pytest tests/data_sources/test_nhanes_registry.py -q`
Expected: FAIL with missing registry implementation

- [ ] **Step 3: Implement the NHANES registry and downloader modules**

```python
BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"


def build_xpt_url(cycle_suffix: str, component: str, table: str) -> str:
    return f"{BASE_URL}/{component}/{table}_{cycle_suffix}.XPT"
```

```python
def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination
```

- [ ] **Step 4: Add a skip-existing download behavior test and implementation**

```python
def test_download_file_skips_existing_file(tmp_path, monkeypatch):
    existing = tmp_path / "demo.xpt"
    existing.write_bytes(b"cached")

    result = download_file("https://example.test/demo.xpt", existing)

    assert result.read_bytes() == b"cached"
```

- [ ] **Step 5: Add the CLI entry point for configured downloads**

```python
if __name__ == "__main__":
    download_from_config(config_path="configs/nhanes.yaml")
```

- [ ] **Step 6: Run the downloader tests**

Run: `python -m pytest tests/data_sources/test_nhanes_registry.py tests/data_sources/test_nhanes_download.py -q`
Expected: PASS

- [ ] **Step 7: Commit the NHANES download task**

```bash
git add configs/nhanes.yaml src/chronic_disease_risk/data_sources scripts/download_nhanes.py tests/data_sources
git commit -m "feat: add nhanes registry and downloader"
```

### Task 4: Implement NHANES XPT Reading, Table Mapping, And Merge Logic

**Files:**
- Create: `configs/nhanes_variables.yaml`
- Create: `src/chronic_disease_risk/data_sources/xpt_reader.py`
- Create: `src/chronic_disease_risk/preprocessing/__init__.py`
- Create: `src/chronic_disease_risk/preprocessing/nhanes_merge.py`
- Create: `scripts/build_nhanes_interim.py`
- Create: `tests/data_sources/test_xpt_reader.py`
- Create: `tests/preprocessing/test_nhanes_merge.py`

- [ ] **Step 1: Write the failing XPT reader test**

```python
import pandas as pd

from chronic_disease_risk.data_sources.xpt_reader import normalize_columns


def test_normalize_columns_lowercases_and_strips_names():
    df = pd.DataFrame(columns=["SEQN", "RIDAGEYR "])
    normalized = normalize_columns(df)
    assert list(normalized.columns) == ["seqn", "ridageyr"]
```

- [ ] **Step 2: Run the reader and merge tests to verify they fail**

Run: `python -m pytest tests/data_sources/test_xpt_reader.py tests/preprocessing/test_nhanes_merge.py -q`
Expected: FAIL with missing reader/merge functions

- [ ] **Step 3: Implement the XPT reader utilities**

```python
def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in df.columns}
    return df.rename(columns=renamed)
```

- [ ] **Step 4: Implement table selection and respondent merge logic**

```python
def merge_nhanes_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for name, frame in tables.items():
        frame = frame.drop_duplicates(subset=["seqn"])
        merged = frame if merged is None else merged.merge(frame, on="seqn", how="inner")
    return merged
```

- [ ] **Step 5: Add variable mapping config for at least the first supported cycle**

```yaml
cycles:
  2017-2018:
    demo: [seqn, ridageyr, riagendr, ridreth3, dmdeduc2]
    lab: [seqn, lbxglu, lbxtr, lbdhdd, lbxins, lbdldl, bmxbmi]
    questionnaire: [seqn]
```

- [ ] **Step 6: Run the reader and merge tests**

Run: `python -m pytest tests/data_sources/test_xpt_reader.py tests/preprocessing/test_nhanes_merge.py -q`
Expected: PASS

- [ ] **Step 7: Commit the NHANES merge task**

```bash
git add configs/nhanes_variables.yaml src/chronic_disease_risk/data_sources/xpt_reader.py src/chronic_disease_risk/preprocessing/nhanes_merge.py scripts/build_nhanes_interim.py tests/data_sources tests/preprocessing
git commit -m "feat: add nhanes reading and merge pipeline"
```

### Task 5: Implement Cohort Filtering, Label Construction, And Time Splits

**Files:**
- Create: `configs/label_rules.yaml`
- Create: `src/chronic_disease_risk/preprocessing/cohort.py`
- Create: `src/chronic_disease_risk/preprocessing/labels.py`
- Create: `src/chronic_disease_risk/preprocessing/splits.py`
- Create: `scripts/build_model_dataset.py`
- Create: `tests/preprocessing/test_cohort.py`
- Create: `tests/preprocessing/test_labels.py`
- Create: `tests/preprocessing/test_splits.py`

- [ ] **Step 1: Write the failing cohort inclusion test**

```python
import pandas as pd

from chronic_disease_risk.preprocessing.cohort import apply_inclusion_rules


def test_apply_inclusion_rules_keeps_adults_with_required_labs():
    df = pd.DataFrame({
        "ridageyr": [19, 45],
        "lbxglu": [100.0, 90.0],
        "lbxtr": [150.0, 120.0],
    })

    filtered = apply_inclusion_rules(df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["ridageyr"] == 45
```

- [ ] **Step 2: Run the preprocessing tests to verify they fail**

Run: `python -m pytest tests/preprocessing/test_cohort.py tests/preprocessing/test_labels.py tests/preprocessing/test_splits.py -q`
Expected: FAIL with missing preprocessing functions

- [ ] **Step 3: Implement cohort inclusion and missing-value guards**

```python
REQUIRED_COLUMNS = ["ridageyr", "lbxglu", "lbxtr", "lbdhdd", "lbxins", "bmxbmi"]


def apply_inclusion_rules(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["ridageyr"].ge(20)
    mask &= df[REQUIRED_COLUMNS].notna().all(axis=1)
    return df.loc[mask].copy()
```

- [ ] **Step 4: Implement config-driven label builders and multimorbidity count**

```python
def add_multimorbidity_label(df: pd.DataFrame, outcome_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["multimorbidity_count"] = df[outcome_columns].sum(axis=1)
    df["multimorbidity_label"] = (df["multimorbidity_count"] >= 2).astype(int)
    return df
```

- [ ] **Step 5: Implement leakage-safe time split logic**

```python
def split_by_cycle(df: pd.DataFrame, train_cycles: set[str], test_cycles: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["cycle"].isin(train_cycles)].copy()
    test_df = df[df["cycle"].isin(test_cycles)].copy()
    return train_df, test_df
```

- [ ] **Step 6: Run all preprocessing tests**

Run: `python -m pytest tests/preprocessing -q`
Expected: PASS

- [ ] **Step 7: Commit the cohort and label task**

```bash
git add configs/label_rules.yaml src/chronic_disease_risk/preprocessing scripts/build_model_dataset.py tests/preprocessing
git commit -m "feat: add cohort filters and outcome labels"
```

### Task 6: Build The Baseline Training And Evaluation Pipeline

**Files:**
- Create: `src/chronic_disease_risk/modeling/__init__.py`
- Create: `src/chronic_disease_risk/modeling/baseline.py`
- Create: `src/chronic_disease_risk/modeling/pipeline.py`
- Create: `src/chronic_disease_risk/evaluation/__init__.py`
- Create: `src/chronic_disease_risk/evaluation/metrics.py`
- Create: `src/chronic_disease_risk/evaluation/reporting.py`
- Create: `scripts/train_baseline.py`
- Create: `tests/modeling/test_baseline.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write the failing baseline training test**

```python
import pandas as pd

from chronic_disease_risk.modeling.baseline import train_logistic_baseline


def test_train_logistic_baseline_returns_sklearn_pipeline():
    df = pd.DataFrame({
        "age": [40, 50, 60, 70],
        "glm7_score": [0.1, 0.3, 0.5, 0.7],
        "target": [0, 0, 1, 1],
    })

    model = train_logistic_baseline(df[["age", "glm7_score"]], df["target"])

    assert hasattr(model, "predict_proba")
```

- [ ] **Step 2: Run the baseline and metrics tests to verify they fail**

Run: `python -m pytest tests/modeling/test_baseline.py tests/evaluation/test_metrics.py -q`
Expected: FAIL with missing training/metric functions

- [ ] **Step 3: Implement the baseline pipeline and metric helpers**

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_logistic_baseline(X, y) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(X, y)
```

```python
def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
```

- [ ] **Step 4: Add report writers for metrics CSV and ROC plot output**

```python
def write_metrics_table(metrics: dict[str, float], destination: Path) -> None:
    pd.DataFrame([metrics]).to_csv(destination, index=False)
```

- [ ] **Step 5: Run the baseline and evaluation tests**

Run: `python -m pytest tests/modeling/test_baseline.py tests/evaluation/test_metrics.py -q`
Expected: PASS

- [ ] **Step 6: Commit the baseline modeling task**

```bash
git add src/chronic_disease_risk/modeling src/chronic_disease_risk/evaluation scripts/train_baseline.py tests/modeling tests/evaluation
git commit -m "feat: add baseline training and evaluation"
```

### Task 7: Add Candidate Models, Model Selection, SHAP, And Website Artifacts

**Files:**
- Modify: `configs/modeling.yaml`
- Create: `src/chronic_disease_risk/modeling/candidates.py`
- Create: `src/chronic_disease_risk/modeling/search.py`
- Create: `src/chronic_disease_risk/interpretability/__init__.py`
- Create: `src/chronic_disease_risk/interpretability/shap_report.py`
- Create: `src/chronic_disease_risk/artifacts/__init__.py`
- Create: `src/chronic_disease_risk/artifacts/exporter.py`
- Create: `scripts/train_candidates.py`
- Create: `scripts/export_artifacts.py`
- Create: `tests/modeling/test_candidates.py`
- Create: `tests/interpretability/test_shap_report.py`
- Create: `tests/artifacts/test_exporter.py`

- [ ] **Step 1: Write the failing candidate model registry test**

```python
from chronic_disease_risk.modeling.candidates import build_candidate_models


def test_build_candidate_models_includes_expected_estimators():
    models = build_candidate_models(random_state=42)
    assert {"logistic_regression", "random_forest", "xgboost", "svm"}.issubset(models)
```

- [ ] **Step 2: Run the candidate model tests to verify they fail**

Run: `python -m pytest tests/modeling/test_candidates.py tests/interpretability/test_shap_report.py tests/artifacts/test_exporter.py -q`
Expected: FAIL with missing candidate/SHAP/export logic

- [ ] **Step 3: Implement the model registry and selection routine**

```python
def build_candidate_models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced"),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_state),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
    }
```

- [ ] **Step 4: Implement SHAP summary generation for tree models and fallback explanation metadata for non-tree models**

```python
def save_shap_summary_plot(model, X: pd.DataFrame, destination: Path) -> None:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(destination, dpi=200, bbox_inches="tight")
```

- [ ] **Step 5: Implement website artifact export**

```python
def export_web_artifact(task_name: str, metrics: dict[str, float], feature_names: list[str], destination: Path) -> None:
    payload = {
        "task": task_name,
        "metrics": metrics,
        "features": feature_names,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 6: Run the candidate, SHAP, and export tests**

Run: `python -m pytest tests/modeling/test_candidates.py tests/interpretability/test_shap_report.py tests/artifacts/test_exporter.py -q`
Expected: PASS

- [ ] **Step 7: Commit the model comparison and export task**

```bash
git add configs/modeling.yaml src/chronic_disease_risk/modeling/candidates.py src/chronic_disease_risk/modeling/search.py src/chronic_disease_risk/interpretability src/chronic_disease_risk/artifacts scripts/train_candidates.py scripts/export_artifacts.py tests/modeling/test_candidates.py tests/interpretability tests/artifacts
git commit -m "feat: add candidate models shap and export artifacts"
```

### Task 8: Add End-To-End Scripts, Smoke Tests, And Researcher Docs

**Files:**
- Modify: `README.md`
- Create: `scripts/run_nhanes_pipeline.py`
- Create: `tests/integration/test_pipeline_smoke.py`
- Create: `tests/fixtures/sample_interim.csv`
- Create: `tests/fixtures/sample_processed.csv`

- [ ] **Step 1: Write the failing smoke test for the minimal offline pipeline**

```python
from pathlib import Path

from chronic_disease_risk.artifacts.exporter import export_web_artifact


def test_pipeline_smoke_writes_metrics_artifact(tmp_path: Path):
    destination = tmp_path / "artifact.json"
    export_web_artifact("diabetes", {"auc": 0.81}, ["age", "glm7_score"], destination)
    assert destination.exists()
```

- [ ] **Step 2: Run the smoke test to verify the final pipeline entry points are wired correctly**

Run: `python -m pytest tests/integration/test_pipeline_smoke.py -q`
Expected: FAIL until the end-to-end path and fixtures are ready

- [ ] **Step 3: Create the top-level orchestration script**

```python
if __name__ == "__main__":
    download_raw_data()
    build_interim_tables()
    build_processed_dataset()
    train_all_tasks()
    generate_explanations()
    export_all_web_artifacts()
```

- [ ] **Step 4: Expand the README with the exact local workflow**

```markdown
1. `python -m pip install -r requirements.txt`
2. `python scripts/download_nhanes.py`
3. `python scripts/build_nhanes_interim.py`
4. `python scripts/build_model_dataset.py`
5. `python scripts/train_candidates.py`
6. `python scripts/export_artifacts.py`
```

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Run the minimal command walkthrough**

Run: `python scripts/run_nhanes_pipeline.py`
Expected: directories are created and artifact files appear under `reports/` and `artifacts/`

- [ ] **Step 7: Commit the orchestration and docs task**

```bash
git add README.md scripts/run_nhanes_pipeline.py tests/integration tests/fixtures
git commit -m "feat: add end-to-end pipeline orchestration"
```

## Notes For Execution

- Keep `CHARLS` out of the first execution pass except for config placeholders and empty directories.
- Do not hardcode literature GLM7 coefficients until they are verified against the source paper; keep them config-driven.
- For the first disease labels, prefer explicit NHANES questionnaire-based rules and document every column choice in `configs/label_rules.yaml`.
- Save every generated dataset and report with cycle/task names so the later web layer can map outputs deterministically.
- If dependency installation or dataset download needs network approval, request it before running those commands.
