from __future__ import annotations

from pathlib import Path

from chronic_disease_risk.config import load_yaml_config, resolve_repo_path
from chronic_disease_risk.features.formulas import compute_aip, compute_tyg, compute_tyg_bmi
from chronic_disease_risk.features.glm7 import GLM7Builder
from chronic_disease_risk.preprocessing.cohort import apply_inclusion_rules
from chronic_disease_risk.preprocessing.labels import add_multimorbidity_label


DEFAULT_LABEL_RULES = {
    "outcomes": {
        "cardiovascular": {"any_of": ["mcq160b", "mcq160c", "mcq160d", "mcq160e", "mcq160f"], "positive_values": [1]},
        "diabetes": {"any_of": ["diq010"], "positive_values": [1, 3]},
        "liver": {"any_of": ["mcq160l"], "positive_values": [1]},
        "cancer": {"any_of": ["mcq220"], "positive_values": [1]},
    },
    "multimorbidity_threshold": 2,
}

DEFAULT_GLM7_CONFIG = {"weights": {}, "intercept": 0.0}
INSULIN_CANDIDATES = ["lbxin", "lbxins"]


def _build_outcome(df, columns: list[str], positive_values: list[int]):
    import pandas as pd

    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0, index=df.index)
    return df[available].isin(positive_values).any(axis=1).astype(int)


def _resolve_insulin_column(df) -> str:
    for column in INSULIN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError("No insulin column found; expected one of lbxin or lbxins")


def _glm7_input_row(row, insulin_column: str) -> dict[str, float]:
    return {
        "age": row["ridageyr"],
        "insulin": row[insulin_column],
        "triglycerides": row["lbxtr"],
        "ldl_c": row["lbdldl"],
        "aip": row["aip"],
        "tyg": row["tyg"],
        "tyg_bmi": row["tyg_bmi"],
    }


def build_processed_dataset(df, glm7_config: dict | None = None, label_rules: dict | None = None):
    processed = apply_inclusion_rules(df)
    label_rules = label_rules or DEFAULT_LABEL_RULES
    glm7_config = glm7_config or DEFAULT_GLM7_CONFIG
    insulin_column = _resolve_insulin_column(processed)

    processed = processed.copy()
    processed["aip"] = processed.apply(lambda row: compute_aip(row["lbxtr"], row["lbdhdd"]), axis=1)
    processed["tyg"] = processed.apply(lambda row: compute_tyg(row["lbxtr"], row["lbxglu"]), axis=1)
    processed["tyg_bmi"] = processed.apply(lambda row: compute_tyg_bmi(row["tyg"], row["bmxbmi"]), axis=1)

    outcome_columns = []
    for outcome_name, rule in label_rules.get("outcomes", {}).items():
        processed[outcome_name] = _build_outcome(processed, rule.get("any_of", []), rule.get("positive_values", [1]))
        outcome_columns.append(outcome_name)

    processed = add_multimorbidity_label(processed, outcome_columns)

    glm7_builder = GLM7Builder(weights=glm7_config.get("weights", {}), intercept=glm7_config.get("intercept", 0.0))
    processed["glm7_score"] = processed.apply(
        lambda row: glm7_builder.transform_row(_glm7_input_row(row, insulin_column))["glm7_score"],
        axis=1,
    )

    return processed


def build_processed_dataset_from_csv(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or Path.cwd()
    import pandas as pd

    interim_path = resolve_repo_path(repo_root, "data/interim/nhanes_interim.csv")
    output_path = resolve_repo_path(repo_root, "data/processed/nhanes_model_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    interim_df = pd.read_csv(interim_path)
    glm7_config = load_yaml_config(resolve_repo_path(repo_root, "configs/glm7.yaml"))
    label_rules = load_yaml_config(resolve_repo_path(repo_root, "configs/label_rules.yaml"))
    processed_df = build_processed_dataset(interim_df, glm7_config=glm7_config, label_rules=label_rules)
    processed_df.to_csv(output_path, index=False)
    return output_path
