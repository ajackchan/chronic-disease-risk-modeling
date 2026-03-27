from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.config import load_yaml_config
from chronic_disease_risk.evaluation.metrics import compute_binary_metrics
from chronic_disease_risk.evaluation.reporting import write_metrics_table


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path.as_posix()}\n{hint}")


def validate_on_charls(
    *,
    charls_dataset_path: Path,
    reports_dir: Path,
    task_names: list[str],
    feature_columns: list[str],
) -> dict[str, Path]:
    """External validation on CHARLS.

    Requirements:
    - `data/processed/charls_model_dataset.csv` must exist.
    - It must contain the same feature columns as `configs/modeling.yaml`.
    - It must contain task label columns (0/1).

    Outputs:
    - reports/tables/charls_external_{task}_metrics.csv
    - reports/tables/charls_external_comparison.csv
    """

    _require_file(
        charls_dataset_path,
        hint=(
            "请先准备 CHARLS 处理后数据: data/processed/charls_model_dataset.csv\n"
            "放置约定见 docs/charls.md"
        ),
    )

    df = pd.read_csv(charls_dataset_path)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise KeyError(
            "CHARLS 数据缺少特征列: "
            + ", ".join(missing_features)
            + "\n请按 docs/charls.md 将列名对齐 configs/modeling.yaml"
        )

    rows: list[dict[str, float | str]] = []
    outputs: dict[str, Path] = {}

    for task in task_names:
        if task not in df.columns:
            raise KeyError(f"CHARLS 数据缺少标签列: {task}")

        model_path = reports_dir / f"candidate_best_{task}.joblib"
        _require_file(model_path, hint="请先在 NHANES 上训练并生成 reports/tables/candidate_best_{task}.joblib")

        X = df[feature_columns]
        y = df[task]

        pipeline = joblib.load(model_path)
        y_prob = pipeline.predict_proba(X)[:, 1]
        metrics = compute_binary_metrics(y.to_numpy(), y_prob)

        out_path = reports_dir / f"charls_external_{task}_metrics.csv"
        write_metrics_table(metrics, out_path)

        rows.append({"task": task, **metrics})
        outputs[task] = out_path

    comparison = pd.DataFrame(rows).sort_values("auc", ascending=False)
    comparison_path = reports_dir / "charls_external_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    outputs["comparison"] = comparison_path

    return outputs


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")

    output = validate_on_charls(
        charls_dataset_path=REPO_ROOT / "data" / "processed" / "charls_model_dataset.csv",
        reports_dir=REPO_ROOT / "reports" / "tables",
        task_names=modeling_config["tasks"],
        feature_columns=modeling_config["feature_columns"],
    )
    print(output)