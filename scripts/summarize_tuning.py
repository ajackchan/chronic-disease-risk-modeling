from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.config import load_yaml_config


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_tuning_results(*, reports_dir: Path, task_names: list[str]) -> Path:
    rows: list[dict[str, object]] = []

    for task in task_names:
        summary_path = reports_dir / f"candidate_{task}_summary.csv"
        if not summary_path.exists():
            rows.append({"task": task, "status": "missing_summary"})
            continue

        summary = pd.read_csv(summary_path)
        best_row = summary.sort_values("auc", ascending=False).iloc[0]

        xgb_auc = None
        tuned_auc = None

        if (summary["model_name"] == "xgboost").any():
            xgb_auc = float(summary.loc[summary["model_name"] == "xgboost", "auc"].iloc[0])
        if (summary["model_name"] == "xgboost_tuned").any():
            tuned_auc = float(summary.loc[summary["model_name"] == "xgboost_tuned", "auc"].iloc[0])

        tuned_params_path = reports_dir / f"candidate_{task}_xgboost_tuned_params.json"
        tuned_payload = _load_json(tuned_params_path) if tuned_params_path.exists() else {}

        rows.append(
            {
                "task": task,
                "best_model_name": str(best_row["model_name"]),
                "best_auc": float(best_row["auc"]),
                "xgboost_auc": xgb_auc,
                "xgboost_tuned_auc": tuned_auc,
                "auc_gain_tuned_minus_base": (tuned_auc - xgb_auc) if (tuned_auc is not None and xgb_auc is not None) else None,
                "tuned_search_mode": tuned_payload.get("search_mode"),
                "tuned_cv_splits": tuned_payload.get("cv_splits"),
                "tuned_best_score_cv_auc": tuned_payload.get("best_score_cv_auc"),
                "tuned_best_params": json.dumps(tuned_payload.get("best_params", {}), ensure_ascii=False),
                "status": "ok",
            }
        )

    out = pd.DataFrame(rows)
    out_path = reports_dir / "tuning_summary.csv"
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")
    reports_dir = REPO_ROOT / "reports" / "tables"

    out_path = summarize_tuning_results(reports_dir=reports_dir, task_names=modeling_config["tasks"])
    print({"tuning_summary": out_path.as_posix()})