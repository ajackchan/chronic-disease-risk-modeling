from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chronic_disease_risk.config import load_yaml_config


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def _odds_ratio_2x2(a: int, b: int, c: int, d: int) -> float | None:
    # a: y=1, high=1
    # b: y=0, high=1
    # c: y=1, high=0
    # d: y=0, high=0
    if min(a, b, c, d) == 0:
        return None
    return float((a * d) / (b * c))


def analyze_glm7(
    *,
    dataset_path: Path,
    reports_dir: Path,
    task_names: list[str],
    threshold: float | None = None,
) -> Path:
    df = pd.read_csv(dataset_path)

    if "glm7_score" not in df.columns:
        raise KeyError("Missing column glm7_score in dataset")

    glm7 = df["glm7_score"].to_numpy(dtype=float)
    if float(np.nanstd(glm7)) == 0.0:
        raise ValueError(
            "glm7_score is constant (std=0). This usually means configs/glm7.yaml has empty weights. "
            "To strictly reproduce GLM7 from the paper, fill in the coefficients/intercept and rebuild the dataset."
        )

    rows: list[dict[str, object]] = []

    for task in task_names:
        if task not in df.columns:
            continue

        y = df[task].to_numpy(dtype=int)
        auc = float(roc_auc_score(y, glm7))

        th = float(threshold) if threshold is not None else _youden_threshold(y, glm7)
        high = (glm7 >= th).astype(int)

        a = int(((y == 1) & (high == 1)).sum())
        b = int(((y == 0) & (high == 1)).sum())
        c = int(((y == 1) & (high == 0)).sum())
        d = int(((y == 0) & (high == 0)).sum())

        or_high = _odds_ratio_2x2(a, b, c, d)
        rows.append(
            {
                "task": task,
                "auc_glm7_score": auc,
                "threshold_used": th,
                "high_risk_rate": float(high.mean()),
                "odds_ratio_high_vs_low": or_high,
                "n": int(len(y)),
            }
        )

    out = pd.DataFrame(rows).sort_values("auc_glm7_score", ascending=False)
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "glm7_analysis.csv"
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")

    threshold = None
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        if idx + 1 >= len(sys.argv):
            print("[ERROR] --threshold requires a value")
            raise SystemExit(2)
        threshold = float(sys.argv[idx + 1])

    try:
        out_path = analyze_glm7(
            dataset_path=REPO_ROOT / "data" / "processed" / "nhanes_model_dataset.csv",
            reports_dir=REPO_ROOT / "reports" / "tables",
            task_names=modeling_config["tasks"],
            threshold=threshold,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(2) from exc

    print({"glm7_analysis": out_path.as_posix()})