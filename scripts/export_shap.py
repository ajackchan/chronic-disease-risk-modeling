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


def _subsample(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)


def _resolve_pipeline(pipeline, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    """Return (estimator, transformed_X) for SHAP.

    Our pipelines are `SimpleImputer -> model`. SHAP should run on the imputed
    feature matrix.
    """

    named_steps = getattr(pipeline, "named_steps", None)
    if not named_steps:
        return pipeline, X

    imputer = named_steps.get("imputer")
    estimator = named_steps.get("model")
    if imputer is None or estimator is None:
        return pipeline, X

    X_imp = imputer.transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=list(X.columns))
    return estimator, X_imp_df


def export_shap_plots(
    *,
    dataset_path: Path,
    reports_dir: Path,
    task_names: list[str],
    feature_columns: list[str],
    random_state: int = 42,
    background_max_rows: int = 200,
    explain_max_rows: int = 2000,
    force: bool = False,
) -> dict[str, str]:
    """Export SHAP summary plots for candidate-best pipelines.

    Output files:
    - reports/tables/candidate_best_{task}_shap.png

    This is intentionally separate from training so you can run it after you have
    artifacts and want explanation figures for the thesis/defense.
    """

    try:
        from chronic_disease_risk.interpretability import save_shap_summary_plot_with_background
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SHAP 依赖未就绪。请先在你的环境里安装 shap (pip install shap)，再运行 scripts/export_shap.py"
        ) from exc

    df = pd.read_csv(dataset_path)
    X_all = df[feature_columns]

    outputs: dict[str, str] = {}
    for task in task_names:
        model_path = reports_dir / f"candidate_best_{task}.joblib"
        if not model_path.exists():
            outputs[task] = f"skip: missing model: {model_path.as_posix()}"
            continue

        destination = reports_dir / f"candidate_best_{task}_shap.png"
        if destination.exists() and not force:
            outputs[task] = f"skip: exists: {destination.as_posix()}"
            continue

        pipeline = joblib.load(model_path)

        # Keep background small so SHAP is fast enough on laptops.
        X_bg = _subsample(X_all, background_max_rows, random_state)
        X_explain = _subsample(X_all, explain_max_rows, random_state + 1)

        estimator, X_bg_imp = _resolve_pipeline(pipeline, X_bg)
        _, X_explain_imp = _resolve_pipeline(pipeline, X_explain)

        save_shap_summary_plot_with_background(
            model=estimator,
            X=X_explain_imp,
            background=X_bg_imp,
            destination=destination,
            title=f"SHAP Summary - {task}",
            max_display=min(20, len(feature_columns)),
        )

        outputs[task] = destination.as_posix()

    return outputs


if __name__ == "__main__":
    modeling_config = load_yaml_config(REPO_ROOT / "configs" / "modeling.yaml")

    force = "--force" in sys.argv

    output = export_shap_plots(
        dataset_path=REPO_ROOT / "data" / "processed" / "nhanes_model_dataset.csv",
        reports_dir=REPO_ROOT / "reports" / "tables",
        task_names=modeling_config["tasks"],
        feature_columns=modeling_config["feature_columns"],
        random_state=modeling_config.get("random_state", 42),
        force=force,
    )
    print(output)