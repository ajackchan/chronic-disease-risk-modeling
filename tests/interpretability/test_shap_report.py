from pathlib import Path

import pandas as pd

from chronic_disease_risk.interpretability.shap_report import save_shap_summary_plot


class _DummyExplainer:
    def __call__(self, X):
        return {"rows": len(X)}


def test_save_shap_summary_plot_writes_output(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "shap.png"
    seen = {}

    monkeypatch.setattr(
        "chronic_disease_risk.interpretability.shap_report.shap.Explainer",
        lambda model, X: _DummyExplainer(),
    )
    monkeypatch.setattr(
        "chronic_disease_risk.interpretability.shap_report.shap.plots.beeswarm",
        lambda values, show=False: seen.setdefault("beeswarm", (values, show)),
    )
    monkeypatch.setattr(
        "chronic_disease_risk.interpretability.shap_report.plt.savefig",
        lambda path, dpi, bbox_inches: Path(path).write_text("plot", encoding="utf-8"),
    )

    save_shap_summary_plot(object(), pd.DataFrame({"age": [40, 50]}), destination)

    assert destination.exists()
    assert seen["beeswarm"][1] is False
