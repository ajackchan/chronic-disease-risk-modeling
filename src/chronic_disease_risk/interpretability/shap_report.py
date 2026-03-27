from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def save_shap_summary_plot(model, X: pd.DataFrame, destination: Path) -> None:
    """Save a SHAP beeswarm plot.

    Notes:
    - For performance, prefer `save_shap_summary_plot_with_background` so you can
      pass a small background sample.
    - This function keeps the original signature for backward compatibility.
    """

    save_shap_summary_plot_with_background(model=model, X=X, destination=destination, background=X)


def save_shap_summary_plot_with_background(
    model,
    X: pd.DataFrame,
    destination: Path,
    background: pd.DataFrame | None = None,
    *,
    title: str | None = None,
    max_display: int = 20,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    background = X if background is None else background

    explainer = shap.Explainer(model, background)
    shap_values = explainer(X)

    # For some classifiers, SHAP returns per-class outputs with shape (n, features, outputs).
    # Beeswarm expects a single output; use the positive class when available.
    try:
        if getattr(shap_values, "values", None) is not None and getattr(shap_values.values, "ndim", 0) == 3:
            shap_values = shap_values[..., 1]
    except Exception:
        pass

    try:
        shap.plots.beeswarm(shap_values, show=False, max_display=max_display)
    except TypeError:
        # Some test stubs and older SHAP versions may not accept max_display.
        shap.plots.beeswarm(shap_values, show=False)

    if title:
        plt.title(title)

    plt.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close()