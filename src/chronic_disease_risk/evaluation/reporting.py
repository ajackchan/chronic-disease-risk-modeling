from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def write_metrics_table(metrics: dict[str, float], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(destination, index=False)


def save_roc_curve_plot(y_true, y_prob, destination: Path, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, destination: Path, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(summary: pd.DataFrame, destination: Path, metric_name: str, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ranked = summary.sort_values(metric_name, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ranked, x="model_name", y=metric_name, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("model_name")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
