from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
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


def save_calibration_curve_plot(y_true, y_prob, destination: Path, title: str) -> None:
    """Save a simple reliability diagram (calibration curve)."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa4b2", linewidth=1, label="理想")
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#32d6c8", label="模型")
    ax.set_title(title)
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际阳性比例")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_decision_curve_plot(y_true, y_prob, destination: Path, title: str) -> None:
    """Decision curve analysis (net benefit) plot.

    net_benefit(t) = TP/n - FP/n * (t/(1-t))

    This provides a quick, defense-friendly visualization. It's not a full clinical
    DCA framework, but is enough for comparing model vs treat-all vs treat-none.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = max(1, len(y_true))

    thresholds = np.linspace(0.01, 0.99, 99)

    def net_benefit_for_probs(probs: np.ndarray) -> np.ndarray:
        out = []
        for t in thresholds:
            y_pred = (probs >= t).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            w = t / (1.0 - t)
            out.append((tp / n) - (fp / n) * w)
        return np.asarray(out)

    nb_model = net_benefit_for_probs(y_prob)
    # treat-all: always predict positive
    nb_all = net_benefit_for_probs(np.ones_like(y_prob))
    # treat-none: always negative -> net benefit 0
    nb_none = np.zeros_like(nb_model)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, nb_model, color="#32d6c8", linewidth=2, label="模型")
    ax.plot(thresholds, nb_all, color="#ffb454", linewidth=2, label="全治疗")
    ax.plot(thresholds, nb_none, color="#9aa4b2", linewidth=1.5, linestyle="--", label="不治疗")

    ax.set_title(title)
    ax.set_xlabel("阈值概率")
    ax.set_ylabel("净获益 (Net Benefit)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)