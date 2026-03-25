from __future__ import annotations


def select_best_model(scores: dict[str, dict[str, float]], metric_name: str = "auc") -> tuple[str, dict[str, float]]:
    return max(scores.items(), key=lambda item: item[1].get(metric_name, float("-inf")))
