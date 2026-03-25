from __future__ import annotations


class GLM7Builder:
    def __init__(self, weights: dict[str, float], intercept: float = 0.0) -> None:
        self.weights = weights
        self.intercept = intercept

    def transform_row(self, row: dict[str, float]) -> dict[str, float]:
        score = self.intercept + sum(row[name] * weight for name, weight in self.weights.items())
        return {**row, "glm7_score": score}
