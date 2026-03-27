from __future__ import annotations

import math


def _mg_dl_to_mmol_l_triglycerides(value_mg_dl: float) -> float:
    # TG conversion: 1 mmol/L = 88.57 mg/dL
    return float(value_mg_dl) / 88.57


def _mg_dl_to_mmol_l_cholesterol(value_mg_dl: float) -> float:
    # Cholesterol (HDL/LDL/TC) conversion: 1 mmol/L = 38.67 mg/dL
    return float(value_mg_dl) / 38.67


def _uu_ml_to_pmol_l_insulin(value_uU_ml: float) -> float:
    # Common clinical conversion.
    return float(value_uU_ml) * 6.945


def compute_glm7_score(
    *,
    age_years: float,
    bmi: float,
    fbg_mg_dl: float,
    insulin_uU_ml: float,
    triglycerides_mg_dl: float,
    ldl_c_mg_dl: float,
    hdl_c_mg_dl: float,
) -> float:
    """Compute GLM7 according to the formula provided in the proposal/paper.

    GLM7 = log10( Age(year) * BMI(kg/m^2) * FBG(mg/dL) * Insulin(pmol/L)
                  * TG(mmol/L) * LDL-c(mmol/L) / HDL-c(mmol/L) )

    Returns NaN if inputs are non-positive.
    """

    tg_mmol_l = _mg_dl_to_mmol_l_triglycerides(triglycerides_mg_dl)
    ldl_mmol_l = _mg_dl_to_mmol_l_cholesterol(ldl_c_mg_dl)
    hdl_mmol_l = _mg_dl_to_mmol_l_cholesterol(hdl_c_mg_dl)
    insulin_pmol_l = _uu_ml_to_pmol_l_insulin(insulin_uU_ml)

    numerator = float(age_years) * float(bmi) * float(fbg_mg_dl) * insulin_pmol_l * tg_mmol_l * ldl_mmol_l
    denominator = float(hdl_mmol_l)

    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")

    ratio = numerator / denominator
    if ratio <= 0.0:
        return float("nan")

    return float(math.log10(ratio))


class GLM7Builder:
    """GLM7 feature builder.

    Backward compatibility:
    - If `weights` is provided (non-empty), it behaves as a linear score.
    - Otherwise, it computes GLM7 using the formula.
    """

    def __init__(self, weights: dict[str, float], intercept: float = 0.0) -> None:
        self.weights = weights
        self.intercept = intercept

    def transform_row(self, row: dict[str, float]) -> dict[str, float]:
        if self.weights:
            score = float(self.intercept) + sum(float(row[name]) * float(weight) for name, weight in self.weights.items())
            return {**row, "glm7_score": float(score)}

        score = compute_glm7_score(
            age_years=row["age"],
            bmi=row["bmi"],
            fbg_mg_dl=row["fbg"],
            insulin_uU_ml=row["insulin"],
            triglycerides_mg_dl=row["triglycerides"],
            ldl_c_mg_dl=row["ldl_c"],
            hdl_c_mg_dl=row["hdl_c"],
        )
        return {**row, "glm7_score": float(score)}