from __future__ import annotations

import math


def _mg_dl_to_mmol_l_triglycerides(value_mg_dl: float) -> float:
    # TG conversion: 1 mmol/L = 88.57 mg/dL
    return float(value_mg_dl) / 88.57


def _mg_dl_to_mmol_l_cholesterol(value_mg_dl: float) -> float:
    # Cholesterol (HDL/LDL/TC) conversion: 1 mmol/L = 38.67 mg/dL
    return float(value_mg_dl) / 38.67


def compute_aip(triglycerides_mg_dl: float, hdl_c_mg_dl: float) -> float:
    """AIP = log10(TG/HDL) using molar concentrations.

    Many clinical datasets store TG/HDL in mg/dL; this converts to mmol/L first.
    """

    tg_mmol_l = _mg_dl_to_mmol_l_triglycerides(triglycerides_mg_dl)
    hdl_mmol_l = _mg_dl_to_mmol_l_cholesterol(hdl_c_mg_dl)
    return math.log10(tg_mmol_l / hdl_mmol_l)


def compute_tyg(triglycerides_mg_dl: float, fasting_glucose_mg_dl: float) -> float:
    # Standard TyG definition typically uses mg/dL.
    return math.log((triglycerides_mg_dl * fasting_glucose_mg_dl) / 2.0)


def compute_tyg_bmi(tyg: float, bmi: float) -> float:
    return float(tyg) * float(bmi)