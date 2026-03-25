from __future__ import annotations

import math


def compute_aip(triglycerides: float, hdl_c: float) -> float:
    return math.log10(triglycerides / hdl_c)


def compute_tyg(triglycerides_mg_dl: float, fasting_glucose_mg_dl: float) -> float:
    return math.log((triglycerides_mg_dl * fasting_glucose_mg_dl) / 2.0)


def compute_tyg_bmi(tyg: float, bmi: float) -> float:
    return tyg * bmi
