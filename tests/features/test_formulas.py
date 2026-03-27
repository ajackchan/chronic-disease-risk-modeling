import math

import pytest

from chronic_disease_risk.features.formulas import (
    compute_aip,
    compute_tyg,
    compute_tyg_bmi,
)


def test_compute_aip_uses_log10_ratio() -> None:
    # AIP uses molar ratio; when inputs are mg/dL, conversion factors apply.
    expected = math.log10((2.0 / 88.57) / (1.0 / 38.67))
    assert compute_aip(2.0, 1.0) == pytest.approx(expected)


def test_compute_tyg_matches_reference_formula() -> None:
    expected = math.log((150.0 * 100.0) / 2.0)

    assert compute_tyg(150.0, 100.0) == pytest.approx(expected)


def test_compute_tyg_bmi_multiplies_tyg_and_bmi() -> None:
    assert compute_tyg_bmi(8.0, 24.0) == pytest.approx(192.0)