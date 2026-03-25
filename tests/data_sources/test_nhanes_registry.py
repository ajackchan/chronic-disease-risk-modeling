from chronic_disease_risk.data_sources.nhanes_registry import build_xpt_url


def test_build_xpt_url_for_demographics_table() -> None:
    url = build_xpt_url(cycle_suffix="J", component="DEMO", table="P_DEMO")

    assert url.endswith("/DEMO/P_DEMO_J.XPT")
