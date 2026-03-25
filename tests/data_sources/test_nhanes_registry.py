from pathlib import Path

from chronic_disease_risk.data_sources.nhanes_registry import build_download_manifest, build_xpt_url


def test_build_xpt_url_matches_cdc_datafiles_pattern() -> None:
    url = build_xpt_url(base_url="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public", cycle_path="2017", table_code="DEMO_J")

    assert url == "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"


def test_build_download_manifest_generates_expected_nhanes_paths(tmp_path: Path) -> None:
    config = {
        "base_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public",
        "cycles": [
            {
                "name": "2017-2018",
                "path": "2017",
                "suffix": "J",
                "files": [
                    {"table": "DEMO"},
                    {"table": "MCQ"},
                ],
            }
        ],
    }

    manifest = build_download_manifest(config, tmp_path)

    assert manifest == [
        {
            "cycle": "2017-2018",
            "table": "DEMO",
            "table_code": "DEMO_J",
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
            "destination": tmp_path / "data" / "raw" / "nhanes" / "2017-2018" / "DEMO_J.xpt",
        },
        {
            "cycle": "2017-2018",
            "table": "MCQ",
            "table_code": "MCQ_J",
            "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.xpt",
            "destination": tmp_path / "data" / "raw" / "nhanes" / "2017-2018" / "MCQ_J.xpt",
        },
    ]
