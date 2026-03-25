from __future__ import annotations

from pathlib import Path


def build_xpt_url(base_url: str, cycle_path: str, table_code: str) -> str:
    return f"{base_url}/{cycle_path}/DataFiles/{table_code}.xpt"


def build_download_manifest(config: dict, repo_root: Path) -> list[dict]:
    manifest: list[dict] = []
    base_url = config["base_url"]

    for cycle in config.get("cycles", []):
        cycle_name = cycle["name"]
        cycle_path = str(cycle["path"])
        suffix = cycle["suffix"]

        for file_config in cycle.get("files", []):
            table = file_config["table"]
            table_code = f"{table}_{suffix}"
            manifest.append(
                {
                    "cycle": cycle_name,
                    "table": table,
                    "table_code": table_code,
                    "url": build_xpt_url(base_url=base_url, cycle_path=cycle_path, table_code=table_code),
                    "destination": repo_root / "data" / "raw" / "nhanes" / cycle_name / f"{table_code}.xpt",
                }
            )

    return manifest
