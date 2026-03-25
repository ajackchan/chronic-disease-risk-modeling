from __future__ import annotations

from pathlib import Path

import requests

from chronic_disease_risk.config import load_yaml_config, resolve_repo_path


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_from_config(config_path: str | Path = "configs/nhanes.yaml", repo_root: str | Path | None = None) -> list[Path]:
    repo_root = Path(repo_root or Path.cwd())
    config = load_yaml_config(resolve_repo_path(repo_root, str(config_path)))
    downloads = config.get("downloads", [])
    written: list[Path] = []

    for item in downloads:
        destination = resolve_repo_path(repo_root, item["destination"])
        written.append(download_file(item["url"], destination))

    return written
