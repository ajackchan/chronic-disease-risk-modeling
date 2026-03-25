from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML file and always return a mapping."""
    with Path(path).open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def resolve_repo_path(repo_root: str | Path, relative_path: str) -> Path:
    """Resolve a repo-relative path into an absolute path."""
    return Path(repo_root).joinpath(relative_path).resolve()
