from pathlib import Path

from chronic_disease_risk.config import load_yaml_config, resolve_repo_path


def test_load_yaml_config_reads_repo_relative_file(tmp_path: Path) -> None:
    config_path = tmp_path / "paths.yaml"
    config_path.write_text("raw_dir: data/raw\n", encoding="utf-8")

    data = load_yaml_config(config_path)

    assert data["raw_dir"] == "data/raw"


def test_resolve_repo_path_returns_absolute_path(tmp_path: Path) -> None:
    resolved = resolve_repo_path(tmp_path, "data/raw")

    assert resolved == tmp_path / "data" / "raw"
