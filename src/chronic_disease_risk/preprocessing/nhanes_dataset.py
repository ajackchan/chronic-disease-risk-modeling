from __future__ import annotations

from pathlib import Path

from chronic_disease_risk.config import load_yaml_config, resolve_repo_path
from chronic_disease_risk.data_sources.xpt_reader import read_xpt
from chronic_disease_risk.preprocessing.nhanes_merge import merge_nhanes_tables


def build_nhanes_interim_dataset(repo_root: Path | None = None) -> Path:
    repo_root = repo_root or Path.cwd()
    nhanes_config = load_yaml_config(resolve_repo_path(repo_root, "configs/nhanes.yaml"))
    variable_config = load_yaml_config(resolve_repo_path(repo_root, "configs/nhanes_variables.yaml"))
    output_path = resolve_repo_path(repo_root, "data/interim/nhanes_interim.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for cycle in nhanes_config.get("cycles", []):
        cycle_name = cycle["name"]
        suffix = cycle["suffix"]
        cycle_tables = variable_config["cycles"][cycle_name]
        loaded_tables = {}

        for file_config in cycle.get("files", []):
            table = file_config["table"]
            table_code = f"{table}_{suffix}"
            xpt_path = repo_root / "data" / "raw" / "nhanes" / cycle_name / f"{table_code}.xpt"
            table_df = read_xpt(xpt_path)
            selected_columns = [column for column in cycle_tables.get(table, []) if column in table_df.columns]
            loaded_tables[table] = table_df[selected_columns].copy()

        merged = merge_nhanes_tables(loaded_tables)
        if not merged.empty:
            merged["cycle"] = cycle_name
            frames.append(merged)

    interim_df = merge_nhanes_tables({}) if not frames else __import__("pandas").concat(frames, ignore_index=True)
    interim_df.to_csv(output_path, index=False)
    return output_path
