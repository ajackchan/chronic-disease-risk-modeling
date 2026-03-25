from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.config import CONFIGS_DIR, REPORTS_DIR
from chronic_disease_risk.config import load_yaml_config


def get_task_names() -> list[str]:
    modeling_config = load_yaml_config(CONFIGS_DIR / 'modeling.yaml')
    return modeling_config.get('tasks', [])


def load_task_overview(task_name: str, reports_dir: Path = REPORTS_DIR) -> dict:
    summary = pd.read_csv(reports_dir / f'candidate_{task_name}_summary.csv')
    best_row = summary.sort_values('auc', ascending=False).iloc[0]
    return {
        'task': task_name,
        'best_model_name': best_row['model_name'],
        'auc': float(best_row['auc']),
        'accuracy': float(best_row['accuracy']),
        'precision': float(best_row['precision']),
        'recall': float(best_row['recall']),
        'f1': float(best_row['f1']),
    }


def build_overview_payload() -> dict:
    tasks = [load_task_overview(task_name) for task_name in get_task_names()]
    return {
        'title': '慢性病风险建模答辩展示台',
        'tasks': tasks,
    }
