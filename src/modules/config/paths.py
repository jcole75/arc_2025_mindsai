from __future__ import annotations

import os
from pathlib import Path


def _get_env_value(name: str) -> str | None:
    """Return environment value from os.environ or .env (via env._ENV) if present."""
    # Deferred import to avoid cycles
    try:
        from .env import _ENV as env_values  # type: ignore
    except Exception:
        env_values = {}
    value = os.environ.get(name, None)
    if value is None:
        value = env_values.get(name)
    if isinstance(value, str):
        value = value.strip()
    return value or None


def get_data_path() -> str:
    """Resolve the ARC data path with sensible overrides.

    Priority:
    1) `ARC_DATA_PATH` env var (or `DATA_PATH`) if set and non-empty
    2) Kaggle dataset default if present: `/kaggle/input/arc-1-5-json/challenges.json`
    3) Local repo data file if present: `./data/arc-agi-1-5_challenges.json`
    4) Fallback to bundled test data: `<project_root>/src/tests/test_data.json`
    """
    # 1) Explicit env overrides
    for key in ("ARC_DATA_PATH", "DATA_PATH"):
        env_path = _get_env_value(key)
        if env_path:
            return env_path

    # 2) Kaggle dataset path (only if it exists; harmless on local)
    kaggle_default = "/kaggle/input/arc-1-5-json/challenges.json"

    if os.path.exists(kaggle_default):
        return kaggle_default

    # 3) Local file shipped with repo (if user synced datasets locally)
    local_repo_path = "./data/arc-agi-1-5_challenges.json"
    # local_repo_path = "./data/arc-agi-2_evaluation_challenges.json"
    # local_repo_path = "./data/arc-agi-1_evaluation_challenges.json"
    # local_repo_path = "./data/arc-agi-2_training_challenges.json"
    if os.path.exists(local_repo_path):
        return local_repo_path

    # 4) Final fallback: small test JSON used by unit tests
    project_root = Path(__file__).resolve().parents[3]
    return str(project_root / "src" / "tests" / "test_data.json")


DATA_PATH = get_data_path()
RELOAD_PATH = "./model_fine_tuned"
SUBMISSION_PATH = "submission.json"
TRAIN_DATA_DISK_PATH = "train_dataset_prepared"
