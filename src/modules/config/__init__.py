"""Public API aggregator for the config package."""

from __future__ import annotations

import os

# Apply environment overrides for symbol encoding before re-exports
import os as _os
import time
from typing import Any, Optional

from . import augmentation as _aug
from . import env as _env
from . import models as _models
from . import paths as _paths
from . import run_logging as _log


_env_map = getattr(_env, "_ENV", {}) if hasattr(_env, "_ENV") else {}
_sym_enabled_raw = _env_map.get("SYMBOL_ENCODING_ENABLED", _os.environ.get("SYMBOL_ENCODING_ENABLED"))
_sym_scheme_raw = _env_map.get("SYMBOL_ENCODING_SCHEME", _os.environ.get("SYMBOL_ENCODING_SCHEME"))
if _sym_enabled_raw is not None or _sym_scheme_raw is not None:
    try:
        se = _models.SYMBOL_ENCODING if isinstance(getattr(_models, "SYMBOL_ENCODING", {}), dict) else {}
    except Exception:
        se = {}
    if _sym_enabled_raw is not None:
        if isinstance(_sym_enabled_raw, bool):
            se["enabled"] = _sym_enabled_raw
        elif isinstance(_sym_enabled_raw, str):
            se["enabled"] = _sym_enabled_raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            se["enabled"] = bool(_sym_enabled_raw)
    if _sym_scheme_raw is not None:
        se["scheme"] = str(_sym_scheme_raw).strip().lower() or se.get("scheme", "letters")
    _models.SYMBOL_ENCODING = se

# Optional: MODEL_PATHS override via environment
_mp_raw = os.environ.get("MODEL_PATHS")
if not _mp_raw:
    # Also check .env loaded values
    try:
        _mp_raw = getattr(_env, "_ENV", {}).get("MODEL_PATHS")
    except Exception:
        _mp_raw = None
if _mp_raw:
    try:
        if isinstance(_mp_raw, str) and _mp_raw.strip().startswith("["):
            import json as _json

            _models.MODEL_PATHS = list(_json.loads(_mp_raw))
        elif isinstance(_mp_raw, str):
            _models.MODEL_PATHS = [p.strip() for p in _mp_raw.split(",") if p.strip()]
        elif isinstance(_mp_raw, (list, tuple)):
            _models.MODEL_PATHS = list(_mp_raw)
    except Exception:
        # Leave defaults if parsing fails
        pass

# Re-exports: environment flags and timers
START_TIME = _env.START_TIME
TOTAL_TIME = _env.TOTAL_TIME
BUFFER_TIME = _env.BUFFER_TIME
DEBUG_MODE = _env.DEBUG_MODE
VERBOSE_LOGGING = _env.VERBOSE_LOGGING
USE_TPU = _env.USE_TPU
USE_FLAX_GPU = _env.USE_FLAX_GPU
USE_FLAX = _env.USE_FLAX
ENVIRONMENT = _env.ENVIRONMENT
IS_KAGGLE = _env.IS_KAGGLE
IS_LOCAL = _env.IS_LOCAL
IS_SKYPILOT = _env.IS_SKYPILOT

TEST_MODE = _env.TEST_MODE
TEST_MODE_NUM_TASKS = _env.TEST_MODE_NUM_TASKS
TEST_MODE_TASK_IDS = _env.TEST_MODE_TASK_IDS
TEST_MODE_TTT_ITEMS = _env.TEST_MODE_TTT_ITEMS
TEST_MODE_INFERENCE_ITEMS = _env.TEST_MODE_INFERENCE_ITEMS
TEST_SIZE = _env.TEST_SIZE
GLOBAL_TTT_ENABLED = _env.GLOBAL_TTT_ENABLED
EXPAND_FACTOR = _env.EXPAND_FACTOR
PRUNE_FACTOR = _env.PRUNE_FACTOR
PRESERVE_FINETUNED_MODEL = _env.PRESERVE_FINETUNED_MODEL
SAVE_STEPS = _env.SAVE_STEPS

debug_print = _env.debug_print
verbose_print = _env.verbose_print

# Re-exports: paths
get_data_path = _paths.get_data_path
DATA_PATH = _paths.DATA_PATH
RELOAD_PATH = _paths.RELOAD_PATH
SUBMISSION_PATH = _paths.SUBMISSION_PATH
TRAIN_DATA_DISK_PATH = _paths.TRAIN_DATA_DISK_PATH

# Re-exports: models and constants
Z_SCORE_THRESHOLD = _models.Z_SCORE_THRESHOLD
MIN_VOTES_SINGLE_PRED = _models.MIN_VOTES_SINGLE_PRED
MIN_VOTES_AMBIGUOUS = _models.MIN_VOTES_AMBIGUOUS
MAX_GRID_SIZE = _models.MAX_GRID_SIZE
MAX_SYMBOLS = _models.MAX_SYMBOLS
MAX_PERMUTATIONS_TO_SAMPLE = _models.MAX_PERMUTATIONS_TO_SAMPLE
SYMBOL_ENCODING = _models.SYMBOL_ENCODING

MODEL_PATHS = _models.MODEL_PATHS
MODEL_SETTINGS = _models.MODEL_SETTINGS
MODEL_CONFIG_MODULES = _models.MODEL_CONFIG_MODULES
MODEL_CONFIG_SOURCES = _models.MODEL_CONFIG_SOURCES
get_model_type = _models.get_model_type
get_nested_setting = _models.get_nested_setting
get_ensemble_settings = _models.get_ensemble_settings

# Re-exports: augmentation
AUGMENTATION_CONFIG = _aug.AUGMENTATION_CONFIG


def generate_auto_run_title() -> str:
    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d_%H%M")
    mode = "TEST" if _env.TEST_MODE else "PROD"
    model_name = (
        "Mixed"
        if len(_models.MODEL_PATHS) > 1
        else os.path.basename(_models.MODEL_PATHS[0]).replace("-", "").replace("_", "")
    )
    dataset_name = "challenges" if "challenges.json" in _paths.DATA_PATH else "eval"
    ttt_suffix = "TTT" if _env.GLOBAL_TTT_ENABLED else "NoTTT"
    return f"ARC_{timestamp}_{mode}_{model_name}_{dataset_name}_{ttt_suffix}"


RUN_TITLE: str | None = generate_auto_run_title()


def deep_update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_notebook_config(
    notebook_config=None, notebook_model_settings=None, notebook_augmentation_config=None, *args, **kwargs
):
    # Update env flags
    if notebook_config:
        _env.TEST_MODE = notebook_config.get("TEST_MODE", _env.TEST_MODE)
        _env.TEST_MODE_NUM_TASKS = notebook_config.get("TEST_MODE_NUM_TASKS", _env.TEST_MODE_NUM_TASKS)
        _env.TEST_MODE_TASK_IDS = notebook_config.get("TEST_MODE_TASK_IDS", _env.TEST_MODE_TASK_IDS)
        _env.TEST_MODE_TTT_ITEMS = notebook_config.get("TEST_MODE_TTT_ITEMS", _env.TEST_MODE_TTT_ITEMS)
        _env.TEST_MODE_INFERENCE_ITEMS = notebook_config.get(
            "TEST_MODE_INFERENCE_ITEMS", _env.TEST_MODE_INFERENCE_ITEMS
        )
        _env.GLOBAL_TTT_ENABLED = notebook_config.get("GLOBAL_TTT_ENABLED", _env.GLOBAL_TTT_ENABLED)
        _env.EXPAND_FACTOR = notebook_config.get("EXPAND_FACTOR", _env.EXPAND_FACTOR)
        _env.PRUNE_FACTOR = notebook_config.get("PRUNE_FACTOR", _env.PRUNE_FACTOR)
        _env.DEBUG_MODE = notebook_config.get("DEBUG_MODE", _env.DEBUG_MODE)
        _env.VERBOSE_LOGGING = notebook_config.get("VERBOSE_LOGGING", _env.VERBOSE_LOGGING)
        _env.USE_TPU = notebook_config.get("USE_TPU", _env.USE_TPU)
        _env.USE_FLAX_GPU = notebook_config.get("USE_FLAX_GPU", _env.USE_FLAX_GPU)
        _env.USE_FLAX = _env.USE_TPU or _env.USE_FLAX_GPU

        if "TOTAL_TIME_HOURS" in notebook_config:
            _env.TOTAL_TIME = int(notebook_config["TOTAL_TIME_HOURS"] * 60 * 60)
        elif "TOTAL_TIME_MINUTES" in notebook_config:
            _env.TOTAL_TIME = int(notebook_config["TOTAL_TIME_MINUTES"] * 60)
        if "BUFFER_TIME_MINUTES" in notebook_config:
            _env.BUFFER_TIME = int(notebook_config["BUFFER_TIME_MINUTES"] * 60)
        if notebook_config.get("RESET_START_TIME"):
            _env.START_TIME = time.time()

        if "MODEL_PATHS" in notebook_config:
            _models.MODEL_PATHS = list(notebook_config["MODEL_PATHS"])
        if "DATA_PATH" in notebook_config:
            _paths.DATA_PATH = notebook_config["DATA_PATH"]

    if notebook_model_settings:
        for model_key, override in notebook_model_settings.items():
            if (
                model_key in _models.MODEL_SETTINGS
                and isinstance(_models.MODEL_SETTINGS[model_key], dict)
                and isinstance(override, dict)
            ):
                deep_update(_models.MODEL_SETTINGS[model_key], override)
            else:
                _models.MODEL_SETTINGS[model_key] = override

    if notebook_augmentation_config:
        for aug_key, aug_override in notebook_augmentation_config.items():
            if (
                aug_key in _aug.AUGMENTATION_CONFIG
                and isinstance(_aug.AUGMENTATION_CONFIG[aug_key], dict)
                and isinstance(aug_override, dict)
            ):
                deep_update(_aug.AUGMENTATION_CONFIG[aug_key], aug_override)
            else:
                _aug.AUGMENTATION_CONFIG[aug_key] = aug_override

    # Regenerate title unless explicitly provided
    if not (notebook_config and "RUN_TITLE" in notebook_config):
        globals()["RUN_TITLE"] = generate_auto_run_title()
    else:
        globals()["RUN_TITLE"] = notebook_config["RUN_TITLE"]

    # Sync re-exports after mutation
    globals().update(
        {
            "TEST_MODE": _env.TEST_MODE,
            "TEST_MODE_NUM_TASKS": _env.TEST_MODE_NUM_TASKS,
            "TEST_MODE_TASK_IDS": _env.TEST_MODE_TASK_IDS,
            "TEST_MODE_TTT_ITEMS": _env.TEST_MODE_TTT_ITEMS,
            "TEST_MODE_INFERENCE_ITEMS": _env.TEST_MODE_INFERENCE_ITEMS,
            "GLOBAL_TTT_ENABLED": _env.GLOBAL_TTT_ENABLED,
            "EXPAND_FACTOR": _env.EXPAND_FACTOR,
            "PRUNE_FACTOR": _env.PRUNE_FACTOR,
            "DEBUG_MODE": _env.DEBUG_MODE,
            "VERBOSE_LOGGING": _env.VERBOSE_LOGGING,
            "USE_TPU": _env.USE_TPU,
            "USE_FLAX_GPU": _env.USE_FLAX_GPU,
            "USE_FLAX": _env.USE_FLAX,
            "TOTAL_TIME": _env.TOTAL_TIME,
            "BUFFER_TIME": _env.BUFFER_TIME,
            "START_TIME": _env.START_TIME,
            "MODEL_PATHS": _models.MODEL_PATHS,
            "DATA_PATH": _paths.DATA_PATH,
            "AUGMENTATION_CONFIG": _aug.AUGMENTATION_CONFIG,
        }
    )


# Re-exports: logging
get_all_run_logs = _log.get_all_run_logs
get_run_logs_dir = _log.get_run_logs_dir
RunLogger = _log.RunLogger
current_run_logger = _log.current_run_logger
