from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL_MODULES = "codet5_660m_arcmega"


def _load_settings(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _format_env_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _apply_env(settings: dict, mode: str) -> None:
    raw_train = Path(settings["TRAIN_DATA_CLEAN_PATH"]).expanduser().resolve()
    os.environ.setdefault("ARC_DATA_PATH", str(raw_train))
    pretrain_dir = Path(settings.get("PRETRAIN_DIR", settings["TRAIN_DATA_CLEAN_PATH"])).expanduser().resolve()
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PRETRAINING_DATA_DIR", str(pretrain_dir))
    os.environ.setdefault("MODEL_CONFIG_MODULES", settings.get("MODEL_CONFIG_MODULES", DEFAULT_MODEL_MODULES))
    model_dir = Path(settings.get("MODEL_DIR", "model")).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RELOAD_PATH", str(model_dir))
    checkpoint_dir = Path(settings.get("CHECKPOINT_DIR", "model/checkpoints")).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CHECKPOINT_DIR", str(checkpoint_dir))
    submission_dir = Path(settings.get("SUBMISSION_DIR", "submission")).expanduser().resolve()
    submission_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SUBMISSION_PATH", str(submission_dir / "submission.json"))
    os.environ.setdefault("RUN_MODE", mode)

    extra_env_keys = {
        "RAW_DATA_DIR": "RAW_DATA_DIR",
        "CLEAN_DATA_DIR": "CLEAN_DATA_DIR",
        "DATA_PATH": "DATA_PATH",
        "ARC_DATA_PATH": "ARC_DATA_PATH",
        "MODEL_PATHS": "MODEL_PATHS",
        "SYMBOL_ENCODING_ENABLED": "SYMBOL_ENCODING_ENABLED",
        "SYMBOL_ENCODING_SCHEME": "SYMBOL_ENCODING_SCHEME",
        "TEST_MODE": "TEST_MODE",
        "TEST_MODE_NUM_TASKS": "TEST_MODE_NUM_TASKS",
        "TEST_MODE_TTT_ITEMS": "TEST_MODE_TTT_ITEMS",
        "TEST_MODE_INFERENCE_ITEMS": "TEST_MODE_INFERENCE_ITEMS",
        "TEST_MODE_TASK_IDS": "TEST_MODE_TASK_IDS",
        "GLOBAL_TTT_ENABLED": "GLOBAL_TTT_ENABLED",
        "GLOBAL_TTT_SAMPLES": "GLOBAL_TTT_SAMPLES",
        "GLOBAL_INF_SAMPLES": "GLOBAL_INF_SAMPLES",
        "DETECT_TEST_MODE_ID": "DETECT_TEST_MODE_ID",
        "SE": "SE",
        "ARC_TORCH_OPTIMIZER": "ARC_TORCH_OPTIMIZER",
        "ARC_TORCH_MUON_LR": "ARC_TORCH_MUON_LR",
        "ARC_TORCH_MUON_MOMENTUM": "ARC_TORCH_MUON_MOMENTUM",
        "ARC_TORCH_MUON_EXCLUDE": "ARC_TORCH_MUON_EXCLUDE",
        "USE_TPU": "USE_TPU",
        "USE_FLAX_GPU": "USE_FLAX_GPU",
        "PRESERVE_FINETUNED_MODEL": "PRESERVE_FINETUNED_MODEL",
        "JAX_PLATFORMS": "JAX_PLATFORMS",
        "FLAX_TARGET_DEVICE": "FLAX_TARGET_DEVICE",
        "XLA_FLAGS": "XLA_FLAGS",
        "JAX_COMPILATION_CACHE_DIR": "JAX_COMPILATION_CACHE_DIR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "XLA_PYTHON_CLIENT_ALLOCATOR",
    }
    for json_key, env_name in extra_env_keys.items():
        if json_key not in settings:
            continue
        formatted = _format_env_value(settings[json_key])
        if formatted is None:
            continue
        os.environ.setdefault(env_name, formatted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MindsAI ARC model")
    parser.add_argument("--settings", default="SETTINGS.json", help="Path to SETTINGS.json")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args to pass to src.main")
    args = parser.parse_args()

    settings_path = Path(args.settings)
    settings = _load_settings(settings_path)
    _apply_env(settings, mode="train")

    cmd = [sys.executable, "-m", "src.main"]
    if args.extra:
        cmd.extend(args.extra)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
