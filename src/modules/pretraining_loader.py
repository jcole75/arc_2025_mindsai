from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

try:
    from .config import env as env_mod
except Exception:  # pragma: no cover - defensive import guard
    env_mod = None

from .pretraining_data import (
    PretrainingDataError,
    read_records_from_path,
    sample_directory_records,
)

try:
    from .hf_pretraining import HuggingFacePretrainError, load_hf_pretraining_records
except ImportError:  # pragma: no cover - optional dependency
    HuggingFacePretrainError = RuntimeError  # type: ignore
    load_hf_pretraining_records = None  # type: ignore


__all__ = [
    "PretrainingLoadError",
    "load_pretraining_records",
]


def _env_lookup(name: str) -> Any:
    value = os.environ.get(name)
    if value is not None:
        return value
    if env_mod is not None and hasattr(env_mod, "_lookup_env"):
        try:
            return env_mod._lookup_env(name)
        except Exception:
            return None
    return None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PretrainingLoadError(RuntimeError):
    """Raised when pretraining data cannot be produced for the configured model."""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(";", ",").split(",")]
        cleaned = [item for item in items if item]
        return cleaned or None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception as exc:
        raise PretrainingLoadError(f"Expected integer value for pretraining configuration, received {value!r}") from exc


def _resolve_path(path_like: str | os.PathLike[str] | None) -> Path | None:
    if not path_like:
        return None
    candidate = Path(path_like)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (Path.cwd(), PROJECT_ROOT):
        resolved = (root / candidate).expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _normalise_limit(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        limit = int(value)
    except Exception as exc:
        raise PretrainingLoadError(f"Invalid pretraining record limit: {value!r}") from exc
    if limit < 0:
        if limit == -1:
            return None
        raise PretrainingLoadError(f"Pretraining record limit must be >= 0 (received {value!r})")
    return limit


def load_pretraining_records(
    training_cfg: dict[str, Any],
    *,
    limit: int | None = None,
    puzzle_limit: int | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve and load pretraining data for a model."""
    rng = rng or random.Random()
    record_limit = _normalise_limit(limit)
    puzzle_limit = _normalise_limit(puzzle_limit)

    hf_cfg = training_cfg.get("hf_pretraining")
    if isinstance(hf_cfg, dict) and _truthy(hf_cfg.get("enabled", True)):
        if load_hf_pretraining_records is None:
            raise PretrainingLoadError("HuggingFace pretraining requested but datasets loader is unavailable.")
        target_records = record_limit or _coerce_int(hf_cfg.get("record_limit")) or None
        if target_records is None:
            raise PretrainingLoadError(
                "HuggingFace pretraining requires PRETRAINING_EXAMPLES or hf_pretraining.record_limit."
            )
        try:
            records, meta = load_hf_pretraining_records(
                hf_cfg,
                record_limit=target_records,
                rng=rng,
                verbose=verbose,
            )
        except HuggingFacePretrainError as exc:
            raise PretrainingLoadError(str(exc)) from exc
        record_limit = target_records
        return records, meta

    pretrain_dir = training_cfg.get("pretrain_dir") or _env_lookup("PRETRAINING_DATA_DIR")
    if pretrain_dir:
        directory = _resolve_path(pretrain_dir)
        if not directory:
            raise PretrainingLoadError(f"Configured pretrain_dir not found: {pretrain_dir}")
        try:
            records, detail = sample_directory_records(directory, record_limit, rng=rng, verbose=verbose)
        except PretrainingDataError as exc:
            raise PretrainingLoadError(str(exc)) from exc
        meta = {
            "source": "pretrain_dir",
            "directory": detail.get("directory"),
            "files_used": detail.get("files_used"),
            "records_requested": detail.get("records_requested"),
            "records_loaded": detail.get("records_loaded"),
        }
        return records, meta

    pretrain_file = (
        training_cfg.get("pretrain_file")
        or training_cfg.get("pretraining_file")
        or _env_lookup("PRETRAINING_DATA_FILE")
    )
    if pretrain_file:
        file_path = _resolve_path(pretrain_file)
        if not file_path:
            raise PretrainingLoadError(f"Configured pretraining file not found: {pretrain_file}")
        try:
            records = read_records_from_path(file_path, None)
        except PretrainingDataError as exc:
            raise PretrainingLoadError(str(exc)) from exc
        if not records:
            raise PretrainingLoadError(f"No usable pretraining records found in file: {file_path}")
        rng.shuffle(records)
        if record_limit is not None:
            records = records[:record_limit]
        meta = {
            "source": "pretrain_file",
            "file": str(file_path),
            "records_requested": record_limit or len(records),
            "records_loaded": len(records),
        }
        return records, meta

    raise PretrainingLoadError("Config does not specify a pretraining source.")
