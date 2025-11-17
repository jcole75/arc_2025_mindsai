"""
Utilities for loading pretraining records from files or directories.

Supports CSV/JSON files (optionally gzipped) that contain prompt-answer pairs
formatted with `prompt` and `correct_answer` columns. The helpers keep memory
usage bounded by only retaining the requested number of examples and shuffle
file order so a directory-backed corpus can be sampled randomly.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


__all__ = [
    "PretrainingDataError",
    "normalize_record",
    "read_records_from_path",
    "sample_directory_records",
]


class PretrainingDataError(RuntimeError):
    """Raised when pretraining data cannot be loaded from the requested source."""


# Supported suffixes (with and without compression) recognised for directory scans.
_SUPPORTED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".ndjson",
}
_SUPPORTED_COMPRESSED_SUFFIXES = {suffix + ".gz" for suffix in _SUPPORTED_SUFFIXES}


def _configure_csv_field_limit() -> None:
    env_limit = os.environ.get("PRETRAINING_CSV_FIELD_LIMIT")
    limit: int | None = None
    if env_limit not in (None, ""):
        try:
            limit = int(float(env_limit))
        except Exception:
            limit = None
    if limit is None or limit <= 0:
        # default to Python's max size guard but fall back if platform disallows it
        limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 2
        except Exception:
            # If other errors occur, stop attempting to adjust
            return


_configure_csv_field_limit()


def _is_supported_file(path: Path) -> bool:
    name = path.name.lower()
    if any(name.endswith(ext) for ext in _SUPPORTED_COMPRESSED_SUFFIXES):
        return True
    return any(name.endswith(ext) for ext in _SUPPORTED_SUFFIXES)


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Ensure a record exposes prompt/correct_answer fields and drop null rows."""
    prompt = record.get("prompt") or record.get("input") or record.get("question")
    answer = record.get("correct_answer") or record.get("target") or record.get("answer")
    if prompt is None or answer is None:
        return None
    cleaned = {
        "prompt": str(prompt),
        "correct_answer": str(answer),
    }
    for key, value in record.items():
        if key in ("prompt", "correct_answer"):
            continue
        cleaned[key] = value
    return cleaned


def _open_text_handle(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text_handle(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            norm = normalize_record(row)
            if norm is None:
                continue
            norm.setdefault("source_path", str(path))
            yield norm


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text_handle(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        iterable: Iterable[Any] = data.get("records") or data.values()
    else:
        iterable = data
    for item in iterable:
        if not isinstance(item, dict):
            continue
        norm = normalize_record(item)
        if norm is None:
            continue
        norm.setdefault("source_path", str(path))
        yield norm


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text_handle(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            norm = normalize_record(obj)
            if norm is None:
                continue
            norm.setdefault("source_path", str(path))
            yield norm


def _determine_format(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".csv", ".jsonl", ".ndjson", ".json"):
        if name.endswith(suffix) or name.endswith(suffix + ".gz"):
            return suffix
    raise PretrainingDataError(f"Unsupported pretraining file format: {path.suffix}")


def read_records_from_path(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read up to `limit` records from a pretraining data file (all records when limit is None)."""
    normalized_limit: int | None
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = int(limit)
        except Exception as exc:
            raise PretrainingDataError(f"Invalid limit for pretraining data: {limit!r}") from exc
        if normalized_limit <= 0:
            return []
    if not path.exists() or not path.is_file():
        raise PretrainingDataError(f"Pretraining file not found: {path}")

    format_hint = _determine_format(path)
    if format_hint == ".csv":
        iterator = _iter_csv_records(path)
    elif format_hint == ".json":
        iterator = _iter_json_records(path)
    else:  # .jsonl / .ndjson
        iterator = _iter_jsonl_records(path)

    records: list[dict[str, Any]] = []
    for record in iterator:
        records.append(record)
        if normalized_limit is not None and len(records) >= normalized_limit:
            break
    return records


def _collect_directory_files(directory: Path) -> list[Path]:
    candidates = [path for path in directory.iterdir() if path.is_file() and _is_supported_file(path)]
    return sorted(candidates)


def sample_directory_records(
    directory: Path,
    limit: int | None = None,
    *,
    rng: random.Random | None = None,
    shuffle_files: bool = True,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Sample records from a directory containing CSV/JSON pretraining data files.

    Files are picked in random order (when `shuffle_files` is True) and read
    just until the requested number of records is gathered. Results are
    shuffled before returning so consecutive calls with the same seed still
    produce varied rows.
    """
    normalized_limit: int | None
    if limit is None:
        normalized_limit = None
    else:
        try:
            normalized_limit = int(limit)
        except Exception as exc:
            raise PretrainingDataError(f"Invalid directory sampling limit: {limit!r}") from exc
        if normalized_limit < 0:
            raise PretrainingDataError(f"Directory sampling limit must be >= 0 (received {limit!r})")
        if normalized_limit == 0:
            return [], {
                "directory": str(directory),
                "files_considered": 0,
                "files_used": [],
                "records_requested": 0,
                "records_loaded": 0,
            }
    if not directory.exists() or not directory.is_dir():
        raise PretrainingDataError(f"Pretraining directory does not exist: {directory}")

    files = _collect_directory_files(directory)
    if not files:
        raise PretrainingDataError(f"No supported pretraining files found in {directory}")

    prng = rng or random.Random()
    if shuffle_files:
        prng.shuffle(files)

    records: list[dict[str, Any]] = []
    files_used: list[str] = []
    file_record_counts: list[tuple[str, int]] = []

    for file_path in files:
        needed = None if normalized_limit is None else normalized_limit - len(records)
        if needed is not None and needed <= 0:
            break
        if verbose:
            if needed is None:
                print(f"   → Reading {file_path} (no limit)")
            else:
                print(f"   → Reading {file_path} (need {needed} records)")
        chunk = read_records_from_path(file_path, needed)
        chunk_count = len(chunk)
        if not chunk:
            continue
        prng.shuffle(chunk)
        records.extend(chunk)
        files_used.append(str(file_path))
        file_record_counts.append((str(file_path), chunk_count))
        if verbose:
            print(f"     Loaded {chunk_count} record(s) from {file_path}")

    if not records:
        raise PretrainingDataError(f"No usable pretraining records found in directory: {directory}")

    prng.shuffle(records)
    if normalized_limit is not None:
        records = records[:normalized_limit]

    meta = {
        "directory": str(directory),
        "files_considered": len(files),
        "files_used": files_used,
        "records_requested": normalized_limit if normalized_limit is not None else len(records),
        "records_loaded": len(records),
        "files_detail": [
            {"path": path, "records": count}
            for path, count in file_record_counts
        ],
    }
    return records, meta
