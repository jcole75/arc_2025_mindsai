from __future__ import annotations

import copy
import math
import os
import random
from typing import Any, Iterable, Iterator

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - handled by caller
    load_dataset = None  # type: ignore

from .grid_utils import grid_to_string, is_grid, makeprompt, output_prefix
from .augmentation_framework import get_n_augs_flexible
from .transformations import (
    GEOM_FUNCTIONS,
    get_colormaps,
    get_symbols,
)


SUPPORTED_GEOMS = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hmirror",
    "vmirror",
    "dmirror",
    "cmirror",
]


class HuggingFacePretrainError(RuntimeError):
    """Raised when HuggingFace-backed pretraining data cannot be produced."""


def load_hf_pretraining_records(
    hf_cfg: dict[str, Any],
    *,
    record_limit: int,
    rng: random.Random | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build prompt/answer records from a HuggingFace dataset.

    Parameters
    ----------
    hf_cfg:
        Configuration dictionary describing dataset settings. Expected keys:
            - dataset_name (str)
            - split (str, default 'train')
            - arc_source ('arc_mindsai' | 'arc_json')
            - non_arc_fraction (float or percentage)
            - arc_mindsai / non_arc_mindsai / arc_json (dict overrides)
            - streaming / shuffle_buffer_size / seed / data_root convenience keys
    record_limit:
        Number of prompt/answer pairs to emit.
    rng:
        Random generator for deterministic shuffling/augmentation.
    verbose:
        Emit debug prints.
    """

    if load_dataset is None:  # pragma: no cover - import guarded earlier
        raise HuggingFacePretrainError("datasets package is not available. Install 'datasets' to enable HF loading.")

    if record_limit <= 0:
        raise HuggingFacePretrainError("record_limit must be > 0 for HuggingFace pretraining.")

    if not isinstance(hf_cfg, dict):
        raise HuggingFacePretrainError("hf_pretraining configuration must be a dictionary.")

    rng = rng or random.Random()
    dataset_name = str(hf_cfg.get("dataset_name") or "").strip()
    if not dataset_name:
        raise HuggingFacePretrainError("hf_pretraining.dataset_name is required.")

    split = str(hf_cfg.get("split") or "train")
    streaming = bool(_coerce_bool(hf_cfg.get("streaming"), default=True))
    shuffle_buffer = _coerce_int(hf_cfg.get("shuffle_buffer_size"))
    base_seed = _coerce_int(hf_cfg.get("seed"), allow_zero=True)
    data_root = hf_cfg.get("data_root")

    arc_source = str(hf_cfg.get("arc_source") or "arc_mindsai").strip().lower()
    if arc_source not in {"arc_mindsai", "arc_json"}:
        raise HuggingFacePretrainError(f"Unsupported arc_source: {arc_source}")

    non_arc_fraction = _coerce_percentage(hf_cfg.get("non_arc_fraction"), default=0.25)
    non_arc_count = int(math.floor(record_limit * non_arc_fraction))
    arc_count = max(0, record_limit - non_arc_count)

    # Partition-specific overrides
    arc_mindsai_cfg = _merge_partition_cfg(
        hf_cfg, "arc_mindsai", split, data_root, default_subdir="arc_mindsai"
    )
    arc_json_cfg = _merge_partition_cfg(hf_cfg, "arc_json", split, data_root, default_subdir="arc_json")
    non_arc_cfg = _merge_partition_cfg(
        hf_cfg, "non_arc_mindsai", split, data_root, default_subdir="non_arc_mindsai"
    )

    records: list[dict[str, Any]] = []
    partition_counts: dict[str, int] = {}

    if arc_count > 0:
        if arc_source == "arc_mindsai":
            arc_records = _sample_text_partition(
                {**arc_mindsai_cfg, "dataset_name": arc_mindsai_cfg.get("dataset_name") or dataset_name},
                arc_count,
                rng,
                streaming,
                shuffle_buffer,
                base_seed,
                verbose=verbose,
                partition_name="arc_mindsai",
            )
        else:
            arc_records = _sample_arc_json_partition(
                {**arc_json_cfg, "dataset_name": arc_json_cfg.get("dataset_name") or dataset_name},
                arc_count,
                rng,
                streaming,
                shuffle_buffer,
                base_seed,
                verbose=verbose,
            )
        records.extend(arc_records)
        partition_counts[arc_source] = len(arc_records)

    if non_arc_count > 0:
        non_arc_records = _sample_text_partition(
            {**non_arc_cfg, "dataset_name": non_arc_cfg.get("dataset_name") or dataset_name},
            non_arc_count,
            rng,
            streaming,
            shuffle_buffer,
            None if base_seed is None else base_seed + 17,
            verbose=verbose,
            partition_name="non_arc_mindsai",
        )
        records.extend(non_arc_records)
        partition_counts["non_arc_mindsai"] = len(non_arc_records)

    if len(records) < record_limit:
        raise HuggingFacePretrainError(
            f"Requested {record_limit} records but only produced {len(records)} from HuggingFace dataset."
        )

    rng.shuffle(records)
    return (
        records[:record_limit],
        {
            "source": "hf_pretraining",
            "dataset_name": dataset_name,
            "split": split,
            "arc_source": arc_source,
            "non_arc_fraction": non_arc_fraction,
            "records_requested": record_limit,
            "records_loaded": len(records),
            "partition_counts": partition_counts,
        },
    )


def _merge_partition_cfg(
    hf_cfg: dict[str, Any],
    key: str,
    split: str,
    data_root: str | None,
    *,
    default_subdir: str,
) -> dict[str, Any]:
    cfg = dict(hf_cfg.get(key) or {})
    cfg.setdefault("split", cfg.get("split") or split)
    cfg.setdefault("dataset_name", cfg.get("dataset_name") or hf_cfg.get("dataset_name"))
    if not cfg.get("data_dir"):
        root = data_root or f"data/{cfg['split']}"
        cfg["data_dir"] = os.path.join(root, default_subdir)
    cfg.setdefault("data_files", None)
    return cfg


def _sample_text_partition(
    part_cfg: dict[str, Any],
    count: int,
    rng: random.Random,
    streaming: bool,
    shuffle_buffer: int | None,
    seed: int | None,
    *,
    verbose: bool,
    partition_name: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rows: list[dict[str, Any]] = []
    iterator = _build_dataset_iterator(part_cfg, streaming, shuffle_buffer, seed)
    for row in iterator:
        prompt = row.get("prompt")
        answer = row.get("correct_answer")
        if prompt is None or answer is None:
            continue
        rows.append({"prompt": str(prompt), "correct_answer": str(answer)})
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise HuggingFacePretrainError(
            f"{partition_name} partition produced {len(rows)} records (requested {count}). "
            "Provide additional data_files or lower PRETRAINING_EXAMPLES."
        )
    if verbose:
        print(f"Loaded {len(rows)} records from HuggingFace partition '{partition_name}'.")
    rng.shuffle(rows)
    return rows[:count]


def _sample_arc_json_partition(
    part_cfg: dict[str, Any],
    count: int,
    rng: random.Random,
    streaming: bool,
    shuffle_buffer: int | None,
    seed: int | None,
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    arc_json_cfg = dict(part_cfg.get("arc_json_settings") or {})
    arc_json_cfg["apply_augs"] = bool(
        _coerce_bool(arc_json_cfg.get("apply_augs") or part_cfg.get("apply_augs"), default=False)
    )
    arc_json_cfg["variants_per_task"] = max(1, _coerce_int(arc_json_cfg.get("variants_per_task")) or 1)
    arc_json_cfg["enable_color_shift"] = bool(
        _coerce_bool(arc_json_cfg.get("enable_color_shift") or True, default=True)
    )
    arc_json_cfg["prompt_format"] = str(arc_json_cfg.get("prompt_format") or "legacy")

    iterator = _build_dataset_iterator(part_cfg, streaming, shuffle_buffer, seed)
    output: list[dict[str, Any]] = []
    formatter = ArcJsonFormatter(arc_json_cfg, rng)

    for raw in iterator:
        task = _normalize_arc_task(raw)
        if not task:
            continue
        records = formatter.task_to_records(task)
        if not records:
            continue
        output.extend(records)
        if len(output) >= count:
            break

    if len(output) < count:
        raise HuggingFacePretrainError(
            f"arc_json partition produced {len(output)} records (requested {count}). "
            "Ensure the dataset split contains enough tasks."
        )

    if verbose:
        print(f"Loaded {len(output)} records from HuggingFace partition 'arc_json'.")
    rng.shuffle(output)
    return output[:count]


def _build_dataset_iterator(
    cfg: dict[str, Any],
    streaming: bool,
    shuffle_buffer: int | None,
    seed: int | None,
) -> Iterator[dict[str, Any]]:
    builder = cfg.get("dataset_name")
    if not builder:
        raise HuggingFacePretrainError("dataset_name is required for each partition.")
    data_dir = cfg.get("data_dir")
    data_files = cfg.get("data_files")
    split = cfg.get("split") or "train"

    try:
        dataset = load_dataset(
            builder,
            split=split,
            data_dir=data_dir,
            data_files=data_files,
            streaming=streaming,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration
        raise HuggingFacePretrainError(
            f"Failed to load HuggingFace dataset '{builder}' (split={split}, data_dir={data_dir}): {exc}"
        ) from exc

    if streaming and shuffle_buffer:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    elif not streaming and shuffle_buffer:
        dataset = dataset.shuffle(seed=seed)

    return iter(dataset)


def _normalize_arc_task(raw: dict[str, Any]) -> dict[str, Any] | None:
    train = raw.get("train")
    test = raw.get("test")
    if not isinstance(train, Iterable) or not isinstance(test, Iterable):
        return None
    normalized = {
        "train": [ex for ex in train if _valid_example(ex)],
        "test": [ex for ex in test if _valid_example(ex)],
    }
    if not normalized["train"] or not normalized["test"]:
        return None
    for key in ("task_id", "metadata"):
        if key in raw:
            normalized[key] = copy.deepcopy(raw[key])
    return normalized


class ArcJsonFormatter:
    """Converts ARC JSON puzzles into prompt/answer records with optional augmentations."""

    def __init__(self, cfg: dict[str, Any], rng: random.Random):
        self.apply_augs = bool(cfg.get("apply_augs"))
        self.use_augmentation_framework = bool(cfg.get("use_augmentation_framework", True))
        self.prompt_format = str(cfg.get("prompt_format") or "legacy")
        self.variants_per_task = max(1, int(cfg.get("variants_per_task") or 1))
        self.enable_color_shift = bool(cfg.get("enable_color_shift", True))
        self.augmentation_use_case = str(cfg.get("augmentation_use_case") or "ttt")
        self.task_pool_size = max(0, int(cfg.get("task_pool_size") or 64))
        self.task_pool: list[dict[str, Any]] = []
        self.rng = rng

    def task_to_records(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        variants = self._generate_variants(task)
        records: list[dict[str, Any]] = []
        for variant in variants:
            train_examples = [ex for ex in variant.get("train", []) if _valid_example(ex)]
            if not train_examples:
                continue
            for idx, test_example in enumerate(variant.get("test", [])):
                if not _valid_example(test_example):
                    continue
                prompt_task = {"train": train_examples, "test": [{"input": test_example["input"]}]}
                prompt = makeprompt(prompt_task, style=self.prompt_format).rstrip() + " "
                target_grid = test_example.get("output")
                if not is_grid(target_grid):
                    continue
                target = " " + output_prefix(target_grid) + grid_to_string(target_grid) + "."
                record = {
                    "prompt": prompt,
                    "correct_answer": target,
                    "metadata": {
                        "source": "arc_json",
                        "variant_index": idx,
                        "apply_augs": self.apply_augs,
                    },
                }
                if task.get("task_id"):
                    record["metadata"]["task_id"] = task["task_id"]
                records.append(record)
        self._update_task_pool(task)
        return records

    def _generate_variants(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.apply_augs:
            return [task]
        variants: list[dict[str, Any]] = []
        if self.use_augmentation_framework:
            variants = self._augment_with_framework(task)
        if not variants:
            variants = self._generate_simple_variants(task)
        return variants or [task]

    def _augment_with_framework(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            augmented_tasks, _, _ = get_n_augs_flexible(
                task,
                max(1, self.variants_per_task),
                self.augmentation_use_case,
                task_pool=self.task_pool if self.task_pool else None,
            )
        except Exception:
            return []
        if not augmented_tasks:
            return []
        return list(augmented_tasks[: self.variants_per_task])

    def _generate_simple_variants(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = []
        for _ in range(self.variants_per_task):
            geom_name = self.rng.choice(SUPPORTED_GEOMS)
            geom_fn = GEOM_FUNCTIONS.get(geom_name)
            if geom_fn is None:
                continue
            color_map = None
            if self.enable_color_shift:
                symbols = get_symbols(task)
                cmap = get_colormaps(len(symbols), 1)[0]
                color_map = {idx: cmap[idx] for idx in range(len(cmap))}
            variants.append(_apply_transform(task, geom_fn, color_map))
        return variants

    def _update_task_pool(self, task: dict[str, Any]) -> None:
        if self.task_pool_size <= 0:
            return
        self.task_pool.append(copy.deepcopy(task))
        if len(self.task_pool) > self.task_pool_size:
            self.task_pool = self.task_pool[-self.task_pool_size:]


def _apply_transform(task: dict[str, Any], geom_fn, color_map: dict[int, int] | None) -> dict[str, Any]:
    transformed = {"train": [], "test": []}

    def _transform_grid(grid: list[list[int]]) -> list[list[int]]:
        out = grid
        if color_map:
            out = [[color_map.get(v, v) for v in row] for row in out]
        return geom_fn(out)

    for ex in task.get("train", []):
        if not _valid_example(ex):
            continue
        transformed["train"].append(
            {
                "input": _transform_grid(ex["input"]),
                "output": _transform_grid(ex["output"]),
            }
        )
    for ex in task.get("test", []):
        if not _valid_example(ex):
            continue
        transformed["test"].append(
            {
                "input": _transform_grid(ex["input"]),
                "output": _transform_grid(ex["output"]),
            }
        )
    for key in ("metadata", "task_id"):
        if key in task:
            transformed[key] = copy.deepcopy(task[key])
    return transformed


def _valid_example(ex: Any) -> bool:
    if not isinstance(ex, dict):
        return False
    inp = ex.get("input")
    out = ex.get("output")
    return is_grid(inp) and is_grid(out)


def _coerce_percentage(value: Any, *, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        val = float(value)
    except Exception:
        return default
    if val > 1.0:
        val = val / 100.0
    if val < 0:
        val = 0.0
    if val > 1:
        val = 1.0
    return val


def _coerce_int(value: Any, allow_zero: bool = False) -> int | None:
    if value in (None, ""):
        return None
    try:
        val = int(value)
    except Exception:
        return None
    if val < 0 or (val == 0 and not allow_zero):
        return None
    return val


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default
