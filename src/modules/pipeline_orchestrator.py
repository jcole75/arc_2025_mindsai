"""Pipeline orchestration helpers extracted from `src/main.py`."""

from __future__ import annotations

from contextlib import suppress
import csv
from dataclasses import dataclass, field
import gc
import glob
import json
import math
import os
from pathlib import Path
import random
import shutil
import re
import sys
import time
import traceback
from typing import Any

import numpy as np

from . import config
from .confidence_filter import apply_confidence_filter
from .config import RunLogger, get_ensemble_settings, get_nested_setting
from .config import env as env_mod
from .config import run_logging as _run_logging
from .data_loader import load_data
from .dataset_generator import make_datasets, prep_ttt_dataset
from .distributed_executor import predict_distributed, train_distributed
from .grid_utils import grid_to_string, is_valid_prediction, output_to_grid
from .pretraining_loader import PretrainingLoadError, load_pretraining_records
from .scoring import get_solutions_path, score_if_solutions_available
from .session_logger import finalize_session_logger, get_session_logger, initialize_session_logger
from .submission_handler import make_submission
from .transformations import apply_decoder_description


def ensure_prompt_prefix(prompt: str | None, prefix: str) -> str:
    text = "" if prompt is None else str(prompt)
    prefix = str(prefix or "")
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text


SCALING_CSV_COLUMNS: list[str] = []


@dataclass
class PhaseContext:
    phase_name: str = ""


class ScalingMetricsCollector:
    def __init__(self, *_, **__):
        self.enabled = False

    def log_phase(self, *_, **__):
        return None

    def finalize(self, *_, **__):
        return None

    def finalize_phase(self, *_, **__):
        return None

    def print_csv_to_logs(self):
        return None


def resolve_task_log_dir() -> Path:
    return Path(".")


def get_scaling_settings(*_, **__):
    return {}


def _evaluate_ood_panel(*_, **__):
    return None


def build_ttft_task_rows(*_, **__):
    return None


def collect_device_summary(*_, **__):
    return {}


def collect_repo_state(*_, **__):
    return {}


def export_scaling_artifacts(*_, **__):
    return None


def load_ood_panel(*_, **__):
    return None


def run_scaling_inference_phase(*_, **__):
    return None


def _is_scaling_enabled() -> bool:
    return bool(getattr(config, "SCALING_ENABLED", False))


try:
    from .utils import (
        average_model_weights,
        cleanup_directories,
        cleanup_ensemble_models,
        debug_print,
        display_working_directory_contents,
        get_model_ensemble_paths,
        setup_kaggle_environment,
        setup_multiprocessing,
        validate_models_for_averaging,
    )
except ImportError:
    from .utils import (
        average_model_weights,
        cleanup_directories,
        cleanup_ensemble_models,
        display_working_directory_contents,
        get_model_ensemble_paths,
        setup_kaggle_environment,
        setup_multiprocessing,
        validate_models_for_averaging,
    )

    def debug_print(*args, **kwargs):
        if getattr(config, "DEBUG_MODE", False) or getattr(config, "VERBOSE_LOGGING", False):
            print(*args, **kwargs)


try:  # pragma: no cover - torch optional
    import torch  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when torch missing
    torch = None

def _find_real_task_dir_for_pretrained():
    """Reuse shared resolver so pretrain logic stays consistent."""
    return resolve_task_log_dir()


# ======================================================================================
# Small utilities
# ======================================================================================
MODEL_FILE_PATTERNS = [
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model-*-of-*.bin",
    "pytorch_model.safetensors",
    "pytorch_model-*.safetensors",
    "model.safetensors",
    "model-*.safetensors",
    "model.bin",
    "flax_model.msgpack",
]


def _safe_gpu_clear() -> None:
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_mini_lr_trial_table(trial_records: list[dict[str, Any]], metric_keys: list[str]) -> None:
    if not trial_records:
        return
    headers = ["trial", "learning_rate", "metric", "status", "origin"]
    rows: list[dict[str, str]] = []
    metric_name = metric_keys[0] if metric_keys else "metric"
    for record in trial_records:
        lr_display = f"{record['learning_rate']:.6g}"
        metric_value = record.get("metric")
        metric_source = record.get("metric_source")
        metric_details = record.get("metric_details")
        if not isinstance(metric_value, (int, float)) and isinstance(metric_details, dict):
            for key in metric_keys:
                if key in metric_details:
                    metric_value = metric_details[key]
                    if not metric_source:
                        metric_source = key
                    break
        if isinstance(metric_value, (int, float)):
            metric_display = f"{metric_value:.6g}"
            if metric_source and metric_keys and metric_source != metric_keys[0]:
                metric_display += f" ({metric_source})"
        else:
            metric_display = "-"
        rows.append(
            {
                "trial": str(record.get("sequence", "?")),
                "learning_rate": lr_display,
                "metric": metric_display,
                "status": record.get("status", "-"),
                "origin": record.get("origin", "-") or "-",
            }
        )
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))
    print("📊 Mini LR trial records:")
    header_line = "   " + "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("   " + "  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("   " + "  ".join(row[header].ljust(widths[header]) for header in headers))
    for record in trial_records:
        extra_parts: list[str] = []
        details = record.get("metric_details")
        if isinstance(details, dict) and details:
            detail_bits: list[str] = []
            seen_keys: set[str] = set()
            for key in metric_keys:
                if key in details and key not in seen_keys:
                    detail_bits.append(f"{key}={details[key]:.6g}")
                    seen_keys.add(key)
            for key, value in details.items():
                if key in seen_keys:
                    continue
                if isinstance(value, (int, float)):
                    detail_bits.append(f"{key}={value:.6g}")
            if detail_bits:
                extra_parts.append("metrics: " + ", ".join(detail_bits))
        if not isinstance(record.get("metric"), (int, float)):
            available = record.get("metric_keys_available")
            if isinstance(available, list) and available:
                preview = ", ".join(str(item) for item in available[:8])
                if len(available) > 8:
                    preview += ", …"
                extra_parts.append(f"available keys: {preview}")
        for part in extra_parts:
            print(f"   ↳ trial {record.get('sequence', '?')}: {part}")
    print(f"   metric column uses: {metric_name}")


def _maybe_generate_mini_lr_chart(
    trial_records: list[dict[str, Any]],
    best_info: dict[str, Any] | None,
    interpolation_summary: dict[str, Any] | None,
    loess_summary: dict[str, Any] | None,
    metric_keys: list[str],
) -> Path | None:
    successful = [
        record
        for record in trial_records
        if record.get("status") == "success" and isinstance(record.get("metric"), (int, float))
    ]
    if not successful:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"⚠️  Mini LR chart skipped (matplotlib unavailable: {exc})", file=sys.stderr)
        return None
    metric_name = metric_keys[0] if metric_keys else "metric"
    lrs = np.array([float(entry["learning_rate"]) for entry in successful], dtype=float)
    metrics = np.array([float(entry["metric"]) for entry in successful], dtype=float)
    log_lrs = np.log10(lrs)
    jitter_log_lrs = np.array(log_lrs, copy=True)
    jitter_step = 0.012
    rounded = np.round(log_lrs, decimals=12)
    for unique_value in np.unique(rounded):
        idxs = np.flatnonzero(rounded == unique_value)
        if len(idxs) <= 1:
            continue
        offsets = jitter_step * (np.arange(len(idxs)) - (len(idxs) - 1) / 2.0)
        jitter_log_lrs[idxs] += offsets
    jittered_lrs = np.power(10.0, jitter_log_lrs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(jittered_lrs, metrics, color="#1f77b4", label="Trials", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel(metric_name)
    ax.set_title("Mini LR Grid Search")
    best_lr = float(best_info["learning_rate"]) if best_info and best_info.get("learning_rate") else None
    best_metric = (
        float(best_info["metric_value"]) if best_info and best_info.get("metric_value") is not None else None
    )
    if best_lr and best_metric is not None:
        ax.scatter([best_lr], [best_metric], color="#d62728", marker="*", s=160, label="Selected LR", zorder=5)
        ax.annotate(
            f"{best_lr:.2e}",
            xy=(best_lr, best_metric),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )
    elif best_lr:
        ax.axvline(best_lr, color="#d62728", linestyle="--", linewidth=1.2, label="Selected LR", zorder=4)
    if interpolation_summary and interpolation_summary.get("coefficients"):
        coeffs = interpolation_summary["coefficients"]
        try:
            a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        except Exception:
            a = b = c = None
        if a is not None:
            lr_min = float(np.min(lrs))
            lr_max = float(np.max(lrs))
            log_lr_space = np.linspace(np.log10(lr_min), np.log10(lr_max), num=200)
            lr_space = np.power(10.0, log_lr_space)
            curve = a * log_lr_space**2 + b * log_lr_space + c
            ax.plot(lr_space, curve, color="#ff7f0e", linewidth=1.5, label="Quadratic fit", zorder=2)
            cov = interpolation_summary.get("coefficient_covariance")
            if cov is not None:
                try:
                    cov_matrix = np.asarray(cov, dtype=float).reshape(3, 3)
                    design = np.stack((log_lr_space**2, log_lr_space, np.ones_like(log_lr_space)), axis=1)
                    variances = np.einsum("ij,jk,ik->i", design, cov_matrix, design)
                    std = np.sqrt(np.clip(variances, a_min=0.0, a_max=None))
                    ax.fill_between(
                        lr_space,
                        curve - 2 * std,
                        curve + 2 * std,
                        color="#ff7f0e",
                        alpha=0.16,
                        linewidth=0,
                        label="Quadratic fit ±2σ",
                        zorder=1.5,
                    )
                except Exception:
                    pass
            predicted_lr = interpolation_summary.get("verification_learning_rate") or interpolation_summary.get(
                "predicted_learning_rate"
            )
            predicted_metric = interpolation_summary.get("verification_metric") or interpolation_summary.get("predicted_metric")
            if (
                predicted_metric is None
                and predicted_lr
                and best_lr
                and abs(predicted_lr - best_lr) / max(predicted_lr, 1e-12) < 1e-6
                and best_metric is not None
            ):
                predicted_metric = best_metric
            if predicted_lr and predicted_metric is not None:
                ax.scatter(
                    [predicted_lr],
                    [predicted_metric],
                    color="#2ca02c",
                    marker="P",
                    s=120,
                    label="Interpolation verification",
                    zorder=6,
                )
            elif predicted_lr:
                ax.axvline(
                    predicted_lr,
                    color="#2ca02c",
                    linestyle=":",
                    linewidth=1.2,
                    label="Interpolation prediction",
                    zorder=4,
                )
            if predicted_lr and predicted_metric is not None:
                ax.annotate(
                    f"{predicted_lr:.2e}",
                    xy=(predicted_lr, predicted_metric),
                    xytext=(-12, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color="#2ca02c",
                    ha="right",
                )
    if loess_summary and loess_summary.get("log10_lrs") and loess_summary.get("metrics"):
        try:
            loess_log = np.asarray(loess_summary["log10_lrs"], dtype=float)
            loess_metrics = np.asarray(loess_summary["metrics"], dtype=float)
            loess_lrs = np.power(10.0, loess_log)
            ax.plot(loess_lrs, loess_metrics, color="#9467bd", linewidth=1.4, label="LOWESS fit", zorder=2.2)
            loess_lr_star = float(loess_summary.get("predicted_learning_rate") or np.nan)
            loess_metric_star = float(loess_summary.get("predicted_metric") or np.nan)
            if np.isfinite(loess_lr_star) and np.isfinite(loess_metric_star):
                ax.scatter(
                    [loess_lr_star],
                    [loess_metric_star],
                    color="#9467bd",
                    marker="X",
                    s=90,
                    label="LOWESS min",
                    zorder=5.5,
                )
                ax.annotate(
                    f"{loess_lr_star:.2e}",
                    xy=(loess_lr_star, loess_metric_star),
                    xytext=(-6, -6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#9467bd",
                    ha="right",
                    va="top",
                )
        except Exception:
            pass
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best")
    out_dir = Path("artifacts") / "mini_lr_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metric_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in metric_name)
    figure_path = out_dir / f"mini_lr_grid_{metric_safe}_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _directory_has_model_weights(directory: str | None) -> bool:
    if not directory or not os.path.isdir(directory):
        return False
    return any(glob.glob(os.path.join(directory, pattern)) for pattern in MODEL_FILE_PATTERNS)


def _estimate_global_batch(per_device_bs: int, grad_accum: int) -> int:
    world_size = 1
    try:
        env_ws = os.environ.get("WORLD_SIZE")
        if env_ws:
            world_size = max(1, int(env_ws))
        elif torch and torch.cuda.is_available():
            detected = torch.cuda.device_count()
            if detected:
                world_size = detected
    except Exception:  # pragma: no cover - defensive
        pass
    per_device_bs = max(1, int(per_device_bs or 1))
    grad_accum = max(1, int(grad_accum or 1))
    return max(1, per_device_bs * grad_accum * world_size)


def _apply_pretraining_step_budget(args: dict, dataset_len: int, explicit_steps: int | None) -> int | str:
    if explicit_steps == -1:
        args["max_steps"] = -1
        return "unlimited"
    if isinstance(explicit_steps, int) and explicit_steps > 0:
        args["max_steps"] = int(explicit_steps)
        return int(explicit_steps)
    if dataset_len <= 0:
        return 0
    per_device = int(args.get("per_device_train_batch_size", 1) or 1)
    grad_accum = int(args.get("gradient_accumulation_steps", 1) or 1)
    global_batch = _estimate_global_batch(per_device, grad_accum)
    auto_steps = max(1, math.ceil(dataset_len / global_batch))
    args["max_steps"] = auto_steps
    return auto_steps


def _determine_save_steps(dataset_len: int, args: dict[str, Any], fallback: int | None = None) -> int:
    env_override = getattr(config, "SAVE_STEPS", None)
    base: int | None = None
    for candidate in (env_override, fallback):
        if candidate is None:
            continue
        try:
            candidate_int = int(candidate)
        except Exception:
            continue
        if candidate_int <= 0:
            continue
        base = max(1, candidate_int)
        break

    per_device = int(args.get("per_device_train_batch_size", 1) or 1)
    grad_accum = int(args.get("gradient_accumulation_steps", 1) or 1)
    global_batch = max(1, _estimate_global_batch(per_device, grad_accum))
    epochs_raw = args.get("num_train_epochs", 1)
    try:
        epochs = max(1, math.ceil(float(epochs_raw)))
    except Exception:
        epochs = 1
    total_updates = 1
    if dataset_len > 0:
        total_updates = max(1, math.ceil(dataset_len / global_batch) * epochs)

    if base is None:
        base = 1 if total_updates <= 1 else max(1, total_updates // 2)
    else:
        base = 1 if total_updates <= 1 else min(base, max(1, total_updates - 1))

    return max(1, base)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _env_lookup(name: str) -> Any:
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return env_mod._lookup_env(name)
    except Exception:
        return None


def _coerce_int(value: Any, *, allow_zero: bool = False) -> int | None:
    if value is None or value == "":
        return None
    try:
        coerced = int(value)
    except Exception:
        return None
    if coerced < 0:
        return None
    if coerced == 0 and not allow_zero:
        return None
    return coerced


def _pretraining_requested(training_cfg: dict[str, Any] | None) -> bool:
    if not isinstance(training_cfg, dict):
        return False
    if training_cfg.get("pretrain_dir") or _env_lookup("PRETRAINING_DATA_DIR"):
        return True
    if (
        training_cfg.get("pretrain_file")
        or training_cfg.get("pretraining_file")
        or _env_lookup("PRETRAINING_DATA_FILE")
    ):
        return True
    return False


def _resolve_pretraining_steps(training_cfg: dict[str, Any]) -> int | None:
    candidates = (
        training_cfg.get("pretraining_steps"),
        os.environ.get("PRETRAINING_STEPS"),
        env_mod.PRETRAINING_STEPS_OVERRIDE,
    )
    for candidate in candidates:
        steps = env_mod.parse_pretraining_steps(candidate)
        if steps is not None:
            return steps
    return None


def _resolve_pretraining_record_limit(training_cfg: dict[str, Any], explicit_steps: int | None) -> int | None:
    candidates = (
        training_cfg.get("pretraining_examples"),
        _env_lookup("PRETRAINING_EXAMPLES"),
        training_cfg.get("pretraining_record_limit"),
        _env_lookup("PRETRAINING_RECORD_LIMIT"),
        env_mod.PRETRAINING_EXAMPLES_OVERRIDE,
    )
    for candidate in candidates:
        limit = env_mod.parse_pretraining_examples(candidate)
        if limit is not None:
            return limit
    return explicit_steps


def _resolve_pretraining_seed(training_cfg: dict[str, Any]) -> int | None:
    for candidate in (training_cfg.get("pretraining_seed"), os.environ.get("PRETRAINING_SEED")):
        seed = _coerce_int(candidate, allow_zero=True)
        if seed is not None:
            return seed
    return None


def _normalize_power_sampling_config(model_settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(model_settings, dict):
        return None
    training_block = model_settings.get("training")
    if not isinstance(training_block, dict):
        return None
    cfg = training_block.get("power_sampling")
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    if _truthy(cfg):
        return {"enabled": True}
    return None


def _resolve_pretrained_output_dir(base_dir: Path, model_basename: str, training_cfg: dict[str, Any]) -> Path:
    flat_flag = training_cfg.get("pretraining_flat_dir")
    if flat_flag is None:
        flat_flag = os.environ.get("PRETRAINING_FLAT_DIR")
    if _truthy(flat_flag):
        return base_dir

    subdir_override = training_cfg.get("pretraining_subdir")
    if isinstance(subdir_override, str) and subdir_override.strip():
        return base_dir / subdir_override.strip()

    return base_dir / model_basename


def _build_pretraining_dataset(records: list[dict[str, Any]]):
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PretrainingLoadError("The 'datasets' package is required for pretraining but is not installed.") from exc

    prompts: list[str] = []
    answers: list[str] = []
    for record in records:
        prompt = record.get("prompt")
        answer = record.get("correct_answer")
        if prompt is None or answer is None:
            continue
        prompts.append(str(prompt))
        answers.append(str(answer))
    if not prompts:
        raise PretrainingLoadError("No valid prompt/correct_answer pairs found in pretraining data.")
    return Dataset.from_dict({"prompt": prompts, "correct_answer": answers})


def _resolve_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _get_setting(model_settings: dict, model_path: str, name: str, section: str | None = None, default=None):
    if section and section in model_settings and name in model_settings[section]:
        return _resolve_scalar(model_settings[section][name])
    if name in model_settings:
        return _resolve_scalar(model_settings[name])
    return _resolve_scalar(get_nested_setting(name, section, default, model_path))


def _select_mask_token_str(tokenizer) -> str:
    try:
        if hasattr(tokenizer, "mask_token") and tokenizer.mask_token:
            return str(tokenizer.mask_token)
        vocab = {}
        try:
            vocab = tokenizer.get_vocab() or {}
        except Exception:  # pragma: no cover - tokenizer quirks
            vocab = {}
        if "<mask>" in vocab:
            return "<mask>"
        if "<extra_id_0>" in vocab:
            return "<extra_id_0>"
    except Exception:  # pragma: no cover
        pass
    return "[MASK]"



def build_inference_settings(model_settings: dict, model_path: str, airv_enabled: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    settings.update(_collect_base_inference_settings(model_settings, model_path, airv_enabled))
    settings.update(_collect_preview_settings(model_settings, model_path))
    prompt_prefix = _get_prompt_prefix_for_model(model_path)
    general_settings = model_settings.get("general", {}) if isinstance(model_settings, dict) else {}
    prompt_format = str((general_settings or {}).get("prompt_format") or "legacy")
    settings["prompt_prefix"] = prompt_prefix
    settings["prompt_general_prefix"] = prompt_prefix
    settings["prompt_refine_prefix"] = prompt_prefix
    settings["prompt_format"] = prompt_format
    solutions_path = get_solutions_path(config.DATA_PATH)
    if solutions_path:
        settings["solutions_path"] = solutions_path
    return settings


def _collect_base_inference_settings(model_settings: dict, model_path: str, airv_enabled: bool) -> dict[str, Any]:
    base_settings: dict[str, Any] = {}
    inference_keys = [
        "eval_batch_size",
        "num_beams",
        "num_return_sequences",
        "top_k",
        "top_p",
        "temperature",
        "max_generation_length",
        "use_mixed_precision_inference",
        "generation_mode",
        "diffusion_steps",
        "diffusion_eps",
        "diffusion_alg",
        "diffusion_alg_temp",
        "coda_quality_profile",
        "coda_quality_presets",
    ]
    for key in inference_keys:
        value = _get_setting(model_settings, model_path, key, "inference")
        if value is not None:
            base_settings[key] = value

    max_input_len = _get_setting(model_settings, model_path, "max_input_length", "general")
    if max_input_len is not None:
        base_settings["max_input_length"] = max_input_len
    max_length = _get_setting(model_settings, model_path, "max_length", "general")
    if max_length is not None:
        base_settings["max_length"] = max_length
    coda_enabled = _get_setting(model_settings, model_path, "coda_enabled", "general")
    if coda_enabled is not None:
        base_settings["coda_enabled"] = coda_enabled
    dream_enabled = _get_setting(model_settings, model_path, "dream_enabled", "general")
    if dream_enabled is not None:
        base_settings["dream_enabled"] = dream_enabled

    base_settings.update(
        {
            "use_torch_compile": _get_setting(model_settings, model_path, "use_torch_compile", "training", False),
            "expand_factor": config.EXPAND_FACTOR if airv_enabled else 0.0,
            "prune_factor": config.PRUNE_FACTOR if airv_enabled else 0.0,
        }
    )
    return base_settings


def _collect_preview_settings(model_settings: dict, model_path: str) -> dict[str, Any]:
    preview_settings: dict[str, Any] = {}
    preview_keys = [
        "preview_interval_batches",
        "preview_max_samples",
        "preview_include_prompt_snippet",
        "preview_show_terminal_grid",
        "preview_snippet_chars",
        "preview_prompt_chars",
        "preview_all_ranks",
    ]
    for key in preview_keys:
        value = _get_setting(model_settings, model_path, key, "inference")
        if value is not None:
            preview_settings[key] = value
    return preview_settings


def log_empty_output_sample(task_key, aug_idx, text_idx, model_name, mp_inference, raw_text, prompt, context=""):
    try:
        logger = get_session_logger()
        if logger is None:
            return
        logger.log_empty_output_sample(
            task_id=task_key,
            augmentation_index=aug_idx,
            text_index=text_idx,
            model_name=model_name,
            used_mixed_precision=mp_inference,
            raw_text=raw_text,
            prompt_snippet=prompt[:500],
            context=context,
        )
    except Exception as exc:
        print(f"⚠️ Failed to log empty sample: {exc}")


def update_augmentation_stats(global_stats: dict, dataset_stats: dict) -> None:
    if not dataset_stats:
        return
    global_stats["total_augmentations_generated"] += dataset_stats.get("total_generated", 0)
    global_stats["successful_augmentations"] += dataset_stats.get("successful", 0)
    global_stats["failed_augmentations"] += dataset_stats.get("failed", 0)
    for k, v in dataset_stats.get("by_type", {}).items():
        global_stats["augmentation_type_breakdown"][k] = global_stats["augmentation_type_breakdown"].get(k, 0) + v


def _rebuild_aggregated_predictions(
    per_model_store: dict,
    model_ensemble_flags: list,
    upto_model_idx: int,
    all_task_keys: list,
) -> dict:
    agg = {k: [] for k in all_task_keys}
    for mi in range(0, upto_model_idx + 1):
        if mi not in per_model_store:
            continue
        include = model_ensemble_flags[mi] if mi < len(model_ensemble_flags) else True
        if not include and upto_model_idx > 0:
            continue
        for tk, grids in per_model_store[mi].items():
            if tk in agg and grids:
                agg[tk].extend(grids)
    return agg


def postprocess_predictions(
    raw_pred_complex: dict[str, Any],
    counters: dict[str, Any],
    ctx: PostprocessContext,
) -> tuple[dict[str, int], dict[str, Any], dict[str, list[Any]]]:
    state = _init_postprocess_state(counters, ctx)
    for task_key, aug_list in raw_pred_complex.items():
        ctx.aggregated_store.setdefault(task_key, [])
        _process_task_predictions(task_key, aug_list, counters, ctx, state)
    return state.run_delta, state.empty_delta, state.run_contrib


def _init_postprocess_state(counters: dict[str, Any], ctx: PostprocessContext) -> PostprocessState:
    run_delta = {"raw_outputs": 0, "parsed_outputs": 0, "valid_outputs": 0}
    empty_delta = {
        "total_empty_outputs": 0,
        "zero_length_outputs": 0,
        "parsing_failed_outputs": 0,
        "mixed_precision_correlation": {
            "with_mp": 0,
            "without_mp": 0,
            "empty_with_mp": 0,
            "empty_without_mp": 0,
        },
        "by_task": {},
    }
    tgt_run = 0 if ctx.averaged_mode else ctx.run_idx
    model_runs = ctx.per_run_store.setdefault(ctx.model_idx, {})
    if tgt_run not in model_runs:
        model_runs[tgt_run] = {k: [] for k in ctx.aggregated_store}
    else:
        for key in ctx.aggregated_store:
            model_runs[tgt_run].setdefault(key, [])

    model_store = ctx.per_model_store.setdefault(ctx.model_idx, {})
    for key in ctx.aggregated_store:
        model_store.setdefault(key, [])

    run_contrib = {k: [] for k in ctx.aggregated_store}
    use_mp = _get_setting(ctx.model_settings, "", "use_mixed_precision_inference", "inference", True)
    return PostprocessState(run_delta, empty_delta, run_contrib, tgt_run, use_mp)


def _process_task_predictions(
    task_key: str,
    aug_list: list[dict[str, Any]],
    counters: dict[str, Any],
    ctx: PostprocessContext,
    state: PostprocessState,
) -> None:
    final_grids: list[Any] = []
    for aug_idx, aug_res in enumerate(aug_list):
        final_grids.extend(
            _process_augmentation(task_key, aug_idx, aug_res, counters, ctx, state)
        )
    _store_final_task_grids(task_key, final_grids, ctx, state)


def _process_augmentation(
    task_key: str,
    aug_idx: int,
    aug_res: dict[str, Any],
    counters: dict[str, Any],
    ctx: PostprocessContext,
    state: PostprocessState,
) -> list[Any]:
    texts = aug_res.get("texts", [])
    decoder = aug_res.get("decoder", {})
    prompt = aug_res.get("prompt", "PROMPT_NOT_AVAILABLE")
    final_grids: list[Any] = []
    for text_idx, txt in enumerate(texts):
        final_grids.extend(
            _process_single_prediction(
                task_key,
                aug_idx,
                text_idx,
                txt,
                prompt,
                decoder,
                counters,
                ctx,
                state,
            )
        )
    return final_grids


def _process_single_prediction(
    task_key: str,
    aug_idx: int,
    text_idx: int,
    txt: str,
    prompt: str,
    decoder: dict[str, Any],
    counters: dict[str, Any],
    ctx: PostprocessContext,
    state: PostprocessState,
) -> list[Any]:
    _increment_raw_counts(state, counters)
    zero_len = len(txt.strip()) == 0
    if zero_len:
        state.empty_delta["zero_length_outputs"] += 1

    parsed = output_to_grid(txt)
    if parsed is None:
        _handle_parse_failure(task_key, aug_idx, text_idx, txt, prompt, zero_len, ctx, state, counters)
        return []

    counters["total_parsed"] += 1
    state.run_delta["parsed_outputs"] += 1

    mixc_dec, post_split_decoder = _extract_mixup_chain(decoder)
    if mixc_dec is not None:
        _process_mixup_prediction(
            task_key,
            parsed,
            mixc_dec,
            post_split_decoder,
            counters,
            ctx,
            state,
        )
        return []

    combine_dec, combine_tail = _extract_combine_chain(decoder)
    if combine_dec is not None:
        combine_results = _process_combine_prediction(
            task_key,
            parsed,
            combine_dec,
            combine_tail,
            counters,
            state,
        )
        if combine_results is not None:
            return combine_results

    final_grid = apply_decoder_description(parsed, decoder)
    if not is_valid_prediction(final_grid):
        if config.DEBUG_MODE or config.VERBOSE_LOGGING:
            debug_print(f"Invalid decoded for {task_key} (aug {aug_idx}, text {text_idx}).")
        return []

    _record_valid_prediction(counters, state)
    return [final_grid]


def _increment_raw_counts(state: PostprocessState, counters: dict[str, Any]) -> None:
    counters["total_raw"] += 1
    state.run_delta["raw_outputs"] += 1
    mp_key = "with_mp" if state.use_mp else "without_mp"
    state.empty_delta["mixed_precision_correlation"][mp_key] += 1


def _handle_parse_failure(
    task_key: str,
    aug_idx: int,
    text_idx: int,
    txt: str,
    prompt: str,
    zero_len: bool,
    ctx: PostprocessContext,
    state: PostprocessState,
    counters: dict[str, Any],
) -> None:
    state.empty_delta["total_empty_outputs"] += 1
    if not zero_len:
        state.empty_delta["parsing_failed_outputs"] += 1
    state.empty_delta["by_task"][task_key] = state.empty_delta["by_task"].get(task_key, 0) + 1
    empty_key = "empty_with_mp" if state.use_mp else "empty_without_mp"
    state.empty_delta["mixed_precision_correlation"][empty_key] += 1

    if counters.get("empty_samples", 0) < counters.get("max_empty_samples", 0):
        tag = "(AVERAGED)" if ctx.averaged_mode else ""
        log_empty_output_sample(task_key, aug_idx, text_idx, ctx.model_basename, state.use_mp, txt, prompt, context=tag)
        counters["empty_samples"] = counters.get("empty_samples", 0) + 1

    if config.DEBUG_MODE or config.VERBOSE_LOGGING:
        debug_print(
            f"output_to_grid None for task {task_key} (aug {aug_idx}, text {text_idx}). "
            f"Prompt: {prompt[:150]}... Raw: {txt[:150]}..."
        )


def _process_mixup_prediction(
    task_key: str,
    parsed: Any,
    mixc_dec: dict[str, Any],
    post_split_decoder: dict | None,
    counters: dict[str, Any],
    ctx: PostprocessContext,
    state: PostprocessState,
) -> None:
    decoded_entries = _decode_mixup_predictions(parsed, mixc_dec, post_split_decoder)
    if not decoded_entries:
        return
    for dest_key, grid in decoded_entries:
        if not is_valid_prediction(grid):
            continue
        _record_valid_prediction(counters, state)
        target_key = dest_key or task_key
        _store_grid_for_task(target_key, grid, ctx, state)


def _record_valid_prediction(counters: dict[str, Any], state: PostprocessState) -> None:
    counters["total_valid"] += 1
    state.run_delta["valid_outputs"] += 1


def _store_grid_for_task(task_key: str, grid: Any, ctx: PostprocessContext, state: PostprocessState) -> None:
    run_store = ctx.per_run_store.setdefault(ctx.model_idx, {}).setdefault(state.tgt_run, {})
    run_store.setdefault(task_key, []).append(grid)

    model_store = ctx.per_model_store.setdefault(ctx.model_idx, {})
    model_store.setdefault(task_key, [])
    model_store[task_key].append(grid)

    if ctx.enable_model_ensemble:
        ctx.aggregated_store.setdefault(task_key, []).append(grid)

    state.run_contrib.setdefault(task_key, []).append(grid)


def _store_final_task_grids(
    task_key: str,
    final_grids: list[Any],
    ctx: PostprocessContext,
    state: PostprocessState,
) -> None:
    if not final_grids:
        return

    run_store = ctx.per_run_store[ctx.model_idx][state.tgt_run]
    run_store.setdefault(task_key, []).extend(final_grids)

    model_store = ctx.per_model_store[ctx.model_idx]
    if ctx.enable_self_ensemble:
        model_store.setdefault(task_key, []).extend(final_grids)
    else:
        model_store[task_key] = list(final_grids)

    if ctx.enable_model_ensemble:
        ctx.aggregated_store.setdefault(task_key, []).extend(final_grids)

    state.run_contrib.setdefault(task_key, []).extend(final_grids)


def _extract_mixup_chain(decoder: dict[str, Any]) -> tuple[dict[str, Any] | None, dict | None]:
    if isinstance(decoder, dict) and decoder.get("type") == "mixup_combine":
        return decoder, None
    if isinstance(decoder, dict) and decoder.get("type") == "chain":
        sequence = decoder.get("sequence") or []
        if sequence and isinstance(sequence[0], dict) and sequence[0].get("type") == "mixup_combine":
            post = sequence[1] if len(sequence) > 1 else None
            return sequence[0], post
    return None, None


def _decode_mixup_predictions(
    parsed: Any,
    mixc_dec: dict[str, Any],
    post_split_decoder: dict | None,
) -> list[tuple[str | None, Any]]:
    try:
        from .augmentation_framework import AugmentationManager

        mgr = AugmentationManager()
        aug = mgr.augmentations.get("mixup_combine")
        if not aug:
            return []
        decoded_list, _ = aug.apply_decoder_multi_task(parsed, mixc_dec)
    except Exception as exc:
        debug_print(f"mixup_combine multi-task decode failed: {exc}")
        return []

    if post_split_decoder:
        with suppress(Exception):
            decoded_list = [apply_decoder_description(grid, post_split_decoder) for grid in decoded_list]

    split_keys = list(mixc_dec.get("task_keys") or [])
    while len(split_keys) < len(decoded_list):
        split_keys.append(None)
    return list(zip(split_keys, decoded_list, strict=False))


def _extract_combine_chain(decoder: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    if not isinstance(decoder, dict):
        return None, None
    if decoder.get("type") == "combine":
        return decoder, []
    if decoder.get("type") == "chain":
        sequence = decoder.get("sequence") or []
        if sequence and isinstance(sequence[0], dict) and sequence[0].get("type") == "combine":
            tail = [s for s in sequence[1:] if isinstance(s, dict)]
            return sequence[0], tail
    return None, None


def _compose_decoder_from_sequence(sequence: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not sequence:
        return None
    if len(sequence) == 1:
        return sequence[0]
    return {"type": "chain", "sequence": sequence}


def _decode_combine_predictions(
    parsed: Any,
    combine_desc: dict[str, Any],
) -> tuple[list[list[list[int]]], list[list[int]] | None]:
    try:
        from .augmentation_framework import AugmentationManager

        mgr = AugmentationManager()
        aug = mgr.augmentations.get("combine")
        if not aug:
            return [], None
        decoded_list, combined = aug.apply_decoder_multi_task(parsed, combine_desc)
        return decoded_list or [], combined
    except Exception as exc:
        debug_print(f"combine multi-task decode failed: {exc}")
        return [], None


def _apply_decoder_sequence(grid: Any, decoder_desc: dict[str, Any] | None) -> Any:
    if decoder_desc is None:
        return grid
    try:
        return apply_decoder_description(grid, decoder_desc)
    except Exception as exc:
        debug_print(f"Failed to apply chained decoder after combine split: {exc}")
        return grid


def _process_combine_prediction(
    task_key: str,
    parsed: Any,
    combine_dec: dict[str, Any],
    post_sequence: list[dict[str, Any]] | None,
    counters: dict[str, Any],
    state: PostprocessState,
) -> list[Any] | None:
    decoded_list, combined = _decode_combine_predictions(parsed, combine_dec)
    if not decoded_list and combined is None:
        return None

    tail_decoder = _compose_decoder_from_sequence(post_sequence)
    expected = len(combine_dec.get("original_sizes") or [])
    splits_success = len(decoded_list) > 1 or (expected > 1 and len(decoded_list) == expected)

    final_candidates: list[Any] = []
    if splits_success:
        for grid in decoded_list:
            processed = _apply_decoder_sequence(grid, tail_decoder)
            if not is_valid_prediction(processed):
                continue
            _record_valid_prediction(counters, state)
            final_candidates.append(processed)
        if final_candidates:
            return final_candidates

    if combined is not None:
        fallback = _apply_decoder_sequence(combined, tail_decoder)
        if is_valid_prediction(fallback):
            _record_valid_prediction(counters, state)
            return [fallback]
        return []

    return None


def _scaling_postprocess_wrapper(
    raw_pred_complex: dict[str, Any],
    model_basename: str,
    model_settings: dict[str, Any],
    per_run_store: dict[int, dict[int, dict[str, list[Any]]]],
    per_model_store: dict[int, dict[str, list[Any]]],
    aggregated_store: dict[str, list[Any]],
    **kwargs: Any,
) -> tuple[dict[str, int], dict[str, Any], dict[str, list[Any]]]:
    counters: dict[str, Any] = kwargs["counters"]
    ctx = PostprocessContext(
        model_basename=model_basename,
        model_settings=model_settings,
        per_run_store=per_run_store,
        per_model_store=per_model_store,
        aggregated_store=aggregated_store,
        model_idx=kwargs["model_idx"],
        run_idx=kwargs["run_idx"],
        enable_self_ensemble=kwargs["enable_self_ensemble"],
        enable_model_ensemble=kwargs["enable_model_ensemble"],
        averaged_mode=kwargs.get("averaged_mode", False),
    )
    return postprocess_predictions(raw_pred_complex, counters, ctx)


# ======================================================================================
# Pipeline orchestrator
# ======================================================================================


@dataclass
class TaskLoadResult:
    all_task_data: dict[str, Any]
    submission_keys: list[str]
    loaded_keys: list[str]
    active_keys: list[str]


@dataclass
class PipelineState:
    aggregated_predictions: dict[str, list[Any]]
    counters: dict[str, Any]
    global_aug_stats: dict[str, Any]
    global_filter_stats: dict[str, Any]
    per_run_predictions: dict[int, dict[int, dict[str, list[Any]]]]
    per_model_predictions: dict[int, dict[str, list[Any]]]
    model_paths_for_tracking: list[str]
    model_ensemble_flags: list[bool]
    last_score: float = 0.0
    last_solved: int = 0
    progress_log: list[dict[str, Any]] = field(default_factory=list)
    error_tracker: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    start_time: float
    session_logger: Any
    run_logger: RunLogger
    is_kaggle: bool
    repo_state: dict[str, Any] = field(default_factory=dict)
    device_summary: dict[str, Any] = field(default_factory=dict)
    scaling_collector: ScalingMetricsCollector | None = None
    world_size: int = 1


def initialize_environment(start_time: float) -> RuntimeContext:
    """Setup session logging, environment, and run logger."""
    session_logger = initialize_session_logger()
    session_logger.start_console_capture()
    banner = "=" * 70 + "\nARC Prize 2025 Solution - MindsAI Team\nAuthor: Jack Cole\n" + "=" * 70
    print(banner)
    session_logger.log_custom_event("session_start", "Starting ARC Prize 2025 solution execution")
    _, is_kaggle = setup_kaggle_environment()
    if is_kaggle:
        display_working_directory_contents()
        session_logger.log_custom_event("environment", "Kaggle")
        os.makedirs("/kaggle/working", exist_ok=True)
    else:
        session_logger.log_custom_event("environment", "Local")

    run_logger = RunLogger()
    print(f"🎯 Starting: {config.RUN_TITLE} | 🆔 Run ID: {run_logger.run_id}")
    session_logger.log_custom_event("run_id", f"Run ID: {run_logger.run_id}")
    run_logger.register_session_artifacts(session_logger.get_artifact_manifest())
    _run_logging.current_run_logger = run_logger
    config.current_run_logger = run_logger
    _apply_persistent_overrides(is_kaggle)
    _maybe_enable_test_mode_via_env_or_taskid()
    setup_multiprocessing()

    return RuntimeContext(
        start_time=start_time,
        session_logger=session_logger,
        run_logger=run_logger,
        is_kaggle=is_kaggle,
    )


def load_tasks(runtime: RuntimeContext) -> TaskLoadResult:
    if config.TEST_MODE:
        print("TEST_MODE: Loading all challenges for submission, subset for processing...")
        all_complete = load_data(config.DATA_PATH, limit=None)
        if config.TEST_MODE_TASK_IDS:
            all_proc = load_data(config.DATA_PATH, task_ids=config.TEST_MODE_TASK_IDS)
        else:
            all_proc = load_data(config.DATA_PATH, limit=config.TEST_MODE_NUM_TASKS, random_seed=42)
        submission_keys = list(all_complete.keys())
        loaded_keys = list(all_proc.keys())
        all_task_data = all_proc
    else:
        all_task_data = load_data(config.DATA_PATH, limit=None)
        submission_keys = loaded_keys = list(all_task_data.keys())
    active_keys = list(loaded_keys)
    if all_task_data:
        print("Creating initial submission file...")
        make_submission({k: [] for k in submission_keys}, submission_keys, config.SUBMISSION_PATH)
    return TaskLoadResult(
        all_task_data=all_task_data,
        submission_keys=submission_keys,
        loaded_keys=loaded_keys,
        active_keys=active_keys,
    )


def prepare_initial_state(tasks: TaskLoadResult) -> PipelineState:
    return PipelineState(
        aggregated_predictions={k: [] for k in tasks.submission_keys},
        counters={
            "total_raw": 0,
            "total_parsed": 0,
            "total_valid": 0,
            "empty_samples": 0,
            "max_empty_samples": 5,
        },
        global_aug_stats={
            "total_augmentations_generated": 0,
            "successful_augmentations": 0,
            "failed_augmentations": 0,
            "augmentation_type_breakdown": {},
        },
        global_filter_stats={
            "total_filtering_calls": 0,
            "total_tasks_filtered": 0,
            "total_correct_filtered": 0,
            "total_remaining_tasks": 0,
            "total_correct_remaining": 0,
            "filtering_history": [],
        },
        per_run_predictions={},
        per_model_predictions={},
        model_paths_for_tracking=[],
        model_ensemble_flags=[],
        error_tracker={
            "training_errors": [],
            "inference_errors": [],
            "total_training_attempts": 0,
            "total_inference_attempts": 0,
            "current_run_errors": [],
        },
    )


def _apply_persistent_overrides(is_kaggle: bool) -> None:
    cfg_file = "/kaggle/working/notebook_config.json" if is_kaggle else "notebook_config.json"
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file) as f:
                persistent = json.load(f)
            if "DATA_PATH" in persistent:
                config.DATA_PATH = persistent["DATA_PATH"]
                print(f"📁 Applied persistent DATA_PATH: {persistent['DATA_PATH']}")
        except Exception as exc:
            print(f"⚠️ Failed to load persistent config: {exc}")
    env_data = os.getenv("ARC_DATA_PATH")
    if env_data:
        config.DATA_PATH = env_data
        print(f"🌍 Applied environment DATA_PATH: {env_data}")
    mp_env = os.getenv("MODEL_PATHS")
    if mp_env:
        try:
            models = (
                json.loads(mp_env)
                if mp_env.strip().startswith("[")
                else [p.strip() for p in mp_env.split(",") if p.strip()]
            )
            if models:
                config.MODEL_PATHS[:] = models
                print(f"🌍 Applied environment MODEL_PATHS: {config.MODEL_PATHS}")
        except Exception as exc:
            print(f"⚠️ Failed to parse MODEL_PATHS env var: {exc}")


_TRUEY_FLAG_VALUES = {"true", "1", "yes", "t", "y"}
_FALSEY_FLAG_VALUES = {"false", "0", "no", "f", "n"}
_PUBLIC_EVAL_SENTINEL_ID = "00576224"


def _enable_test_mode_and_overrides(reason: str) -> None:
    config.TEST_MODE = True
    for model_path, settings in config.MODEL_SETTINGS.items():
        training_cfg = settings.get("training")
        if isinstance(training_cfg, dict):
            training_cfg["target_total_ttt_items"] = config.TEST_MODE_TTT_ITEMS
        inference_cfg = settings.get("inference")
        if isinstance(inference_cfg, dict):
            inference_cfg["target_total_inference_items"] = config.TEST_MODE_INFERENCE_ITEMS
    config.DATA_PATH = config.get_data_path()
    print(f"🧪 TEST_MODE enabled ({reason})")


def _maybe_enable_test_mode_via_env_or_taskid() -> None:
    env_flag = os.getenv("TEST_MODE")
    if env_flag is not None:
        normalized = env_flag.strip().lower()
        if normalized in _TRUEY_FLAG_VALUES:
            _enable_test_mode_and_overrides("env var")
            return
        if normalized not in _FALSEY_FLAG_VALUES:
            return

    detect_flag = os.getenv("DETECT_TEST_MODE_ID")
    if detect_flag is not None and detect_flag.strip().lower() in _FALSEY_FLAG_VALUES:
        return

    candidate_paths: list[str] = []
    if config.DATA_PATH:
        candidate_paths.append(config.DATA_PATH)
    fallback_path = config.get_data_path()
    if fallback_path and fallback_path not in candidate_paths:
        candidate_paths.append(fallback_path)

    for data_path in candidate_paths:
        if not data_path or not os.path.isfile(data_path):
            continue
        try:
            with open(data_path, encoding="utf-8") as handle:
                contents = handle.read()
        except OSError as exc:
            print(f"Warning: Could not read {data_path} to detect test mode: {exc}")
            continue
        if _PUBLIC_EVAL_SENTINEL_ID in contents:
            _enable_test_mode_and_overrides(f"found task {_PUBLIC_EVAL_SENTINEL_ID} in {data_path}")
            return


def prepare_scaling(runtime: RuntimeContext) -> None:
    runtime.repo_state = collect_repo_state()
    runtime.device_summary = collect_device_summary(torch)
    runtime.scaling_collector = None
    try:
        runtime.run_logger.record_repo_state(runtime.repo_state)
    except Exception as exc:
        print(f"⚠️ Failed to snapshot git state: {exc}")
    scaling_enabled_raw = getattr(config, "SCALING_ENABLED", False)
    scaling_enabled = bool(scaling_enabled_raw)
    print(f"🔍 Debug: SCALING_ENABLED={scaling_enabled_raw} (type: {type(scaling_enabled_raw)})")
    print(f"🔍 Debug: SCALING_ENABLED env var='{os.environ.get('SCALING_ENABLED', 'NOT_SET')}'")
    if scaling_enabled:
        try:
            runtime.scaling_collector = ScalingMetricsCollector()
            print("✅ Scaling metrics collector initialized successfully")
        except Exception as exc:
            print(f"⚠️ Scaling metrics collector initialization failed: {exc}")
            runtime.scaling_collector = None
    else:
        print(f"⚠️ Scaling disabled: SCALING_ENABLED={scaling_enabled_raw}")
    world_size_env = os.environ.get("WORLD_SIZE")
    try:
        runtime.world_size = (
            int(world_size_env)
            if world_size_env
            else (torch.cuda.device_count() if torch and torch.cuda.is_available() else 1)
        )
    except Exception:
        runtime.world_size = 1


def prepare_runtime(runtime: RuntimeContext) -> None:
    prepare_scaling(runtime)


def process_models(runtime: RuntimeContext, tasks: TaskLoadResult, state: PipelineState) -> None:
    print(
        f"Starting main loop with {len(config.MODEL_PATHS)} model(s). "
        f"Total tasks loaded: {len(tasks.loaded_keys)}. Test mode: {config.TEST_MODE}."
    )
    for model_idx, model_path_current in enumerate(config.MODEL_PATHS):
        if (time.time() - runtime.start_time + config.BUFFER_TIME) >= config.TOTAL_TIME:
            print("Time limit approaching. Stopping.", file=sys.stderr)
            break
        _process_single_model(runtime, tasks, state, model_idx, model_path_current)

# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------


@dataclass
class ModelContext:
    model_idx: int
    model_path: str
    model_basename: str
    settings: dict[str, Any]
    get: Any  # getter function _S(name, section=None, default=None)
    is_causal: bool
    is_coda: bool
    ttt_enabled: bool
    airv_enabled: bool
    self_ensemble_runs: int
    task_group_size: int | None
    model_averaging: bool
    ensemble: dict[str, Any]
    scaling_enabled_for_model: bool
    prompt_prefix: str
    prompt_format: str
    power_sampling_cfg: dict[str, Any] | None


@dataclass
class PostprocessContext:
    model_basename: str
    model_settings: dict[str, Any]
    per_run_store: dict[int, dict[int, dict[str, list[Any]]]]
    per_model_store: dict[int, dict[str, list[Any]]]
    aggregated_store: dict[str, list[Any]]
    model_idx: int
    run_idx: int
    enable_self_ensemble: bool
    enable_model_ensemble: bool
    averaged_mode: bool = False


@dataclass
class PostprocessState:
    run_delta: dict[str, int]
    empty_delta: dict[str, Any]
    run_contrib: dict[str, list[Any]]
    tgt_run: int
    use_mp: bool


def _get_phase_markers(mc: ModelContext) -> dict[str, str]:
    general_block = mc.settings.get("general") if isinstance(mc.settings, dict) else {}
    if not isinstance(general_block, dict):
        general_block = {}
    markers = general_block.get("phase_markers")
    return markers if isinstance(markers, dict) else {}


def _print_phase_marker(mc: ModelContext, phase: str) -> bool:
    markers = _get_phase_markers(mc)
    marker = markers.get(phase)
    if isinstance(marker, str) and marker.strip():
        print(marker)
        return True
    return False


def _format_collection(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _format_coda_training_summary(mc: ModelContext) -> str | None:
    training_block = mc.settings.get("training") if isinstance(mc.settings, dict) else {}
    if not isinstance(training_block, dict):
        return None
    coda_cfg = training_block.get("coda")
    if not isinstance(coda_cfg, dict) or not coda_cfg:
        return None
    pieces: list[str] = []
    mode = coda_cfg.get("training_mode")
    if mode:
        pieces.append(f"mode={mode}")
    mask_cfg = coda_cfg.get("masking_schedule")
    if isinstance(mask_cfg, dict):
        mask_blocks = mask_cfg.get("mask_block_sizes")
        block_prob = mask_cfg.get("block_masking_probability")
        prefix_prob = mask_cfg.get("prefix_probability")
        truncate_prob = mask_cfg.get("truncate_probability")
        if mask_blocks:
            pieces.append(f"mask_blocks={_format_collection(mask_blocks)}")
        if block_prob is not None:
            pieces.append(f"block_prob={block_prob}")
        if prefix_prob is not None:
            pieces.append(f"prefix_prob={prefix_prob}")
        if truncate_prob is not None:
            pieces.append(f"truncate_prob={truncate_prob}")
    sampling_eps = coda_cfg.get("sampling_eps")
    if sampling_eps:
        pieces.append(f"sampling_eps={_format_collection(sampling_eps)}")
    return f"  • CoDA training: {'; '.join(pieces)}" if pieces else None


def _format_coda_inference_summary(settings: dict[str, Any]) -> str | None:
    if not isinstance(settings, dict):
        return None
    mode = settings.get("generation_mode")
    if not isinstance(mode, str) or mode.strip().lower() != "diffusion":
        return None
    pieces: list[str] = []
    profile = settings.get("coda_quality_profile")
    if profile:
        pieces.append(f"profile={profile}")
    steps = settings.get("diffusion_steps")
    if steps is not None:
        pieces.append(f"steps={steps}")
    diff_alg = settings.get("diffusion_alg")
    if diff_alg:
        pieces.append(f"alg={diff_alg}")
    diff_eps = settings.get("diffusion_eps")
    if diff_eps is not None:
        pieces.append(f"eps={diff_eps}")
    temp = settings.get("temperature")
    if temp is not None:
        pieces.append(f"temperature={temp}")
    top_p = settings.get("top_p")
    if top_p is not None:
        pieces.append(f"top_p={top_p}")
    num_returns = settings.get("num_return_sequences")
    if num_returns is not None:
        pieces.append(f"return_sequences={num_returns}")
    return f"  • CoDA inference: {'; '.join(pieces)}" if pieces else None


def _format_coda_quality_preset(settings: dict[str, Any]) -> str | None:
    if not isinstance(settings, dict):
        return None
    profile = settings.get("coda_quality_profile")
    presets = settings.get("coda_quality_presets")
    if not (profile and isinstance(presets, dict)):
        return None
    preset_cfg = presets.get(profile)
    if not isinstance(preset_cfg, dict):
        return None
    pieces: list[str] = []
    for key in ("diffusion_steps", "temperature", "top_p"):
        value = preset_cfg.get(key)
        if value is not None:
            pieces.append(f"{key}={value}")
    return f"  • Active quality preset '{profile}': {', '.join(pieces)}" if pieces else None


# -----------------------------------------------------------------------------
# Helper utilities (pure or side-effect light)
# -----------------------------------------------------------------------------

def _get_prompt_prefix_for_model(model_path: str) -> str:
    settings = getattr(config, "MODEL_SETTINGS", {}).get(model_path, {}) or {}
    general_block = settings.get("general", {}) or {}
    prompt_cfg = general_block.get("prompt_settings") or {}
    return str(prompt_cfg.get("general_prefix") or "")


def _model_getter(current_model_settings: dict, model_path: str):
    """Return a closure to fetch nested settings consistently."""
    def _S(name: str, section: str | None = None, default=None):
        return _get_setting(current_model_settings, model_path, name, section, default)
    return _S


def _summarize_model_context(model_idx: int, total_models: int, mc: ModelContext) -> None:
    print(
        "\n"
        + "=" * 70
        + f"\n🚀 {config.RUN_TITLE}"
        + f"\n📍 Model {model_idx + 1}/{total_models}: {mc.model_basename}"
        + f"\n Type: {'CAUSAL' if mc.is_causal else 'SEQ2SEQ'}"
        + f"\n Architecture: {'CoDA diffusion LM' if mc.is_coda else 'Standard'}"
        + f"\n TTT: {mc.ttt_enabled}, AIRV: {mc.airv_enabled}, Self-Ensemble Runs: {mc.self_ensemble_runs}"
    )
    if mc.task_group_size is not None:
        print(f" Task Group Size: {mc.task_group_size}")
    print(
        f" Model Averaging: {mc.model_averaging}\n"
        f" Ensemble: Self={mc.ensemble['enable_self_ensemble']}, "
        f"Model={mc.ensemble['enable_model_ensemble']}, RunTrack={mc.ensemble['enable_run_tracking']}"
    )
    if mc.ensemble.get("iterative_ensemble_mode", False):
        print(f" Iterative Ensemble Mode: ENABLED (fine-tuned reuse at {config.RELOAD_PATH})")
    if mc.is_causal:
        print(f" Max Length: {mc.get('max_length', 'general', 4096)}")
    else:
        print(
            f" Max Input/Target: {mc.get('max_input_length', 'general', 2048)}, "
            f"{mc.get('max_target_length', 'general', 512)}"
        )
    general_notes = mc.settings.get("general", {}) if isinstance(mc.settings, dict) else {}
    notes = general_notes.get("notes") if isinstance(general_notes, dict) else None
    if notes:
        items = notes if isinstance(notes, (list, tuple)) else [notes]
        for note in items:
            if isinstance(note, str) and note.strip():
                print(f" Note: {note}")
    print("=" * 70)


def _prepare_task_groups(
    active_task_keys: list[str], task_group_size: int | None
) -> tuple[list[list[str]] | None, int | None]:
    if task_group_size and task_group_size > 0 and active_task_keys:
        ks = list(active_task_keys)
        groups = [ks[i : i + task_group_size] for i in range(0, len(ks), task_group_size)]
        print(f" Created {len(groups)} group(s) from {len(ks)} tasks")
        return groups, len(ks)
    return None, None


def _build_train_args(
    mc: ModelContext,
    *,
    seed: int,
    enable_tf: bool,
    phase: str = "train",
) -> dict[str, Any]:
    train_keys = [
        "learning_rate",
        "weight_decay",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "logging_steps",
        "dataloader_num_workers",
        "trainable_layer_patterns",
        "label_smoothing",
        "per_device_eval_batch_size",
        "pretraining_objective",
        "pretraining_objective_settings",
    ]
    args = {k: mc.get(k, "training") for k in train_keys if mc.get(k, "training") is not None}
    settings_blob = args.get("pretraining_objective_settings")
    if isinstance(settings_blob, dict):
        try:
            args["pretraining_objective_settings"] = json.dumps(settings_blob)
        except Exception:
            args.pop("pretraining_objective_settings", None)
    elif settings_blob is not None:
        args["pretraining_objective_settings"] = str(settings_blob)
    for key in ["max_input_length", "max_target_length", "max_length"]:
        value = mc.get(key, "general")
        if value is not None:
            args[key] = value
    use_bf16 = mc.get("use_bf16", "training", False)
    use_fp16 = mc.get("use_fp16", "training", False)
    args.update({
        "use_mixed_precision": (use_bf16 or use_fp16),
        "use_bf16": use_bf16,
        "use_fp16": use_fp16,
        "use_gradient_checkpointing": mc.get("gradient_checkpointing", "training", False),
        "per_device_train_batch_size": mc.get("train_batch_size", "training", 1),
        "use_torch_compile": mc.get("use_torch_compile", "training", False),
        "seed": seed,
        "warmup_ratio": mc.get("warmup_ratio", "training", 0.1),
        "tokenizer_dropout_enabled": mc.get("tokenizer_dropout", "general", {}).get("train", {}).get("enabled", False),
        "tokenizer_dropout_rate": mc.get("tokenizer_dropout", "general", {}).get("train", {}).get("rate", 0.1),
        "tokenizer_dropout_apply_to_labels": mc.get("tokenizer_dropout", "general", {}).get("train", {}).get("apply_to_labels", False),
        "enable_token_filtering": enable_tf,
        "is_coda_model": mc.is_coda,
    })

    lr_reset_value = mc.get("lr_reset_examples", "training")
    if lr_reset_value is None:
        env_keys = ["LR_RESET_EXAMPLES"]
        if phase == "pretraining":
            env_keys.insert(0, "PRETRAINING_LR_RESET_EXAMPLES")
        for env_key in env_keys:
            env_val = _env_lookup(env_key)
            if env_val not in (None, ""):
                lr_reset_value = env_val
                break
    if lr_reset_value not in (None, ""):
        try:
            lr_reset_int = int(float(lr_reset_value))
        except Exception:
            lr_reset_int = 0
        if lr_reset_int and lr_reset_int > 0:
            args["lr_reset_examples"] = int(lr_reset_int)

    def _collect_str_list(value) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
            return parts or None
        if isinstance(value, (list, tuple, set)):
            out = [str(item).strip() for item in value if str(item).strip()]
            return out or None
        text = str(value).strip()
        return [text] if text else None

    def _collect_int_list(value) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]
        collected: list[int] = []
        for item in raw:
            if item is None:
                continue
            try:
                collected.append(int(item))
            except Exception:
                raise ValueError(f"LoRA layers_to_transform entry '{item}' could not be parsed as an integer.")
        return collected or None

    lora_cfg = mc.get("lora", "training")
    if isinstance(lora_cfg, dict) and lora_cfg.get("enabled"):
        args["use_lora"] = True
        for src, dest, caster in [
            ("r", "lora_r", int),
            ("alpha", "lora_alpha", float),
            ("dropout", "lora_dropout", float),
            ("bias", "lora_bias", str),
            ("task_type", "lora_task_type", str),
            ("init_lora_weights", "lora_init_lora_weights", None),
            ("scaling", "lora_scaling", float),
            ("layers_pattern", "lora_layers_pattern", str),
        ]:
            val = lora_cfg.get(src)
            if val is not None:
                if caster is None:
                    args[dest] = val
                else:
                    try:
                        args[dest] = caster(val)
                    except Exception:
                        raise ValueError(f"LoRA config value '{src}'={val!r} could not be cast with {caster.__name__}.")
        if lora_cfg.get("use_dora"):
            args["lora_use_dora"] = True
        for src, dest in [
            ("target_modules", "lora_target_modules"),
            ("modules_to_save", "lora_modules_to_save"),
        ]:
            values = _collect_str_list(lora_cfg.get(src))
            if values:
                args[dest] = values
        layers_to_transform = _collect_int_list(lora_cfg.get("layers_to_transform"))
        if layers_to_transform:
            args["lora_layers_to_transform"] = layers_to_transform
        rank_pattern = lora_cfg.get("rank_pattern")
        if isinstance(rank_pattern, dict) and rank_pattern:
            args["lora_rank_pattern"] = json.dumps(rank_pattern)

    # Allow external log dir wiring
    tr_log_dir = os.environ.get("ARC_TRAINING_LOG_DIR") or (
        os.environ.get("ARC_LOG_DIR") and os.path.join(os.environ.get("ARC_LOG_DIR"), "training_logs")
    )
    if tr_log_dir:
        args["log_dir"] = tr_log_dir
    return args


def _make_ttt_and_infer_datasets(
    mc: ModelContext,
    all_task_data: dict[str, Any],
    cur_run_keys: list[str],
    *,
    airv_enabled_for_run: bool,
) -> tuple[list, dict[str, Any], dict[str, Any]]:
    enable_tf = mc.get("enable_token_filtering", "general", True)
    max_in = mc.get("max_input_length", "general", 2048)
    max_tg = mc.get("max_target_length", "general", 512)

    need_ttt = mc.ttt_enabled
    target_ttt_cfg = mc.get("target_total_ttt_items", "training", 0) if need_ttt else 0
    target_inf_cfg = mc.get("target_total_inference_items", "inference", 0)
    # Apply global overrides when provided (>0)
    try:
        ttt_override = int(getattr(config, "GLOBAL_TTT_SAMPLES", 0) or 0)
    except Exception:
        ttt_override = 0
    try:
        inf_override = int(getattr(config, "GLOBAL_INF_SAMPLES", 0) or 0)
    except Exception:
        inf_override = 0
    target_ttt = ttt_override if (need_ttt and ttt_override > 0) else target_ttt_cfg
    target_inf = inf_override if (inf_override > 0) else target_inf_cfg

    data_for_keys = {k: all_task_data[k] for k in cur_run_keys if k in all_task_data}
    ttt_items, inf_data, ds_stats = make_datasets(
        data_for_keys,
        list(data_for_keys.keys()),
        target_ttt,
        target_inf,
        max_in,
        max_tg,
        mc.model_path,
        airv_enabled_for_model=airv_enabled_for_run,
        enable_token_filtering=enable_tf,
    )
    return ttt_items, inf_data, ds_stats


def _prep_ttt_dataset(ttt_items) -> Any | None:
    if not ttt_items:
        return None
    train_ds, _ = prep_ttt_dataset(ttt_items, config.TEST_SIZE)
    return train_ds if len(train_ds) > 0 else None


def _resolve_mini_lr_grid_config(mc: ModelContext) -> dict[str, Any] | None:
    training_cfg = mc.settings.get("training") if isinstance(mc.settings, dict) else {}
    if not isinstance(training_cfg, dict):
        return None
    raw_cfg = training_cfg.get("mini_lr_grid")
    if raw_cfg is None:
        return None
    if isinstance(raw_cfg, dict):
        cfg = dict(raw_cfg)
        enabled = cfg.pop("enabled", True)
        if not _truthy(enabled):
            return None
        cfg["enabled"] = True
        test_override = None
        for key in ("max_steps_test_mode", "test_mode_max_steps", "max_steps_when_test_mode"):
            if key in cfg and cfg[key] is not None:
                test_override = cfg[key]
                break
        if config.TEST_MODE and test_override is not None:
            override_val = _coerce_int_value(test_override)
            if override_val is not None:
                cfg["max_steps"] = override_val
        return cfg
    if isinstance(raw_cfg, (list, tuple)):
        values = [val for val in raw_cfg if val is not None]
        if not values:
            return None
        return {"enabled": True, "values": list(values)}
    if _truthy(raw_cfg):
        return {"enabled": True}
    return None


def _coerce_float_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"none", "nan", "null"}:
            return None
        if text in {"inf", "infinity"}:
            return float("inf")
        try:
            return float(text)
        except Exception:
            return None
    return None


def _coerce_int_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "nan", "null"}:
            return None
        try:
            return int(float(text))
        except Exception:
            return None
    return None


def _limit_lr_grid_dataset(train_ds, limit: int | None):
    if limit is None or limit <= 0:
        return train_ds
    try:
        length = len(train_ds)  # type: ignore[arg-type]
    except Exception:
        length = None
    if length is not None and length <= limit:
        return train_ds
    try:
        from datasets import Dataset, DatasetDict  # type: ignore
    except Exception:  # pragma: no cover - datasets optional
        Dataset = DatasetDict = None  # type: ignore
    try:
        if Dataset is not None and isinstance(train_ds, Dataset):
            return train_ds.select(range(limit))
        if DatasetDict is not None and isinstance(train_ds, DatasetDict):
            return train_ds.select(range(limit))
    except Exception:
        pass
    if isinstance(train_ds, list):
        return train_ds[:limit]
    if isinstance(train_ds, tuple):
        return train_ds[:limit]
    if hasattr(train_ds, 'select'):
        try:
            return train_ds.select(range(limit))
        except Exception:
            return train_ds
    return train_ds



def _normalize_learning_rate_candidate(raw: Any, base_lr: float) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if not text:
            return None
        if text in {"base", "default"}:
            return base_lr
        if text.endswith("x"):
            try:
                multiplier = float(text[:-1])
            except Exception:
                multiplier = None
            if multiplier is not None and multiplier > 0:
                return base_lr * multiplier
    value = _coerce_float_value(raw)
    if value is not None and value > 0:
        return float(value)
    return None


def _resolve_lr_grid_candidates(base_lr: float, grid_cfg: dict[str, Any]) -> list[float]:
    include_base = grid_cfg.get("include_base", True)
    max_trials = _coerce_int_value(grid_cfg.get("max_trials") or grid_cfg.get("limit"))
    candidates: list[float] = []

    values = grid_cfg.get("values")
    if isinstance(values, (list, tuple)):
        for raw in values:
            candidate = _normalize_learning_rate_candidate(raw, base_lr)
            if candidate is not None:
                candidates.append(candidate)

    multipliers = grid_cfg.get("multipliers")
    if isinstance(multipliers, (list, tuple)):
        for raw in multipliers:
            mult = _coerce_float_value(raw)
            if mult is not None and mult > 0:
                candidates.append(base_lr * mult)

    if not candidates:
        num_trials = _coerce_int_value(
            grid_cfg.get("num_trials")
            or grid_cfg.get("num_steps")
            or grid_cfg.get("steps")
            or grid_cfg.get("k")
        )
        if num_trials and num_trials > 0:
            min_lr = _normalize_learning_rate_candidate(
                grid_cfg.get("min_lr") or grid_cfg.get("min"), base_lr
            )
            max_lr = _normalize_learning_rate_candidate(
                grid_cfg.get("max_lr") or grid_cfg.get("max"), base_lr
            )
            spacing = str(grid_cfg.get("spacing") or grid_cfg.get("mode") or "log").strip().lower()
            if min_lr is None and max_lr is None:
                spread = _coerce_float_value(
                    grid_cfg.get("spread")
                    or grid_cfg.get("ratio")
                    or grid_cfg.get("step_factor")
                    or grid_cfg.get("range_multiplier")
                )
                if spread is None or spread <= 1.0:
                    spread = 3.0
                min_lr = base_lr / spread
                max_lr = base_lr * spread
            if min_lr is None:
                min_lr = max(base_lr * 0.5, 1e-7)
            if max_lr is None:
                max_lr = max(base_lr * 1.5, min_lr * 1.1)
            if num_trials == 1:
                candidates = [base_lr if include_base else float(max(min_lr, min(max_lr, base_lr)))]
            else:
                min_lr = max(min_lr, 1e-9)
                max_lr = max(max_lr, min_lr * 1.01)
                if spacing == "linear":
                    candidates = [float(v) for v in np.linspace(min_lr, max_lr, num_trials)]
                else:
                    candidates = [float(v) for v in np.geomspace(min_lr, max_lr, num_trials)]

    if include_base and base_lr > 0:
        candidates.append(base_lr)

    normalized: list[float] = []
    seen: set[str] = set()
    for value in candidates:
        if not value or value <= 0:
            continue
        key = f"{value:.12g}"
        if key in seen:
            continue
        seen.add(key)
        normalized.append(float(value))

    if max_trials and max_trials > 0:
        normalized = normalized[: max_trials]

    return normalized


def _resolve_lr_grid_metric_keys(grid_cfg: dict[str, Any]) -> list[str]:
    metric_raw = grid_cfg.get("metric") or grid_cfg.get("metrics")
    if isinstance(metric_raw, str):
        keys = [metric_raw]
    elif isinstance(metric_raw, (list, tuple)):
        keys = [str(m) for m in metric_raw if m is not None]
    else:
        keys = []
    default_keys = [
        "loss_train_min",
        "loss_eval_min",
        "loss_train_final",
        "train_loss",
    ]
    for key in default_keys:
        if key not in keys:
            keys.append(key)
    return keys


def _resolve_lr_grid_metric_mode(grid_cfg: dict[str, Any]) -> str:
    mode_raw = (
        grid_cfg.get("metric_mode")
        or grid_cfg.get("metric_direction")
        or grid_cfg.get("metric_goal")
        or grid_cfg.get("metric_preference")
    )
    if isinstance(mode_raw, str):
        mode = mode_raw.strip().lower()
        if mode in {"max", "maximize", "higher", "desc", "decreasing"}:
            return "max"
        if mode in {"min", "minimize", "lower", "asc", "ascending"}:
            return "min"
    maximize_flag = grid_cfg.get("maximize_metric") or grid_cfg.get("metric_maximize") or grid_cfg.get("prefer_higher")
    try:
        if maximize_flag is not None and bool(maximize_flag):
            return "max"
    except Exception:
        pass
    return "min"


def _resolve_metric_candidate(metrics: dict[str, Any], key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None

    def _get_nested_value(blob: dict[str, Any], dotted_key: str) -> Any:
        current: Any = blob
        for part in dotted_key.split("."):
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current[part]
        return current

    candidate = metrics.get(key) if "." not in key else _get_nested_value(metrics, key)
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except Exception:
            return None
    return None


def _extract_metric_value(metrics: dict[str, Any] | None, metric_keys: list[str]) -> float | None:
    if not metrics or not isinstance(metrics, dict):
        return None
    for key in metric_keys:
        candidate = _resolve_metric_candidate(metrics, key)
        if candidate is not None:
            return candidate
    return None


def _copy_best_reload_path(src: Path, dest: Path) -> None:
    if dest.exists():
        cleanup_directories([str(dest)])
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _run_ttt_mini_lr_grid(
    mc: ModelContext,
    train_ds,
    base_args: dict[str, Any],
    base_lr: float,
    grid_cfg: dict[str, Any],
    candidates: list[float],
) -> tuple[bool, dict[str, Any] | None] | None:
    total = len(candidates)
    if total == 0:
        return None
    max_steps_raw = grid_cfg.get("max_steps") or grid_cfg.get("train_steps") or grid_cfg.get("steps")
    max_steps_override = _coerce_int_value(max_steps_raw) if max_steps_raw is not None else None
    # Skip grid search if only the base learning rate is present
    if total == 1 and abs(candidates[0] - base_lr) <= max(1e-9, base_lr * 1e-6):
        return None

    metric_keys = _resolve_lr_grid_metric_keys(grid_cfg)
    metric_mode = _resolve_lr_grid_metric_mode(grid_cfg)
    def _score_for_metric(value: Any) -> float:
        if value is None:
            return float("inf")
        try:
            numeric = float(value)
        except Exception:
            return float("inf")
        if math.isnan(numeric):
            return float("inf")
        return -numeric if metric_mode == "max" else numeric
    mode_label = "maximize" if metric_mode == "max" else "minimize"
    print(f"⚙️  Mini LR grid will {mode_label} metric '{metric_keys[0]}' (fallback order preserved).")
    if max_steps_override is not None and max_steps_override > 0:
        print(f"⚙️  Mini LR grid limiting each trial to {max_steps_override} training step(s)")
    base_reload_path = Path(config.RELOAD_PATH)
    trial_records: list[dict[str, Any]] = []
    best_info: dict[str, Any] | None = None
    interpolation_summary: dict[str, Any] | None = None
    loess_summary: dict[str, Any] | None = None

    def _run_single_trial(lr_candidate: float, dir_suffix: str, origin: str) -> dict[str, Any]:
        nonlocal best_info
        trial_dir = base_reload_path.parent / dir_suffix
        if trial_dir.exists():
            cleanup_directories([str(trial_dir)])
        trial_args = dict(base_args)
        trial_args["learning_rate"] = lr_candidate
        if max_steps_override is not None and max_steps_override > 0:
            trial_args["max_steps"] = max_steps_override
        trial_args["save_strategy"] = "no"
        trial_args["save_total_limit"] = 1
        trial_args["save_optimizer"] = False

        metrics = None
        metric_value = None
        metric_source: str | None = None
        metric_details: dict[str, float] = {}
        available_metric_keys: list[str] = []
        status = "success"
        error_text = None
        try:
            metrics = train_distributed(mc.model_path, train_ds, str(trial_dir), trial_args)
            if isinstance(metrics, dict):
                try:
                    available_metric_keys = sorted(str(k) for k in metrics.keys())
                except Exception:
                    available_metric_keys = [str(k) for k in metrics.keys()]
                for key in metric_keys:
                    candidate_val = _resolve_metric_candidate(metrics, key)
                    if candidate_val is None:
                        continue
                    metric_details[key] = candidate_val
                    if metric_value is None:
                        metric_value = candidate_val
                        metric_source = key
            if metric_value is not None:
                source_name = metric_source or metric_keys[0]
                print(f"     ↳ metric[{source_name}]={metric_value:.6g}")
            else:
                message = "     ↳ metric unavailable; using fallback ordering"
                if available_metric_keys:
                    preview = ", ".join(available_metric_keys[:8])
                    if len(available_metric_keys) > 8:
                        preview += ", …"
                    message += f" (available keys: {preview})"
                print(message)
        except Exception as exc:  # noqa: PERF203 - propagate detailed failure
            status = "failed"
            error_text = str(exc)
            print(f"     ↳ Trial failed: {error_text}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            _safe_gpu_clear()

        record = {
            "learning_rate": lr_candidate,
            "metric": metric_value,
            "status": status,
            "path": str(trial_dir),
            "origin": origin,
        }
        if error_text:
            record["error"] = error_text
        if metric_source:
            record["metric_source"] = metric_source
        if metric_details:
            record["metric_details"] = dict(metric_details)
        if available_metric_keys:
            record["metric_keys_available"] = available_metric_keys[:12]
        record["sequence"] = len(trial_records) + 1
        trial_records.append(record)

        keep_dir = False
        if status == "success":
            score = _score_for_metric(metric_value)
            if best_info is None or score < best_info["score"]:
                previous_best_path = Path(best_info["path"]) if best_info else None
                best_info = {
                    "score": score,
                    "learning_rate": lr_candidate,
                    "metric_value": metric_value,
                    "metrics": metrics,
                    "path": str(trial_dir),
                    "origin": origin,
                    "metric_mode": metric_mode,
                }
                keep_dir = True
                if previous_best_path and previous_best_path.exists():
                    cleanup_directories([str(previous_best_path)])
        if not keep_dir and trial_dir.exists():
            cleanup_directories([str(trial_dir)])
        return record

    print(f"⚙️  Mini LR grid search evaluating {total} learning rate(s)...")
    for idx, lr_candidate in enumerate(candidates, start=1):
        lr_display = f"{lr_candidate:.6g}"
        print(f"   • Trial {idx}/{total}: learning_rate={lr_display}")
        _run_single_trial(lr_candidate, f"{base_reload_path.name}__mini_lr_trial{idx}", "grid")

    successful_trials = [
        entry
        for entry in trial_records
        if entry["status"] == "success" and isinstance(entry.get("metric"), (int, float))
    ]

    interpolation_mode = str(grid_cfg.get("interpolation") or "none").strip().lower()
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
    except Exception:
        lowess = None  # type: ignore[assignment]
    if lowess and len(successful_trials) >= 3:
        xs_lowess = np.log10([float(entry["learning_rate"]) for entry in successful_trials])
        ys_lowess = np.array([float(entry["metric"]) for entry in successful_trials], dtype=float)
        frac = 0.6 if len(successful_trials) >= 5 else 0.75
        try:
            smoothed = lowess(ys_lowess, xs_lowess, frac=frac, return_sorted=True)
        except Exception as exc:
            print(f"⚠️  Mini LR LOWESS smoothing failed ({exc}).", file=sys.stderr)
        else:
            if smoothed is not None and len(smoothed):
                smooth_x = np.array(smoothed[:, 0], dtype=float)
                smooth_y = np.array(smoothed[:, 1], dtype=float)
                if metric_mode == "max":
                    best_idx = int(np.argmax(smooth_y))
                else:
                    best_idx = int(np.argmin(smooth_y))
                loess_summary = {
                    "mode": "lowess",
                    "frac": float(frac),
                    "log10_lrs": smooth_x.tolist(),
                    "metrics": smooth_y.tolist(),
                    "predicted_log10_learning_rate": float(smooth_x[best_idx]),
                    "predicted_learning_rate": float(10 ** smooth_x[best_idx]),
                    "predicted_metric": float(smooth_y[best_idx]),
                }
    if interpolation_mode == "quadratic":
        if len(successful_trials) < 3:
            print("⚠️  Mini LR interpolation skipped (need ≥ 3 successful trials).", file=sys.stderr)
        else:
            try:
                xs = np.log10([float(entry["learning_rate"]) for entry in successful_trials])
                ys = [float(entry["metric"]) for entry in successful_trials]
                try:
                    coeffs, cov = np.polyfit(xs, ys, deg=2, cov=True)  # type: ignore[misc]
                except TypeError:
                    coeffs = np.polyfit(xs, ys, deg=2)
                    cov = None
            except Exception as exc:
                print(f"⚠️  Mini LR interpolation failed (polyfit error: {exc})", file=sys.stderr)
            else:
                a, b, c = coeffs
                if (metric_mode == "min" and a <= 0) or (metric_mode == "max" and a >= 0):
                    shape_desc = "non-convex" if metric_mode == "min" else "non-concave"
                    print(f"⚠️  Mini LR interpolation skipped ({shape_desc} fit).", file=sys.stderr)
                else:
                    x_star = -b / (2 * a)
                    metric_star = float(a * x_star**2 + b * x_star + c)
                    raw_lr_star = float(10 ** x_star)
                    min_lr = min(entry["learning_rate"] for entry in successful_trials)
                    max_lr = max(entry["learning_rate"] for entry in successful_trials)
                    lr_star = max(min_lr, min(max_lr, raw_lr_star))
                    nearest = min(successful_trials, key=lambda entry: abs(entry["learning_rate"] - lr_star))
                    rel_diff = abs(lr_star - nearest["learning_rate"]) / max(lr_star, 1e-12)
                    min_rel = grid_cfg.get("interpolation_min_rel_distance")
                    if min_rel is None:
                        min_rel = grid_cfg.get("interpolation_min_rel_dist")
                    if min_rel is None:
                        min_rel = grid_cfg.get("interpolation_min_rel")
                    min_rel = float(min_rel) if min_rel is not None else 0.05
                    interpolation_summary = {
                        "mode": "quadratic",
                        "coefficients": [float(a), float(b), float(c)],
                        "predicted_learning_rate": lr_star,
                        "predicted_raw_learning_rate": raw_lr_star,
                        "predicted_log10_learning_rate": float(x_star),
                        "predicted_metric": metric_star,
                        "nearest_sample_learning_rate": nearest["learning_rate"],
                        "nearest_sample_metric": nearest.get("metric"),
                        "relative_distance_to_nearest": rel_diff,
                        "min_relative_distance_required": min_rel,
                    }
                    if cov is not None:
                        interpolation_summary["coefficient_covariance"] = np.asarray(cov, dtype=float).tolist()
                    verify_flag = grid_cfg.get("interpolation_verify")
                    verify_interp = True if verify_flag is None else _truthy(verify_flag)
                    interpolation_summary["verification_enabled"] = bool(verify_interp)
                    if rel_diff < min_rel:
                        interpolation_summary["status"] = "skipped_near_sample"
                        print(
                            "ℹ️  Mini LR interpolation suggestion is within the configured proximity of an observed trial; "
                            "skipping verification run."
                        )
                    elif not verify_interp:
                        interpolation_summary["status"] = "predicted"
                        print(
                            f"ℹ️  Mini LR interpolation suggests learning_rate={lr_star:.6g} "
                            "(verification disabled)."
                        )
                    else:
                        print(
                            f"   • Interpolation trial: learning_rate={lr_star:.6g} "
                            f"(raw prediction {raw_lr_star:.6g})"
                        )
                        interp_record = _run_single_trial(
                            lr_star, f"{base_reload_path.name}__mini_lr_interp", "interpolation"
                        )
                        interp_status = "verification_success" if interp_record["status"] == "success" else "verification_failed"
                        interpolation_summary.update(
                            {
                                "status": interp_status,
                                "verification_status": interp_record["status"],
                                "verification_metric": interp_record.get("metric"),
                                "verification_learning_rate": interp_record["learning_rate"],
                            }
                        )

    if best_info is None:
        print("⚠️  Mini LR grid search failed for all trials; falling back to standard training.", file=sys.stderr)
        return None

    summary = f"✅ Mini LR grid best learning_rate={best_info['learning_rate']:.6g}"
    if best_info["metric_value"] is not None:
        trend_label = "max" if metric_mode == "max" else "min"
        summary += f" (metric={best_info['metric_value']:.6g}, mode={trend_label})"
    else:
        summary += " (metric unavailable)"
    print(summary)
    _print_mini_lr_trial_table(trial_records, metric_keys)
    chart_path = _maybe_generate_mini_lr_chart(
        trial_records,
        best_info,
        interpolation_summary,
        loess_summary,
        metric_keys,
    )
    if chart_path:
        print(f"📈 Mini LR chart saved to: {chart_path}")

    best_path = Path(best_info["path"])
    _copy_best_reload_path(best_path, Path(config.RELOAD_PATH))
    best_metrics = dict(best_info.get("metrics") or {})
    trials_for_report: list[dict[str, Any]] = []
    for entry in trial_records:
        report_entry = {
            "learning_rate": entry["learning_rate"],
            "status": entry["status"],
            "origin": entry.get("origin"),
        }
        if entry.get("metric") is not None:
            report_entry["metric"] = entry["metric"]
        if entry.get("metric_source"):
            report_entry["metric_source"] = entry["metric_source"]
        if entry.get("metric_details"):
            report_entry["metric_details"] = entry["metric_details"]
        if entry.get("metric_keys_available"):
            report_entry["metric_keys_available"] = entry["metric_keys_available"]
        if entry.get("error"):
            report_entry["error"] = entry["error"]
        trials_for_report.append(report_entry)

    best_metrics["mini_lr_grid"] = {
        "metric_keys": metric_keys,
        "best_learning_rate": best_info["learning_rate"],
        "best_metric": best_info["metric_value"],
        "metric_mode": metric_mode,
        "trials": trials_for_report,
    }
    if interpolation_summary:
        best_metrics["mini_lr_grid"]["interpolation"] = interpolation_summary
    if loess_summary:
        best_metrics["mini_lr_grid"]["loess"] = loess_summary
    return True, best_metrics, best_info, trial_records


def _run_ttt(mc: ModelContext, runtime, run_logger, run_num: int, train_ds) -> tuple[bool, dict[str, Any] | None]:
    if not train_ds:
        print("TTT dataset empty. Skipping fine-tuning.", file=sys.stderr)
        return False, None
    print(f"Starting TTT with {len(train_ds)} examples…")
    if mc.is_coda:
        _print_phase_marker(mc, "ttt")
        summary = _format_coda_training_summary(mc)
        if summary:
            print(summary)
    args = _build_train_args(
        mc,
        seed=int(time.time() * 1000) % 100000 + run_num,
        enable_tf=mc.get("enable_token_filtering", "general", True),
        phase="ttt",
    )
    # Save strategy
    save_default = mc.get("save_steps", "training", None)
    save_steps = _determine_save_steps(len(train_ds), args, save_default)
    args.update({"save_steps": save_steps, "save_strategy": "steps"})
    print(f"  • Checkpoint interval: every {save_steps} step(s) (SAVE_STEPS={config.SAVE_STEPS})")
    # Add tracking
    if hasattr(run_logger, "run_id"):
        args["run_id"] = run_logger.run_id

    raw_grid_cfg = mc.settings.get("training", {}).get("mini_lr_grid") if isinstance(mc.settings, dict) else None
    grid_cfg = _resolve_mini_lr_grid_config(mc)
    base_lr_value = _coerce_float_value(args.get("learning_rate"))

    allowed_columns = {"prompt", "correct_answer"}
    extra_columns = [col for col in getattr(train_ds, "column_names", []) if col not in allowed_columns]
    train_ds_core = train_ds.remove_columns(extra_columns) if extra_columns else train_ds
    if grid_cfg:
        if base_lr_value is None:
            print("⚠️  Mini LR grid requested but base learning rate is missing; skipping sweep.", file=sys.stderr)
        else:
            candidates = _resolve_lr_grid_candidates(base_lr_value, grid_cfg)
            if not candidates:
                print(
                    "⚠️  Mini LR grid produced no candidate learning rates; falling back to standard TTFT.",
                    file=sys.stderr,
                )
            else:
                subset_limit = _coerce_int_value(
                    grid_cfg.get("ttt_items")
                    or grid_cfg.get("limit_examples")
                    or grid_cfg.get("max_examples")
                    or grid_cfg.get("sample_size")
                )
                grid_train_ds = _limit_lr_grid_dataset(train_ds_core, subset_limit)
                if subset_limit and subset_limit > 0:
                    try:
                        actual_len = len(grid_train_ds)  # type: ignore[arg-type]
                    except Exception:
                        actual_len = None
                    msg_subset = f"⚙️  Mini LR grid limiting TTFT dataset to {subset_limit} example(s)"
                    if actual_len is not None and actual_len != subset_limit:
                        msg_subset += f" (actual: {actual_len})"
                    print(msg_subset)

                grid_result = _run_ttt_mini_lr_grid(mc, grid_train_ds, args, base_lr_value, grid_cfg, candidates)
                if grid_result is not None:
                    success, metrics, best_info, trial_records = grid_result
                    best_lr = None
                    mini_metrics = metrics.get("mini_lr_grid") if isinstance(metrics, dict) else None
                    if isinstance(mini_metrics, dict):
                        best_lr = mini_metrics.get("best_learning_rate")
                    finalize_setting = grid_cfg.get("finalize_full_run")
                    finalize_full_run = True if finalize_setting is None else _truthy(finalize_setting)
                    best_path = Path(best_info["path"])
                    if success and finalize_full_run and best_lr:
                        print(
                            f"🔁 Re-running TTFT on full dataset with best learning rate {best_lr:.6g} "
                            "(mini LR grid finalize)."
                        )
                        final_args = dict(args)
                        final_args["learning_rate"] = best_lr
                        final_args["save_strategy"] = args.get("save_strategy", "steps")
                        final_args["save_optimizer"] = args.get("save_optimizer", True)
                        if "save_total_limit" in args:
                            final_args["save_total_limit"] = args["save_total_limit"]
                        if args.get("max_steps") and args["max_steps"] > 0:
                            final_args["max_steps"] = args["max_steps"]
                        else:
                            final_args["max_steps"] = -1
                        final_metrics = None
                        try:
                            final_metrics = train_distributed(mc.model_path, train_ds_core, config.RELOAD_PATH, final_args)
                            print("TTT (final) successful. Using:", config.RELOAD_PATH)
                            if isinstance(mini_metrics, dict):
                                mini_metrics["finalize_full_run"] = {
                                    "learning_rate": best_lr,
                                    "metrics": final_metrics,
                                }
                            if isinstance(metrics, dict):
                                if final_metrics:
                                    metrics["final_ttt_metrics"] = final_metrics
                            elif final_metrics:
                                metrics = final_metrics
                        except Exception as exc:
                            print(
                                f"⚠️  Final TTFT run with best learning rate failed: {exc}. "
                                "Continuing with best mini-grid weights.",
                                file=sys.stderr,
                            )
                            traceback.print_exc(file=sys.stderr)
                        finally:
                            if best_path.exists():
                                cleanup_directories([str(best_path)])
                    else:
                        if best_path.exists():
                            cleanup_directories([str(best_path)])
                    if success:
                        print("TTT successful. Using:", config.RELOAD_PATH)
                    else:
                        print("TTT mini grid search failed to produce a model.", file=sys.stderr)
                    _safe_gpu_clear()
                    return success, metrics
    elif raw_grid_cfg:
        print("⚠️  Mini LR grid configured but disabled; skipping sweep.", file=sys.stderr)

    try:
        metrics = train_distributed(mc.model_path, train_ds_core, config.RELOAD_PATH, args)
        print("TTT successful. Using:", config.RELOAD_PATH)
        return True, metrics
    except Exception as e:
        msg = f"TTT failed for {mc.model_basename}: {e}"
        print(msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, None
    finally:
        _safe_gpu_clear()


def _infer_for_payload(model_for_run: str, inf_payload: dict[str, Any], mc: ModelContext) -> dict[str, Any]:
    if not inf_payload:
        print("No inference data; skipping inference.")
        return {}
    print(f"Running inference on {len(inf_payload)} active tasks…")
    settings = build_inference_settings(mc.settings, mc.model_path, mc.airv_enabled)
    if mc.is_coda:
        mode = settings.get("generation_mode")
        if isinstance(mode, str) and mode.strip().lower() == "diffusion":
            _print_phase_marker(mc, "inference")
            summary = _format_coda_inference_summary(settings)
            if summary:
                print(summary)
            preset_summary = _format_coda_quality_preset(settings)
            if preset_summary:
                print(preset_summary)
    try:
        return predict_distributed(model_for_run, inf_payload, list(inf_payload.keys()), settings)
    except Exception as e:
        msg = f"Inference failed for {mc.model_basename}: {e}"
        print(f"🔴 {msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {}
    finally:
        _safe_gpu_clear()


def _postprocess_and_aggregate(
    raw_pred: dict[str, Any],
    mc: ModelContext,
    state,
    model_idx: int,
    run_num: int,
    averaged_mode: bool = False,
) -> dict[str, int]:
    ctx = PostprocessContext(
        model_basename=mc.model_basename,
        model_settings=mc.settings,
        per_run_store=state.per_run_predictions,
        per_model_store=state.per_model_predictions,
        aggregated_store=state.aggregated_predictions,
        model_idx=model_idx,
        run_idx=run_num,
        enable_self_ensemble=mc.ensemble["enable_self_ensemble"],
        enable_model_ensemble=False,
        averaged_mode=averaged_mode,
    )
    run_delta, _, _ = postprocess_predictions(raw_pred, state.counters, ctx)
    # Rebuild global aggregation explicitly after each run
    state.aggregated_predictions = _rebuild_aggregated_predictions(
        state.per_model_predictions,
        state.model_ensemble_flags,
        model_idx,
        list(state.aggregated_predictions.keys()),
    )
    return run_delta


def _score_and_log(
    runtime,
    mc: ModelContext,
    tasks,
    state,
    run_logger,
    run_num: int,
    run_stats: dict[str, Any],
    *,
    skip_visuals: bool,
) -> None:
    print(f"\n📊 Scoring progress after Run {run_num + 1} …")
    run_aug_meta = {
        "total_model_outputs": state.counters["total_raw"],
        "unparseable_outputs": state.counters["total_raw"] - state.counters["total_parsed"],
        "run_info": f"Model {mc.model_basename}, Run {run_num + 1}/{mc.self_ensemble_runs}",
    }
    inter = score_if_solutions_available(
        config.SUBMISSION_PATH,
        config.DATA_PATH,
        tasks.loaded_keys,
        run_aug_meta,
        run_logger.run_data,
        state.aggregated_predictions,
        skip_visuals=skip_visuals,
    )
    if not inter:
        print("📝 No scoring available")
        return

    overall = inter.get("overall_stats", {})
    task_results = inter.get("task_results", [])
    overall_score = overall.get("overall_top2_score", 0.0)
    solved = sum(1 for t in task_results if t.get("task_score", 0) > 0)
    total_tasks = overall.get("total_tasks", len(tasks.loaded_keys))
    score_delta = overall_score - state.last_score
    solved_delta = solved - state.last_solved
    solved_pct = (solved / total_tasks * 100) if total_tasks else 0.0

    print(
        "🎯 Current Score: "
        f"{overall_score:.3f} ({score_delta:+.3f}) | "
        f"Solved: {solved}/{total_tasks} ({solved_delta:+d}) ({solved_pct:.1f}%)"
    )

    run_errors = state.error_tracker["current_run_errors"].copy()
    state.error_tracker["current_run_errors"].clear()
    state.progress_log.append(
        {
            "checkpoint": f"Model {mc.model_basename} Run {run_num + 1}",
            "score": overall_score,
            "score_delta": score_delta,
            "solved": solved,
            "solved_delta": solved_delta,
            "total": total_tasks,
            "active_remaining": len(tasks.active_keys),
            "time_elapsed": time.time() - runtime.start_time,
            "errors": run_errors,
            "has_errors": len(run_errors) > 0,
        }
    )
    state.last_score, state.last_solved = overall_score, solved

    # Update run_stats for the logger
    run_stats["score"] = overall_score
    run_stats["solved_tasks"] = solved
    run_stats["total_tasks"] = total_tasks


def _filter_active_tasks(
    state,
    active_task_keys: list[str],
    current_model_settings: dict,
) -> tuple[list[str], dict[str, Any]]:
    print("🔍 Applying confidence-based filtering…")
    active_task_keys, filt = apply_confidence_filter(
        state.aggregated_predictions,
        active_task_keys,
        current_model_settings,
        config.DATA_PATH,
    )
    print(f"Active tasks remaining: {len(active_task_keys)}")
    return active_task_keys, (filt or {})


# -----------------------------------------------------------------------------
# Pretraining hook extracted intact to a helper to reduce core function size
# -----------------------------------------------------------------------------

def _maybe_run_pretraining_hook(runtime, mc: ModelContext, state) -> str:
    """Run optional pretraining before TTT/inference phases."""
    training_cfg = mc.settings.get("training") if isinstance(mc.settings, dict) else {}
    if not _pretraining_requested(training_cfg):
        return mc.model_path
    if config.USE_FLAX:
        print("⚠️ Pretraining skipped: TPU/Flax path not supported by GPU trainer.")
        return mc.model_path

    explicit_steps = _resolve_pretraining_steps(training_cfg or {})
    record_limit = _resolve_pretraining_record_limit(training_cfg or {}, explicit_steps)
    loader_limit = None if record_limit == -1 else record_limit
    if record_limit == 0:
        print("ℹ️ Pretraining skipped: pretraining record limit set to 0.")
        return mc.model_path

    seed = _resolve_pretraining_seed(training_cfg or {})
    rng = random.Random(seed) if seed is not None else random.Random()
    verbose = getattr(config, "VERBOSE_LOGGING", False) or _truthy(os.environ.get("PRETRAINING_VERBOSE"))

    try:
        records, meta = load_pretraining_records(
            training_cfg or {},
            limit=loader_limit,
            rng=rng,
            verbose=verbose,
        )
    except PretrainingLoadError as exc:
        message = str(exc).strip()
        if message and message != "Config does not specify a pretraining source.":
            print(f"⚠️ Pretraining skipped: {message}")
        return mc.model_path

    if not records:
        print("⚠️ Pretraining skipped: no records available after sampling.")
        return mc.model_path

    try:
        dataset = _build_pretraining_dataset(records)
    except PretrainingLoadError as exc:
        print(f"⚠️ Pretraining skipped: {exc}")
        return mc.model_path

    prompt_prefix = getattr(mc, "prompt_prefix", "")
    if prompt_prefix:
        try:
            def _apply_prompt_prefixes(example):
                example["prompt"] = ensure_prompt_prefix(example.get("prompt"), prompt_prefix)
                return example

            dataset = dataset.map(_apply_prompt_prefixes)
        except Exception as map_err:
            print(f"⚠️ Pretraining prefix map failed: {map_err}")

    dataset_len = len(dataset)
    if dataset_len == 0:
        print("⚠️ Pretraining skipped: dataset empty after normalization.")
        return mc.model_path

    meta = dict(meta)
    meta["records_loaded"] = dataset_len

    print("\n=== Pretraining Phase ===")
    source_label = meta.get("source", "unknown source")
    limit_display: str | int = "auto"
    if record_limit == -1:
        limit_display = "unlimited"
    elif record_limit is not None:
        limit_display = record_limit

    print(
        f"Preparing {dataset_len} record(s) from {source_label} "
        f"(limit={limit_display})"
    )
    files_detail = meta.get("files_detail") or []
    if files_detail:
        print(f"  • Files used: {len(files_detail)}")
        max_lines = 5
        for entry in files_detail[:max_lines]:
            file_path = entry.get("path", "")
            display = Path(file_path).name if file_path else str(file_path)
            try:
                count = int(entry.get("records", 0))
            except Exception:
                count = entry.get("records")
            print(f"     - {display}: {count} record(s)")
        remaining = len(files_detail) - max_lines
        if remaining > 0:
            print(f"     - … {remaining} more file(s)")
    elif meta.get("files_used"):
        print(f"  • Files used: {len(meta['files_used'])}")
    if meta.get("records_generated") and meta["records_generated"] != dataset_len:
        print(f"  • Sampled {dataset_len} of {meta['records_generated']} available records")

    args = _build_train_args(
        mc,
        seed=seed if seed is not None else int(time.time() * 1000) % 100000,
        enable_tf=mc.get("enable_token_filtering", "general", True),
        phase="pretraining",
    )
    args["pretraining"] = True
    args["pretraining_metadata"] = meta
    if hasattr(runtime.run_logger, "run_id"):
        args["run_id"] = runtime.run_logger.run_id
    args["pretraining_records"] = dataset_len
    objective_name = args.get("pretraining_objective")
    if objective_name:
        print(f"  • Objective: {str(objective_name).upper()}")
        settings_blob = args.get("pretraining_objective_settings")
        if settings_blob:
            try:
                settings_dict = json.loads(settings_blob) if isinstance(settings_blob, str) else settings_blob
            except Exception:
                settings_dict = None
            if isinstance(settings_dict, dict):
                mix_cfg = settings_dict.get("mixture") if "mixture" in settings_dict else settings_dict
                if isinstance(mix_cfg, dict):
                    mix_parts: list[str] = []
                    for key, cfg in mix_cfg.items():
                        name = str(key).upper()
                        weight = cfg.get("weight")
                        if name in {"R", "X"}:
                            mix_parts.append(
                                f"{name}:{weight} (noise={cfg.get('noise_density')}, span={cfg.get('mean_span_length')})"
                            )
                        elif name == "S":
                            mix_parts.append(
                                f"{name}:{weight} (prefix=[{cfg.get('min_prefix_ratio')}, {cfg.get('max_prefix_ratio')}])"
                            )
                        else:
                            mix_parts.append(f"{name}:{weight}")
                    if mix_parts:
                        print(f"  • Objective mix: {', '.join(mix_parts)}")

    steps_applied = _apply_pretraining_step_budget(args, dataset_len, explicit_steps)
    save_default = mc.get("save_steps", "training", None)
    save_steps = _determine_save_steps(dataset_len, args, save_default)
    args.update({"save_strategy": "steps", "save_steps": save_steps})
    if steps_applied == "unlimited":
        print(f"  • Step budget: unlimited (manual stop required, save every {save_steps} step(s))")
    else:
        print(f"  • Step budget: {steps_applied} step(s) (save every {save_steps} step(s))")

    pretrained_root = Path(os.environ.get("ARC_PRETRAINED_DIR") or "model_pretrained").expanduser()
    pretrained_root.mkdir(parents=True, exist_ok=True)
    target_dir = _resolve_pretrained_output_dir(pretrained_root, mc.model_basename, training_cfg or {}).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_str = str(target_dir)
    args["metrics_output"] = os.path.join(target_dir_str, "pretraining_metrics.json")
    print(f"  • Output directory: {target_dir_str}")

    source_model_path = mc.model_path
    metrics = None
    try:
        metrics = train_distributed(source_model_path, dataset, target_dir_str, args)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"❌ Pretraining failed for {mc.model_basename}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return mc.model_path
    finally:
        _safe_gpu_clear()

    if metrics:
        print(f"  • Training metrics: {metrics}")
    print(f"✅ Pretraining complete. Using {target_dir_str} for subsequent phases.")

    if runtime.session_logger is not None:
        runtime.session_logger.log_custom_event(
            "pretraining",
            f"Pretrained {mc.model_basename}",
            {
                "records": dataset_len,
                "source": source_label,
                "output_dir": target_dir_str,
                "steps": steps_applied,
            },
        )

    new_model_path = target_dir_str
    config.MODEL_SETTINGS.setdefault(new_model_path, mc.settings)
    config.MODEL_PATHS[mc.model_idx] = new_model_path
    if state.model_paths_for_tracking:
        state.model_paths_for_tracking[-1] = new_model_path
    try:
        runtime.run_logger.data["models"][mc.model_idx]["model_path"] = new_model_path
        runtime.run_logger.data["models"][mc.model_idx]["model_basename"] = os.path.basename(new_model_path)
        runtime.run_logger._save()
    except Exception:
        pass

    mc.model_path = new_model_path
    return new_model_path


# -----------------------------------------------------------------------------
# Main orchestration refactor
# -----------------------------------------------------------------------------

def _process_single_model(  # noqa: PLR0912, PLR0915 - parity with original control flow
    runtime,
    tasks,
    state,
    model_idx: int,
    model_path_current: str,
) -> None:
    """Refactored single-model processing pipeline.

    High-level flow:
      1) Create ModelContext + banner
      2) (Optional) pretraining hook
      3) Choose execution mode:
           a) Model-averaging mode (TTT → average → single inference)
           b) Standard mode (TTT + Inference per run)
      4) After runs: per-model scoring summary + scaling finalize
      5) Persist state last_score/solved and active tasks
    """
    run_logger = runtime.run_logger
    session_logger = runtime.session_logger
    scaling_collector = runtime.scaling_collector
    repo_state = runtime.repo_state
    device_summary = runtime.device_summary
    world_size = runtime.world_size
    start_time = runtime.start_time

    per_run_predictions = state.per_run_predictions
    per_model_predictions = state.per_model_predictions
    model_paths_for_tracking = state.model_paths_for_tracking
    counters = state.counters
    global_aug_stats = state.global_aug_stats
    global_filter_stats = state.global_filter_stats

    active_task_keys = tasks.active_keys
    all_task_data = tasks.all_task_data
    all_task_keys_for_submission = tasks.submission_keys
    all_task_keys_loaded = tasks.loaded_keys

    total_models = len(config.MODEL_PATHS)

    current_model_settings = config.MODEL_SETTINGS.get(model_path_current, {})
    setting_getter = _model_getter(current_model_settings, model_path_current)

    model_basename = os.path.basename(model_path_current)
    logged_model_idx = run_logger.log_model_start(model_path_current, current_model_settings)
    session_logger.log_model_usage(model_basename, current_model_settings)

    prompt_prefix = _get_prompt_prefix_for_model(model_path_current)

    prompt_format = str(setting_getter("prompt_format", "general", "legacy") or "legacy").lower()

    mc = ModelContext(
        model_idx=model_idx,
        model_path=model_path_current,
        model_basename=model_basename,
        settings=current_model_settings,
        get=setting_getter,
        is_causal=(setting_getter("model_type", "general", "seq2seq") == "causal_lm"),
        is_coda=bool(
            setting_getter("coda_enabled", "general", False)
            or str(setting_getter("generation_mode", "inference", "") or "").strip().lower() == "diffusion"
        ),
        ttt_enabled=setting_getter("ttt_enabled", "general", True) and config.GLOBAL_TTT_ENABLED,
        airv_enabled=setting_getter("airv_enabled", "general", True),
        self_ensemble_runs=int(setting_getter("self_ensemble_count", "training", 1) or 1),
        task_group_size=setting_getter("task_group_size", "training", None),
        model_averaging=bool(setting_getter("model_averaging", "training", False)),
        ensemble=get_ensemble_settings(model_path_current),
        scaling_enabled_for_model=bool(
            _is_scaling_enabled()
            or (get_scaling_settings(model_path_current) or {}).get("enabled_default", False)
        ),
        prompt_prefix=prompt_prefix,
        prompt_format=prompt_format,
        power_sampling_cfg=_normalize_power_sampling_config(current_model_settings),
    )

    print(
        "🔍 Debug: "
        f"model_scaling_enabled={mc.scaling_enabled_for_model}, SCALING_ENABLED={_is_scaling_enabled()}, "
        f"scaling_settings={get_scaling_settings(model_path_current)}"
    )
    print(f"🔍 Debug: scaling_collector is {'None' if scaling_collector is None else 'initialized'}")

    _summarize_model_context(model_idx, total_models, mc)

    model_paths_for_tracking.append(mc.model_path)
    per_run_predictions[model_idx] = {}
    per_model_predictions[model_idx] = {k: [] for k in all_task_keys_for_submission}

    ttft_phase_context: PhaseContext | None = None
    last_ttt_items_for_scaling: list[Any] | None = None
    last_ttft_train_metrics: dict[str, Any] | None = None
    ood_panel_info: dict[str, Any] | None = None
    if mc.scaling_enabled_for_model and scaling_collector is not None:
        try:
            ttft_phase_context = PhaseContext(
                run_logger.run_id,
                "ttft_airv",
                {
                    "model_name": mc.model_basename,
                    "scaling_enabled": True,
                },
                time_start=time.time(),
            )
            scaling_collector.start_phase(ttft_phase_context)
        except Exception as exc:
            print(f"⚠️ Scaling phase start failed: {exc}")
            ttft_phase_context = None
        try:
            ood_cfg = (get_scaling_settings(mc.model_path) or {}).get("ood_panel", {})
            if ood_cfg.get("enabled"):
                ood_panel_info = load_ood_panel(ood_cfg)
        except Exception as exc:
            print(f"⚠️ OOD panel load failed: {exc}")

    # ---- Pretraining hook (parity) ----
    mc.model_path = _maybe_run_pretraining_hook(runtime, mc, state)

    # ---- Optional task grouping ----
    model_task_groups, _grouping_total = _prepare_task_groups(active_task_keys, mc.task_group_size)

    # ---- Execution modes ----
    if (
        mc.model_averaging
        and mc.ttt_enabled
        and mc.self_ensemble_runs > 1
        and not mc.model_basename.startswith("local-dummy")
    ):
        # Model-averaging path: N TTT iterations → (optional) average → single inference
        print(f"\n🔄 Model Averaging Mode: {mc.self_ensemble_runs} TTT iterations then single inference")
        ensemble_paths = get_model_ensemble_paths(config.RELOAD_PATH, mc.self_ensemble_runs)
        successful_paths: list[str] = []

        print(f"\n=== TTT Training Phase ({mc.self_ensemble_runs} iterations) ===")
        for run_num in range(mc.self_ensemble_runs):
            if (time.time() - start_time + config.BUFFER_TIME) >= config.TOTAL_TIME:
                print(f"Time limit during TTT run {run_num + 1}. Stopping.", file=sys.stderr)
                break
            if model_task_groups and run_num < len(model_task_groups):
                cur_run_keys = model_task_groups[run_num]
            else:
                cur_run_keys = active_task_keys
            run_output_path = ensemble_paths[run_num]
            print(
                "\n--- TTT Run "
                f"{run_num + 1}/{mc.self_ensemble_runs} "
                f"({len(cur_run_keys)} tasks) -> {run_output_path} ---"
            )

            # Build datasets
            ttt_items, _, ds_stats = _make_ttt_and_infer_datasets(
                mc,
                all_task_data,
                cur_run_keys,
                airv_enabled_for_run=mc.airv_enabled,
            )
            update_augmentation_stats(global_aug_stats, ds_stats)
            if ttt_items:
                last_ttt_items_for_scaling = ttt_items
            train_ds = _prep_ttt_dataset(ttt_items)

            success = False
            if train_ds:
                args = _build_train_args(
                    mc,
                    seed=int(time.time() * 1000) % 100000 + run_num,
                    enable_tf=mc.get("enable_token_filtering", "general", True),
                    phase="ttt",
                )
                save_default = mc.get("save_steps", "training", None)
                save_steps = _determine_save_steps(len(train_ds), args, save_default)
                args.update({"save_steps": save_steps, "save_strategy": "steps"})
                print(f"  • Checkpoint interval: every {save_steps} step(s) (SAVE_STEPS={config.SAVE_STEPS})")
                if hasattr(run_logger, "run_id"):
                    args["run_id"] = run_logger.run_id
                try:
                    metrics = train_distributed(mc.model_path, train_ds, run_output_path, args)
                    if metrics:
                        last_ttft_train_metrics = metrics
                    saved = _directory_has_model_weights(run_output_path)
                    if saved:
                        print(f" ✅ TTT ensemble {run_num + 1} completed successfully")
                        success = True
                    else:
                        contents = os.listdir(run_output_path) if os.path.exists(run_output_path) else []
                        print(f" ❌ TTT ensemble {run_num + 1} failed (no model file). Files: {contents}")
                except Exception as e:
                    print(f" ❌ TTT run error: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                finally:
                    _safe_gpu_clear()

            final_model_path = run_output_path if success else mc.model_path
            successful_paths.append(final_model_path)

        # Average paths if applicable
        if len(successful_paths) >= 2:
            print(f"\n=== Averaging {len(successful_paths)} models ===")
            _safe_gpu_clear()
            averaged_path = f"{config.RELOAD_PATH}_averaged"
            if validate_models_for_averaging(successful_paths):
                ok = average_model_weights(successful_paths, averaged_path, device="cpu")
                model_for_infer = averaged_path if ok else successful_paths[0]
                if ok:
                    print(f"✅ Averaging successful -> {averaged_path}")
                    cleanup_ensemble_models(successful_paths, keep_averaged=True)
                else:
                    print("❌ Averaging failed. Using first successful model.")
            else:
                print("❌ Model validation failed. Using first successful model.")
                model_for_infer = successful_paths[0]
        elif len(successful_paths) == 1:
            print(f"⚠️ Only one successful TTT model. Using it directly: {successful_paths[0]}")
            model_for_infer = successful_paths[0]
        else:
            print(f"❌ No successful TTT models. Using original: {mc.model_path}")
            model_for_infer = mc.model_path

        # Single inference pass with averaged/best model
        data_for_keys = {k: all_task_data[k] for k in active_task_keys if k in all_task_data}
        target_inf = mc.get("target_total_inference_items", "inference", 0)
        max_in, max_tg = mc.get("max_input_length", "general", 2048), mc.get("max_target_length", "general", 512)
        enable_tf = mc.get("enable_token_filtering", "general", True)
        _, inf_data, ds_stats = make_datasets(
            data_for_keys,
            list(data_for_keys.keys()),
            0,
            target_inf,
            max_in,
            max_tg,
            mc.model_path,
            airv_enabled_for_model=mc.airv_enabled,
            enable_token_filtering=enable_tf,
        )
        update_augmentation_stats(global_aug_stats, ds_stats)
        inf_payload = {k: v for k, v in inf_data.items() if k in active_task_keys and v.get("tasks")}
        run_stats = {
            "start_time": time.time(),
            "ttt_items": len(successful_paths),
            "inference_items": sum(len(v.get("tasks", [])) for v in inf_payload.values()) if inf_payload else 0,
            "active_tasks_before": len(active_task_keys),
            "raw_outputs": 0,
            "parsed_outputs": 0,
            "valid_outputs": 0,
        }
        if inf_payload:
            print("Running inference on averaged/best model…")
            raw_pred = _infer_for_payload(model_for_infer, inf_payload, mc)
            run_delta = _postprocess_and_aggregate(raw_pred, mc, state, model_idx, 0, averaged_mode=True)
            run_stats.update(run_delta)
            make_submission(state.aggregated_predictions, all_task_keys_for_submission, config.SUBMISSION_PATH)
            active_task_keys, filt = _filter_active_tasks(state, active_task_keys, current_model_settings)
            if filt:
                global_filter_stats.update({
                    "total_filtering_calls": global_filter_stats["total_filtering_calls"] + 1,
                    "total_tasks_filtered": filt.get("filtered_total", 0),
                    "total_correct_filtered": filt.get("filtered_correct", 0),
                    "total_remaining_tasks": filt.get("remaining_total", 0),
                    "total_correct_remaining": filt.get("remaining_correct", 0),
                })
                global_filter_stats["filtering_history"].append({
                    "model": mc.model_basename,
                    "run": 1,
                    "stage": "model_averaging",
                    "filtered_correct": filt.get("filtered_correct", 0),
                    "filtered_total": filt.get("filtered_total", 0),
                    "remaining_correct": filt.get("remaining_correct", 0),
                    "remaining_total": filt.get("remaining_total", 0),
                    "current_threshold": filt.get("current_threshold"),
                    "optimal_threshold": filt.get("optimal_threshold"),
                })
            run_logger.log_model_run(logged_model_idx, 0, run_stats)
        else:
            print("No active tasks for inference.")
        run_logger.log_model_end(logged_model_idx)

    else:
        # Standard multi-run path
        print(f"\n📝 Standard Mode: {mc.self_ensemble_runs} combined TTT+Inference run(s)")
        # Build run schedule if grouped
        if model_task_groups:
            group_count = len(model_task_groups)
            if mc.self_ensemble_runs > group_count:
                schedule = list(range(group_count)) + [0] * (mc.self_ensemble_runs - group_count)
            else:
                schedule = list(range(group_count))
        else:
            schedule = list(range(mc.self_ensemble_runs))

        planned_runs = len(schedule)
        zero_shot_logged = False
        airv_only_logged = False
        ttft_only_logged = False

        for s_idx, selector in enumerate(schedule):
            run_num = s_idx
            if (time.time() - start_time + config.BUFFER_TIME) >= config.TOTAL_TIME:
                print(f"Time limit during run {run_num + 1} for {mc.model_basename}. Stopping.", file=sys.stderr)
                break

            base_airv = mc.get("airv_enabled", "general", True)
            iterative_mode = mc.ensemble.get("iterative_ensemble_mode", False)
            airv_last_only = mc.ensemble.get("airv_last_cycle_only", False) if iterative_mode else False
            airv_enabled_run = (
                (run_num == planned_runs - 1 and base_airv)
                if airv_last_only
                else base_airv
            )

            run_start = time.time()
            cur_run_keys = model_task_groups[selector] if model_task_groups else active_task_keys
            per_run_predictions[model_idx][run_num] = {k: [] for k in all_task_keys_for_submission}
            if not cur_run_keys:
                print("No active tasks. Skipping.", file=sys.stderr)
                break

            # Datasets
            ttt_items, inf_data, ds_stats = _make_ttt_and_infer_datasets(
                mc,
                all_task_data,
                cur_run_keys,
                airv_enabled_for_run=airv_enabled_run,
            )
            update_augmentation_stats(global_aug_stats, ds_stats)
            if ttt_items:
                last_ttt_items_for_scaling = ttt_items

            run_stats = {
                "start_time": run_start,
                "ttt_items": len(ttt_items or []),
                "inference_items": sum(len(v.get("tasks", [])) for v in inf_data.values()),
                "active_tasks_before": len(cur_run_keys),
                "raw_outputs": 0,
                "parsed_outputs": 0,
                "valid_outputs": 0,
            }

            # Iterative ensemble support: choose start model
            iterative_ens = mc.ensemble.get("iterative_ensemble_mode", False)
            if iterative_ens and mc.task_group_size and mc.task_group_size > 0:
                # Alternate between base and fine-tuned
                if not hasattr(_process_single_model, "_alt_state"):
                    _process_single_model._alt_state = {"next_use_original": True}  # type: ignore[attr-defined]
                alt = _process_single_model._alt_state  # type: ignore[attr-defined]
                if alt["next_use_original"] or not os.path.exists(config.RELOAD_PATH):
                    start_model = mc.model_path
                    alt["next_use_original"] = False
                    if os.path.exists(config.RELOAD_PATH) and not config.PRESERVE_FINETUNED_MODEL:
                        print(f"🔧 Cleaning stale fine-tuned directory: {config.RELOAD_PATH}")
                        cleanup_directories([config.RELOAD_PATH])
                    os.makedirs(config.RELOAD_PATH, exist_ok=True)
                    with open(os.path.join(config.RELOAD_PATH, "BASE_MODEL_SOURCE.txt"), "w") as fh:
                        fh.write(f"run={run_num + 1}\nbase_model_path={start_model}\n")
                else:
                    start_model = config.RELOAD_PATH
                    alt["next_use_original"] = True
                    print(f"🔁 Using FINE-TUNED base for run {run_num + 1}: {start_model}")
            else:
                start_model = (
                    mc.model_path
                    if run_num == 0
                    else (config.RELOAD_PATH if os.path.exists(config.RELOAD_PATH) else mc.model_path)
                )

            model_for_run = start_model

            # Scaling: optional zero-shot + AIRV-only bookkeeping
            if mc.scaling_enabled_for_model and scaling_collector is not None:
                try:
                    if not zero_shot_logged:
                        run_scaling_inference_phase(
                            "zero_shot", mc.model_path, mc.settings, all_task_data,
                            all_task_keys_for_submission, active_task_keys, scaling_collector, run_logger,
                            repo_state, device_summary, world_size, airv_enabled=False, data_path=config.DATA_PATH,
                            get_setting=mc.get, build_inference_settings_fn=build_inference_settings,
                            postprocess_fn=_scaling_postprocess_wrapper, ood_panel_info=ood_panel_info,
                        )
                        zero_shot_logged = True
                except Exception as exc:
                    print(f"⚠️ Zero-shot scaling phase failed: {exc}")
                try:
                    if not airv_only_logged and mc.airv_enabled:
                        _print_phase_marker(mc, "airv")
                        run_scaling_inference_phase(
                            "airv_only", mc.model_path, mc.settings, all_task_data,
                            all_task_keys_for_submission, active_task_keys, scaling_collector, run_logger,
                            repo_state, device_summary, world_size, airv_enabled=True, data_path=config.DATA_PATH,
                            get_setting=mc.get, build_inference_settings_fn=build_inference_settings,
                            postprocess_fn=_scaling_postprocess_wrapper, ood_panel_info=ood_panel_info,
                        )
                        airv_only_logged = True
                except Exception as exc:
                    print(f"⚠️ AIRV-only scaling phase failed: {exc}")

            # TTT
            train_ds = _prep_ttt_dataset(ttt_items)
            ttft_train_metrics = None
            if mc.ttt_enabled and train_ds and not mc.model_basename.startswith("local-dummy"):
                ok, ttft_train_metrics = _run_ttt(mc, runtime, run_logger, run_num, train_ds)
                if ok:
                    if ttft_train_metrics:
                        last_ttft_train_metrics = ttft_train_metrics
                    model_for_run = config.RELOAD_PATH
                    if mc.scaling_enabled_for_model and scaling_collector is not None and not ttft_only_logged:
                        try:
                            run_scaling_inference_phase(
                                "ttft_only", model_for_run, mc.settings, all_task_data,
                                all_task_keys_for_submission, active_task_keys, scaling_collector, run_logger,
                                repo_state, device_summary, world_size, airv_enabled=False, data_path=config.DATA_PATH,
                                get_setting=mc.get, build_inference_settings_fn=build_inference_settings,
                                postprocess_fn=_scaling_postprocess_wrapper, extra_metrics=ttft_train_metrics,
                                ttft_task_rows=build_ttft_task_rows(ttt_items, ttft_train_metrics),
                                ood_panel_info=ood_panel_info,
                            )
                            ttft_only_logged = True
                        except Exception as exc:
                            print(f"⚠️ TTFT-only scaling phase failed: {exc}")
            _safe_gpu_clear()

            # Inference
            inf_payload = {k: v for k, v in inf_data.items() if k in active_task_keys and v.get("tasks")}
            if inf_payload:
                if mc.model_basename.startswith("local-dummy"):
                    raw_pred = {
                        k: [
                            {
                                "texts": ["5 2 2 01 01 10", "5 2 2 01 10 01"],
                                "decoder": {"geom_name": "identity", "colmap_inv": {0: 0, 1: 1}},
                            }
                        ]
                        for k in inf_payload
                    }
                else:
                    raw_pred = _infer_for_payload(model_for_run, inf_payload, mc)
                run_delta = _postprocess_and_aggregate(raw_pred, mc, state, model_idx, run_num)
                run_stats.update(run_delta)
            else:
                print("No inference data; skipping inference.")

            # Cleanup fine-tuned model if not needed
            grouped_alt = (
                mc.ensemble.get("iterative_ensemble_mode", False)
                and mc.task_group_size
                and mc.task_group_size > 0
            )
            if not grouped_alt and model_for_run == config.RELOAD_PATH and os.path.exists(config.RELOAD_PATH):
                if not config.PRESERVE_FINETUNED_MODEL:
                    print(f"Cleaning fine-tuned model directory: {config.RELOAD_PATH}")
                    cleanup_directories([config.RELOAD_PATH])
                else:
                    print(f"🔒 PRESERVE_FINETUNED_MODEL=True; keeping {config.RELOAD_PATH}")

            # Write submission
            print("Updating submission…")
            make_submission(state.aggregated_predictions, all_task_keys_for_submission, config.SUBMISSION_PATH)

            # Score + filter
            _score_and_log(
                runtime,
                mc,
                tasks,
                state,
                run_logger,
                run_num,
                run_stats,
                skip_visuals=mc.scaling_enabled_for_model,
            )

            if active_task_keys:
                active_task_keys, filt = _filter_active_tasks(state, active_task_keys, current_model_settings)
                if filt:
                    global_filter_stats["total_filtering_calls"] += 1
                    global_filter_stats["total_tasks_filtered"] = filt.get("filtered_total", 0)
                    global_filter_stats["total_correct_filtered"] = filt.get("filtered_correct", 0)
                    global_filter_stats["total_remaining_tasks"] = filt.get("remaining_total", 0)
                    global_filter_stats["total_correct_remaining"] = filt.get("remaining_correct", 0)
                    entry = {
                        "model": mc.model_basename,
                        "run": run_num + 1,
                        "stage": "post_scoring",
                        "filtered_correct": filt.get("filtered_correct", 0),
                        "filtered_total": filt.get("filtered_total", 0),
                        "remaining_correct": filt.get("remaining_correct", 0),
                        "remaining_total": filt.get("remaining_total", 0),
                        "current_threshold": filt.get("current_threshold"),
                        "optimal_threshold": filt.get("optimal_threshold"),
                    }
                    if "vote_counts" in filt:
                        entry["vote_counts"] = filt["vote_counts"]
                    global_filter_stats["filtering_history"].append(entry)

            run_dur = time.time() - run_start
            run_stats["duration"] = run_dur
            run_stats["active_tasks_after"] = len(active_task_keys)
            run_logger.log_model_run(logged_model_idx, run_num + 1, run_stats)
            session_logger.log_run_completion(
                run_stats.get("score", 0),
                {
                    "model": mc.model_basename,
                    "run_number": run_num + 1,
                    "duration": run_dur,
                    "solved_tasks": run_stats.get("solved_tasks", 0),
                    "total_tasks": run_stats.get("total_tasks", 0),
                    "active_tasks_remaining": len(active_task_keys),
                },
            )
            elapsed = time.time() - start_time
            print(
                f"Run {run_num + 1} complete in {run_dur:.1f}s | "
                f"elapsed {elapsed:.1f}s | active remaining {len(active_task_keys)}"
            )

        run_logger.log_model_end(logged_model_idx)

    # ---- Per-model scoring summary (preserved) ----
    print(
        f"\n🎯 SCORING SUMMARY AFTER MODEL {model_idx + 1}/{len(config.MODEL_PATHS)}: {mc.model_basename}\n"
        f"{'=' * 60}"
    )
    model_aug_meta = {
        "total_model_outputs": counters["total_raw"],
        "unparseable_outputs": counters["total_raw"] - counters["total_parsed"],
        "model_info": f"After all runs for {mc.model_basename}",
    }
    run_log_data = run_logger.run_data
    mres = score_if_solutions_available(
        config.SUBMISSION_PATH,
        config.DATA_PATH,
        all_task_keys_loaded,
        model_aug_meta,
        run_log_data,
        state.aggregated_predictions,
        skip_visuals=mc.scaling_enabled_for_model,
    )
    if mres:
        overall = mres.get("overall_stats", {})
        score = overall.get("overall_top2_score", 0.0)
        top1_score = overall.get("overall_top1_score")
        task_res = mres.get("task_results", [])
        solved = sum(1 for t in task_res if t.get("task_score", 0) > 0)
        total_tasks = overall.get("total_tasks", len(all_task_keys_loaded))
        solved_pct = (solved / total_tasks * 100) if total_tasks else 0.0
        print(
            f"🏆 Final Score: {score:.3f}\n"
            f"✅ Tasks Solved: {solved}/{total_tasks} ({solved_pct:.1f}%)\n"
            f"🔄 Active Remaining: {len(active_task_keys)}\n{'=' * 60}"
        )
    else:
        top1_score = None
        score = None
        solved = None
        total_tasks = None
        print("📝 No scoring available for this model\n{'=' * 60}")

    if (
        mc.scaling_enabled_for_model
        and scaling_collector is not None
        and ttft_phase_context is not None
    ):
        try:
            phase_end = time.time()
            devices_per_host = device_summary.get("devices_per_host")
            host_count = None
            try:
                if devices_per_host and devices_per_host > 0:
                    host_count = max(1, world_size // devices_per_host)
            except Exception:
                host_count = None
            metrics_row = {
                "time_start": ttft_phase_context.time_start,
                "time_end": phase_end,
                "commit_sha": repo_state.get("commit_sha"),
                "git_branch": repo_state.get("git_branch"),
                "git_dirty": repo_state.get("git_dirty"),
                "host_count": host_count,
                "devices_per_host": devices_per_host,
                "world_size": world_size,
                "device_type": device_summary.get("device_type"),
                "vram_gb": device_summary.get("vram_gb"),
                "driver_version": device_summary.get("driver_version"),
                "cuda_rocm_xla_version": device_summary.get("cuda_rocm_xla_version"),
                "scaling_enabled": True,
                "scaling_phases": "ttft_airv",
                "model_name": mc.model_basename,
                "ttt_enabled": mc.ttt_enabled,
                "airv_enabled": mc.airv_enabled,
                "items_solved": solved,
                "items_scored": total_tasks,
                "items_failed": (total_tasks - solved) if (total_tasks is not None and solved is not None) else None,
                "top1_pass": top1_score,
                "top2_pass": score,
                "beam_size": mc.get("num_beams", "inference", 1),
                "temperature": mc.get("temperature", "inference", None),
                "top_p": mc.get("top_p", "inference", None),
                "dataset_mixture_json": json.dumps({}),
                "delta_vs_prev_phase_json": json.dumps({}),
                "determinism_flags_json": json.dumps({"beam_size": mc.get("num_beams", "inference", 1)}),
                "shape_metadata_json": json.dumps({"model_type": mc.get("model_type", "general", "seq2seq")}),
                "precision_metadata_json": json.dumps({
                    "use_bf16": mc.get("use_bf16", "training", False),
                    "use_fp16": mc.get("use_fp16", "training", False),
                }),
            }
            if last_ttft_train_metrics:
                for key, value in last_ttft_train_metrics.items():
                    if key in SCALING_CSV_COLUMNS and value is not None:
                        metrics_row[key] = value
            ood_metrics_final: dict[str, Any] = {}
            if ood_panel_info:
                ood_metrics_final = _evaluate_ood_panel(
                    "ttft_airv",
                    mc.model_path,
                    mc.settings,
                    ood_panel_info,
                    airv_enabled=mc.airv_enabled,
                    baseline_score=score,
                )
                if ood_metrics_final.get("ood_panel_loss") is not None:
                    metrics_row["ood_panel_loss"] = ood_metrics_final["ood_panel_loss"]
                if ood_metrics_final.get("ood_offset_vs_indist") is not None:
                    metrics_row["ood_offset_vs_indist"] = ood_metrics_final["ood_offset_vs_indist"]
            summary_payload = {
                "run_id": run_logger.run_id,
                "phase": "ttft_airv",
                "model": mc.model_basename,
                "items_solved": solved,
                "items_scored": total_tasks,
                "score_top2": score,
                "duration_s": phase_end - ttft_phase_context.time_start,
                "ood_panel_loss": ood_metrics_final.get("ood_panel_loss") if ood_metrics_final else None,
            }
            final_ttft_rows = (
                build_ttft_task_rows(last_ttt_items_for_scaling, last_ttft_train_metrics)
                if last_ttft_train_metrics and last_ttt_items_for_scaling
                else None
            )
            scaling_collector.finalize_phase(
                metrics_row,
                summary=summary_payload,
                ttft_tasks=final_ttft_rows,
            )
        except Exception as exc:
            print(f"⚠️ Scaling metrics finalize failed: {exc}")

    tasks.active_keys = active_task_keys


def _print_finalization_banner(runtime: RuntimeContext, active_task_count: int) -> float:
    print("\n" + "=" * 70 + "\nALL MODEL PROCESSING COMPLETE\n" + "=" * 70)
    if _is_scaling_enabled() and runtime.scaling_collector is not None:
        try:
            runtime.scaling_collector.print_csv_to_logs()
        except Exception as exc:
            print(f"⚠️ Failed to print CSV data to logs: {exc}")
    total_time = time.time() - runtime.start_time
    print(
        f"Total execution time: {total_time:.1f}s\n"
        f"Active tasks remaining: {active_task_count}\n"
        f"Final submission: {config.SUBMISSION_PATH}"
    )
    return total_time


def _generate_submission_metadata(
    runtime: RuntimeContext,
    tasks: TaskLoadResult,
    state: PipelineState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregated_predictions = state.aggregated_predictions
    counters = state.counters
    global_aug_stats = state.global_aug_stats
    global_filter_stats = state.global_filter_stats
    per_run_predictions = state.per_run_predictions
    per_model_predictions = state.per_model_predictions
    model_paths_for_tracking = state.model_paths_for_tracking
    run_logger = runtime.run_logger

    make_submission(aggregated_predictions, tasks.submission_keys, config.SUBMISSION_PATH)
    print("Submission generation finished.\n--- Final Post-processing Stats (All Runs) ---")
    print(f"Total raw model outputs: {counters['total_raw']}")
    if counters["total_raw"] > 0:
        parse_perc = counters["total_parsed"] / counters["total_raw"] * 100
        dec_total_perc = counters["total_valid"] / counters["total_raw"] * 100
        print(
            f"Parsed by output_to_grid: {counters['total_parsed']} ({parse_perc:.2f}%)\n"
            f"Decoded & validated (of total raw): {counters['total_valid']} ({dec_total_perc:.2f}%)"
        )
    else:
        print(f"Parsed by output_to_grid: {counters['total_parsed']}\nDecoded & validated: {counters['total_valid']}")

    overall_stats = {
        "total_raw_outputs": counters["total_raw"],
        "total_parsed_outputs": counters["total_parsed"],
        "total_valid_outputs": counters["total_valid"],
    }
    task_stats = {
        "total_tasks_loaded": len(tasks.loaded_keys),
        "active_tasks_remaining": len(tasks.active_keys),
        "tasks_processed": len(tasks.loaded_keys) - len(tasks.active_keys),
    }
    run_logger.log_overall_stats(overall_stats)
    run_logger.log_task_stats(task_stats)

    granular_data = {
        "per_run_predictions": per_run_predictions,
        "per_model_predictions": per_model_predictions,
        "model_paths_for_tracking": model_paths_for_tracking,
        "ensemble_settings_per_model": [get_ensemble_settings(mp) for mp in model_paths_for_tracking],
    }
    aug_meta: dict[str, Any] = {
        "total_model_outputs": counters["total_raw"],
        "unparseable_outputs": counters["total_raw"] - counters["total_parsed"],
        "total_augmentations_generated": global_aug_stats["total_augmentations_generated"],
        "successful_augmentations": global_aug_stats["successful_augmentations"],
        "failed_augmentations": global_aug_stats["failed_augmentations"],
        "augmentation_type_breakdown": global_aug_stats["augmentation_type_breakdown"],
        "parsing_error_breakdown": {
            "empty_output": counters["total_raw"] - counters["total_parsed"],
            "invalid_format": 0,
            "grid_validation_failed": counters["total_parsed"] - counters["total_valid"],
        },
        "filtering_stats": global_filter_stats,
    }
    return aug_meta, granular_data


def _score_submission_and_update_progress(
    runtime: RuntimeContext,
    tasks: TaskLoadResult,
    state: PipelineState,
    aug_meta: dict[str, Any],
    granular_data: dict[str, Any],
) -> dict[str, Any] | None:
    run_logger = runtime.run_logger
    scaling_collector = runtime.scaling_collector
    aggregated_predictions = state.aggregated_predictions
    all_task_keys_loaded = tasks.loaded_keys

    scoring_result = score_if_solutions_available(
        config.SUBMISSION_PATH,
        config.DATA_PATH,
        all_task_keys_loaded,
        aug_meta,
        run_logger.run_data,
        aggregated_predictions,
        granular_scoring_data=granular_data,
        skip_visuals=bool(scaling_collector),
    )

    if scoring_result:
        run_logger.log_scoring_results(scoring_result)
        overall = scoring_result.get("overall_stats", {})
        task_results = scoring_result.get("task_results", [])
        state.last_score = overall.get("overall_top2_score", state.last_score)
        state.last_solved = sum(1 for t in task_results if t.get("task_score", 0) > 0)
    return scoring_result


def _display_progress_summary(runtime: RuntimeContext, state: PipelineState) -> None:
    progress_log = state.progress_log
    if not progress_log:
        return

    error_tracker = state.error_tracker
    run_logger = runtime.run_logger

    print("\n" + "=" * 90 + "\n📊 PROGRESS SUMMARY TABLE\n" + "=" * 90)
    print(
        
            f"{'Checkpoint':<25} {'Score':<8} {'Delta':<8} "
            f"{'Solved':<8} {'Delta':<6} {'Active':<7} "
            f"{'Time(s)':<8} {'Status':<9}"
        
    )
    print("-" * 90)
    total_errors = 0
    for entry in progress_log:
        checkpoint = entry["checkpoint"][:24]
        score = f"{entry['score']:.3f}"
        sdelta = f"{entry['score_delta']:+.3f}" if entry["score_delta"] != 0 else "+0.000"
        solved = f"{entry['solved']}/{entry['total']}"
        sdd = f"{entry['solved_delta']:+d}" if entry["solved_delta"] != 0 else "+0"
        active = str(entry["active_remaining"])
        elapsed = f"{entry['time_elapsed']:.1f}"
        status = "❌ ERROR" if entry.get("has_errors", False) else "✅ OK"
        if entry.get("has_errors", False):
            total_errors += 1
        print(f"{checkpoint:<25} {score:<8} {sdelta:<8} {solved:<8} {sdd:<6} {active:<7} {elapsed:<8} {status}")
    print("=" * 90)
    if len(progress_log) > 1:
        total_score_gain = progress_log[-1]["score"] - progress_log[0]["score"]
        total_solved_gain = progress_log[-1]["solved"] - progress_log[0]["solved"]
        success_rate = ((len(progress_log) - total_errors) / len(progress_log)) * 100
        print(
            f"🚀 Total Progress: Score +{total_score_gain:.3f} | "
            f"Tasks +{total_solved_gain} | Time {progress_log[-1]['time_elapsed']:.1f}s"
        )
        print(
            f"📈 Success Rate: {success_rate:.1f}% ({len(progress_log) - total_errors}/{len(progress_log)}) | "
            f"Errors: {total_errors}"
        )
        if total_errors > 0:
            print("\n⚠️ ERROR SUMMARY:")
            print(
                " Training Errors: "
                f"{len(error_tracker['training_errors'])}\n Inference Errors: {len(error_tracker['inference_errors'])}"
            )
            recent = [
                f"{entry['checkpoint']}: {msg}"
                for entry in progress_log
                if entry.get("has_errors")
                for msg in entry.get("errors", [])
            ][-3:]
            if recent:
                print(" Recent Errors:")
                for line in recent:
                    print(f" • {line}")
    try:
        csv_dir = Path(getattr(run_logger, "logs_dir", "run_logs"))
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{getattr(run_logger, 'run_id', 'run')}_progress.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "checkpoint",
                    "score",
                    "score_delta",
                    "solved",
                    "solved_delta",
                    "active",
                    "time_seconds",
                    "status",
                ]
            )
            for entry in progress_log:
                writer.writerow(
                    [
                        entry["checkpoint"],
                        f"{entry['score']:.3f}",
                        f"{entry['score_delta']:+.3f}",
                        f"{entry['solved']}/{entry['total']}",
                        f"{entry['solved_delta']:+d}",
                        entry["active_remaining"],
                        f"{entry['time_elapsed']:.1f}",
                        "ERROR" if entry.get("has_errors") else "OK",
                    ]
                )
        print(f"🗂️ Progress summary CSV saved to {csv_path}")
    except Exception as exc:
        print(f"⚠️ Failed to write progress summary CSV: {exc}")
    print("=" * 90)


def _finalize_run_artifacts(runtime: RuntimeContext) -> None:
    run_logger = runtime.run_logger
    session_logger = runtime.session_logger

    if session_logger is not None:
        run_logger.register_session_artifacts(session_logger.get_artifact_manifest())
    run_logger.finalize_run("completed")
    try:
        export_scaling_artifacts(run_logger.run_id)
    except Exception as exc:
        print(f"⚠️ Failed to stage scaling artifacts locally: {exc}")
    print(
        "\n"
        + "=" * 70
        + f"\n✅ COMPLETED: {config.RUN_TITLE}\n🆔 Run ID: {run_logger.run_id}\n📁 Run logs: {run_logger.run_log_file}"
    )
    scoring_vis_dir = "/kaggle/working/scoring_visualizations" if runtime.is_kaggle else "scoring_visualizations"
    if os.path.exists(os.path.join(scoring_vis_dir, "scoring_report.html")):
        port = 8000
        print(
            f"\n🌐 START WEB SERVER FOR SCORING VISUALIZATIONS:\n"
            f" cd {scoring_vis_dir}\n"
            f" python -m http.server {port}\n"
            f" Then open: http://localhost:{port}/scoring_report.html\n"
            f" Or use: make serve PORT={port}"
        )
    session_logger.log_custom_event("session_end", "Finalizing session and creating archive")
    zip_path: str | None = None
    try:
        zip_path = finalize_session_logger()
    finally:
        if session_logger is not None:
            session_logger.stop_console_capture()
    if zip_path:
        run_logger.register_session_artifacts({"session_archive": zip_path})
    print(f"\n📦 Session archive created: {zip_path}" if zip_path else "\n⚠️ Failed to create session archive")
    print("=" * 70)


def finalize_pipeline(runtime: RuntimeContext, tasks: TaskLoadResult, state: PipelineState) -> None:
    aggregated_predictions = state.aggregated_predictions
    active_task_keys = tasks.active_keys

    _print_finalization_banner(runtime, len(active_task_keys))
    aug_meta, granular_data = _generate_submission_metadata(runtime, tasks, state)
    _score_submission_and_update_progress(
        runtime,
        tasks,
        state,
        aug_meta,
        granular_data,
    )
    _display_progress_summary(runtime, state)
    _finalize_run_artifacts(runtime)

    state.aggregated_predictions = aggregated_predictions
