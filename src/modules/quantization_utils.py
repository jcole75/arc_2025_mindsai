"""Helpers for configuring quantized model loading (bitsandbytes)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from contextlib import suppress

import torch

try:  # transformers>=4.31 exposes BitsAndBytesConfig
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore


def _resolve_torch_dtype(dtype_name: Any) -> torch.dtype | None:
    """Convert a user-specified dtype string to a torch.dtype when possible."""
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if not isinstance(dtype_name, str):
        return None
    candidate = getattr(torch, dtype_name, None)
    if isinstance(candidate, torch.dtype):
        return candidate
    lookup = dtype_name.strip().lower()
    aliases: dict[str, str] = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    lookup = aliases.get(lookup, lookup)
    for attr in dir(torch):
        value = getattr(torch, attr)
        if isinstance(value, torch.dtype) and attr.lower() == lookup:
            return value
    return None


def _bitsandbytes_available() -> bool:
    with suppress(Exception):
        import bitsandbytes  # type: ignore  # noqa: F401

        return True
    return False


def build_bitsandbytes_kwargs(
    settings: dict[str, Any] | None,
    *,
    rank: int | None,
    for_training: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate a config block into kwargs for HF model loaders.

    Returns (kwargs, info). info["enabled"] indicates whether quantization
    will be attempted. info["message"] contains a human-readable reason when
    quantization is skipped.
    """
    info: Dict[str, Any] = {"enabled": False, "message": None}
    if not isinstance(settings, dict) or not settings.get("enabled"):
        info["message"] = "bitsandbytes disabled or not configured"
        return {}, info

    allow_training = bool(settings.get("allow_training", False))
    if for_training and not allow_training:
        info["message"] = "bitsandbytes disabled for training"
        return {}, info

    load_in_4bit = bool(settings.get("load_in_4bit"))
    load_in_8bit = bool(settings.get("load_in_8bit"))
    if not load_in_4bit and not load_in_8bit:
        info["message"] = "bitsandbytes config missing load_in_4bit/8bit flags"
        return {}, info

    if not _bitsandbytes_available():
        info["message"] = "bitsandbytes package unavailable"
        return {}, info

    compute_dtype = _resolve_torch_dtype(settings.get("compute_dtype"))
    quant_type = str(settings.get("quant_type") or "nf4")
    use_double = bool(settings.get("use_double_quant", True))

    kwargs: Dict[str, Any] = {}
    if BitsAndBytesConfig is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double,
            bnb_4bit_quant_type=quant_type,
        )
    else:
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["bnb_4bit_use_double_quant"] = use_double
            kwargs["bnb_4bit_quant_type"] = quant_type
            if compute_dtype is not None:
                kwargs["bnb_4bit_compute_dtype"] = compute_dtype
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
    if compute_dtype is not None and load_in_4bit:
        kwargs.setdefault("torch_dtype", compute_dtype)

    device_map = settings.get("device_map")
    if device_map is None:
        if torch.cuda.is_available():
            device_index = max(0, int(rank or 0))
            device_map = {"": device_index}
        else:
            device_map = {"": "cpu"}
    kwargs["device_map"] = device_map

    max_memory = settings.get("max_memory")
    if isinstance(max_memory, dict):
        kwargs["max_memory"] = max_memory

    if "low_cpu_mem_usage" not in settings:
        kwargs.setdefault("low_cpu_mem_usage", True)
    else:
        kwargs["low_cpu_mem_usage"] = bool(settings["low_cpu_mem_usage"])

    info["enabled"] = True
    info["message"] = "bitsandbytes quantization enabled"
    return kwargs, info
