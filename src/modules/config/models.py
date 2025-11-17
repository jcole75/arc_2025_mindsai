from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
from importlib import import_module
import json
import os
from typing import Any

from . import env as env_mod
from .model_common import (
    MAX_GRID_SIZE as _MAX_GRID_SIZE,
)
from .model_common import (
    MAX_PERMUTATIONS_TO_SAMPLE as _MAX_PERMUTATIONS_TO_SAMPLE,
)
from .model_common import (
    MAX_SYMBOLS as _MAX_SYMBOLS,
)
from .model_common import (
    MIN_VOTES_AMBIGUOUS as _MIN_VOTES_AMBIGUOUS,
)
from .model_common import (
    MIN_VOTES_SINGLE_PRED as _MIN_VOTES_SINGLE_PRED,
)
from .model_common import (
    SYMBOL_ENCODING as _SYMBOL_ENCODING,
)
from .model_common import (
    Z_SCORE_THRESHOLD as _Z_SCORE_THRESHOLD,
)


Z_SCORE_THRESHOLD = _Z_SCORE_THRESHOLD
MIN_VOTES_SINGLE_PRED = _MIN_VOTES_SINGLE_PRED
MIN_VOTES_AMBIGUOUS = _MIN_VOTES_AMBIGUOUS
MAX_GRID_SIZE = _MAX_GRID_SIZE
MAX_SYMBOLS = _MAX_SYMBOLS
MAX_PERMUTATIONS_TO_SAMPLE = _MAX_PERMUTATIONS_TO_SAMPLE
SYMBOL_ENCODING = _SYMBOL_ENCODING

_MODEL_CONFIG_PACKAGE = __name__.rsplit(".", 1)[0]
_MODEL_CONFIG_BASE = f"{_MODEL_CONFIG_PACKAGE}.model_configs"

DEFAULT_MODEL_CONFIG_MODULES: list[str] = [
    f"{_MODEL_CONFIG_BASE}.codet5_small",
    f"{_MODEL_CONFIG_BASE}.qwen3_06b",
    f"{_MODEL_CONFIG_BASE}.codet5_large",
    f"{_MODEL_CONFIG_BASE}.codet5_large_v2",
    f"{_MODEL_CONFIG_BASE}.codet5_large_v3",
    f"{_MODEL_CONFIG_BASE}.coda_instruct_b200",
    f"{_MODEL_CONFIG_BASE}.coda_instruct_kaggle",
]

_ENV_MODULE_OVERRIDE = False


def _parse_module_list(raw: Any) -> list[str] | None:
    """Parse a raw environment value describing module names."""
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = [part.strip() for part in text.split(",")]
            else:
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed]
                return [str(parsed).strip()]
            return [p for p in parsed if p]
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [str(raw).strip()]


def _normalize_module_name(name: str) -> str:
    if not name:
        return ""
    candidate = name.strip()
    if not candidate:
        return ""
    if candidate.startswith("."):  # relative to model_configs
        return f"{_MODEL_CONFIG_BASE}{candidate}"
    if candidate.startswith("model_configs."):
        return f"{_MODEL_CONFIG_PACKAGE}.{candidate}"
    if "." not in candidate:
        return f"{_MODEL_CONFIG_BASE}.{candidate}"
    return candidate


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _get_config_module_list() -> list[str]:
    global _ENV_MODULE_OVERRIDE
    env_map = getattr(env_mod, "_ENV", {}) if hasattr(env_mod, "_ENV") else {}
    raw = os.environ.get("MODEL_CONFIG_MODULES")
    if raw is None:
        raw = env_map.get("MODEL_CONFIG_MODULES")
    module_names = _parse_module_list(raw)
    if module_names:
        normalized = [_normalize_module_name(name) for name in module_names]
        normalized = _dedupe_preserve_order(normalized)
        normalized = [name for name in normalized if name]
        if normalized:
            _ENV_MODULE_OVERRIDE = True
            return normalized
    _ENV_MODULE_OVERRIDE = False
    return list(DEFAULT_MODEL_CONFIG_MODULES)


def _iter_module_entries(module: Any) -> Iterable[tuple[str, dict[str, Any], bool]]:
    if hasattr(module, "MODEL_SETTINGS"):
        settings = module.MODEL_SETTINGS
        defaults = set(getattr(module, "DEFAULT_MODEL_PATHS", []))
        if isinstance(settings, dict):
            for path, cfg in settings.items():
                yield path, cfg, path in defaults
            return
    if hasattr(module, "MODEL_CONFIG"):
        path = getattr(module, "MODEL_PATH", None)
        cfg = module.MODEL_CONFIG
        include = bool(getattr(module, "DEFAULT_ACTIVE", False))
        if not path:
            raise ValueError(f"Model config module {module.__name__} missing MODEL_PATH")
        if not isinstance(cfg, dict):
            raise ValueError(f"Model config module {module.__name__} has invalid MODEL_CONFIG")
        yield path, cfg, include
        return
    raise ValueError(f"Model config module {module.__name__} exposes no model settings")


def _load_model_settings(
    module_names: Sequence[str],
    include_all_paths: bool,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    model_paths: list[str] = []
    default_paths: list[str] = []
    settings: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for module_name in module_names:
        module = import_module(module_name)
        for path, cfg, include in _iter_module_entries(module):
            if path not in model_paths:
                model_paths.append(path)
            settings[path] = copy.deepcopy(cfg)
            sources[path] = module_name
            if include and path not in default_paths:
                default_paths.append(path)
    selected_paths = list(model_paths) if include_all_paths else default_paths or model_paths
    if not selected_paths and model_paths:
        selected_paths = [model_paths[0]]
    return selected_paths, settings, sources


MODEL_CONFIG_MODULES = _get_config_module_list()
MODEL_PATHS, MODEL_SETTINGS, MODEL_CONFIG_SOURCES = _load_model_settings(
    MODEL_CONFIG_MODULES,
    include_all_paths=_ENV_MODULE_OVERRIDE,
)


def _resolve_scalar(value: Any) -> Any:
    """If a config value is provided as a list/tuple (legacy multi-value), return the first element.
    Leaves other types unchanged. Returns None for empty lists.
    """
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def get_model_type(model_path: str | None = None) -> str:
    if model_path is None and MODEL_PATHS:
        model_path = MODEL_PATHS[0]

    # First try with the exact model path
    model_type = MODEL_SETTINGS.get(model_path or "", {}).get("general", {}).get("model_type")
    if model_type:
        return model_type

    # If not found and MODEL_PATHS exists, fall back to the first model path
    if MODEL_PATHS:
        fallback_model_type = MODEL_SETTINGS.get(MODEL_PATHS[0], {}).get("general", {}).get("model_type")
        if fallback_model_type:
            return fallback_model_type

    # Try to detect from the actual model config if it's a local path
    if model_path and os.path.exists(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Check for CoDA config or other causal LM architectures
                arch_type = config.get("architectures", [])
                model_type_str = config.get("model_type", "")
                if "CoDA" in str(arch_type) or "CoDA" in model_type_str:
                    return "causal_lm"
                # Check for common causal LM architectures
                causal_lm_archs = ["LlamaForCausalLM", "GPTNeoXForCausalLM", "GPT2LMHeadModel",
                                   "GPTJForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM"]
                if any(arch in str(arch_type) for arch in causal_lm_archs):
                    return "causal_lm"
            except:
                pass

    return "seq2seq"


def get_nested_setting(setting_name: str, section: str, default_value: Any, model_path: str | None = None) -> Any:
    if model_path is None and MODEL_PATHS:
        model_path = MODEL_PATHS[0]
    mc = MODEL_SETTINGS.get(model_path or "", {})
    sec = mc.get(section, {}) if isinstance(mc, dict) else {}
    if setting_name in sec:
        return _resolve_scalar(sec[setting_name])
    if mc.get("general", {}).get("model_type") == "causal_lm":
        if setting_name == "max_input_length":
            return get_nested_setting("max_length", section, 4096, model_path)
        if setting_name == "max_target_length":
            max_len = get_nested_setting("max_length", section, 4096, model_path)
            base = default_value if default_value is not None else 512
            return min(max_len // 4, base)
        if setting_name == "max_generation_length":
            return default_value if default_value is not None else 600
    return _resolve_scalar(default_value)


def get_ensemble_settings(model_path: str | None = None) -> dict[str, Any]:
    if model_path is None and MODEL_PATHS:
        model_path = MODEL_PATHS[0]
    defaults = {
        "enable_self_ensemble": True,
        "enable_model_ensemble": True,
        "enable_run_tracking": False,
        "iterative_ensemble_mode": False,
        # Ensure this option is surfaced to callers when present in model config
        "airv_last_cycle_only": False,
    }
    cfg = MODEL_SETTINGS.get(model_path or "", {}).get("ensembling", {})
    out = defaults.copy()
    out.update({k: _resolve_scalar(v) for k, v in cfg.items() if k in defaults})
    return {k: _resolve_scalar(v) for k, v in out.items()}
