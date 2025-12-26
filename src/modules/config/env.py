from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any


# Timing and logging toggles
START_TIME = time.time()
TOTAL_TIME = 999 * 60 * 60  # 12 hours
BUFFER_TIME = 15 * 60  # 15 minutes

DEBUG_MODE = False
VERBOSE_LOGGING = False


def _strip_inline_comment(value: str) -> str:
    """Remove inline comments starting with '#' unless inside quotes.

    Example: "2 # comment" -> "2"; "'abc # not comment'" -> "'abc # not comment'".
    """
    in_single = False
    in_double = False
    out_chars = []
    for ch in value:
        if ch == "'" and not in_double:
            in_single = not in_single
            out_chars.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out_chars.append(ch)
            continue
        if ch == "#" and not in_single and not in_double:
            break
        out_chars.append(ch)
    return "".join(out_chars).strip()


def _unquote(value: str) -> str:
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def _load_env_file() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    env_file = project_root / ".env"
    env_vars: dict[str, Any] = {}
    if env_file.exists():
        try:
            for raw in env_file.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Allow shell-style `export KEY=VALUE`
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = _strip_inline_comment(v.strip())
                    v = _unquote(v)
                    # Store booleans directly for convenience; other types parsed later
                    if isinstance(v, str) and v.lower() in ("true", "false"):
                        env_vars[k] = v.lower() == "true"
                    else:
                        env_vars[k] = v
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")
    return env_vars


_ENV = _load_env_file()


def _lookup_env(env_name: str):
    if env_name in os.environ:
        return os.environ.get(env_name)
    return _ENV.get(env_name)


_UNLIMITED_TOKENS = {"unlimited", "infinite", "infinity", "inf", "no_limit", "nolimit"}


def _parse_int_with_unlimited(raw: Any, *, allow_zero: bool = False) -> int | None:
    """Parse an integer-like value while allowing a '-1'/'unlimited' sentinel."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = int(raw)
        if value == -1:
            return -1
        if value < 0:
            return None
        if value == 0 and not allow_zero:
            return None
        return value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "null", "auto"}:
            return None
        if lowered in _UNLIMITED_TOKENS:
            return -1
        try:
            value = int(text)
        except Exception:
            return None
        if value == -1:
            return -1
        if value < 0:
            return None
        if value == 0 and not allow_zero:
            return None
        return value
    try:
        value = int(raw)
    except Exception:
        return None
    if value == -1:
        return -1
    if value < 0:
        return None
    if value == 0 and not allow_zero:
        return None
    return value


_BOOL_TRUE_VALUES = {"true", "1", "yes", "on", "enable", "enabled", "test", "testing", "t", "y"}
_BOOL_FALSE_VALUES = {"false", "0", "no", "off", "disable", "disabled", "f", "n"}


def _get_bool(env_name: str, default: bool) -> bool:
    raw = _lookup_env(env_name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if lower in _BOOL_TRUE_VALUES:
            return True
        if lower in _BOOL_FALSE_VALUES:
            return False
    return bool(raw)


USE_TPU = _get_bool("USE_TPU", False)
USE_FLAX_GPU = _get_bool("USE_FLAX_GPU", False)
USE_FLAX = USE_TPU or USE_FLAX_GPU

# Optional logging toggles from env
DEBUG_MODE = _get_bool("DEBUG_MODE", DEBUG_MODE)
VERBOSE_LOGGING = _get_bool("VERBOSE_LOGGING", VERBOSE_LOGGING)

def _detect_environment() -> tuple[str, bool, bool, bool]:
    """
    Determine the current execution environment.

    Precedence:
      1. Explicit ENVIRONMENT value from os.environ or .env file.
      2. Well-known host environment variables.
      3. Default to local.
    """

    def _normalize(value: str) -> str:
        return value.strip().lower()

    override = _lookup_env("ENVIRONMENT")
    if isinstance(override, bool):
        override = "kaggle" if override else "local"
    environment = _normalize(str(override)) if override else None

    if environment in {"kaggle", "local", "skypilot"}:
        pass
    else:
        environment = None

    if environment is None:
        kaggle_hints = (
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            os.environ.get("KAGGLE_URL_BASE"),
            os.environ.get("KAGGLE_USER_SECRETS_TOKEN"),
            os.environ.get("KAGGLE_CONTAINER_NAME"),
        )
        skypilot_hints = (
            os.environ.get("SKYPILOT_CLUSTER"),
            os.environ.get("SKYPILOT_TASK_ID"),
        )
        if any(hints for hints in kaggle_hints):
            environment = "kaggle"
        elif any(hints for hints in skypilot_hints):
            environment = "skypilot"
        else:
            environment = "local"

    env_value = environment if environment in {"kaggle", "local", "skypilot"} else "local"
    return (
        env_value,
        env_value == "kaggle",
        env_value == "local",
        env_value == "skypilot",
    )


ENVIRONMENT, IS_KAGGLE, IS_LOCAL, IS_SKYPILOT = _detect_environment()


# Test mode and related toggles
def _get_int(env_name: str, default: int) -> int:
    raw = _lookup_env(env_name)
    try:
        return int(raw) if raw is not None else default
    except Exception:
        return default


TEST_MODE = _get_bool("TEST_MODE", False)
TEST_MODE_NUM_TASKS = _get_int("TEST_MODE_NUM_TASKS", 10)
# Parse explicit task IDs only if provided; otherwise leave empty to use count-based selection
_ids_raw = os.environ.get("TEST_MODE_TASK_IDS", _ENV.get("TEST_MODE_TASK_IDS"))
if isinstance(_ids_raw, str) and _ids_raw.strip():
    TEST_MODE_TASK_IDS = [s.strip() for s in _ids_raw.split(",") if s.strip()]
elif isinstance(_ids_raw, (list, tuple)):
    TEST_MODE_TASK_IDS = [str(s).strip() for s in _ids_raw if str(s).strip()]
else:
    TEST_MODE_TASK_IDS = []
TEST_MODE_TTT_ITEMS = _get_int("TEST_MODE_TTT_ITEMS", 200)
TEST_MODE_INFERENCE_ITEMS = _get_int("TEST_MODE_INFERENCE_ITEMS", 200)

TEST_SIZE = 0.0
GLOBAL_TTT_ENABLED = _get_bool("GLOBAL_TTT_ENABLED", True)
EXPAND_FACTOR = 0
PRUNE_FACTOR = 0

PRESERVE_FINETUNED_MODEL = _get_bool("PRESERVE_FINETUNED_MODEL", False)


# -----------------------------------------------------------------------------
# Global run/sample settings (rename from LARGE_* to GLOBAL_* per request)
# -----------------------------------------------------------------------------
def _get_optional_int(env_name: str, default: int | None) -> int | None:
    raw = _lookup_env(env_name)
    if raw is None:
        return default
    try:
        if isinstance(raw, str) and raw.strip().lower() in ("none", ""):
            return None
        return int(raw)
    except Exception:
        return default


def parse_pretraining_examples(value: Any) -> int | None:
    """Parse a pretraining example limit, allowing '-1' or 'unlimited'."""
    return _parse_int_with_unlimited(value, allow_zero=True)


def parse_pretraining_steps(value: Any) -> int | None:
    """Parse a pretraining step limit, allowing '-1' or 'unlimited'."""
    return _parse_int_with_unlimited(value, allow_zero=False)


# Defaults based on user request
GLOBAL_TTT_SAMPLES = _get_int("GLOBAL_TTT_SAMPLES", 42000)
GLOBAL_INF_SAMPLES = _get_int("GLOBAL_INF_SAMPLES", 20000)
SE = _get_int("SE", 1)
GROUP_SIZE = _get_optional_int("GROUP_SIZE", None)
SAVE_STEPS = _get_optional_int("SAVE_STEPS", None)


PRETRAINING_EXAMPLES_OVERRIDE = parse_pretraining_examples(_lookup_env("PRETRAINING_EXAMPLES"))
PRETRAINING_STEPS_OVERRIDE = parse_pretraining_steps(_lookup_env("PRETRAINING_STEPS"))


# Optional timing overrides from env
_total_hours = _lookup_env("TOTAL_TIME_HOURS")
_total_minutes = _lookup_env("TOTAL_TIME_MINUTES")
_buffer_minutes = _lookup_env("BUFFER_TIME_MINUTES")
try:
    if _total_hours is not None:
        TOTAL_TIME = int(float(_total_hours) * 60 * 60)
    elif _total_minutes is not None:
        TOTAL_TIME = int(float(_total_minutes) * 60)
except Exception:
    pass
try:
    if _buffer_minutes is not None:
        BUFFER_TIME = int(float(_buffer_minutes) * 60)
except Exception:
    pass


def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def verbose_print(*args, **kwargs):
    if VERBOSE_LOGGING:
        print("[VERBOSE]", *args, **kwargs)
