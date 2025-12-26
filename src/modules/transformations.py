# ===== src/modules/transformations.py =====
"""
Minimal geometric + color utilities for ARC Prize 2025.

Exposed API (used elsewhere):
- identity, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror
- GEOM_FUNCTIONS, GEOM_FUNCTION_NAMES
- get_symbols(task) -> List[int]
- get_colormaps(n_colors, n_maps) -> List[List[int]]
- mix_augms(items1_list, items2_list, expansion_factor, pruning_factor)
- apply_decoder_description(grid, decoder_desc)  # legacy entry point kept for main.py
\n+Back-compat shims:
- get_n_augs(task, n, use_case='inference', task_pool=None) -> (tasks, decs)
"""

from collections.abc import Callable
from random import choice, sample, shuffle
from typing import Any


try:
    from .config import MAX_SYMBOLS, debug_print
    from .grid_utils import is_grid
except ImportError:  # script-style fallback
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import MAX_SYMBOLS, debug_print  # type: ignore
    from grid_utils import is_grid  # type: ignore

# -----------------------------------------------------------------------------
# Geometric transforms
# -----------------------------------------------------------------------------


def identity(g: list[list[int]]) -> list[list[int]]:
    return g


def rot90(g: list[list[int]]) -> list[list[int]]:
    # 90° clockwise
    return [list(r) for r in zip(*g[::-1], strict=False)]


def rot180(g: list[list[int]]) -> list[list[int]]:
    return [row[::-1] for row in g[::-1]]


def rot270(g: list[list[int]]) -> list[list[int]]:
    # 90° counter-clockwise
    return [list(r) for r in zip(*g, strict=False)][::-1]


def hmirror(g: list[list[int]]) -> list[list[int]]:
    # flip vertically (top<->bottom)
    return g[::-1]


def vmirror(g: list[list[int]]) -> list[list[int]]:
    # flip horizontally (left<->right)
    return [row[::-1] for row in g]


def dmirror(g: list[list[int]]) -> list[list[int]]:
    # main-diagonal mirror (transpose)
    return [list(r) for r in zip(*g, strict=False)]


def cmirror(g: list[list[int]]) -> list[list[int]]:
    # counter-diagonal mirror (anti-diagonal)
    return [list(row) for row in zip(*(r[::-1] for r in g[::-1]), strict=False)]


GEOM_FUNCTIONS: dict[str, Callable[[list[list[int]]], list[list[int]]]] = {
    "identity": identity,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "hmirror": hmirror,
    "vmirror": vmirror,
    "dmirror": dmirror,
    "cmirror": cmirror,
}
GEOM_FUNCTION_NAMES: dict[Callable[[list[list[int]]], list[list[int]]], str] = {v: k for k, v in GEOM_FUNCTIONS.items()}

# -----------------------------------------------------------------------------
# Decoder (legacy entry point maintained for compatibility)
# -----------------------------------------------------------------------------


def apply_decoder_description(grid_to_decode: Any, decoder_desc: dict) -> list[list[int]]:
    """
    Legacy shim used by older code paths (e.g., main.py).
    - If decoder_desc has 'type', delegate to augmentation_framework.apply_decoder_description_flexible.
    - Otherwise, apply legacy (geom_name + colmap_inv) decoding.
    Returns [[0]] on invalid inputs.
    """
    # Validate input early
    if grid_to_decode is None or not is_grid(grid_to_decode):
        debug_print(f"apply_decoder_description: invalid/None grid. Decoder={decoder_desc}")
        return [[0]]

    # Prefer new flexible decoder when available
    try:
        if "type" in decoder_desc:
            from .augmentation_framework import apply_decoder_description_flexible  # local import avoids circularity

            out = apply_decoder_description_flexible(grid_to_decode, decoder_desc)
            return out if is_grid(out) else [[0]]
    except Exception as e:
        debug_print(f"apply_decoder_description: flexible path failed ({e}); falling back to legacy.")

    # Legacy: color inverse map + geometric function
    geom_name = decoder_desc.get("geom_name")
    colmap_inv = decoder_desc.get("colmap_inv", {})
    if not geom_name and not colmap_inv:
        # nothing to do; return as-is if valid
        return grid_to_decode

    geom_fn = GEOM_FUNCTIONS.get(geom_name or "identity", identity)
    try:
        # color inverse
        mapped = [[colmap_inv.get(v, v) for v in row] for row in grid_to_decode]
        if not is_grid(mapped):
            debug_print("apply_decoder_description: grid invalid after color map.")
            return [[0]]
        # geometric
        decoded = geom_fn(mapped)
        return decoded if is_grid(decoded) else [[0]]
    except Exception as e:
        debug_print(f"apply_decoder_description: legacy decode error: {e}")
        return [[0]]


# -----------------------------------------------------------------------------
# Color + symbol utilities
# -----------------------------------------------------------------------------


def get_symbols(task: dict) -> list[int]:
    """Collect unique symbols from all grids in a task."""
    syms = set()
    for ex in task.get("train", []) or []:
        for key in ("input", "output"):
            g = ex.get(key)
            if is_grid(g):
                for row in g:
                    syms.update(row)
    test0 = (task.get("test") or [{}])[0]
    g_in = test0.get("input")
    if is_grid(g_in):
        for row in g_in:
            syms.update(row)
    return sorted(syms)


def get_colormaps(n_colors: int, n_maps: int) -> list[list[int]]:
    """
    Return n_maps permutations of [0..MAX_SYMBOLS-1].
    We dedupe the first n_colors slice to encourage variety without heavy bookkeeping.
    """
    n_maps = max(1, int(n_maps))
    base = list(range(MAX_SYMBOLS))
    colormaps, seen = [], set()
    attempts, max_attempts = 0, max(200, n_maps * 20)
    k = max(0, min(int(n_colors), MAX_SYMBOLS))

    while len(colormaps) < n_maps and attempts < max_attempts:
        attempts += 1
        m = base[:]
        shuffle(m)
        key = tuple(m[:k])
        if k and key in seen:
            continue
        seen.add(key)
        colormaps.append(m)

    while len(colormaps) < n_maps:
        colormaps.append(choice(colormaps) if colormaps else base[:])
    return colormaps


# -----------------------------------------------------------------------------
# Pair mixing utility (public API used elsewhere)
# -----------------------------------------------------------------------------


def mix_augms(
    items1_list: list, items2_list: list, expansion_factor: float, pruning_factor: float
) -> tuple[list, list]:
    """
    Pair up two lists, optionally expand via random duplicates and prune.
    Returns (items1_selected, items2_selected) with the same length.
    """
    n1, n2 = len(items1_list), len(items2_list)
    n = max(n1, n2)
    if n == 0:
        return [], []
    a = (items1_list + [items1_list[-1]] * (n - n1)) if n1 else [None] * n
    b = (items2_list + [items2_list[-1]] * (n - n2)) if n2 else [None] * n
    paired = list(zip(a, b, strict=False))
    base_len = min(n1, n2)
    if base_len and expansion_factor > 0:
        idxs = sample(range(base_len), min(int(base_len * expansion_factor), base_len))
        paired += [paired[i] for i in idxs] if idxs else []
    keep = max(0, int(len(paired) * (1.0 - pruning_factor))) if 0.0 <= pruning_factor < 1.0 else len(paired)
    selected = [paired[i] for i in sorted(sample(range(len(paired)), keep))] if keep and paired else []
    return (list(t) for t in zip(*selected, strict=False)) if selected else ([], [])


# -----------------------------------------------------------------------------
# Backward compatibility shim for augmentation API
# -----------------------------------------------------------------------------


def get_n_augs(task: dict, n: int, use_case: str = "inference", task_pool: Any = None) -> tuple[list[dict], list[dict]]:
    """
    Legacy wrapper for the old transformations.get_n_augs API.
    Delegates to augmentation_framework.get_n_augs_flexible and returns (tasks, decoders).
    """
    try:
        from .augmentation_framework import get_n_augs_flexible

        tasks, decs, _stats = get_n_augs_flexible(task, n, use_case, task_pool)
        return tasks, decs
    except Exception as e:
        debug_print(f"get_n_augs shim error: {e}")
        return [], []
