"""
Flexible augmentation framework for ARC Prize 2025 solution.

Key public objects:
- Classes: AugmentationBase, GeometricAugmentation, ColorAugmentation, OrderAugmentation,
          NoiseAugmentation, CropAugmentation, DuplicateAugmentation, MixupAugmentation,
          InputOutputSwapAugmentation, CombinedAugmentation, CombineAugmentation,

          AugmentationManager.
- Functions: get_n_augs_flexible, apply_decoder_description_flexible

"""

import ast
from collections import Counter
from random import choice, randint, shuffle, uniform
import re
from typing import Any


try:
    from .config import AUGMENTATION_CONFIG, MAX_SYMBOLS, debug_print
    from .grid_utils import is_grid
    from .transformations import (
        GEOM_FUNCTION_NAMES,
        GEOM_FUNCTIONS,
        cmirror,
        dmirror,
        get_colormaps,
        get_symbols,
        hmirror,
        identity,
        rot90,
        rot180,
        rot270,
        vmirror,
    )
except ImportError:  # local, script-style usage
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import AUGMENTATION_CONFIG, MAX_SYMBOLS, debug_print
    from grid_utils import is_grid
    from transformations import (
        GEOM_FUNCTION_NAMES,
        GEOM_FUNCTIONS,
        cmirror,
        dmirror,
        get_colormaps,
        get_symbols,
        hmirror,
        identity,
        rot90,
        rot180,
        rot270,
        vmirror,
    )

# -----------------------------------------------------------------------------
# Shared helpers (minimize duplication)
# -----------------------------------------------------------------------------

_GEOM_ENCODERS = [identity, rot90, rot180, rot270, hmirror, vmirror, dmirror, cmirror]
_GEOM_DECODERS = [identity, rot270, rot180, rot90, hmirror, vmirror, dmirror, cmirror]
_GEOM_PAIRS = list(zip(_GEOM_ENCODERS, _GEOM_DECODERS, strict=False))


def _valid_task(task: dict) -> bool:
    test = task.get("test", [])
    return (
        isinstance(test, list)
        and len(test) > 0
        and isinstance(test[0], dict)
        and "input" in test[0]
        and is_grid(test[0]["input"])
    )


def _valid_example(ex: dict) -> bool:
    return isinstance(ex, dict) and "input" in ex and "output" in ex and is_grid(ex["input"]) and is_grid(ex["output"])


def _deepcopy_grid(g: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in g] if g else g


def _map_grid_colors(g: list[list[int]], cmap: dict[int, int]) -> list[list[int]]:
    return [[cmap.get(v, v) for v in row] for row in g]


def _dominant_color(g: list[list[int]]) -> int:
    if not g:
        return 0
    counts: dict[int, int] = {}
    for row in g:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else 0


def _pad_to(grid: list[list[int]], h: int, w: int, fill: int | None = None) -> list[list[int]]:
    """Pad grid to (h, w) with fill (defaults to grid dominant color)."""
    if not grid:
        return [[0] * w for _ in range(h)]
    fill = _dominant_color(grid) if fill is None else fill
    gh, gw = len(grid), len(grid[0])
    out = []
    for r in range(max(h, gh)):
        row = grid[r][:] if r < gh else [fill] * gw
        if len(row) < w:
            row = row + [fill] * (w - len(row))
        out.append(row)
    return [row[:w] for row in out[:h]]


def _apply_to_task(task: dict, fn) -> dict:
    """Apply fn(grid) to every grid in train/test of the task; return new task."""
    out: dict[str, Any] = {"train": [], "test": []}
    for split in ("train",):
        for ex in task.get(split, []) or []:
            new_ex = {}
            if "input" in ex:
                new_ex["input"] = fn(ex["input"]) if is_grid(ex["input"]) else ex["input"]
            if "output" in ex:
                new_ex["output"] = fn(ex["output"]) if is_grid(ex["output"]) else ex["output"]
            out["train"].append(new_ex)
    for ex in task.get("test", []) or []:
        new_ex = {}
        if "input" in ex:
            new_ex["input"] = fn(ex["input"]) if is_grid(ex["input"]) else ex["input"]
        if "output" in ex:
            new_ex["output"] = fn(ex["output"]) if is_grid(ex["output"]) else ex["output"]
        out["test"].append(new_ex)
    return out


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------


class AugmentationBase:
    """Base class for all augmentation types."""

    def __init__(self, name: str, reversible: bool = True):
        self.name = name
        self.reversible = reversible

    # Interface
    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:  # pragma: no cover
        raise NotImplementedError

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:  # pragma: no cover
        raise NotImplementedError

    # Shared validation helpers (available to subclasses)
    @staticmethod
    def valid_task(task: dict) -> bool:
        return _valid_task(task)

    @staticmethod
    def valid_example(ex: dict) -> bool:
        return _valid_example(ex)


# -----------------------------------------------------------------------------
# Simple, reversible geometric/color augs
# -----------------------------------------------------------------------------


class GeometricAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("geometric", reversible=True)
        self.transform_pairs = _GEOM_PAIRS

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        train = task.get("train", [])
        augmented, decs = [], []
        pairs = self.transform_pairs[:]
        shuffle(pairs)
        for i in range(num_variants):
            try:
                enc, dec = pairs[i % len(pairs)]
                t_train = [
                    {"input": enc(ex["input"]), "output": enc(ex["output"])} for ex in train if self.valid_example(ex)
                ]
                if not t_train:
                    continue
                shuffle(t_train)
                t_test_in = enc(task["test"][0]["input"])  # validated above
                if not is_grid(t_test_in):
                    continue
                augmented.append({"train": t_train, "test": [{"input": t_test_in}]})
                decs.append(
                    {
                        "type": "geometric",
                        "geom_name": GEOM_FUNCTION_NAMES.get(dec, "identity"),
                        "colmap_inv": {},
                    }
                )
            except Exception as e:
                debug_print(f"Error in geometric augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        try:
            geom = decoder_desc.get("geom_name", "identity")
            return GEOM_FUNCTIONS.get(geom, identity)(grid)
        except Exception as e:
            debug_print(f"Error applying geometric decoder: {e}")
            return [[0]]


class ColorAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("color", reversible=True)

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        task_symbols = get_symbols(task)
        color_maps = get_colormaps(len(task_symbols), max(1, num_variants))
        augmented, decs = [], []
        for i, cmap in enumerate(color_maps[:num_variants]):
            try:
                forward = {orig: cmap[orig] for orig in range(MAX_SYMBOLS)}
                inverse = {v: k for k, v in forward.items()}

                def apply_map(g, cmap_forward=forward):
                    return _map_grid_colors(g, cmap_forward)

                t_train = [
                    {"input": apply_map(ex["input"]), "output": apply_map(ex["output"])}
                    for ex in task.get("train", [])
                    if self.valid_example(ex)
                ]
                if not t_train:
                    continue
                shuffle(t_train)
                t_test_in = apply_map(task["test"][0]["input"])  # validated above
                augmented.append({"train": t_train, "test": [{"input": t_test_in}]})
                decs.append({"type": "color", "geom_name": "identity", "colmap_inv": inverse})
            except Exception as e:
                debug_print(f"Error in color augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        try:
            inv = decoder_desc.get("colmap_inv", {})
            return _map_grid_colors(grid, inv)
        except Exception as e:
            debug_print(f"Error applying color decoder: {e}")
            return [[0]]


# -----------------------------------------------------------------------------
# Non-reversible simple augs
# -----------------------------------------------------------------------------


class OrderAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("order", reversible=False)

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        train = task.get("train", [])
        if not train or not self.valid_task(task):
            return [], []
        augmented, decs = [], []
        base = list(range(len(train)))
        seen = set()
        for _ in range(num_variants * 2):  # extra variety
            shuffle(base)
            key = tuple(base)
            if key in seen:
                continue
            seen.add(key)
            reordered = [train[i] for i in base if i < len(train)]
            if not reordered:
                continue
            augmented.append({"train": reordered, "test": task["test"][:]})
            decs.append({"type": "order", "permutation": list(key)})
            if len(augmented) >= num_variants:
                break
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


class NoiseAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("noise", reversible=False)

    def _noisify(self, g: list[list[int]], p: float) -> list[list[int]]:
        return [[(randint(0, 9) if uniform(0, 1) < p else v) for v in row] for row in g]

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        augmented, decs = [], []
        for i in range(num_variants):
            try:
                p = uniform(0.01, 0.05)
                t_train = [
                    {"input": self._noisify(ex["input"], p), "output": ex["output"]}
                    for ex in task.get("train", [])
                    if self.valid_example(ex)
                ]
                if not t_train:
                    continue
                shuffle(t_train)
                t_test_in = self._noisify(task["test"][0]["input"], p)
                augmented.append({"train": t_train, "test": [{"input": t_test_in}]})
                decs.append({"type": "noise", "noise_prob": p})
            except Exception as e:
                debug_print(f"Error in noise augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


class CropAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("crop", reversible=False)

    def _min_dims(self, task: dict) -> tuple[int, int]:
        mh, mw = 10**9, 10**9
        for ex in task.get("train", []) or []:
            if self.valid_example(ex):
                mh = min(mh, len(ex["input"]), len(ex["output"]))
                mw = min(mw, len(ex["input"][0]), len(ex["output"][0]))
        ti = task["test"][0]["input"]
        mh, mw = min(mh, len(ti)), min(mw, len(ti[0]))
        return int(mh), int(mw)

    def _crop(self, g: list[list[int]], h: int, w: int) -> list[list[int]] | None:
        if len(g) < h or len(g[0]) < w:
            return None
        r0 = randint(0, len(g) - h)
        c0 = randint(0, len(g[0]) - w)
        return [row[c0 : c0 + w] for row in g[r0 : r0 + h]]

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        augmented, decs = [], []
        for i in range(num_variants):
            try:
                mh, mw = self._min_dims(task)
                if mh < 3 or mw < 3:
                    continue
                ch = randint(max(2, mh - 2), mh)
                cw = randint(max(2, mw - 2), mw)
                t_train = []
                for ex in task.get("train", []) or []:
                    if self.valid_example(ex):
                        ci, co = self._crop(ex["input"], ch, cw), self._crop(ex["output"], ch, cw)
                        if ci and co:
                            t_train.append({"input": ci, "output": co})
                if not t_train:
                    continue
                shuffle(t_train)
                t_test_in = self._crop(task["test"][0]["input"], ch, cw)
                if not t_test_in:
                    continue
                augmented.append({"train": t_train, "test": [{"input": t_test_in}]})
                decs.append({"type": "crop", "crop_height": ch, "crop_width": cw})
            except Exception as e:
                debug_print(f"Error in crop augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


class DuplicateAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("duplicate", reversible=False)

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        augmented, decs = [], []
        for i in range(num_variants):
            try:
                train_copy = [ex.copy() for ex in task.get("train", [])]
                shuffle(train_copy)
                augmented.append({"train": train_copy, "test": [ex.copy() for ex in task.get("test", [])]})
                decs.append({"type": "duplicate", "copy_number": i + 1})
            except Exception as e:
                debug_print(f"Error in duplicate augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


class MixupAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("mixup", reversible=False)
        self.other_tasks: list[dict] = []
        cfg = AUGMENTATION_CONFIG.get("mixup", {})
        self.overlay_mode: str = str(cfg.get("overlay_mode", "replace_nonzero"))
        try:
            self.shift_inc: int = int(cfg.get("color_shift_increment", 1))
        except Exception:
            self.shift_inc = 1

    def set_task_pool(self, tasks: list[dict]):
        self.other_tasks = tasks or []

    def _mix(self, g1: list[list[int]], g2: list[list[int]], align: str = "center") -> list[list[int]] | None:
        """Overlay smaller grid onto larger at alignment.
        Modes:
          - replace_nonzero: foreground non-zero cells replace background.
          - add_mod: add foreground to background modulo 10 in overlap.
        align in {topleft, topright, bottomleft, bottomright, center}.
        """
        try:
            if not g1 or not g2:
                return None
            h1, w1 = len(g1), (len(g1[0]) if g1 and g1[0] else 0)
            h2, w2 = len(g2), (len(g2[0]) if g2 and g2[0] else 0)
            area1, area2 = h1 * w1, h2 * w2
            bg, fg = (g1, g2) if area1 >= area2 else (g2, g1)
            Hb, Wb = len(bg), (len(bg[0]) if bg and bg[0] else 0)
            Hf, Wf = len(fg), (len(fg[0]) if fg and fg[0] else 0)

            if align == "topleft":
                r0, c0 = 0, 0
            elif align == "topright":
                r0, c0 = 0, max(0, Wb - Wf)
            elif align == "bottomleft":
                r0, c0 = max(0, Hb - Hf), 0
            elif align == "bottomright":
                r0, c0 = max(0, Hb - Hf), max(0, Wb - Wf)
            else:  # center
                r0 = max(0, (Hb - Hf) // 2)
                c0 = max(0, (Wb - Wf) // 2)

            out = [row[:] for row in bg]
            for r in range(Hf):
                rr = r0 + r
                if rr >= Hb:
                    break
                for c in range(Wf):
                    cc = c0 + c
                    if cc >= Wb:
                        break
                    if self.overlay_mode == "add_mod":
                        out[rr][cc] = (out[rr][cc] + fg[r][c]) % 10
                    else:
                        v = fg[r][c]
                        if v != 0:
                            out[rr][cc] = v
            return out
        except Exception as e:
            debug_print(f"Error mixing grids: {e}")
            return None

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task) or not self.other_tasks:
            return [], []
        augmented, decs = [], []
        for i in range(num_variants):
            try:
                # Try to select an 'other' task different from the current task (by test grid)
                other = None
                attempts = 0
                base_test = (task.get("test") or [{}])[0].get("input")
                while attempts < 10 and self.other_tasks:
                    candidate = choice(self.other_tasks)
                    other_test = (candidate.get("test") or [{}])[0].get("input")
                    if other_test != base_test:
                        other = candidate
                        break
                    attempts += 1
                if other is None:
                    other = choice(self.other_tasks)
                if not self.valid_task(other):
                    continue
                align_choice = choice(["topleft", "topright", "bottomleft", "bottomright", "center"])
                mixed_train = []
                for j, ex in enumerate(task.get("train", [])):
                    if not self.valid_example(ex) or j >= len(other.get("train", [])):
                        continue
                    oex = other["train"][j]
                    if not self.valid_example(oex):
                        continue
                    # Optional color shift of foreground before overlay
                    oi = oex["input"]
                    oo = oex.get("output")
                    if self.shift_inc:

                        def _shift(g):
                            return [[(v + self.shift_inc) % 10 for v in row] for row in g]

                        oi = _shift(oi)
                        if oo is not None:
                            oo = _shift(oo)
                    mi = self._mix(ex["input"], oi, align_choice)
                    mo = self._mix(ex["output"], oo, align_choice) if oo is not None else None
                    if mi is not None and mo is not None:
                        mixed_train.append({"input": mi, "output": mo})
                if not mixed_train:
                    continue
                shuffle(mixed_train)
                toi = other.get("test", [{}])[0].get("input", task["test"][0]["input"]) or task["test"][0]["input"]
                if self.shift_inc:
                    toi = [[(v + self.shift_inc) % 10 for v in row] for row in toi]
                ti = self._mix(task["test"][0]["input"], toi, align_choice)
                if not ti:
                    continue
                augmented.append({"train": mixed_train, "test": [{"input": ti}]})
                decs.append(
                    {"type": "mixup", "mix_ratio": 0.5, "alignment": align_choice, "overlay_mode": self.overlay_mode}
                )
            except Exception as e:
                debug_print(f"Error in mixup augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


class InputOutputSwapAugmentation(AugmentationBase):
    def __init__(self, swap_probability: float = 0.5):
        super().__init__("input_output_swap", reversible=False)
        self.swap_probability = swap_probability

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            return [], []
        augmented, decs = [], []
        for _ in range(num_variants):
            try:
                swapped, decisions = [], []
                for ex in task.get("train", []) or []:
                    if not self.valid_example(ex):
                        continue
                    do_swap = uniform(0, 1) < self.swap_probability
                    decisions.append(do_swap)
                    if do_swap:
                        swapped.append({"input": _deepcopy_grid(ex["output"]), "output": _deepcopy_grid(ex["input"])})
                    else:
                        swapped.append({"input": _deepcopy_grid(ex["input"]), "output": _deepcopy_grid(ex["output"])})
                if not swapped:
                    continue
                shuffle(swapped)
                ti = _deepcopy_grid(task["test"][0]["input"])  # unchanged test
                augmented.append({"train": swapped, "test": [{"input": ti}]})
                decs.append(
                    {
                        "type": "input_output_swap",
                        "swap_probability": self.swap_probability,
                        "swap_decisions": decisions,
                    }
                )
            except Exception as e:
                debug_print(f"Error in input-output swap augmentation: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        return grid


# -----------------------------------------------------------------------------
# Combined (color+geom) reversible aug
# -----------------------------------------------------------------------------


class CombinedAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("geometric_color", reversible=True)
        self.geom_pairs = _GEOM_PAIRS

    def augment(self, task: dict, num_variants: int) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            debug_print("  Task validation failed for combined augmentation")
            return [], []
        augmented, decs = [], []
        pairs = self.geom_pairs[:]
        shuffle(pairs)
        maps = get_colormaps(len(get_symbols(task)), max(1, num_variants))
        shuffle(maps)
        for i in range(num_variants):
            try:
                enc, dec = pairs[i % len(pairs)]
                cmap = maps[i % len(maps)]
                forward = {orig: cmap[orig] for orig in range(MAX_SYMBOLS)}
                inverse = {v: k for k, v in forward.items()}

                def color(g, cmap_forward=forward):
                    return _map_grid_colors(g, cmap_forward)

                t_train = []
                for ex in task.get("train", []) or []:
                    if self.valid_example(ex):
                        t_train.append({"input": enc(color(ex["input"])), "output": enc(color(ex["output"]))})
                if not t_train:
                    continue
                shuffle(t_train)
                ti = enc(color(task["test"][0]["input"]))
                if not is_grid(ti):
                    continue
                augmented.append({"train": t_train, "test": [{"input": ti}]})
                decs.append(
                    {
                        "type": "geometric_color",
                        "geom_name": GEOM_FUNCTION_NAMES.get(dec, "identity"),
                        "colmap_inv": inverse,
                    }
                )
            except Exception as e:
                debug_print(f"Error in combined augmentation {i}: {e}")
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        try:
            grid = _map_grid_colors(grid, decoder_desc.get("colmap_inv", {}))
            geom = decoder_desc.get("geom_name", "identity")
            return GEOM_FUNCTIONS.get(geom, identity)(grid)
        except Exception as e:
            debug_print(f"Error applying combined decoder: {e}")
            return [[0]]


# -----------------------------------------------------------------------------
# Combine multiple tasks into one mega-task (reversible via split + reverse)
# -----------------------------------------------------------------------------


class CombineAugmentation(AugmentationBase):
    def __init__(self):
        super().__init__("combine", reversible=True)
        cfg = AUGMENTATION_CONFIG.get("combine", {})
        self._enforce_seq_shift: bool = bool(cfg.get("enforce_sequential_color_shift", True))
        try:
            self._shift_inc: int = int(cfg.get("color_shift_increment", 1))
        except Exception:
            self._shift_inc = 1

    # ---- Public API ----

    def augment(
        self,
        task: dict,
        num_variants: int,
        method: str | None = None,
        num_to_combine: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            debug_print("  Task validation failed for combine augmentation")
            return [], []
        cfg = AUGMENTATION_CONFIG.get("combine", {})
        max_combine = cfg.get("max_tasks_to_combine", 3)
        methods = cfg.get(
            "test_board_combination_methods",
            ["horizontal", "vertical", "color_separator_horizontal", "color_separator_vertical"],
        )
        augment_train_pairs = cfg.get("augment_train_pairs", True)
        augmented, decs = [], []
        attempts = 5
        for i in range(num_variants):
            success = False
            for _ in range(attempts):
                try:
                    m_raw = method or choice(methods)
                    k = num_to_combine or randint(2, max(2, int(max_combine)))
                    # Resolve grid placeholder to a concrete shape if selected
                    if m_raw.startswith("grid") and not re.match(r"grid_\d+x\d+", m_raw):
                        r, c = self._choose_grid_dims(max(2, k))
                        k = r * c
                        m = f"grid_{r}x{c}"
                    else:
                        m = m_raw
                    result = self._create_combined_task(task, k, m, augment_train_pairs)
                    if not result or result[0] is None:
                        continue
                    combined_task, original_sizes, variant_transforms = result
                    if not combined_task or not combined_task.get("test") or not original_sizes:
                        continue
                    decs.append(
                        {
                            "type": "combine",
                            "method": m,
                            "num_combined": k,
                            "original_sizes": original_sizes,
                            "separator_color": 9 if m.startswith("color_separator") else None,
                            "variant_transformations": variant_transforms,
                        }
                    )
                    augmented.append(combined_task)
                    success = True
                    break
                except Exception as e:
                    debug_print(f"Error in combine augmentation variant {i + 1}: {e}")
            if not success:
                debug_print(
                    f"  Failed to generate combine variant {i + 1} after {attempts} attempts - excluding this item"
                )
        return augmented, decs

    def apply_decoder(self, grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
        try:
            candidates, _ = self.apply_decoder_multi_task(grid, decoder_desc)
            valid = [g for g in candidates if is_grid(g)]
            if not valid:
                return grid
            votes = Counter(str(g) for g in valid)
            best_s, _ = votes.most_common(1)[0]
            try:
                best = ast.literal_eval(best_s)
                return best if is_grid(best) else valid[0]
            except Exception:
                return valid[0]
        except Exception as e:
            debug_print(f"Error applying combine decoder with voting: {e}")
            return grid

    def apply_decoder_multi_task(
        self, model_output: Any, decoder_desc: dict
    ) -> tuple[list[list[list[int]]], list[list[int]] | None]:
        try:
            combined = model_output
            if isinstance(model_output, str):
                combined = self._parse_combined_grid_from_text(model_output, decoder_desc)
            if not combined:
                debug_print("Could not parse or obtain a valid combined grid from model output.")
                return [], None
            method = decoder_desc.get("method", "horizontal")
            sizes = decoder_desc.get("original_sizes", [])
            sep = decoder_desc.get("separator_color", 9)
            transforms = decoder_desc.get("variant_transformations", [])
            # If dimensions mismatch, attempt lenient adjustment to expected dims
            exp_h, exp_w = self._expected_dims(method, sizes)
            act_h, act_w = (len(combined), len(combined[0]) if combined else 0)
            if exp_h and exp_w and (act_h != exp_h or act_w != exp_w):
                # Adjust by padding or cropping to expected dims (best-effort salvage)
                combined = self._fit_to_dims(combined, exp_h, exp_w)
            split, ok = self._split_combined_grid(combined, method, sizes, sep)
            if not ok and method.startswith("color_separator"):
                # Fallback: ignore separators and split by expected sizes directly
                split, ok = self._fallback_split_without_separators(combined, method, sizes)
            if not ok or not split:
                return [combined], combined
            if len(split) != len(transforms):
                debug_print("Warning: Mismatch between split grids and transform info; returning raw splits.")
                return split, combined
            decoded = [
                self._reverse_transformation(g, t) if is_grid(g) else g for g, t in zip(split, transforms, strict=False)
            ]
            # If a global pre-augmentation was applied uniformly, reverse it per split
            try:
                gp = decoder_desc.get("global_pre")
                if gp and isinstance(gp, dict):
                    mgr = AugmentationManager()
                    decoded = [mgr.apply_decoder(g, gp) for g in decoded]
            except Exception:
                pass
            return decoded, combined
        except Exception as e:
            debug_print(f"Error applying multi-task combine decoder: {e}")
        return ([model_output] if isinstance(model_output, list) else []), None

    # ---- Internal helpers ----

    @staticmethod
    def _fit_to_dims(grid: list[list[int]], h: int, w: int) -> list[list[int]]:
        """Best-effort adjust grid to exact (h, w) via crop/pad."""
        if not grid:
            return [[0] * w for _ in range(h)]
        gh, gw = len(grid), len(grid[0])
        # Crop if too large
        g2 = [row[: min(gw, w)] for row in grid[: min(gh, h)]]
        # Pad if too small
        return _pad_to(g2, h, w)

    @staticmethod
    def _fallback_split_without_separators(
        grid: list[list[int]], method: str, sizes: list[dict]
    ) -> tuple[list[list[list[int]]], bool]:
        """Split combined grid by expected sizes ignoring separators (lenient mode)."""
        if not grid or not sizes:
            return [], False
        H, W = len(grid), len(grid[0])
        parts: list[list[list[int]]] = []
        try:
            if method.endswith("horizontal"):  # stacked vertically
                r = 0
                for s in sizes:
                    h = int(s.get("height", 0))
                    if h <= 0:
                        return [], False
                    parts.append(grid[r : min(r + h, H)])
                    r += h
                # Pad/crop each part to its expected width using dominant color
                for i, s in enumerate(sizes):
                    w = int(s.get("width", 0))
                    parts[i] = _pad_to([row[: min(len(row), w)] for row in parts[i]], len(parts[i]), w)
            else:  # vertical side-by-side
                c = 0
                for s in sizes:
                    w = int(s.get("width", 0))
                    if w <= 0:
                        return [], False
                    seg = [row[c : min(c + w, W)] for row in grid]
                    h = int(s.get("height", 0))
                    seg = _pad_to(seg, h, len(seg[0]) if seg and seg[0] else w)
                    parts.append(seg)
                    c += w
            return parts, True
        except Exception:
            return [], False

    def _parse_combined_grid_from_text(self, text: str, decoder_desc: dict) -> list[list[int]] | None:
        try:
            # Allow letter-encoded symbols by decoding them back to digits first
            try:
                from .grid_utils import decode_symbols_in_text

                text = decode_symbols_in_text(text)
            except Exception:
                pass
            cleaned = re.sub(r"[^0-9 .]", "", text)
            parts = [p for p in cleaned.split(".")[0].strip().split(" ") if p]
            if len(parts) <= 4:
                debug_print(f"Strict parse failed: Not enough parts in output ({len(parts)} <= 4).")
                return None
            row_tokens = parts[4:]
            grid = [[int(d) for d in row] for row in row_tokens if row.isdigit()]
            if not is_grid(grid):
                debug_print("Strict parse failed: Parsed grid invalid.")
                return None
            method = decoder_desc.get("method", "horizontal")
            sizes = decoder_desc.get("original_sizes", [])
            if not sizes:
                debug_print("Strict parse failed: Decoder has no original_sizes.")
                return None
            exp_h, exp_w = self._expected_dims(method, sizes)
            act_h, act_w = (len(grid), len(grid[0]) if grid else 0)
            if (act_h, act_w) != (exp_h, exp_w):
                debug_print(f"Strict parse failed: Dimension mismatch. Expected {exp_w}x{exp_h}, got {act_w}x{act_h}.")
                return None
            return grid
        except Exception as e:
            debug_print(f"Exception in _parse_combined_grid_from_text: {e}")
            return None

    @staticmethod
    def _expected_dims(method: str, sizes: list[dict]) -> tuple[int, int]:
        n = len(sizes)
        # Grid method: parse grid_RxC
        if method.startswith("grid_"):
            try:
                shp = method.split("_", 1)[1]
                r_str, c_str = shp.split("x")
                R, C = int(r_str), int(c_str)
            except Exception:
                R, C = 1, n
            # Row heights: max height per row; Col widths: max width per col
            row_heights = [0] * R
            col_widths = [0] * C
            for idx, s in enumerate(sizes[: R * C]):
                rr, cc = divmod(idx, C)
                row_heights[rr] = max(row_heights[rr], int(s.get("height", 0)))
                col_widths[cc] = max(col_widths[cc], int(s.get("width", 0)))
            h = sum(row_heights)
            w = sum(col_widths)
        elif method == "horizontal":
            h = max(s.get("height", 0) for s in sizes)
            w = sum(s.get("width", 0) for s in sizes)
        elif method == "vertical":
            h = sum(s.get("height", 0) for s in sizes)
            w = max(s.get("width", 0) for s in sizes)
        elif method == "color_separator_horizontal":  # stacked vertically with separator rows
            h = sum(s.get("height", 0) for s in sizes) + (n - 1)
            w = max(s.get("width", 0) for s in sizes)
        elif method == "color_separator_vertical":  # side-by-side with separator cols
            h = max(s.get("height", 0) for s in sizes)
            w = sum(s.get("width", 0) for s in sizes) + (n - 1)
        else:  # fallback to horizontal
            h = max(s.get("height", 0) for s in sizes)
            w = sum(s.get("width", 0) for s in sizes)
        return h, w

    @staticmethod
    def _choose_grid_dims(k: int) -> tuple[int, int]:
        """Pick near-square grid dims R x C for k items, R,C>=1, R<=C.
        Prefer factors when possible; else fall back to 1 x k.
        """
        if k <= 0:
            return 1, 1
        # Perfect square near sqrt
        best = (1, k)
        diff = k - 1
        r = 1
        while r * r <= k:
            if k % r == 0:
                c = k // r
                if c < r:
                    r, c = c, r
                if c - r < diff:
                    diff = c - r
                    best = (r, c)
            r += 1
        return best

    def _create_combined_task(
        self, task: dict, num_to_combine: int, method: str, augment_train_pairs: bool = True
    ) -> tuple[dict | None, list[dict], list[dict]]:
        import copy

        original_train = task.get("train", [])
        variation = choice(["train_more", "test_more", "same"])  # variety strategy
        train_k = num_to_combine if variation != "test_more" else max(2, num_to_combine - 1)
        test_k = num_to_combine if variation != "train_more" else max(2, num_to_combine - 1)
        # Enforce exact count for grid layouts
        if isinstance(method, str) and method.startswith("grid_"):
            try:
                shp = method.split("_", 1)[1]
                R, C = map(int, shp.split("x"))
                required = max(1, R * C)
                train_k = required
                test_k = required
            except Exception:
                pass

        # Build training variants (optionally enforce sequential color shifts)
        train_variants: list[dict] = []
        if self._enforce_seq_shift:
            for i in range(train_k):
                if i == 0:
                    v = copy.deepcopy(task)
                else:
                    shift = (i * self._shift_inc) % 10
                    v = self._apply_color_shift(task, shift)
                train_variants.append(v)
        else:
            for i in range(train_k):
                v = copy.deepcopy(task) if i == 0 else self._create_task_variant(task, i, "enhanced")
                train_variants.append(v)

        # Build test variants + transform info
        test_variants, transforms, sizes = [], [], []
        if self._enforce_seq_shift:
            for i in range(test_k):
                if i == 0:
                    v = copy.deepcopy(task)
                    info = {"type": "original", "params": None, "variant_index": 0}
                else:
                    shift = (i * self._shift_inc) % 10
                    v = self._apply_color_shift(task, shift)
                    info = {"type": "color_shift", "params": shift, "variant_index": i}
                test_variants.append(v)
                transforms.append(info)
                ti = v.get("test", [{}])[0].get("input", [])
                if ti:
                    sizes.append({"width": len(ti[0]) if ti and ti[0] else 1, "height": len(ti)})
        else:
            for i in range(test_k):
                if i == 0:
                    v, info = copy.deepcopy(task), {"type": "original", "params": None, "variant_index": 0}
                else:
                    v, info = self._create_task_variant_with_transform_info(task, i, "enhanced")
                test_variants.append(v)
                transforms.append(info)
                ti = v.get("test", [{}])[0].get("input", [])
                if ti:
                    sizes.append({"width": len(ti[0]) if ti and ti[0] else 1, "height": len(ti)})

        # Build combined training pairs if requested
        combined_train: list[dict] = []
        if augment_train_pairs and train_variants:
            # gather aligned examples across variants
            cols = [tv.get("train", []) for tv in train_variants]
            max_n = max((len(c) for c in cols), default=0)
            for idx in range(max_n):
                in_boards, out_boards = [], []
                for col in cols:
                    ex = col[idx] if idx < len(col) else (col[-1] if col else {})
                    if ex:
                        in_boards.append(ex.get("input", []))
                        out_boards.append(ex.get("output", []))
                ci, _sizes_i, ok_i = self._validate_and_combine_grids(in_boards, method, 9)
                co, _, ok_o = self._validate_and_combine_grids(out_boards, method, 9)
                if ok_i and ok_o:
                    combined_train.append({"input": ci, "output": co})
        else:
            combined_train = original_train.copy() if original_train else []

        # Combine test input/output
        test_inputs = [tv.get("test", [{}])[0].get("input", []) for tv in test_variants]
        test_outputs = [
            tv.get("test", [{}])[0].get("output", [])
            for tv in test_variants
            if tv.get("test", [{}])[0].get("output") is not None
        ]

        cti, sizes_checked, ok = self._validate_and_combine_grids(test_inputs, method, 9)
        if not ok:
            return None, [], []  # always return triple
        cto, _, ok_o = (None, [], False)
        if test_outputs and len(test_outputs) == len(test_inputs):
            cto, _, ok_o = self._validate_and_combine_grids(test_outputs, method, 9)
            if not ok_o:
                debug_print(f"Failed to combine test outputs for method {method}; proceeding without output")
                cto = None

        shuffle(combined_train)
        test_case = {"input": cti}
        if cto is not None:
            test_case["output"] = cto
        combined_task = {"train": combined_train, "test": [test_case]}
        return combined_task, sizes_checked, transforms

    # Transform variants (kept compatible with original)
    def _create_task_variant(self, task: dict, variant_index: int, mode: str = "simple") -> dict:
        import copy

        variant = copy.deepcopy(task)
        if mode == "enhanced":
            transformations = [
                ("color_shift", 1),
                ("color_shift", 2),
                ("color_shift", 3),
                ("mirror_horizontal", None),
                ("mirror_vertical", None),
                ("rotate_90", None),
                ("color_invert", None),
                ("color_swap", (1, 2)),
                ("color_swap", (0, 3)),
            ]
            if 1 <= variant_index <= len(transformations):
                t, p = transformations[variant_index - 1]
                return self._apply_transformation(variant, t, p)
        else:
            if variant_index == 1:
                return self._apply_color_shift(variant, 1)
            if variant_index == 2:
                return self._apply_color_shift(variant, 2)
        return variant

    def _create_task_variant_with_transform_info(
        self, task: dict, variant_index: int, mode: str = "simple"
    ) -> tuple[dict, dict]:
        v = self._create_task_variant(task, variant_index, mode)
        if mode == "enhanced" and variant_index > 0:
            transformations = [
                ("color_shift", 1),
                ("color_shift", 2),
                ("color_shift", 3),
                ("mirror_horizontal", None),
                ("mirror_vertical", None),
                ("rotate_90", None),
                ("color_invert", None),
                ("color_swap", (1, 2)),
                ("color_swap", (0, 3)),
            ]
            t, p = transformations[min(variant_index - 1, len(transformations) - 1)]
            return v, {"type": t, "params": p, "variant_index": variant_index}
        if variant_index in (1, 2):
            return v, {"type": "color_shift", "params": variant_index, "variant_index": variant_index}
        return v, {"type": "original", "params": None, "variant_index": variant_index}

    def _apply_transformation(self, task: dict, ttype: str, param) -> dict:
        if ttype == "color_shift":
            return self._apply_color_shift(task, param or 0)
        if ttype == "mirror_horizontal":
            return self._apply_grid_xform(task, lambda g: [row[::-1] for row in g])
        if ttype == "mirror_vertical":
            return self._apply_grid_xform(task, lambda g: g[::-1])
        if ttype == "rotate_90":

            def rot(g):
                if not g:
                    return g
                r, c = len(g), len(g[0])
                out = [[0] * r for _ in range(c)]
                for i in range(r):
                    for j in range(c):
                        out[j][r - 1 - i] = g[i][j]
                return out

            return self._apply_grid_xform(task, rot)
        if ttype == "color_invert":
            return self._apply_grid_xform(task, lambda g: [[(9 - v) % 10 for v in row] for row in g])
        if ttype == "color_swap":
            a, b = param

            def swap(g):
                def f(v):
                    return b if v == a else (a if v == b else v)

                return [[f(v) for v in row] for row in g]

            return self._apply_grid_xform(task, swap)
        return task

    def _apply_color_shift(self, task: dict, shift: int) -> dict:
        def s(g):
            return [[(v + shift) % 10 for v in row] for row in g]

        return self._apply_grid_xform(task, s)

    @staticmethod
    def _apply_grid_xform(task: dict, fn) -> dict:
        return _apply_to_task(task, fn)

    # Reverse individual transform for each split grid
    def _reverse_transformation(self, grid: list[list[int]], info: dict) -> list[list[int]]:
        t = (info or {}).get("type", "original")
        p = (info or {}).get("params")
        try:
            if t == "original":
                return grid
            if t == "color_shift":
                shift = (-(p or 0)) % 10
                return [[(v + shift) % 10 for v in row] for row in grid]
            if t == "mirror_horizontal":
                return [row[::-1] for row in grid]
            if t == "mirror_vertical":
                return grid[::-1]
            if t == "rotate_90":
                # rotate 270 (3x 90)
                def rot90(g):
                    if not g:
                        return g
                    r, c = len(g), len(g[0])
                    out = [[0] * r for _ in range(c)]
                    for i in range(r):
                        for j in range(c):
                            out[j][r - 1 - i] = g[i][j]
                    return out

                out = grid
                for _ in range(3):
                    out = rot90(out)
                return out
            if t == "color_invert":
                return [[(9 - v) % 10 for v in row] for row in grid]
            if t == "color_swap":
                if p and len(p) == 2:
                    a, b = p

                    def swap(v):
                        return b if v == a else (a if v == b else v)

                    return [[swap(v) for v in row] for row in grid]
                return grid
            debug_print(f"Unknown transformation type for reversal: {t}")
            return grid
        except Exception as e:
            debug_print(f"Error reversing transformation {t}: {e}")
            return grid

    # ---- Safe combine/split primitives ----

    def _validate_and_combine_grids(
        self, grids: list[list[list[int]]], method: str, sep_color: int = 9
    ) -> tuple[list[list[int]], list[dict], bool]:
        if not grids or len(grids) < 2:
            debug_print(f"Combine validation failed: Need >=2 grids, got {len(grids) if grids else 0}")
            return [], [], False
        valid, sizes = [], []
        for i, g in enumerate(grids):
            if not g or not isinstance(g, list) or not all(isinstance(r, list) and r for r in g):
                debug_print(f"Combine validation failed: Grid {i} invalid/empty")
                continue
            w = len(g[0])
            if not all(len(r) == w for r in g):
                debug_print(f"Combine validation failed: Grid {i} row length mismatch")
                continue
            h = len(g)
            if w > 100 or h > 100:
                debug_print(f"Combine validation failed: Grid {i} too large ({w}x{h})")
                continue
            valid.append(g)
            sizes.append({"width": w, "height": h})
        if len(valid) < 2:
            debug_print(f"Combine validation failed: Only {len(valid)} valid grids after filtering")
            return [], [], False

        # pre-limit checks
        if method.startswith("grid_"):
            try:
                shp = method.split("_", 1)[1]
                R, C = map(int, shp.split("x"))
            except Exception:
                R, C = 1, len(valid)
            # Compute expected overall dims based on row/col maxima
            row_heights = [0] * R
            col_widths = [0] * C
            for idx, s in enumerate(sizes[: R * C]):
                rr, cc = divmod(idx, C)
                row_heights[rr] = max(row_heights[rr], s["height"])
                col_widths[cc] = max(col_widths[cc], s["width"])
            W, H = sum(col_widths), sum(row_heights)
            if W > 200 or H > 200:
                debug_print(f"Combine validation failed: Combined grid too large ({W}x{H}) for grid {R}x{C}")
                return [], [], False
        elif method == "horizontal":
            W, H = sum(s["width"] for s in sizes), max(s["height"] for s in sizes)
            if W > 200 or H > 100:
                debug_print(f"Combine validation failed: Combined horizontal too large ({W}x{H})")
                return [], [], False
        elif method == "vertical":
            W, H = max(s["width"] for s in sizes), sum(s["height"] for s in sizes)
            if W > 100 or H > 200:
                debug_print(f"Combine validation failed: Combined vertical too large ({W}x{H})")
                return [], [], False
        elif method == "color_separator_horizontal":
            W, H = max(s["width"] for s in sizes), sum(s["height"] for s in sizes) + (len(valid) - 1)
            if W > 100 or H > 200:
                debug_print(f"Combine validation failed: Combined sep-horiz too large ({W}x{H})")
                return [], [], False
        elif method == "color_separator_vertical":
            W, H = sum(s["width"] for s in sizes) + (len(valid) - 1), max(s["height"] for s in sizes)
            if W > 200 or H > 100:
                debug_print(f"Combine validation failed: Combined sep-vert too large ({W}x{H})")
                return [], [], False
        else:
            debug_print(f"Combine validation failed: Unknown method '{method}'")
            return [], [], False

        # actual combine
        if method.startswith("grid_"):
            out = self._combine_grid(valid, method)
        elif method == "horizontal":
            out = self._combine_h(valid)
        elif method == "vertical":
            out = self._combine_v(valid)
        elif method == "color_separator_horizontal":
            out = self._combine_with_sep(valid, sep_color, direction="horizontal")
        else:  # color_separator_vertical
            out = self._combine_with_sep(valid, sep_color, direction="vertical")

        if not out or not all(isinstance(r, list) and r for r in out):
            debug_print("Combine failed: Invalid combined grid structure")
            return [], [], False
        W0 = len(out[0])
        if not all(len(r) == W0 for r in out):
            debug_print("Combine failed: Combined grid has inconsistent row lengths")
            return [], [], False
        return out, sizes, True

    def _combine_h(self, grids: list[list[list[int]]]) -> list[list[int]]:
        H = max(len(g) for g in grids)
        out: list[list[int]] = []
        for r in range(H):
            row: list[int] = []
            for g in grids:
                dom = _dominant_color(g)
                if r < len(g):
                    row.extend(g[r])
                else:
                    row.extend([dom] * (len(g[0]) if g else 1))
            out.append(row)
        return out

    def _combine_v(self, grids: list[list[list[int]]]) -> list[list[int]]:
        W = max(len(g[0]) if g else 0 for g in grids)
        out: list[list[int]] = []
        for g in grids:
            dom = _dominant_color(g)
            for row in g:
                out.append(row + [dom] * (W - len(row)))
        return out

    def _combine_with_sep(self, grids: list[list[list[int]]], sep: int, direction: str) -> list[list[int]]:
        if direction == "horizontal":  # stack vertically with separator rows
            W = max(len(g[0]) if g else 0 for g in grids)
            out: list[list[int]] = []
            for i, g in enumerate(grids):
                if i > 0:
                    out.append([sep] * W)
                dom = _dominant_color(g)
                for row in g:
                    out.append(row + [dom] * (W - len(row)))
            return out
        else:  # vertical separators between side-by-side blocks
            H = max(len(g) if g else 0 for g in grids)
            out: list[list[int]] = [[] for _ in range(H)]

            def pad_row(row, W):
                return row + [_dominant_color([row])] * (W - len(row))

            # seed with first grid
            first = grids[0]
            dom0 = _dominant_color(first)
            W0 = len(first[0]) if first else 1
            for r in range(H):
                if r < len(first):
                    out[r] = first[r][:]
                else:
                    out[r] = [dom0] * W0
            # append others with sep
            for g in grids[1:]:
                for r in range(H):
                    out[r].append(sep)
                dom = _dominant_color(g)
                W = len(g[0]) if g else 1
                for r in range(H):
                    if r < len(g):
                        out[r].extend(g[r])
                    else:
                        out[r].extend([dom] * W)
            return out

    def _combine_grid(self, grids: list[list[list[int]]], method: str) -> list[list[int]]:
        # Parse shape
        try:
            shp = method.split("_", 1)[1]
            R, C = map(int, shp.split("x"))
        except Exception:
            R, C = 1, len(grids)
        total = R * C
        cells = grids[:total]
        # If not enough cells, pad with copies of the last grid to fill the grid
        if len(cells) < total:
            filler = cells[-1] if cells else ([[0]],)
            while len(cells) < total:
                cells.append(filler)
        # Compute per-row height and per-col width
        row_heights = [0] * R
        col_widths = [0] * C
        for idx, g in enumerate(cells):
            rr, cc = divmod(idx, C)
            row_heights[rr] = max(row_heights[rr], len(g))
            col_widths[cc] = max(col_widths[cc], len(g[0]) if g and g[0] else 0)
        H = sum(row_heights)
        W = sum(col_widths)
        # Initialize canvas with 0s (will get overwritten fully)
        out: list[list[int]] = [[0] * W for _ in range(H)]
        # Place each grid in its cell aligned top-left, pad with its dominant color
        r_off = 0
        for rr in range(R):
            c_off = 0
            for cc in range(C):
                idx = rr * C + cc
                if idx >= len(cells):
                    break
                g = cells[idx]
                gh = len(g)
                gw = len(g[0]) if g and g[0] else 0
                cell_h = row_heights[rr]
                cell_w = col_widths[cc]
                dom = _dominant_color(g)
                for r in range(cell_h):
                    for c in range(cell_w):
                        if r < gh and c < gw:
                            out[r_off + r][c_off + c] = g[r][c]
                        else:
                            out[r_off + r][c_off + c] = dom
                c_off += cell_w
            r_off += row_heights[rr]
        return out

    def _split_combined_grid(
        self, grid: list[list[int]], method: str, sizes: list[dict], sep_color: int = 9
    ) -> tuple[list[list[list[int]]], bool]:
        if not grid or not sizes:
            debug_print("Split failed: Empty grid or missing sizes")
            return [], False
        if method.startswith("grid_"):
            return self._split_grid(grid, method, sizes)
        if method == "horizontal":
            return self._split_h(grid, sizes)
        if method == "vertical":
            return self._split_v(grid, sizes)
        if method == "color_separator_horizontal":
            return self._split_sep(grid, sizes, sep_color, direction="horizontal")
        if method == "color_separator_vertical":
            return self._split_sep(grid, sizes, sep_color, direction="vertical")
        debug_print(f"Split failed: Unknown method '{method}'")
        return [], False

    def _split_grid(self, grid: list[list[int]], method: str, sizes: list[dict]) -> tuple[list[list[list[int]]], bool]:
        # Parse shape
        try:
            shp = method.split("_", 1)[1]
            R, C = map(int, shp.split("x"))
        except Exception:
            return [], False
        total = R * C
        if not sizes or len(sizes) < total:
            return [], False
        # Compute row heights and col widths using expected maxima (as in combine)
        row_heights = [0] * R
        col_widths = [0] * C
        for idx, s in enumerate(sizes[:total]):
            rr, cc = divmod(idx, C)
            row_heights[rr] = max(row_heights[rr], int(s.get("height", 0)))
            col_widths[cc] = max(col_widths[cc], int(s.get("width", 0)))
        parts: list[list[list[int]]] = []
        r0 = 0
        idx = 0
        for rr in range(R):
            c0 = 0
            for cc in range(C):
                if idx >= total:
                    break
                cell_h = row_heights[rr]
                cell_w = col_widths[cc]
                # Slice the cell region
                block = [row[c0 : c0 + cell_w] for row in grid[r0 : r0 + cell_h]]
                # Crop back to original size for this cell
                s = sizes[idx]
                h = int(s.get("height", 0))
                w = int(s.get("width", 0))
                block = [row[:w] for row in block[:h]]
                parts.append(block)
                c0 += cell_w
                idx += 1
            r0 += row_heights[rr]
        return parts, True

    def _split_h(self, grid: list[list[int]], sizes: list[dict]) -> tuple[list[list[list[int]]], bool]:
        parts, c = [], 0
        Wtot = len(grid[0])
        for s in sizes:
            w = int(s.get("width", 0))
            h = int(s.get("height", 0))
            if w <= 0 or c + w > Wtot:
                debug_print(f"Split horizontal failed at col {c} width {w}")
                return [], False
            block_full = [row[c : c + w] for row in grid]
            # Crop to original height (top-aligned)
            parts.append(block_full[:h] if h > 0 else block_full)
            c += w
        return parts, True

    def _split_v(self, grid: list[list[int]], sizes: list[dict]) -> tuple[list[list[list[int]]], bool]:
        parts, r = [], 0
        Htot = len(grid)
        for s in sizes:
            h = int(s.get("height", 0))
            w = int(s.get("width", 0))
            if h <= 0 or r + h > Htot:
                debug_print(f"Split vertical failed at row {r} height {h}")
                return [], False
            block_rows = grid[r : r + h]
            # Crop each row to original width (left-aligned)
            parts.append([row[:w] for row in block_rows] if w > 0 else block_rows)
            r += h
        return parts, True

    def _split_sep(
        self, grid: list[list[int]], sizes: list[dict], sep: int, direction: str
    ) -> tuple[list[list[list[int]]], bool]:
        if direction == "horizontal":  # stacked vertically with separator rows
            parts, r = [], 0
            for i, s in enumerate(sizes):
                if i > 0:
                    if r >= len(grid):
                        debug_print("Split sep-horiz failed: missing separator row")
                        return [], False
                    if not all(v == sep for v in grid[r]):
                        debug_print(f"Split sep-horiz warn: row {r} not all sep color {sep}")
                    r += 1
                h = int(s.get("height", 0))
                w = int(s.get("width", 0))
                if h <= 0 or r + h > len(grid):
                    debug_print("Split sep-horiz failed: invalid height")
                    return [], False
                rows = grid[r : r + h]
                parts.append([row[:w] for row in rows] if w > 0 else rows)
                r += h
            return parts, True
        else:  # vertical separators between blocks
            parts, c = [], 0
            Wtot = len(grid[0]) if grid else 0
            for i, s in enumerate(sizes):
                if i > 0:
                    if c >= Wtot:
                        debug_print("Split sep-vert failed: missing separator col")
                        return [], False
                    for row in grid:
                        if row[c] != sep:
                            debug_print(f"Split sep-vert warn: col {c} not sep {sep}")
                            break
                    c += 1
                w = int(s.get("width", 0))
                h = int(s.get("height", 0))
                if w <= 0 or c + w > Wtot:
                    debug_print("Split sep-vert failed: invalid width")
                    return [], False
                block_full = [row[c : c + w] for row in grid]
                parts.append(block_full[:h] if h > 0 else block_full)
                c += w
            return parts, True


# -----------------------------------------------------------------------------
# Mixup-Combine: Combine multiple different tasks side-by-side (reversible)
# -----------------------------------------------------------------------------


class MixupCombineAugmentation(CombineAugmentation):
    """Combine multiple different tasks by placing their boards adjacent.

    This behaves like Mixup (draws from a task pool) but uses Combine's
    concatenation strategies (horizontal/vertical/separators/grid) instead of
    overlaying. It is reversible via the inherited split + reverse transforms.
    """

    def __init__(self):
        super().__init__()
        # Override identity and config
        self.name = "mixup_combine"
        self.other_tasks: list[dict] = []
        cfg = AUGMENTATION_CONFIG.get("mixup_combine", {})
        self._enforce_seq_shift = bool(cfg.get("enforce_sequential_color_shift", True))
        try:
            self._shift_inc = int(cfg.get("color_shift_increment", 1))
        except Exception:
            self._shift_inc = 1

    # Public API to set the available pool of other tasks
    def set_task_pool(self, tasks: list[dict]):
        self.other_tasks = tasks or []

    def augment(
        self,
        task: dict,
        num_variants: int,
        method: str | None = None,
        num_to_combine: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        if not self.valid_task(task):
            debug_print("  Task validation failed for mixup_combine augmentation")
            return [], []
        if not self.other_tasks:
            debug_print("  mixup_combine requires a non-empty task pool; skipping")
            return [], []

        cfg = AUGMENTATION_CONFIG.get("mixup_combine", {})
        max_combine = cfg.get("max_tasks_to_combine", 2)
        methods = cfg.get(
            "test_board_combination_methods",
            ["horizontal", "vertical", "color_separator_horizontal", "color_separator_vertical", "grid"],
        )
        augment_train_pairs = cfg.get("augment_train_pairs", True)
        # If a pre-augmentation is requested (e.g., geometric_color), apply it
        # consistently to ALL selected tasks for the combined riddle.
        pre_aug_name = str(cfg.get("aug_before_with") or "").strip()

        augmented, decs = [], []
        attempts = 5
        for i in range(num_variants):
            success = False
            for _ in range(attempts):
                try:
                    # Choose method and number to combine
                    m_raw = method or choice(methods)
                    k = num_to_combine or randint(2, max(2, int(max_combine)))
                    if m_raw.startswith("grid") and not re.match(r"grid_\d+x\d+", m_raw):
                        R, C = self._choose_grid_dims(max(2, k))
                        k = R * C
                        m = f"grid_{R}x{C}"
                    else:
                        m = m_raw

                    # Select k DISTINCT tasks (by test input signature)
                    def _sig(t):
                        try:
                            ti = (t.get("test") or [{}])[0].get("input")
                            return str(ti)
                        except Exception:
                            return None

                    def _task_and_key(entry):
                        if isinstance(entry, dict) and "task" in entry:
                            return entry["task"], entry.get("task_key")
                        return entry, None

                    selected: list[dict] = []
                    selected_keys: list[str | None] = []
                    seen = set()
                    # Always attempt to include the base task first
                    s0 = _sig(task)
                    if s0:
                        selected.append(task)
                        selected_keys.append(None)  # base task key unknown at this scope
                        seen.add(s0)
                    pool = self.other_tasks[:]
                    shuffle(pool)
                    for cand in pool:
                        if len(selected) >= k:
                            break
                        t_cand, key_cand = _task_and_key(cand)
                        if not self.valid_task(t_cand):
                            continue
                        sc = _sig(t_cand)
                        if not sc or sc in seen:
                            continue
                        selected.append(t_cand)
                        selected_keys.append(key_cand)
                        seen.add(sc)
                    # If still short and duplicates not allowed, retry/skip
                    if (
                        len(selected) < k
                        and bool(cfg.get("require_distinct_tasks", True))
                        and not bool(cfg.get("allow_duplicates_if_insufficient", False))
                    ):
                        continue
                    # Else, fill remaining slots (may include duplicates)
                    while len(selected) < k and self.other_tasks:
                        t_fill, k_fill = _task_and_key(choice(self.other_tasks))
                        selected.append(t_fill)
                        selected_keys.append(k_fill)

                    if len(selected) < 2:
                        continue

                    # Optionally apply a UNIFORM pre-augmentation to all selected tasks
                    global_pre_decoder = None
                    if pre_aug_name == "geometric_color":
                        # Choose a single geom+color mapping for entire riddle
                        try:
                            # Pick a random geometric pair and a color map of full symbol space
                            enc_geom, dec_geom = choice(_GEOM_PAIRS)
                            # Sample a single color permutation and apply to ALL selected tasks
                            try:
                                maps = get_colormaps(MAX_SYMBOLS, 1)
                                cmap_list = maps[0] if maps else list(range(MAX_SYMBOLS))
                            except Exception:
                                cmap_list = list(range(MAX_SYMBOLS))
                            cmap = {i: (cmap_list[i] if i < len(cmap_list) else i) for i in range(MAX_SYMBOLS)}
                            inv = {v: k for k, v in cmap.items()}

                            def _apply_geocolor(t: dict, encoder=enc_geom, color_map=cmap) -> dict:
                                def f(g, local_encoder=encoder, local_map=color_map):
                                    return local_encoder(_map_grid_colors(g, local_map))

                                return _apply_to_task(t, f)

                            selected = [_apply_geocolor(t) for t in selected]
                            # Record inverse so decoder can reverse per split
                            global_pre_decoder = {
                                "type": "geometric_color",
                                "geom_name": GEOM_FUNCTION_NAMES.get(dec_geom, "identity"),
                                "colmap_inv": inv,
                            }
                        except Exception:
                            global_pre_decoder = None

                    # Build per-task transformed variants + transform descriptors
                    test_variants: list[dict] = []
                    transforms: list[dict] = []
                    sizes: list[dict] = []
                    if self._enforce_seq_shift:
                        for idx, task_i in enumerate(selected):
                            if idx == 0:
                                v = _apply_to_task(task_i, identity)
                                info = {"type": "original", "params": None, "variant_index": idx}
                            else:
                                shift = (idx * self._shift_inc) % 10
                                v = self._apply_color_shift(task_i, shift)
                                info = {"type": "color_shift", "params": shift, "variant_index": idx}
                            test_variants.append(v)
                            ti = (v.get("test") or [{}])[0].get("input")
                            if is_grid(ti):
                                sizes.append({"width": len(ti[0]) if ti and ti[0] else 1, "height": len(ti)})
                            transforms.append(info)
                    else:
                        for idx, task_i in enumerate(selected):
                            v, info = self._create_task_variant_with_transform_info(task_i, idx, "enhanced")
                            test_variants.append(v)
                            ti = (v.get("test") or [{}])[0].get("input")
                            if is_grid(ti):
                                sizes.append({"width": len(ti[0]) if ti and ti[0] else 1, "height": len(ti)})
                            transforms.append(info)

                    # Combine training pairs across tasks if requested
                    combined_train: list[dict] = []
                    if augment_train_pairs:
                        cols = [tv.get("train", []) for tv in test_variants]
                        max_n = max((len(c) for c in cols), default=0)
                        for idx in range(max_n):
                            in_boards, out_boards = [], []
                            for col in cols:
                                ex = col[idx] if idx < len(col) else (col[-1] if col else {})
                                if ex:
                                    in_boards.append(ex.get("input", []))
                                    out_boards.append(ex.get("output", []))
                            ci, _, ok_i = self._validate_and_combine_grids(in_boards, m, 9)
                            co, _, ok_o = self._validate_and_combine_grids(out_boards, m, 9)
                            if ok_i and ok_o:
                                combined_train.append({"input": ci, "output": co})

                    # Combine test input/output
                    test_inputs = [(tv.get("test") or [{}])[0].get("input", []) for tv in test_variants]
                    test_outputs = [(tv.get("test") or [{}])[0].get("output") for tv in test_variants]
                    test_outputs = [to for to in test_outputs if to is not None]

                    cti, sizes_checked, ok = self._validate_and_combine_grids(test_inputs, m, 9)
                    if not ok:
                        continue
                    cto, _, ok_o = (None, [], False)
                    if test_outputs and len(test_outputs) == len(test_inputs):
                        cto, _, ok_o = self._validate_and_combine_grids(test_outputs, m, 9)
                        if not ok_o:
                            cto = None

                    shuffle(combined_train)
                    test_case = {"input": cti}
                    if cto is not None:
                        test_case["output"] = cto
                    combined_task = {"train": combined_train, "test": [test_case]}

                    # Decoder description
                    dec = {
                        "type": "mixup_combine",
                        "method": m,
                        "num_combined": len(test_variants),
                        "original_sizes": sizes_checked,
                        "separator_color": 9 if m.startswith("color_separator") else None,
                        "variant_transformations": transforms,
                        # Map each split to a specific source task key when available
                        "task_keys": selected_keys,
                    }
                    if global_pre_decoder:
                        dec["global_pre"] = global_pre_decoder
                    augmented.append(combined_task)
                    decs.append(dec)
                    success = True
                    break
                except Exception as e:
                    debug_print(f"Error in mixup_combine augmentation variant {i + 1}: {e}")
            if not success:
                debug_print(
                    "  Failed to generate mixup_combine variant "
                    f"{i + 1} after {attempts} attempts - excluding this item"
                )
        return augmented, decs


# Manager & convenience wrappers
# -----------------------------------------------------------------------------


class AugmentationManager:
    def __init__(self):
        swap_raw = AUGMENTATION_CONFIG.get("input_output_swap", {}).get("swap_probability", 0.5)
        if isinstance(swap_raw, (list, tuple)):
            swap_raw = swap_raw[0] if swap_raw else 0.5
        try:
            swap_prob = float(swap_raw)
        except Exception:
            swap_prob = 0.5
        self.augmentations: dict[str, AugmentationBase] = {
            "geometric": GeometricAugmentation(),
            "color": ColorAugmentation(),
            "geometric_color": CombinedAugmentation(),
            "order": OrderAugmentation(),
            "noise": NoiseAugmentation(),
            "crop": CropAugmentation(),
            "duplicate": DuplicateAugmentation(),
            "mixup": MixupAugmentation(),
            "mixup_combine": MixupCombineAugmentation(),
            "input_output_swap": InputOutputSwapAugmentation(swap_prob),
            "combine": CombineAugmentation(),
        }

    def get_augmentations_for_use_case(self, use_case: str) -> list[AugmentationBase]:
        enabled: list[AugmentationBase] = []
        for name, cfg in AUGMENTATION_CONFIG.items():
            # Some configs may use list values from legacy sweeps—fall back to the first entry.
            def _as_bool(v, default=False):
                try:
                    if isinstance(v, (list, tuple)):
                        return bool(v[0]) if v else bool(default)
                    return bool(v)
                except Exception:
                    return bool(default)

            if not _as_bool(cfg.get("enabled", False), False):
                continue
            if not _as_bool(cfg.get(f"use_for_{use_case}", False), False):
                continue
            aug = self.augmentations.get(name)
            if aug is not None:
                enabled.append(aug)
        return enabled

    @staticmethod
    def _weight_for(name: str, use_case: str) -> float:
        cfg = AUGMENTATION_CONFIG.get(name, {})
        raw = cfg.get(f"weight_{use_case}", cfg.get("weight", 1.0))
        # Accept list-valued weights from legacy configs by using the first element.
        try:
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else 1.0
            return float(raw)
        except Exception:
            return 1.0

    def set_task_pool_for_mixup(self, tasks: list[dict]):
        """Populate task pools for mixup-style augs.

        Accepts either a list of raw task dicts or a list of structured entries
        like {'task_key': <key>, 'task': <task_dict>}.
        - For 'mixup': only task dicts are used.
        - For 'mixup_combine': structured entries are preserved when available
          so decoders can record source task keys.
        """
        # Normalize to both forms
        if not isinstance(tasks, list):
            return
        # Raw task dicts for mixup
        plain_tasks = [t.get("task", t) if isinstance(t, dict) else t for t in tasks]
        aug1 = self.augmentations.get("mixup")
        if isinstance(aug1, MixupAugmentation):
            aug1.set_task_pool(plain_tasks)
        # Structured entries for mixup_combine
        aug2 = self.augmentations.get("mixup_combine")
        if isinstance(aug2, MixupCombineAugmentation):
            aug2.set_task_pool(tasks)

    def generate_augmentations(
        self, task: dict, n: int, use_case: str, task_pool: list[dict] | None = None
    ) -> tuple[list[dict], list[dict], dict]:
        stats = {"total_requested": n, "total_generated": 0, "successful": 0, "failed": 0, "by_type": {}}
        if task_pool:
            self.set_task_pool_for_mixup(task_pool)
        enabled = self.get_augmentations_for_use_case(use_case)
        if not enabled:
            debug_print(f"No augmentations enabled for use case: {use_case}")
            return [], [], stats
        # Improved stochastic allocation: floor + highest fractional remainder
        raw = [self._weight_for(a.name, use_case) for a in enabled]
        s = sum(raw) or 1.0
        probs = [w / s for w in raw]
        base = [int(n * p) for p in probs]
        rem = n - sum(base)
        fracs = [(i, (n * probs[i]) - base[i]) for i in range(len(enabled))]
        fracs.sort(key=lambda x: x[1], reverse=True)
        counts = base[:]
        for i in range(rem):
            counts[fracs[i % len(fracs)][0]] += 1

        all_tasks, all_decs = [], []
        for aug, k in zip(enabled, counts, strict=False):
            if k <= 0:
                continue
            aug_stat = {"requested": k, "generated": 0, "failed": 0}
            try:
                cfg = AUGMENTATION_CONFIG.get(aug.name, {})
                pre_name = cfg.get("aug_before_with")
                if pre_name:
                    pre_aug = self.augmentations.get(pre_name)
                    if not pre_aug:
                        tasks, decs = aug.augment(task, k)
                    else:
                        # Generate k pre-augmentations, then apply current aug to each
                        pre_tasks, pre_decs = pre_aug.augment(task, k)
                        tasks, decs = [], []
                        for pt, pd in zip(pre_tasks, pre_decs, strict=False):
                            t2, d2 = aug.augment(pt, 1)
                            if t2:
                                tasks.append(t2[0])
                                # Compose decoder chain: apply current decoder then pre decoder
                                if d2:
                                    decs.append(
                                        {
                                            "type": "chain",
                                            "sequence": [d2[0], pd],  # decode in order
                                        }
                                    )
                                else:
                                    decs.append(pd)
                else:
                    tasks, decs = aug.augment(task, k)
                aug_stat["generated"] = len(tasks)
                stats["successful"] += len(tasks)
                all_tasks.extend(tasks)
                all_decs.extend(decs)
            except Exception as e:
                debug_print(f"Error in {aug.name} augmentation: {e}")
                aug_stat["failed"] = k
                stats["failed"] += k
            stats["by_type"][aug.name] = aug_stat
        stats["total_generated"] = len(all_tasks)

        paired = list(zip(all_tasks, all_decs, strict=False))
        shuffle(paired)
        if paired:
            t, d = zip(*paired, strict=False)
            return list(t), list(d), stats
        return [], [], stats

    def apply_decoder(self, grid: list[list[int]], desc: dict) -> list[list[int]]:
        t = desc.get("type", "geometric")
        if t == "chain":
            out = grid
            for sub in desc.get("sequence", []):
                out = self.apply_decoder(out, sub)
            return out
        aug = self.augmentations.get(t)
        return aug.apply_decoder(grid, desc) if aug else grid

    @staticmethod
    def _distribute(total: int, weights: list[float]) -> list[int]:
        if not weights:
            return []
        base = [int(total * w) for w in weights]
        r = total - sum(base)
        for i in range(r):
            base[i % len(base)] += 1
        return base


# Backward compatibility wrappers


def get_n_augs_flexible(
    task: dict, n: int, use_case: str = "inference", task_pool: list[dict] | None = None
) -> tuple[list[dict], list[dict], dict]:
    mgr = AugmentationManager()
    return mgr.generate_augmentations(task, n, use_case, task_pool)


def apply_decoder_description_flexible(grid: list[list[int]], decoder_desc: dict) -> list[list[int]]:
    mgr = AugmentationManager()
    return mgr.apply_decoder(grid, decoder_desc)
