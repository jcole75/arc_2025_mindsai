#!/usr/bin/env python3
"""
Confidence-based filtering module for ARC Prize 2025 solution
Filters tasks based on prediction confidence.

Refactor highlights:
- Consolidated repeated logic (task id parsing, vote counting, correctness checks).
- Fixed vote-count/active-task misalignment when some tasks lack predictions.
- Made pixel-diff strictly integer (no float('inf')), handling shape mismatches.
- Removed unused imports; minimized recomputation and dict scans.
- Added small guardrails and clarified prints; preserved public API.
"""

from collections.abc import Callable, Iterable

import numpy as np

from .config import MIN_VOTES_AMBIGUOUS, MIN_VOTES_SINGLE_PRED, Z_SCORE_THRESHOLD
from .grid_utils import is_valid_prediction
from .scoring import get_solutions_path, load_solutions


# ----------------------------- small utilities -----------------------------


def _base_task_id(task_key: str) -> str:
    """Strip trailing _N from task keys to map to ground-truth id."""
    if "_" in task_key:
        tail = task_key.rsplit("_", 1)[-1]
        if tail.isdigit():
            return task_key.rsplit("_", 1)[0]
    return task_key


def _to_tuple(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Hashable representation for counting votes."""
    return tuple(tuple(r) for r in grid)


def _vote_counts(preds_for_task: Iterable[list[list[int]]]) -> dict[tuple[tuple[int, ...], ...], int]:
    """Count only valid predictions."""
    counts: dict[tuple[tuple[int, ...], ...], int] = {}
    for g in preds_for_task:
        if is_valid_prediction(g):
            t = _to_tuple(g)
            counts[t] = counts.get(t, 0) + 1
    return counts


def _top_k_by_votes(counts: dict, k: int) -> list[tuple[tuple[tuple[int, ...], ...], int]]:
    """Return top-k (grid_tuple, votes) sorted by votes desc."""
    if not counts:
        return []
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]


def _any_match(grid: list[list[int]], solutions: list[list[list[int]]]) -> bool:
    return any(grid == s for s in solutions)


# ----------------------------- quality metrics -----------------------------


def _count_pixel_differences(pred_grid: list[list[int]], solution_grid: list[list[int]]) -> int:
    """
    Count differing pixels between two (possibly different-sized) grids.
    Returns an integer (no infinities).
    """
    if not pred_grid or not solution_grid:
        # If either missing, consider all pixels in the other grid as mismatched.
        return (len(pred_grid) * (len(pred_grid[0]) if pred_grid else 0)) + (
            len(solution_grid) * (len(solution_grid[0]) if solution_grid else 0)
        )

    max_rows = max(len(pred_grid), len(solution_grid))
    max_cols = max(len(pred_grid[0]), len(solution_grid[0]))
    diff = 0
    for i in range(max_rows):
        for j in range(max_cols):
            pv = pred_grid[i][j] if i < len(pred_grid) and j < len(pred_grid[i]) else None
            sv = solution_grid[i][j] if i < len(solution_grid) and j < len(solution_grid[i]) else None
            if pv != sv:
                diff += 1
    return diff


def _assess_prediction_quality(pred_grid: list[list[int]], correct_solutions: list[list[list[int]]]) -> tuple[str, int]:
    """
    Quality categories: 'correct','1-off','2-off','3-off','few-off','many-off','invalid'
    Returns (category, min_pixel_diff)
    """
    if not is_valid_prediction(pred_grid):
        return "invalid", 10**9

    min_diff = 10**9
    for sol in correct_solutions:
        if pred_grid == sol:
            return "correct", 0
        d = _count_pixel_differences(pred_grid, sol)
        if d < min_diff:
            min_diff = d

    if min_diff == 1:
        return "1-off", 1
    if min_diff == 2:
        return "2-off", 2
    if min_diff == 3:
        return "3-off", 3
    if min_diff <= 10:
        return "few-off", min_diff
    return "many-off", min_diff


# ----------------------------- correctness checks -----------------------------


def _task_has_any_correct(task_key: str, predictions: dict[str, list], solutions_data: dict) -> bool:
    base_id = _base_task_id(task_key)
    sols = solutions_data.get(base_id)
    if not isinstance(sols, list):
        return False

    return any(
        is_valid_prediction(g) and _any_match(g, sols) for g in predictions.get(task_key, [])
    )


def _task_top1_correct(task_key: str, predictions: dict[str, list], solutions_data: dict) -> bool:
    base_id = _base_task_id(task_key)
    sols = solutions_data.get(base_id)
    if not isinstance(sols, list):
        return False

    for g in predictions.get(task_key, []):
        if is_valid_prediction(g):
            return _any_match(g, sols)
    return False


def _task_top_voted_correct(task_key: str, predictions: dict[str, list], solutions_data: dict) -> bool:
    base_id = _base_task_id(task_key)
    sols = solutions_data.get(base_id)
    if not isinstance(sols, list):
        return False

    counts = _vote_counts(predictions.get(task_key, []))
    if not counts:
        return False
    (best_tuple, _votes) = max(counts.items(), key=lambda x: x[1])
    best_grid = [list(r) for r in best_tuple]
    return _any_match(best_grid, sols)


def _task_top2_correct(task_key: str, predictions: dict[str, list], solutions_data: dict) -> bool:
    base_id = _base_task_id(task_key)
    sols = solutions_data.get(base_id)
    if not isinstance(sols, list):
        return False

    counts = _vote_counts(predictions.get(task_key, []))
    for t, _ in _top_k_by_votes(counts, 2):
        g = [list(r) for r in t]
        if _any_match(g, sols):
            return True
    return False


# ----------------------------- main API -----------------------------


def apply_confidence_filter(
    predictions: dict[str, list],
    active_task_keys: list[str],
    model_specific_settings: dict,
    data_path: str | None = None,
) -> tuple[list[str], dict | None]:
    """
    Filters tasks based on prediction confidence, returning keys that still need processing.
    The `predictions` dict should already be aggregated according to ensemble settings.
    """
    print("\n--- Applying Confidence-Based Task Filtering ---")

    def gs(name, section=None, default=None):
        """Nested setting getter with fallback."""
        if section and section in model_specific_settings and name in model_specific_settings[section]:
            return model_specific_settings[section][name]
        return model_specific_settings.get(name, default)

    z_thr = gs("z_score_filtering_threshold", "inference", Z_SCORE_THRESHOLD)
    min_votes_single = gs("min_votes_for_single_pred_filter", "inference", MIN_VOTES_SINGLE_PRED)
    min_votes_ambig = gs("min_votes_for_ambiguous_filter", "inference", MIN_VOTES_AMBIGUOUS)

    # Pass 1: gather per-task vote stats
    task_stats: dict[str, dict] = {}
    top_vote_counts_all: list[int] = []

    for tk in active_task_keys:
        counts = _vote_counts(predictions.get(tk, []))
        if not counts:
            task_stats[tk] = {"has_preds": False, "top_votes": 0, "vote_list": [], "n_unique": 0}
            continue
        vote_list = sorted(counts.values(), reverse=True)
        top = vote_list[0]
        task_stats[tk] = {"has_preds": True, "top_votes": top, "vote_list": vote_list, "n_unique": len(vote_list)}
        top_vote_counts_all.append(top)

    # Cross-task z-scores
    z_scores: dict[str, float] = {}
    if len(top_vote_counts_all) > 1:
        mean_top = float(np.mean(top_vote_counts_all))
        std_top = float(np.std(top_vote_counts_all))
        if std_top > 1e-9:
            for tk in active_task_keys:
                st = task_stats[tk]
                if st["has_preds"]:
                    z_scores[tk] = (st["top_votes"] - mean_top) / std_top

    # Pass 2: apply filtering rules
    remaining: list[str] = []
    filtered_count = 0

    for tk in active_task_keys:
        st = task_stats[tk]
        if not st["has_preds"]:
            remaining.append(tk)
            continue

        vote_list = st["vote_list"]
        n_unique = st["n_unique"]
        filtered = False

        # Case 1: strong cross-task dominance (z-score)
        z = z_scores.get(tk)
        if z is not None and z > z_thr:
            filtered = True
            print(f"  Task {tk}: filtered by z-score {z:.2f} > {z_thr} (top votes: {st['top_votes']})")

        # Case 2: only one unique valid prediction with enough votes
        if not filtered and n_unique == 1 and vote_list[0] >= min_votes_single:
            filtered = True
            print(f"  Task {tk}: filtered by single prediction threshold {vote_list[0]} >= {min_votes_single}")

        # Case 3: ambiguous top-2 tie, both high, separated from rest
        if not filtered and n_unique >= 2:
            v1, v2 = vote_list[0], vote_list[1]
            if v1 == v2 and v1 >= min_votes_ambig:
                if n_unique == 2:
                    filtered = True
                    print(f"  Task {tk}: filtered by ambiguous pair threshold {v1} >= {min_votes_ambig}")
                else:
                    v3 = vote_list[2]
                    if v2 > v3:
                        filtered = True
                        print(f"  Task {tk}: filtered by ambiguous pair with separation {v1}={v2} > {v3}")

        if filtered:
            filtered_count += 1
        else:
            remaining.append(tk)

    # Optional: load solutions for accuracy stats & colored displays
    filter_stats = None
    solutions = None
    if data_path:
        spath = get_solutions_path(data_path)
        if spath:
            solutions = load_solutions(spath)

    if solutions:
        filter_stats = _calculate_filter_accuracy_stats(
            predictions=predictions,
            all_active_tasks=active_task_keys,
            remaining_tasks=remaining,
            solutions_data=solutions,
            z_scores=z_scores,
            current_threshold=z_thr,
            task_stats=task_stats,
        )

    # Summary prints
    if top_vote_counts_all:
        print(
            "Cross-task vote statistics: "
            f"mean={np.mean(top_vote_counts_all):.2f}, "
            f"std={np.std(top_vote_counts_all):.2f}"
        )
        if solutions:
            print(
                "Top vote counts (color-coded by quality): "
                + _create_colored_vote_display(active_task_keys, predictions, solutions, quality_mode="top1")
            )
            print(
                "Top vote counts (color-coded by pass@2 quality): "
                + _create_colored_vote_display(active_task_keys, predictions, solutions, quality_mode="top2")
            )
        else:
            print(f"Top vote counts: {sorted(top_vote_counts_all, reverse=True)}")

    print(f"Confidence filtering: {filtered_count} tasks met criteria and were filtered.")
    print(f"{len(remaining)} tasks remain active for further processing.")

    if filter_stats:
        _print_filter_accuracy_stats(filter_stats)

    return remaining, filter_stats


# ----------------------------- stats & threshold search -----------------------------


def _find_optimal_threshold(
    all_tasks: list[str], z_scores: dict[str, float], correctness_fn: Callable[[str], bool]
) -> float | None:
    """
    Choose threshold maximizing precision (target 100%) then recall.
    Returns None if no z-scores or no positive case exists.
    """
    if not z_scores:
        return None

    # Precompute correctness of each task under the chosen notion (top-1 or top-2)
    is_correct = {t: correctness_fn(t) for t in all_tasks}
    positives = sum(is_correct.values())
    # If nothing is ever correct, any threshold yields 0 precision/recall
    # We still scan thresholds and return the best by the scoring rule.
    thresholds = sorted(set(z_scores.values()), reverse=True)

    best_thr, best_score = None, -1.0
    for thr in thresholds:
        filtered = [t for t in all_tasks if z_scores.get(t, -1e9) > thr]
        if not filtered:
            continue
        corr = sum(1 for t in filtered if is_correct.get(t))
        prec = corr / len(filtered)
        rec = (corr / positives) if positives > 0 else 0.0

        # Prefer perfect precision; among equals, higher recall
        score = (100.0 + 10.0 * rec) if prec == 1.0 else (50.0 * prec + 5.0 * rec)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr


def _calculate_filter_accuracy_stats(
    predictions: dict[str, list],
    all_active_tasks: list[str],
    remaining_tasks: list[str],
    solutions_data: dict,
    z_scores: dict[str, float],
    current_threshold: float,
    task_stats: dict[str, dict],
) -> dict:
    filtered_tasks = [t for t in all_active_tasks if t not in remaining_tasks]

    # By design: filtered uses pass@all; remaining uses top-voted (pass@1) and pass@2
    filtered_correct = sum(_task_has_any_correct(t, predictions, solutions_data) for t in filtered_tasks)
    remaining_correct_top1 = sum(_task_top_voted_correct(t, predictions, solutions_data) for t in remaining_tasks)
    remaining_correct_top2 = sum(_task_top2_correct(t, predictions, solutions_data) for t in remaining_tasks)

    # Optimal thresholds (top-1 and top-2 notions)
    thr_top1 = _find_optimal_threshold(
        all_active_tasks, z_scores, lambda t: _task_top_voted_correct(t, predictions, solutions_data)
    )
    thr_top2 = _find_optimal_threshold(
        all_active_tasks, z_scores, lambda t: _task_top2_correct(t, predictions, solutions_data)
    )

    def _summarize(thr: float | None, correctness_checker: Callable[[str], bool]):
        if thr is None:
            return 0, 0, 0, 0
        opt_filtered = [t for t in all_active_tasks if z_scores.get(t, -1e9) > thr]
        opt_remaining = [t for t in all_active_tasks if t not in opt_filtered]
        opt_f_corr = sum(correctness_checker(t) for t in opt_filtered)
        opt_r_corr = sum(correctness_checker(t) for t in opt_remaining)
        return opt_f_corr, len(opt_filtered), opt_r_corr, len(opt_remaining)

    (opt_f_corr_1, opt_f_tot_1, opt_r_corr_1, opt_r_tot_1) = _summarize(
        thr_top1, lambda t: _task_top_voted_correct(t, predictions, solutions_data)
    )
    (opt_f_corr_2, opt_f_tot_2, opt_r_corr_2, opt_r_tot_2) = _summarize(
        thr_top2, lambda t: _task_top2_correct(t, predictions, solutions_data)
    )

    # Build vote breakdowns (mapped by each task's own top_votes to avoid ordering bugs)
    all_votes = []
    filt_votes = []
    rem_votes = []
    for tk in all_active_tasks:
        top_v = task_stats[tk]["top_votes"]
        z = z_scores.get(tk, 0.0)
        qual, pix = _task_vote_quality(tk, predictions, solutions_data, mode="top1")
        item = {"task_key": tk, "vote_count": top_v, "z_score": z, "quality": qual, "pixel_diff": pix}
        all_votes.append(item)
        (filt_votes if tk in filtered_tasks else rem_votes).append(item)

    all_votes.sort(key=lambda x: x["vote_count"], reverse=True)
    filt_votes.sort(key=lambda x: x["vote_count"], reverse=True)
    rem_votes.sort(key=lambda x: x["vote_count"], reverse=True)

    return {
        "filtered_correct": filtered_correct,
        "filtered_total": len(filtered_tasks),
        "remaining_correct": remaining_correct_top1,
        "remaining_correct_top2": remaining_correct_top2,
        "remaining_total": len(remaining_tasks),
        "current_threshold": current_threshold,
        "optimal_threshold": thr_top1,
        "optimal_threshold_top2": thr_top2,
        "optimal_filtered_correct": opt_f_corr_1,
        "optimal_filtered_total": opt_f_tot_1,
        "optimal_remaining_correct": opt_r_corr_1,
        "optimal_remaining_total": opt_r_tot_1,
        "optimal_filtered_correct_top2": opt_f_corr_2,
        "optimal_filtered_total_top2": opt_f_tot_2,
        "optimal_remaining_correct_top2": opt_r_corr_2,
        "optimal_remaining_total_top2": opt_r_tot_2,
        "vote_counts": {
            "all": [x["vote_count"] for x in all_votes],
            "filtered": [x["vote_count"] for x in filt_votes],
            "non_filtered": [x["vote_count"] for x in rem_votes],
            "all_detailed": all_votes,
            "filtered_detailed": filt_votes,
            "non_filtered_detailed": rem_votes,
            "all_colored": _create_colored_vote_display(all_active_tasks, predictions, solutions_data, "top1"),
            "filtered_colored": _create_colored_vote_display(
                [x["task_key"] for x in filt_votes], predictions, solutions_data, "top1"
            )
            if filt_votes
            else "None",
            "non_filtered_colored": _create_colored_vote_display(
                [x["task_key"] for x in rem_votes], predictions, solutions_data, "top1"
            )
            if rem_votes
            else "None",
        },
    }


def _print_filter_accuracy_stats(s: dict):
    fc, ft = s["filtered_correct"], s["filtered_total"]
    rc, rt = s["remaining_correct"], s["remaining_total"]
    rc2 = s.get("remaining_correct_top2", rc)

    fa = (fc / ft * 100) if ft else 0.0
    ra = (rc / rt * 100) if rt else 0.0
    ra2 = (rc2 / rt * 100) if rt else 0.0

    print(f"  📊 Filtering accuracy: {fc}/{ft} filtered tasks were correct ({fa:.1f}%)")
    print(f"  📊 Remaining accuracy: {rc}/{rt} remaining tasks are correct ({ra:.1f}%)")
    print(f"  📊 Remaining accuracy (pass@2): {rc2}/{rt} remaining tasks are correct ({ra2:.1f}%)")

    vc = s.get("vote_counts", {})
    if vc:
        print("  🗳️  Vote counts breakdown:")
        print(f"     All tasks: {vc.get('all', [])}")
        if vc.get("filtered"):
            print(f"     Filtered tasks: {vc['filtered']} (color-coded by quality)")
        if vc.get("non_filtered"):
            print(f"     Non-filtered tasks: {vc['non_filtered']} (color-coded by quality)")
        if vc.get("all_colored"):
            print(f"     Color-coded (all): {vc['all_colored']}")
        if vc.get("filtered_colored") not in (None, "None"):
            print(f"     Color-coded (filtered): {vc['filtered_colored']}")
        if vc.get("non_filtered_colored") not in (None, "None"):
            print(f"     Color-coded (non-filtered): {vc['non_filtered_colored']}")

    # Optimal thresholds (top-1)
    ot, curr = s.get("optimal_threshold"), s["current_threshold"]
    if ot is not None:
        ofc, oft = s["optimal_filtered_correct"], s["optimal_filtered_total"]
        if oft > 0:
            acc = ofc / oft * 100
            if acc == 100.0:
                print(f"  🎯 Optimal z-score threshold: {ot:.2f} (current: {curr:.2f})")
                print(
                    "     This finds the threshold for 100% correct filtering, "
                    f"finding {oft} tasks with perfect precision"
                )
            else:
                print(f"  🎯 Optimal z-score threshold: {ot:.2f} (current: {curr:.2f})")
                print(f"     This would filter {oft} tasks with {acc:.1f}% accuracy")
        else:
            print(f"  🎯 Optimal z-score threshold: {ot:.2f} (current: {curr:.2f})")
            print("     This threshold prioritizes 100% correct filtering over quantity")
    else:
        print("  ⚠️ Could not determine optimal z-score threshold (no suitable 100% correct threshold found)")

    # Optimal thresholds (top-2)
    ot2 = s.get("optimal_threshold_top2")
    if ot2 is not None:
        ofc2, oft2 = s["optimal_filtered_correct_top2"], s["optimal_filtered_total_top2"]
        if oft2 > 0:
            acc2 = ofc2 / oft2 * 100
            if acc2 == 100.0:
                print(f"  🎯 Optimal z-score threshold (pass@2): {ot2:.2f} (current: {curr:.2f})")
                print(
                    "     This finds the threshold for 100% correct filtering (pass@2), "
                    f"finding {oft2} tasks with perfect precision"
                )
            else:
                print(f"  🎯 Optimal z-score threshold (pass@2): {ot2:.2f} (current: {curr:.2f})")
                print(f"     This would filter {oft2} tasks with {acc2:.1f}% accuracy (pass@2)")
        else:
            print(f"  🎯 Optimal z-score threshold (pass@2): {ot2:.2f} (current: {curr:.2f})")
            print("     This threshold prioritizes 100% correct filtering over quantity (pass@2)")
    else:
        print("  ⚠️ Could not determine optimal z-score threshold for pass@2 (no suitable 100% correct threshold found)")


# ----------------------------- displays -----------------------------

_COLORS = {
    "correct": "\033[92m",  # bright green
    "1-off": "\033[96m",  # cyan
    "2-off": "\033[94m",  # blue
    "3-off": "\033[93m",  # yellow
    "few-off": "\033[95m",  # magenta
    "many-off": "\033[91m",  # red
    "invalid": "\033[90m",  # dark gray
    "unknown": "\033[37m",  # light gray
    "reset": "\033[0m",
}

_QUALITY_PRIORITY = {
    "correct": 10,
    "1-off": 9,
    "2-off": 8,
    "3-off": 7,
    "few-off": 6,
    "many-off": 5,
    "invalid": 0,
    "unknown": 0,
}


def _task_vote_quality(
    task_key: str, predictions: dict[str, list], solutions_data: dict, mode: str = "top1"
) -> tuple[str, int]:
    """
    Returns (quality_label, pixel_diff) using either top-1 or pass@2 evaluation of voted preds.
    """
    base_id = _base_task_id(task_key)
    sols = solutions_data.get(base_id)
    if not isinstance(sols, list):
        return "unknown", 0

    counts = _vote_counts(predictions.get(task_key, []))
    if not counts:
        return "invalid", 0

    top_preds = _top_k_by_votes(counts, 2 if mode == "top2" else 1)
    best_q, best_d = "many-off", 10**9
    for t, _v in top_preds:
        g = [list(r) for r in t]
        q, d = _assess_prediction_quality(g, sols)
        if (_QUALITY_PRIORITY.get(q, 0) > _QUALITY_PRIORITY.get(best_q, 0)) or (q == best_q and d < best_d):
            best_q, best_d = q, d
    return best_q, best_d


def _create_colored_vote_display(
    task_keys: list[str], predictions: dict[str, list], solutions_data: dict, quality_mode: str = "top1"
) -> str:
    """
    Color-coded string of each task's top vote count, colored by quality.
    Sorts by vote count desc for a compact overview.
    """
    items = []
    used = set()

    # Compute per-task (top_votes, quality)
    per_task = []
    for tk in task_keys:
        counts = _vote_counts(predictions.get(tk, []))
        top_votes = max(counts.values()) if counts else 0
        q, _ = _task_vote_quality(tk, predictions, solutions_data, mode=("top2" if quality_mode == "top2" else "top1"))
        per_task.append((tk, top_votes, q))

    per_task.sort(key=lambda x: x[1], reverse=True)

    for _tk, v, q in per_task:
        used.add(q)
        color = _COLORS.get(q, _COLORS["unknown"])
        items.append(f"{color}{v}{_COLORS['reset']}")

    legend = []
    for q in ["correct", "1-off", "2-off", "3-off", "few-off", "many-off", "invalid", "unknown"]:
        if q in used:
            legend.append(f"{_COLORS.get(q, _COLORS['unknown'])}{q}{_COLORS['reset']}")

    s = f"[{', '.join(items)}]"
    if legend:
        s += f" (Legend: {', '.join(legend)})"
    return s
