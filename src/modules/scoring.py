"""
Scoring module for ARC Prize 2025 solution.
Evaluates submission against ground truth solutions when available.

# SCORING SYSTEM REFERENCE GUIDE

## Key Files for Scoring System Development:
1. `src/modules/scoring.py` - Main scoring logic, HTML report generation (THIS FILE)
2. `src/modules/grid_utils.py` - Grid parsing and validation functions
3. `src/modules/inference_worker.py` - Prediction collection and processing
4. `src/modules/submission_handler.py` - Final submission creation
5. `src/main.py` - Integration point where scoring is called

## Key Functions for Scoring Features:
- `score_submission()` - Core scoring logic, accepts augmentation_metadata parameter
- `generate_html_report()` - HTML visualization with augmentation stats
- `print_scoring_summary()` - Console output summary
- `score_if_solutions_available()` - Main entry point called from main.py

## Data Flow for Augmentation Tracking:
1. Inference worker processes augmentations and tracks metadata in inference_worker.py:300+
2. Main.py collects overall stats and parsing errors in main.py:350+
3. Augmentation metadata passed to scoring via score_if_solutions_available()
4. HTML report displays augmentation grid and non-parseable statistics

## HTML Report Sections:
- Overall Stats (#overall-stats)
- Augmentation Stats (#augmentation-stats)
- Position Stats (#position-stats)
- Task Details (#task-details)
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    _MATPLOTLIB_AVAILABLE = False

from . import config


MAX_ATTEMPTS = 5  # support up to top-5


# ------------------------------- Small utilities -------------------------------


def _base_task_id(task_key: str) -> str:
    """Return base task id (strip trailing '_N' if present)."""
    parts = task_key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else task_key


def _avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def calculate_row_pixel_accuracy(pred_row: list[int], true_row: list[int]) -> float:
    """
    Pixel accuracy for one row; compares overlapping positions and penalizes
    missing predictions by dividing by len(true_row).
    """
    if not true_row:
        return 0.0
    m = min(len(pred_row), len(true_row))
    matches = sum(1 for i in range(m) if pred_row[i] == true_row[i])
    return matches / len(true_row)


def rowwise_pixel_accuracy(pred: list[list[int]], true: list[list[int]]) -> float:
    """
    Pixel accuracy on a row-by-row basis; compares only overlapping regions
    and penalizes missing rows via the per-row denominator.
    """
    if not true:
        return 0.0
    accs = []
    for i, true_row in enumerate(true):
        pred_row = pred[i] if i < len(pred) else []
        accs.append(calculate_row_pixel_accuracy(pred_row, true_row))
    return _avg(accs)


def _empty_task_result(task_id: str, attempted: bool, reason: str) -> dict:
    """Standard zeroed-out result blob for a task."""
    return {
        "task_id": task_id,
        "top1_score": 0.0,
        "top2_score": 0.0,
        "top3_score": 0.0,
        "top4_score": 0.0,
        "top5_score": 0.0,
        "top_max_score": 0.0,
        "max_attempts_used": 0.0,
        "pixel_accuracy_top1": 0.0,
        "pixel_accuracy_top2": 0.0,
        "pixel_accuracy_top3": 0.0,
        "pixel_accuracy_top4": 0.0,
        "pixel_accuracy_top5": 0.0,
        "task_score": 0.0,
        "attempted": attempted,
        "reason": reason,
    }


def get_solutions_path(data_path: str) -> str | None:
    """
    Derive solutions file path from the data path by swapping 'challenges'→'solutions'.
    Return None if not found.
    """
    p = Path(data_path)
    candidate = p.parent / p.name.replace("challenges", "solutions")
    return str(candidate) if candidate.exists() else None


def load_solutions(solutions_path: str) -> dict | None:
    try:
        with open(solutions_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load solutions from {solutions_path}: {e}")
        return None


def calculate_average_correct_position(
    vote_metadata: dict, solutions: dict, attempted_task_keys: list[str]
) -> float | None:
    """
    Average (1-indexed) position of correct answers in vote rankings.
    Uses string equality only (no eval for safety/perf).
    """
    if not vote_metadata or not solutions:
        return None

    attempted_bases = {_base_task_id(k) for k in attempted_task_keys}
    positions: list[int] = []

    for task_id in attempted_bases:
        solution_data = solutions.get(task_id)
        if not solution_data or "test" not in solution_data:
            continue

        for idx, correct_output in enumerate(solution_data["test"]):
            task_key_for_item = f"{task_id}_{idx}"
            votes = vote_metadata.get(task_key_for_item, {}).get("vote_counts", {})
            if not votes:
                continue

            correct_str = str(correct_output)
            # sort predictions by vote count desc
            for pos, (pred_str, _) in enumerate(sorted(votes.items(), key=lambda x: x[1], reverse=True), 1):
                if pred_str == correct_str:
                    positions.append(pos)
                    break

    return _avg(positions) if positions else None


# ------------------------------- Core scoring -------------------------------


def score_submission(
    submission_path: str,
    solutions_path: str,
    attempted_task_keys: list[str],
    augmentation_metadata: dict | None = None,
    vote_metadata: dict | None = None,
) -> dict:
    """
    Score a submission against ground-truth solutions. Returns:
      {
        "task_results": [...],
        "item_results": [...],
        "overall_stats": {...},
        "position_stats": {...},
        (optional) "augmentation_stats": {...},
        (optional) "vote_metadata": {...}
      }
    """
    try:
        with open(submission_path) as f:
            submission = json.load(f)
        with open(solutions_path) as f:
            solutions = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load files: {e}"}

    attempted_bases = {_base_task_id(k) for k in attempted_task_keys}
    task_results, item_results = [], []
    item_position_stats: dict[int, dict[str, list[float]]] = {}
    attempted_task_scores, all_task_scores = [], []

    for task_id, correct_outputs in solutions.items():
        was_attempted = task_id in attempted_bases

        preds_for_task = submission.get(task_id)
        if preds_for_task is None:
            tr = _empty_task_result(task_id, was_attempted, "Missing from submission")
            task_results.append(tr)
            all_task_scores.append(0.0)
            if was_attempted:
                attempted_task_scores.append(0.0)
            continue

        if len(preds_for_task) != len(correct_outputs):
            tr = _empty_task_result(task_id, was_attempted, "Mismatch in number of outputs")
            task_results.append(tr)
            all_task_scores.append(0.0)
            if was_attempted:
                attempted_task_scores.append(0.0)
            continue

        weight = 1.0 / max(1, len(correct_outputs))
        top_scores = [0.0] * MAX_ATTEMPTS
        top_pixel_acc: list[list[float]] = [[] for _ in range(MAX_ATTEMPTS)]

        for idx, (pred, correct) in enumerate(zip(preds_for_task, correct_outputs, strict=False)):
            attempts = (
                [
                    pred.get("attempt_1", []),
                    pred.get("attempt_2", []),
                    pred.get("attempt_3", []),
                    pred.get("attempt_4", []),
                    pred.get("attempt_5", []),
                ]
                if isinstance(pred, dict)
                else [[] for _ in range(MAX_ATTEMPTS)]
            )

            is_correct = [attempts[k] == correct for k in range(MAX_ATTEMPTS)]
            pixel_accs = [rowwise_pixel_accuracy(attempts[k], correct) for k in range(MAX_ATTEMPTS)]

            for k in range(MAX_ATTEMPTS):
                if any(is_correct[: k + 1]):
                    top_scores[k] += weight
                top_pixel_acc[k].append(max(pixel_accs[: k + 1]))

            # per-position stats (backward-compat fields)
            is_top1, is_top2 = any(is_correct[:1]), any(is_correct[:2])
            pixel_top1 = max(pixel_accs[:1]) if pixel_accs else 0.0
            pixel_top2 = max(pixel_accs[:2]) if len(pixel_accs) >= 2 else pixel_top1

            pos = item_position_stats.setdefault(
                idx,
                {
                    "top1_scores": [],
                    "top2_scores": [],
                    "pixel_top1": [],
                    "pixel_top2": [],
                    "weights": [],
                    "top1_contributions": [],
                    "top2_contributions": [],
                },
            )
            pos["top1_scores"].append(float(is_top1))
            pos["top2_scores"].append(float(is_top2))
            pos["pixel_top1"].append(pixel_top1)
            pos["pixel_top2"].append(pixel_top2)
            pos["weights"].append(weight)
            pos["top1_contributions"].append(weight if is_top1 else 0.0)
            pos["top2_contributions"].append(weight if is_top2 else 0.0)

            item_results.append(
                {
                    "task_id": task_id,
                    "item_index": idx,
                    "top1_score": float(is_top1),
                    "top2_score": float(is_top2),
                    "pixel_accuracy_top1": pixel_top1,
                    "pixel_accuracy_top2": pixel_top2,
                    "attempted": was_attempted,
                }
            )

        # finalize task-level aggregates
        top1_score, top2_score = top_scores[0], top_scores[1]
        avg_pixel_acc = [_avg(a) for a in top_pixel_acc]

        # True Top-Max via vote metadata (unique prediction set) if available
        if vote_metadata:
            true_top_max = 0.0
            total_unique_attempts = 0
            for idx, correct in enumerate(correct_outputs):
                task_key = f"{task_id}_{idx}"
                v = vote_metadata.get(task_key, {})
                uniques = v.get("all_unique_grids", []) or []
                total_unique_attempts = max(total_unique_attempts, len(uniques))
                if any(u == correct for u in uniques):
                    true_top_max += weight
            top_max_score = true_top_max
            max_attempts_used = total_unique_attempts if total_unique_attempts > 0 else MAX_ATTEMPTS
        else:
            top_max_score = top_scores[-1]
            max_attempts_used = MAX_ATTEMPTS

        task_results.append(
            {
                "task_id": task_id,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "top3_score": top_scores[2] if len(top_scores) > 2 else top2_score,
                "top4_score": top_scores[3] if len(top_scores) > 3 else top2_score,
                "top5_score": top_scores[4] if len(top_scores) > 4 else top2_score,
                "top_max_score": top_max_score,
                "max_attempts_used": max_attempts_used,
                "pixel_accuracy_top1": avg_pixel_acc[0] if len(avg_pixel_acc) > 0 else 0.0,
                "pixel_accuracy_top2": avg_pixel_acc[1] if len(avg_pixel_acc) > 1 else 0.0,
                "pixel_accuracy_top3": avg_pixel_acc[2]
                if len(avg_pixel_acc) > 2
                else (avg_pixel_acc[1] if len(avg_pixel_acc) > 1 else 0.0),
                "pixel_accuracy_top4": avg_pixel_acc[3]
                if len(avg_pixel_acc) > 3
                else (avg_pixel_acc[1] if len(avg_pixel_acc) > 1 else 0.0),
                "pixel_accuracy_top5": avg_pixel_acc[4]
                if len(avg_pixel_acc) > 4
                else (avg_pixel_acc[1] if len(avg_pixel_acc) > 1 else 0.0),
                "task_score": top2_score,
                "attempted": was_attempted,
            }
        )
        all_task_scores.append(top2_score)
        if was_attempted:
            attempted_task_scores.append(top2_score)

    attempted_tasks = [r for r in task_results if r["attempted"]]
    # helper to reduce boilerplate when averaging across result rows

    def average_for(key: str, rows: list[dict[str, Any]]) -> float:
        return _avg([row[key] for row in rows]) if rows else 0.0

    overall_stats = {
        "total_tasks": len(solutions),
        "attempted_tasks": len(attempted_tasks),
        "attempted_percentage": (len(attempted_tasks) / len(solutions) * 100) if solutions else 0.0,
        # attempted only
        "attempted_top1_score": average_for("top1_score", attempted_tasks),
        "attempted_top2_score": average_for("top2_score", attempted_tasks),
        "attempted_top3_score": average_for("top3_score", attempted_tasks),
        "attempted_top4_score": average_for("top4_score", attempted_tasks),
        "attempted_top5_score": average_for("top5_score", attempted_tasks),
        "attempted_top_max_score": average_for("top_max_score", attempted_tasks),
        "attempted_max_attempts_avg": average_for("max_attempts_used", attempted_tasks),
        "attempted_pixel_accuracy_top1": average_for("pixel_accuracy_top1", attempted_tasks),
        "attempted_pixel_accuracy_top2": average_for("pixel_accuracy_top2", attempted_tasks),
        "attempted_pixel_accuracy_top3": average_for("pixel_accuracy_top3", attempted_tasks),
        "attempted_pixel_accuracy_top4": average_for("pixel_accuracy_top4", attempted_tasks),
        "attempted_pixel_accuracy_top5": average_for("pixel_accuracy_top5", attempted_tasks),
        # overall (unattempted count as 0)
        "overall_top1_score": average_for("top1_score", task_results),
        "overall_top2_score": average_for("top2_score", task_results),
        "overall_top3_score": average_for("top3_score", task_results),
        "overall_top4_score": average_for("top4_score", task_results),
        "overall_top5_score": average_for("top5_score", task_results),
        "overall_top_max_score": average_for("top_max_score", task_results),
        "overall_max_attempts_avg": average_for("max_attempts_used", task_results),
        "overall_pixel_accuracy_top1": average_for("pixel_accuracy_top1", task_results),
        "overall_pixel_accuracy_top2": average_for("pixel_accuracy_top2", task_results),
        "overall_pixel_accuracy_top3": average_for("pixel_accuracy_top3", task_results),
        "overall_pixel_accuracy_top4": average_for("pixel_accuracy_top4", task_results),
        "overall_pixel_accuracy_top5": average_for("pixel_accuracy_top5", task_results),
    }

    # per-position aggregates
    position_stats = {}
    for pos, stats in item_position_stats.items():
        w_sum = sum(stats["weights"]) if stats["weights"] else 0.0
        position_stats[f"position_{pos}"] = {
            "count": len(stats["top1_scores"]),
            "top1_accuracy": _avg(stats["top1_scores"]),
            "top2_accuracy": _avg(stats["top2_scores"]),
            "pixel_accuracy_top1": _avg(stats["pixel_top1"]),
            "pixel_accuracy_top2": _avg(stats["pixel_top2"]),
            "top1_total_contribution": sum(stats["top1_contributions"]),
            "top2_total_contribution": sum(stats["top2_contributions"]),
            "total_possible_contribution": w_sum,
            "average_weight_per_item": (w_sum / len(stats["weights"])) if stats["weights"] else 0.0,
        }

    # optional: average correct answer position in vote rankings
    if vote_metadata:
        avg_pos = calculate_average_correct_position(vote_metadata, solutions, attempted_task_keys)
        if avg_pos is not None:
            overall_stats["average_correct_answer_position"] = avg_pos
            print(f"📍 Average correct answer position calculated: {avg_pos:.2f}")
        else:
            print("⚠️ Average correct answer position calculation returned None")

    result = {
        "task_results": task_results,
        "item_results": item_results,
        "overall_stats": overall_stats,
        "position_stats": position_stats,
    }
    if augmentation_metadata:
        result["augmentation_stats"] = augmentation_metadata
    if vote_metadata:
        result["vote_metadata"] = vote_metadata
    return result


# ------------------------------- Console summary -------------------------------


def print_scoring_summary(scoring_results: dict):
    if "error" in scoring_results:
        print(f"❌ Scoring Error: {scoring_results['error']}")
        return

    stats = scoring_results["overall_stats"]
    position_stats = scoring_results["position_stats"]

    print("\n" + "=" * 60)
    print("🎯 SCORING RESULTS")
    print("=" * 60)

    print("\n📊 Task Coverage:")
    print(f"  • Total tasks in solutions: {stats['total_tasks']}")
    print(f"  • Tasks attempted: {stats['attempted_tasks']}")
    print(f"  • Coverage: {stats['attempted_percentage']:.1f}%")

    print("\n🎯 Accuracy (Attempted Tasks Only):")
    a1 = int(stats["attempted_top1_score"] * stats["attempted_tasks"])
    a2 = int(stats["attempted_top2_score"] * stats["attempted_tasks"])
    amax = int(stats["attempted_top_max_score"] * stats["attempted_tasks"])
    print(
        f"  • Top-1 Accuracy: {stats['attempted_top1_score']:.3f} ({stats['attempted_top1_score'] * 100:.1f}%) - {a1}/{stats['attempted_tasks']} tasks"
    )
    print(
        f"  • Top-2 Accuracy: {stats['attempted_top2_score']:.3f} ({stats['attempted_top2_score'] * 100:.1f}%) - {a2}/{stats['attempted_tasks']} tasks"
    )
    print(
        f"  • Top-Max Accuracy: {stats['attempted_top_max_score']:.3f} ({stats['attempted_top_max_score'] * 100:.1f}%) - {amax}/{stats['attempted_tasks']} tasks (avg {stats['attempted_max_attempts_avg']:.1f} attempts)"
    )
    print(
        f"  • Pixel Accuracy (Top-1): {stats['attempted_pixel_accuracy_top1']:.3f} ({stats['attempted_pixel_accuracy_top1'] * 100:.1f}%)"
    )
    print(
        f"  • Pixel Accuracy (Top-2): {stats['attempted_pixel_accuracy_top2']:.3f} ({stats['attempted_pixel_accuracy_top2'] * 100:.1f}%)"
    )
    if "average_correct_answer_position" in stats:
        print(f"  • Average correct answer position: {stats['average_correct_answer_position']:.1f} (1=highest votes)")

    print("\n🌍 Overall Accuracy (All Tasks):")
    o1 = int(stats["overall_top1_score"] * stats["total_tasks"])
    o2 = int(stats["overall_top2_score"] * stats["total_tasks"])
    omax = int(stats["overall_top_max_score"] * stats["total_tasks"])
    print(
        f"  • Top-1 Accuracy: {stats['overall_top1_score']:.3f} ({stats['overall_top1_score'] * 100:.1f}%) - {o1}/{stats['total_tasks']} tasks"
    )
    print(
        f"  • Top-2 Accuracy: {stats['overall_top2_score']:.3f} ({stats['overall_top2_score'] * 100:.1f}%) - {o2}/{stats['total_tasks']} tasks"
    )
    print(
        f"  • Top-Max Accuracy: {stats['overall_top_max_score']:.3f} ({stats['overall_top_max_score'] * 100:.1f}%) - {omax}/{stats['total_tasks']} tasks (avg {stats['overall_max_attempts_avg']:.1f} attempts)"
    )
    print(
        f"  • Pixel Accuracy (Top-1): {stats['overall_pixel_accuracy_top1']:.3f} ({stats['overall_pixel_accuracy_top1'] * 100:.1f}%)"
    )
    print(
        f"  • Pixel Accuracy (Top-2): {stats['overall_pixel_accuracy_top2']:.3f} ({stats['overall_pixel_accuracy_top2'] * 100:.1f}%)"
    )

    if position_stats:
        print("\n📈 Accuracy by Test Item Position:")
        t1c = t2c = tpc = 0.0
        for pos_key in sorted(position_stats.keys(), key=lambda x: int(x.split("_")[1])):
            d = position_stats[pos_key]
            pos = pos_key.split("_")[1]
            r1 = int(d["top1_accuracy"] * d["count"])
            r2 = int(d["top2_accuracy"] * d["count"])
            t1c += d["top1_total_contribution"]
            t2c += d["top2_total_contribution"]
            tpc += d["total_possible_contribution"]
            print(f"  • Position {pos} ({d['count']} items, avg weight: {d['average_weight_per_item']:.3f}):")
            print(f"    - Top-1: {d['top1_accuracy']:.3f} ({d['top1_accuracy'] * 100:.1f}%) - {r1}/{d['count']} items")
            print(f"    - Top-2: {d['top2_accuracy']:.3f} ({d['top2_accuracy'] * 100:.1f}%) - {r2}/{d['count']} items")
            print(
                f"    - Contributes {d['top1_total_contribution']:.3f}/{d['total_possible_contribution']:.3f} points (Top-1) | "
                f"{d['top2_total_contribution']:.3f}/{d['total_possible_contribution']:.3f} points (Top-2)"
            )
        if tpc > 0:
            print("\n📊 Total Contribution Summary:")
            print(f"  • All positions combined: {t1c:.3f}/{tpc:.3f} (Top-1) | {t2c:.3f}/{tpc:.3f} (Top-2)")
            print(f"  • Overall efficiency: {t1c / tpc * 100:.1f}% (Top-1) | {t2c / tpc * 100:.1f}% (Top-2)")


# ------------------------------- Visuals -------------------------------


def plot_grid_with_errors(ax, grid, correct_grid, title: str = "", gridlines: bool = True):
    """
    Plot a predicted grid and overlay red X on incorrect pixels when shapes match.
    """
    import numpy as np

    from .visuals import cmap_values, draw_grid, norm

    g = np.array(grid, dtype=int)
    gt = np.array(correct_grid, dtype=int)

    ax.imshow(g, cmap=cmap_values, norm=norm, origin="upper")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if gridlines:
        draw_grid(ax, g, 0.5)
    ax.set_aspect("equal", adjustable="box")

    if g.shape == gt.shape:
        rows, cols = g.shape
        for i in range(rows):
            for j in range(cols):
                if g[i, j] != gt[i, j]:
                    ax.plot([j - 0.3, j + 0.3], [i - 0.3, i + 0.3], "r-", linewidth=1.5, alpha=0.8)
                    ax.plot([j - 0.3, j + 0.3], [i + 0.3, i - 0.3], "r-", linewidth=1.5, alpha=0.8)


def plot_task_with_attempts(
    task_data: dict,
    task_id: str,
    submission_data: dict,
    solutions_data: dict,
    pixel_acc: float,
    rank: int,
    save_location: str | None = None,
    vote_metadata: dict | None = None,
):
    """
    2-row layout:
      Row 1: Train inputs + Test inputs + Attempt 1 (with error overlays)
      Row 2: Train outputs + Correct test outputs + Attempt 2 (with error overlays)
    """
    from .visuals import plot_grid

    if not isinstance(task_data, dict):
        print(f"   ⚠️ Invalid task_data for {task_id}: expected dict, got {type(task_data).__name__}")
        return

    train_examples = task_data.get("train", []) or []
    test_examples = task_data.get("test", []) or []
    n_train, n_test = len(train_examples), len(test_examples)

    if n_train == 0 and n_test == 0:
        print(f"   ⚠️ No examples found for task {task_id}")
        return

    train_cols = n_train
    test_cols = n_test * 2 if n_test > 0 else 0
    total_cols = train_cols + test_cols
    if total_cols == 0:
        print(f"   ⚠️ No examples to display for task {task_id}")
        return

    fig, axs = plt.subplots(2, total_cols, figsize=(max(1, total_cols) * 2.5, 6.5))
    if total_cols == 1:
        axs = axs.reshape(2, 1)
    plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.05, right=0.95, top=0.85, bottom=0.1)

    def _safe_blank(ax, text: str):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    col = 0
    # train examples
    for i, ex in enumerate(train_examples):
        plot_grid(axs[0, col], ex["input"], title=f"Train {i + 1} Input", gridlines=True)
        plot_grid(axs[1, col], ex["output"], title=f"Train {i + 1} Output", gridlines=True)
        col += 1

    # test examples
    submission_key = task_id if task_id in submission_data else _base_task_id(task_id)
    solutions_key = task_id if task_id in solutions_data else _base_task_id(task_id)

    if n_test > 0 and (submission_key in submission_data) and (solutions_key in solutions_data):
        predictions = submission_data[submission_key]
        correct_outputs = solutions_data[solutions_key]

        for i, (ex, pred, correct) in enumerate(zip(test_examples, predictions, correct_outputs, strict=False)):
            plot_grid(axs[0, col], ex["input"], title=f"Test {i + 1} Input", gridlines=True)
            plot_grid(axs[1, col], correct, title=f"Test {i + 1} Output", gridlines=True)
            col += 1

            if isinstance(pred, dict):
                a1, a2 = pred.get("attempt_1", []), pred.get("attempt_2", [])
            else:
                a1 = a2 = []

            if a1:
                plot_grid_with_errors(axs[0, col], a1, correct, title="Attempt 1", gridlines=True)
            else:
                _safe_blank(axs[0, col], "No Attempt 1")

            if a2:
                plot_grid_with_errors(axs[1, col], a2, correct, title="Attempt 2", gridlines=True)
            else:
                _safe_blank(axs[1, col], "No Attempt 2")
            col += 1
    elif n_test > 0:
        for i, ex in enumerate(test_examples):
            plot_grid(axs[0, col], ex["input"], title=f"Test {i + 1} Input", gridlines=True)
            if "output" in ex:
                plot_grid(axs[1, col], ex["output"], title=f"Test {i + 1} Output", gridlines=True)
            else:
                _safe_blank(axs[1, col], "Unknown Output")
            col += 1
            axs[0, col].axis("off")
            axs[1, col].axis("off")
            col += 1

    fig.suptitle(f"Rank {rank}: {task_id} (Pixel Acc: {pixel_acc:.3f})", fontsize=14, y=0.95)
    try:
        if save_location:
            fig.savefig(save_location, dpi=150, bbox_inches="tight")
        else:
            plt.show()
    except Exception as e:
        print(f"   ⚠️ Failed to render/save plot for {task_id}: {e}")
    finally:
        plt.close(fig)


def visualize_sorted_tasks_by_pixel_accuracy(
    scoring_results: dict, data_path: str, submission_path: str, top_n: int = 10, run_log_data: dict | None = None
):
    """
    Save (or display) top-N task visuals sorted by attempted status first, then pixel acc.
    """
    if "error" in scoring_results:
        print(f"❌ Cannot visualize due to scoring error: {scoring_results['error']}")
        return

    # output dir (Kaggle-friendly)
    plot_output_dir = (
        "/kaggle/working/scoring_visualizations" if config.IS_KAGGLE else "scoring_visualizations"
    )
    try:
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"\n🖼️ Saving task visualizations to: {os.path.abspath(plot_output_dir)}")
        can_save = True
    except OSError as e:
        print(f"\n⚠️ Could not create directory '{plot_output_dir}': {e}. Plots will not be saved.")
        can_save = False

    # load data blobs
    try:
        with open(data_path) as f:
            challenges_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load challenges data: {e}")
        return

    try:
        with open(submission_path) as f:
            submission_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load submission data: {e}")
        submission_data = {}

    solutions_data = {}
    sol_path = get_solutions_path(data_path)
    if sol_path:
        try:
            with open(sol_path) as f:
                solutions_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load solutions data: {e}")

    # sort results and iterate
    tasks = scoring_results["task_results"]
    sorted_tasks = sorted(tasks, key=lambda x: (not x["attempted"], -x["pixel_accuracy_top2"]))

    print(f"\n🎨 TOP {min(top_n, len(sorted_tasks))} TASKS (ATTEMPTED FIRST, THEN BY PIXEL ACCURACY)")
    print("=" * 80)

    for i, tr in enumerate(sorted_tasks[:top_n]):
        task_id = tr["task_id"]
        pix = tr["pixel_accuracy_top2"]
        top2 = tr["top2_score"]
        attempted = tr["attempted"]
        print(f"\n📋 Rank {i + 1}: {task_id}")
        print(f"   Pixel Accuracy: {pix:.3f} | Top-2 Score: {top2:.3f} | Attempted: {'✓' if attempted else '✗'}")

        task_data = challenges_data.get(task_id)
        if task_data is None and "_" in task_id and task_id.rsplit("_", 1)[-1].isdigit():
            task_data = challenges_data.get(_base_task_id(task_id))

        if isinstance(task_data, dict) and (("train" in task_data) or ("test" in task_data)):
            save_as = os.path.join(plot_output_dir, f"rank_{i + 1}_task_{task_id}.png") if can_save else None
            try:
                plot_task_with_attempts(
                    task_data, task_id, submission_data, solutions_data, pix, i + 1, save_location=save_as
                )
            except Exception as e:
                print(f"   ⚠️ Failed to visualize or save task {task_id}: {e}")
                if "no display name" in str(e).lower() or "could not connect to display" in str(e).lower():
                    print("      Hint: This might be a headless environment. Plots are saved if possible.")
        else:
            print(f"   ⚠️ Task {task_id} not found or invalid in challenges data")

    if can_save:
        html_path = generate_html_report(
            scoring_results, data_path, submission_path, plot_output_dir, top_n, run_log_data
        )
        if html_path:
            print(f"📄 Open the HTML report in your browser: file://{os.path.abspath(html_path)}")


# ------------------------------- Granular scoring -------------------------------


def perform_granular_scoring(solutions_path: str, attempted_task_keys: list[str], granular_scoring_data: dict):
    """Optional deep-dive scoring for ensembles."""
    import tempfile

    from .submission_handler import make_submission

    print(f"\n{'=' * 70}\n🎯 GRANULAR ENSEMBLE SCORING ANALYSIS\n{'=' * 70}")

    solutions = load_solutions(solutions_path)
    if not solutions:
        print("❌ No solutions available for granular scoring")
        return

    per_run_predictions = granular_scoring_data["per_run_predictions"]
    per_model_predictions = granular_scoring_data["per_model_predictions"]
    model_paths = granular_scoring_data["model_paths_for_tracking"]
    ensemble_settings = granular_scoring_data["ensemble_settings_per_model"]

    any_run_tracking = any(s["enable_run_tracking"] for s in ensemble_settings)
    any_model_only_ensemble = any(not s["enable_model_ensemble"] for s in ensemble_settings)

    if any_run_tracking:
        print(f"\n📊 INDIVIDUAL RUN SCORES\n{'=' * 50}")
        for m_idx, model_path in enumerate(model_paths):
            if not ensemble_settings[m_idx]["enable_run_tracking"]:
                continue
            model_name = os.path.basename(model_path)
            print(f"\n🔹 Model: {model_name}")
            if m_idx not in per_run_predictions:
                continue
            for run_idx, run_preds in per_run_predictions[m_idx].items():
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    temp_submission_path = tmp.name
                try:
                    make_submission(run_preds, attempted_task_keys, temp_submission_path)
                    run_scores = score_submission(temp_submission_path, solutions_path, attempted_task_keys)
                    # Back-compat placeholders; detailed fields live inside run_scores['overall_stats']
                    acc = run_scores.get("accuracy_percentage", 0)
                    px = run_scores.get("pixel_accuracy_percentage", 0)
                    solved = run_scores.get("solved_count", 0)
                    print(
                        f"    Run {run_idx + 1}: {solved:2d}/{len(attempted_task_keys)} tasks ({acc:5.1f}%), Pixel: {px:5.1f}%"
                    )
                finally:
                    os.unlink(temp_submission_path)

    if any_model_only_ensemble:
        print(f"\n📊 PER-MODEL SCORES\n{'=' * 50}")
        for m_idx, model_path in enumerate(model_paths):
            if ensemble_settings[m_idx]["enable_model_ensemble"]:
                continue
            model_name = os.path.basename(model_path)
            model_preds = per_model_predictions[m_idx]
            import tempfile

            from .submission_handler import make_submission

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                temp_submission_path = tmp.name
            try:
                make_submission(model_preds, attempted_task_keys, temp_submission_path)
                model_scores = score_submission(temp_submission_path, solutions_path, attempted_task_keys)
                acc = model_scores.get("accuracy_percentage", 0)
                px = model_scores.get("pixel_accuracy_percentage", 0)
                solved = model_scores.get("solved_count", 0)
                print(f"🔹 {model_name}: {solved:2d}/{len(attempted_task_keys)} tasks ({acc:5.1f}%), Pixel: {px:5.1f}%")
            finally:
                os.unlink(temp_submission_path)


# ------------------------------- Orchestration -------------------------------


def score_if_solutions_available(
    submission_path: str,
    data_path: str,
    attempted_task_keys: list[str],
    augmentation_metadata: dict | None = None,
    run_log_data: dict | None = None,
    predictions: dict[str, list] | None = None,
    granular_scoring_data: dict | None = None,
    *,
    skip_visuals: bool = False,
):
    """
    Main entry: run scoring if a matching solutions file exists for `data_path`.
    """
    solutions_path = get_solutions_path(data_path)
    if not solutions_path:
        print(f"\n📝 No solutions file found for scoring (based on {data_path})")
        return
    if not os.path.exists(submission_path):
        print(f"\n❌ Submission file not found: {submission_path}")
        return

    print(f"\n🔍 Solutions file found: {solutions_path}")
    print(f"📊 Scoring submission: {submission_path}")

    vote_metadata = None
    if predictions:
        print(
            f"🔍 Predictions available: {len(predictions)} tasks, {sum(len(v) for v in predictions.values())} total predictions"
        )
        from .submission_handler import extract_vote_metadata

        vote_metadata = extract_vote_metadata(predictions, attempted_task_keys)
        print(f"🗳️ Extracted vote metadata for {len(vote_metadata)} tasks")
    else:
        print("⚠️ No predictions provided - vote metadata extraction skipped")

    if granular_scoring_data:
        perform_granular_scoring(solutions_path, attempted_task_keys, granular_scoring_data)

    scoring_results = score_submission(
        submission_path, solutions_path, attempted_task_keys, augmentation_metadata, vote_metadata
    )
    print_scoring_summary(scoring_results)
    if not skip_visuals:
        visualize_sorted_tasks_by_pixel_accuracy(
            scoring_results, data_path, submission_path, top_n=10, run_log_data=run_log_data
        )
    return scoring_results


# ============================== HTML TEMPLATES ===============================

# Shared minimal CSS/JS (unchanged behavior; compacted)
_COMMON_CSS = """
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:20px;background:#f5f5f5;color:#333}
.container{max-width:1400px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,.1)}
.header{text-align:center;border-bottom:3px solid #4CAF50;padding-bottom:20px;margin-bottom:30px}
.header h1{color:#2E7D32;margin:0;font-size:2.5em}.header .subtitle{color:#666;font-size:1.2em;margin:10px 0 5px 0;font-style:italic}
.navigation{position:sticky;top:20px;background:#fff;padding:15px;border:2px solid #4CAF50;border-radius:10px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,.1)}
.navigation a{color:#4CAF50;text-decoration:none;margin-right:15px;font-weight:700}.navigation a:hover{text-decoration:underline}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:30px}
.stats-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:20px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,.1)}
.stats-card h3{margin:0 0 15px 0;font-size:1.3em;border-bottom:2px solid rgba(255,255,255,.3);padding-bottom:10px}
.stats-card .metric{display:flex;justify-content:space-between;margin:8px 0;font-size:1.1em}
.position-stats{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%)}
.summary-section{margin:30px 0;padding:20px;background:#f8f9fa;border-left:5px solid #4CAF50;border-radius:0 10px 10px 0}
.task-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:30px;margin-top:30px}
.task-card{background:#fff;border:2px solid #ddd;border-radius:10px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,.1);transition:.2s}
.task-card:hover{transform:translateY(-5px);box-shadow:0 8px 15px rgba(0,0,0,.2)}
.task-header{background:linear-gradient(135deg,#74b9ff,#0984e3);color:#fff;padding:15px;font-weight:700;font-size:1.2em}
.task-stats{padding:15px;background:#f8f9fa;border-bottom:1px solid #eee}
.task-stats .stat{display:inline-block;margin-right:20px;margin-bottom:5px}.task-stats .stat .label{font-weight:700;color:#666}.task-stats .stat .value{color:#333;margin-left:5px}
.task-image{text-align:center;padding:20px}.task-image img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:5px}
.attempted{background:linear-gradient(135deg,#00b894,#55a3ff)}.not-attempted{background:linear-gradient(135deg,#fd79a8,#fdcb6e)}
.score-high{color:#00b894;font-weight:700}.score-medium{color:#f39c12;font-weight:700}.score-low{color:#e74c3c;font-weight:700}
.footer{text-align:center;margin-top:40px;padding:20px;border-top:2px solid #ddd;color:#666;font-size:.9em}
.run-summary-table{width:100%;border-collapse:collapse;margin:20px 0;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,.1)}
.run-summary-table th{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:12px 8px;text-align:center;font-weight:700;font-size:.8em}
.run-summary-table td{padding:10px 6px;text-align:center;border-bottom:1px solid #eee;font-size:.8em}
.run-summary-table tr:hover{background:#f8f9fa}.run-summary-table .checkpoint-name{text-align:left;font-weight:500;max-width:160px;min-width:140px}
.run-summary-table .score-value{font-weight:700;color:#2E7D32}.delta-positive{color:#4CAF50;font-weight:700}.delta-negative{color:#f44336;font-weight:700}.delta-zero{color:#666}
.vote-counts{margin:15px 0;padding:15px;background:#f8f9fa;border-radius:8px;border-left:4px solid #007bff}
.vote-badge{padding:4px 8px;border-radius:12px;font-size:.85em;font-weight:700;cursor:pointer;transition:.2s;border:2px solid transparent}
.vote-badge:hover{transform:scale(1.05);border-color:#333}
.vote-badge.correct{background:#28a745;color:#fff;border-color:#1e7e34}
.vote-badge.one-off{background:#6f42c1;color:#fff;border-color:#5a2d91}
.vote-badge.two-off{background:#fd7e14;color:#fff;border-color:#e55100}
.vote-badge.few-off{background:#e83e8c;color:#fff;border-color:#d91a72}
.vote-badge.many-off{background:#dc3545;color:#fff;border-color:#c82333}
.vote-badge.invalid{background:#6c757d;color:#fff;border-color:#5a6268}
.vote-expand-btn{background:#007bff;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:.9em;margin-top:8px}
.vote-expand-btn:hover{background:#0056b3}
.vote-details{display:none;margin-top:15px;padding:15px;background:#fff;border-radius:6px;border:1px solid #dee2e6}
.vote-details.expanded{display:block}
.prediction-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin-top:15px}
.prediction-item{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;padding:12px}
.prediction-header{font-weight:700;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.prediction-votes{background:#007bff;color:#fff;padding:2px 6px;border-radius:10px;font-size:.8em}
.prediction-grid-display{font-family:monospace;font-size:.8em;background:#fff;padding:8px;border-radius:4px;border:1px solid #dee2e6;overflow-x:auto}
"""

_COMMON_JS = """
function scrollToTask(taskId){const el=document.getElementById('task-'+taskId);if(el){el.scrollIntoView({behavior:'smooth',block:'center'});el.style.background='#fffacd';setTimeout(()=>{el.style.background='';},2000);}}
function toggleVoteDetails(taskId){const d=document.getElementById('vote-details-'+taskId);const b=document.getElementById('vote-expand-btn-'+taskId);if(d.classList.contains('expanded')){d.classList.remove('expanded');b.textContent='Show All Predictions';}else{d.classList.add('expanded');b.textContent='Hide Predictions';}}
"""


def _html_shell(title: str, body_html: str, width_1600: bool = False) -> str:
    """Thin wrapper to assemble a full HTML page."""
    container_css = _COMMON_CSS if not width_1600 else _COMMON_CSS.replace("max-width:1400px", "max-width:1600px")
    head = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        f"<title>{title}</title><style>{container_css}</style>"
        f"<script>{_COMMON_JS}</script></head><body>"
    )
    return head + body_html + "</body></html>"


# ----------------------------- Tiny render helpers ----------------------------


def _nav_links() -> str:
    return (
        "<div class='navigation'>"
        "<a href='leaderboard.html'>🏆 Run Leaderboard</a>"
        "<a href='#run-summary'>⏱️ Run Progress Summary</a>"
        "<a href='#overall-stats'>📊 Overall Statistics</a>"
        "<a href='#augmentation-stats'>🔄 Augmentation Stats</a>"
        "<a href='#filtering-stats'>🔍 Filtering Stats</a>"
        "<a href='#position-stats'>📈 Position Statistics</a>"
        "<a href='#task-details'>🎨 Task Details</a>"
        "</div>"
    )


def _page_header(title: str, subtitle: str) -> str:
    return (
        "<div class='header'>"
        f"<h1>{title}</h1>"
        f"<p class='subtitle'>{subtitle}</p>"
        f"<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        "</div>"
    )


def _metric_row(label: str, value_html: str) -> str:
    return f"<div class='metric'><span class='label'>{label}</span><span class='value'>{value_html}</span></div>"


def _stats_card(title: str, metrics_html: str, extra_classes: str = "") -> str:
    cls = "stats-card" + (f" {extra_classes}" if extra_classes else "")
    return f"<div class='{cls}'><h3>{title}</h3>{metrics_html}</div>"


def _wrap_container(inner: str) -> str:
    return f"<div class='container'>{inner}</div>"


def _footer(note_html: str) -> str:
    return f"<div class='footer'>{note_html}</div>"


# -------------------------- Leaderboard templates -----------------------------


def _render_run_card(rank: int, run_log: dict) -> str:
    run_id = run_log.get("run_id", "unknown")
    run_title = run_log.get("run_title", "Unknown Run")
    status = run_log.get("status", "unknown")
    start_time = run_log.get("start_time_formatted") or "Unknown"
    scoring = run_log.get("scoring_results") or {}
    stats = scoring.get("overall_stats", {}) if scoring else {}
    top2 = stats.get("overall_top2_score", 0.0)
    top1 = stats.get("overall_top1_score", 0.0)
    attempted = stats.get("attempted_tasks", 0)
    total = stats.get("total_tasks", 0)
    duration = run_log.get("total_duration", 0.0)
    duration_str = f"{duration / 60:.1f} min" if duration else "Unknown"
    env = (run_log.get("environment") or "unknown").title()
    test_mode = run_log.get("test_mode", False)
    model_names = [os.path.basename(m.get("model_path", "")) for m in run_log.get("models", [])]
    rank_cls = "rank-1" if rank == 1 else "rank-2" if rank == 2 else "rank-3" if rank == 3 else ""
    status_cls = f"status-{status}"

    models_html = (
        "".join(f"<span class='model-badge'>{n}</span>" for n in model_names) or "<span class='model-badge'>None</span>"
    )
    header = (
        f"<div class='run-header' onclick=\"window.open('run_details_{run_id}.html','_blank')\">"
        f"<div class='run-title'>Rank {rank}: {run_title}</div>"
        f"<div class='run-id'>ID: {run_id}</div>"
        f"<span style='float:right;'>📅 {start_time[:16] if start_time != 'Unknown' else 'Unknown'} | 🎯 {top2:.3f} | ⏱️ {duration_str}</span>"
        "</div>"
    )

    def stat(label: str, value: str) -> str:
        return (
            f"<div class='stat-item'><span class='stat-label'>{label}:</span>"
            f"<span class='stat-value'>{value}</span></div>"
        )
    stats_html = (
        "<div class='run-stats'>"
        + stat("Top-2 Score", f"{top2:.3f} ({top2 * 100:.1f}%)")
        + stat("Top-1 Score", f"{top1:.3f} ({top1 * 100:.1f}%)")
        + stat("Tasks Attempted", f"{attempted}/{total}")
        + stat("Status", status.title())
        + stat("Environment", f"{env} {'(Test Mode)' if test_mode else '(Production)'}")
        + stat("Duration", duration_str)
        + "</div>"
    )

    return (
        f"<div class='run-card {rank_cls} {status_cls}'>"
        f"{header}{stats_html}"
        f"<div style='padding:0 20px 20px;'><div class='model-list'><strong>Models:</strong> {models_html}</div></div>"
        f"</div>"
    )


def generate_leaderboard_html(run_logs_list: list[dict], output_dir: str) -> str | None:
    if not run_logs_list:
        print("⚠️ No run logs found for leaderboard generation")
        return None

    def score_key(log):
        scoring = log.get("scoring_results") or {}
        return (scoring.get("overall_stats") or {}).get("overall_top2_score", 0.0)

    runs = sorted(run_logs_list, key=score_key, reverse=True)
    header = _page_header("🏆 Run Leaderboard", "ARC Prize 2025 - All runs sorted by performance (Top-2 accuracy)")
    nav = (
        "<div class='navigation'>"
        "<a href='#' onclick='location.reload()' class='nav-button'>🔄 Refresh</a>"
        "<a href='scoring_report.html' class='nav-button'>📊 Latest Run Details</a>"
        "</div>"
    )
    cards = "".join(_render_run_card(i + 1, r) for i, r in enumerate(runs))
    body = (
        f"{header}{nav}"
        f"<div class='leaderboard'>{cards}</div>"
        + _footer(
            "<p><strong>Run Leaderboard</strong><br><em>ARC Prize 2025 Solution</em><br>Auto-refreshes every 30 seconds</p>"
        )
        + "<script>setTimeout(function(){location.reload();},30000);</script>"
    )
    html = _html_shell("ARC Prize 2025 - Run Leaderboard", _wrap_container(body), width_1600=True)
    path = os.path.join(output_dir, "leaderboard.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n🏆 Leaderboard generated: {os.path.abspath(path)}")
        return path
    except Exception as e:
        print(f"\n⚠️ Failed to generate leaderboard: {e}")
        return None


# --------------------------- Report page templates ----------------------------


def _stats_section_overall(stats: dict) -> str:
    # Card 1: coverage
    cov = (
        _metric_row("Total tasks in solutions:", f"{stats['total_tasks']}")
        + _metric_row("Tasks attempted:", f"{stats['attempted_tasks']}")
        + _metric_row("Coverage:", f"{stats['attempted_percentage']:.1f}%")
    )
    # Card 2: attempted accuracy
    att = (
        _metric_row(
            "Top-1 Accuracy:",
            f"{stats['attempted_top1_score']:.3f} ({stats['attempted_top1_score'] * 100:.1f}%) - {int(stats['attempted_top1_score'] * stats['attempted_tasks'])}/{stats['attempted_tasks']} tasks",
        )
        + _metric_row(
            "Top-2 Accuracy:",
            f"{stats['attempted_top2_score']:.3f} ({stats['attempted_top2_score'] * 100:.1f}%) - {int(stats['attempted_top2_score'] * stats['attempted_tasks'])}/{stats['attempted_tasks']} tasks",
        )
        + _metric_row(
            "Top-Max Accuracy:",
            f"{stats['attempted_top_max_score']:.3f} ({stats['attempted_top_max_score'] * 100:.1f}%) - {int(stats['attempted_top_max_score'] * stats['attempted_tasks'])}/{stats['attempted_tasks']} tasks (avg {stats['attempted_max_attempts_avg']:.1f} attempts)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-1):",
            f"{stats['attempted_pixel_accuracy_top1']:.3f} ({stats['attempted_pixel_accuracy_top1'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-2):",
            f"{stats['attempted_pixel_accuracy_top2']:.3f} ({stats['attempted_pixel_accuracy_top2'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-3):",
            f"{stats['attempted_pixel_accuracy_top3']:.3f} ({stats['attempted_pixel_accuracy_top3'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-4):",
            f"{stats['attempted_pixel_accuracy_top4']:.3f} ({stats['attempted_pixel_accuracy_top4'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-5):",
            f"{stats['attempted_pixel_accuracy_top5']:.3f} ({stats['attempted_pixel_accuracy_top5'] * 100:.1f}%)",
        )
    )
    # Card 3: overall accuracy
    ov = (
        _metric_row(
            "Top-1 Accuracy:",
            f"{stats['overall_top1_score']:.3f} ({stats['overall_top1_score'] * 100:.1f}%) - {int(stats['overall_top1_score'] * stats['total_tasks'])}/{stats['total_tasks']} tasks",
        )
        + _metric_row(
            "Top-2 Accuracy:",
            f"{stats['overall_top2_score']:.3f} ({stats['overall_top2_score'] * 100:.1f}%) - {int(stats['overall_top2_score'] * stats['total_tasks'])}/{stats['total_tasks']} tasks",
        )
        + _metric_row(
            "Top-Max Accuracy:",
            f"{stats['overall_top_max_score']:.3f} ({stats['overall_top_max_score'] * 100:.1f}%) - {int(stats['overall_top_max_score'] * stats['total_tasks'])}/{stats['total_tasks']} tasks (avg {stats['overall_max_attempts_avg']:.1f} attempts)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-1):",
            f"{stats['overall_pixel_accuracy_top1']:.3f} ({stats['overall_pixel_accuracy_top1'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-2):",
            f"{stats['overall_pixel_accuracy_top2']:.3f} ({stats['overall_pixel_accuracy_top2'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-3):",
            f"{stats['overall_pixel_accuracy_top3']:.3f} ({stats['overall_pixel_accuracy_top3'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-4):",
            f"{stats['overall_pixel_accuracy_top4']:.3f} ({stats['overall_pixel_accuracy_top4'] * 100:.1f}%)",
        )
        + _metric_row(
            "Pixel Accuracy (Top-5):",
            f"{stats['overall_pixel_accuracy_top5']:.3f} ({stats['overall_pixel_accuracy_top5'] * 100:.1f}%)",
        )
    )
    return f"<div id='overall-stats' class='stats-grid'>{_stats_card('📊 Task Coverage', cov)}{_stats_card('🎯 Attempted Tasks Accuracy', att)}{_stats_card('🌍 Overall Accuracy (All Tasks)', ov)}</div>"


def _run_summary_table(run_log_data: dict, scoring_results: dict) -> str:
    if not run_log_data or "models" not in run_log_data:
        return ""
    rows, cumulative_time = [], 0.0
    stats = scoring_results.get("overall_stats", {})
    prev_score, prev_solved = 0.0, 0

    for model in run_log_data["models"]:
        name = model.get("model_basename", "Unknown Model")
        for run in model.get("runs", []):
            duration = run.get("duration", 0.0)
            cumulative_time += duration
            score = run.get("final_score", run.get("score"))
            solved = run.get("final_solved", run.get("solved_tasks"))
            if score is None or solved is None:
                score = stats.get("overall_top2_score", 0.0)
                solved = sum(1 for t in scoring_results.get("task_results", []) if t.get("task_score", 0.0) > 0)
            s_delta = score - prev_score
            solved_delta = solved - prev_solved
            s_cls = "delta-positive" if s_delta > 0 else ("delta-negative" if s_delta < 0 else "delta-zero")
            so_cls = "delta-positive" if solved_delta > 0 else ("delta-negative" if solved_delta < 0 else "delta-zero")
            total_tasks = run.get("total_tasks", stats.get("total_tasks", 0))
            rows.append(
                f"<tr><td class='checkpoint-name'>Model {name} Run {run.get('run_number', '')}</td>"
                f"<td class='score-value'>{score:.3f}</td>"
                f"<td class='{s_cls}'>{'+' if s_delta >= 0 else ''}{s_delta:.3f}</td>"
                f"<td class='score-value'>{solved}/{total_tasks}</td>"
                f"<td class='{so_cls}'>{'+' if solved_delta >= 0 else ''}{solved_delta}</td>"
                f"<td>{run.get('active_tasks_after', 0)}</td>"
                f"<td>{run.get('ttt_items', 0)}</td>"
                f"<td>{run.get('inference_items', 0)}</td>"
                f"<td>{cumulative_time:.1f}</td></tr>"
            )
            prev_score, prev_solved = score, solved

    header = (
        "<div id='run-summary' class='summary-section'><h2>⏱️ Run Progress Summary</h2>"
        "<p>Progress summary showing model runs, scores, task completion, and processed items.</p>"
        "<table class='run-summary-table'><thead>"
        "<tr><th>Checkpoint</th><th>Score</th><th>Delta</th><th>Solved</th><th>Delta</th>"
        "<th>Active</th><th>TTT Items</th><th>Inference Items</th><th>Time (s)</th></tr>"
        "</thead><tbody>"
    )
    footer = (
        "</tbody></table>"
        f"<p style='font-size:.9em;color:#666;margin-top:10px;'>📊 Total Progress: Score {prev_score:.3f} | "
        f"Tasks solved {prev_solved}/{stats.get('total_tasks', 0)} | Time {cumulative_time:.1f}s</p></div>"
    )
    return header + "".join(rows) + footer


def _augmentation_section(augmentation_stats: dict) -> str:
    if not augmentation_stats:
        return ""
    total = augmentation_stats.get("total_augmentations_generated", 0)
    succ = augmentation_stats.get("successful_augmentations", 0)
    fail = augmentation_stats.get("failed_augmentations", 0)
    succ_pct = (succ / total * 100) if total else 0.0
    fail_pct = (fail / total * 100) if total else 0.0
    metrics = (
        _metric_row("Total Augmentations Generated:", str(total))
        + _metric_row("Successful Augmentations:", f"{succ} ({succ_pct:.1f}%)")
        + _metric_row("Failed Augmentations:", f"{fail} ({fail_pct:.1f}%)")
    )
    extra = ""
    breakdown = augmentation_stats.get("augmentation_type_breakdown", {})
    if breakdown:
        lines = "".join(
            _metric_row(
                k.replace("_", " ").title() + ":",
                f"{v.get('count', 0)} items ({v.get('success_rate', 0):.1f}% success)",
            )
            for k, v in breakdown.items()
        )
        extra = f"<div style='margin-top:15px;'><strong>By Augmentation Type:</strong>{lines}</div>"
    left = _stats_card("🔄 Augmentation Statistics", metrics + extra)

    # Filtering stats (optional)
    filt = augmentation_stats.get("filtering_stats", {})
    if filt.get("total_filtering_calls", 0) <= 0:
        return f"<div id='augmentation-stats' class='stats-grid'>{left}</div>"

    tc = filt.get("total_filtering_calls", 0)
    tf = filt.get("total_tasks_filtered", 0)
    tcf = filt.get("total_correct_filtered", 0)
    tr = filt.get("total_remaining_tasks", 0)
    tcr = filt.get("total_correct_remaining", 0)
    fa = (tcf / tf * 100) if tf else 0.0
    ra = (tcr / tr * 100) if tr else 0.0
    right_metrics = (
        _metric_row("Total Filtering Calls:", f"{tc}")
        + _metric_row("Tasks Filtered (Correct):", f"{tcf}/{tf} ({fa:.1f}%)")
        + _metric_row("Tasks Remaining (Correct):", f"{tcr}/{tr} ({ra:.1f}%)")
    )

    history = ""
    for e in filt.get("filtering_history", []):
        model = e.get("model", "Unknown")
        runn = e.get("run", 0)
        stage = e.get("stage", "Unknown")
        fcor, ftot = e.get("filtered_correct", 0), e.get("filtered_total", 0)
        rcor, rtot = e.get("remaining_correct", 0), e.get("remaining_total", 0)
        facc = (fcor / ftot * 100) if ftot else 0.0
        racc = (rcor / rtot * 100) if rtot else 0.0
        cur = e.get("current_threshold", 0.0)
        opt = e.get("optimal_threshold")
        opt_txt = (
            f"(optimal: {opt:.1f} - 100% CORRECT filtering achieved!)"
            if (opt is not None and ftot and fcor == ftot)
            else (f"(optimal: {opt:.1f})" if opt is not None else "(optimal: not determined)")
        )
        history += (
            "<div style='font-size:.9em;padding:8px;border-left:3px solid #4CAF50;margin:5px 0;background:#f8f9fa;'>"
            f"<strong>{model} Run {runn}</strong> ({stage})<br>"
            f"Filtered: {fcor}/{ftot} ({facc:.1f}%) | Remaining: {rcor}/{rtot} ({racc:.1f}%)<br>"
            f"Threshold: {cur:.1f} {opt_txt}</div>"
        )
    right = _stats_card(
        "🔍 Confidence Filtering Statistics",
        right_metrics
        + (
            f"<div style='margin-top:15px;'><strong>Filtering History:</strong><div style='max-height:400px;overflow-y:auto;margin-top:10px;'>{history}</div></div>"
            if history
            else ""
        ),
    )
    return f"<div id='augmentation-stats' class='stats-grid'>{left}{right}</div>"


def _position_stats_section(position_stats: dict) -> str:
    if not position_stats:
        return ""
    items = []
    for key in sorted(position_stats.keys(), key=lambda x: int(x.split("_")[1])):
        d = position_stats[key]
        pos = key.split("_")[1]
        items.append(
            _metric_row(
                f"Position {pos} ({d['count']} items):",
                f"Top-1: {d['top1_accuracy']:.3f} ({int(d['top1_accuracy'] * d['count'])}/{d['count']}) | "
                f"Top-2: {d['top2_accuracy']:.3f} ({int(d['top2_accuracy'] * d['count'])}/{d['count']})",
            )
        )
    return f"<div id='position-stats' class='stats-grid'>{_stats_card('📈 Accuracy by Test Item Position', ''.join(items), extra_classes='position-stats')}</div>"


def _quality_label(grid, correct):
    if not grid or not correct:
        return "invalid"
    try:
        if grid == correct:
            return "correct"
        pa = rowwise_pixel_accuracy(grid, correct)
        if pa >= 0.95:
            return "one-off"
        if pa >= 0.85:
            return "two-off"
        if pa >= 0.5:
            return "few-off"
        return "many-off"
    except Exception:
        return "invalid"


def _vote_badges(task_id: str, vote_metadata: dict, solutions_data: dict) -> str:
    # Build compact badges per test item (top vote).
    entries = []
    for idx in range(10):
        tk = f"{task_id}_{idx}"
        v = vote_metadata.get(tk)
        if not v:
            continue
        uniques, counts = (v.get("all_unique_grids") or []), (v.get("vote_counts") or {})
        total = v.get("total_predictions", 0)
        if not uniques or not counts:
            continue
        top_grid = max(uniques, key=lambda g: counts.get(str(g), 0))
        top_votes = counts.get(str(top_grid), 0)
        correct = solutions_data.get(task_id, [])
        correct = correct[idx] if idx < len(correct) else None
        q = _quality_label(top_grid, correct)
        entries.append(
            f"<span class='vote-badge {q}' title='Item {idx}: {len(uniques)} unique, top {top_votes} votes, quality={q}'>Item {idx}: {top_votes}/{total}</span>"
        )
    if not entries:
        return ""
    legend = (
        "<div style='margin-top:8px;font-size:.8em;color:#666;'><strong>Quality Legend:</strong>"
        "<span class='vote-badge correct' style='font-size:.7em;'>Correct</span>"
        "<span class='vote-badge one-off' style='font-size:.7em;'>1-off</span>"
        "<span class='vote-badge two-off' style='font-size:.7em;'>2-off</span>"
        "<span class='vote-badge few-off' style='font-size:.7em;'>Few-off</span>"
        "<span class='vote-badge many-off' style='font-size:.7em;'>Many-off</span>"
        "<span class='vote-badge invalid' style='font-size:.7em;'>Invalid</span></div>"
    )
    return (
        "<div class='vote-counts'><h4>🗳️ Prediction Vote Counts</h4>"
        f"<div class='vote-count-summary'>{''.join(entries)}</div>"
        f"{legend}"
        f"<button class='vote-expand-btn' id='vote-expand-btn-{task_id}' onclick=\"toggleVoteDetails('{task_id}')\">Show All Predictions</button>"
        f"<div class='vote-details' id='vote-details-{task_id}'><h5>Detailed Prediction Analysis</h5><div class='prediction-grid'>"
    )


def _vote_prediction_cards(task_id: str, vote_metadata: dict, solutions_data: dict) -> str:
    blocks = []
    for idx in range(10):
        tk = f"{task_id}_{idx}"
        v = vote_metadata.get(tk)
        if not v:
            continue
        uniques, counts = (v.get("all_unique_grids") or []), (v.get("vote_counts") or {})
        if not uniques or not counts:
            continue
        correct = solutions_data.get(task_id, [])
        correct = correct[idx] if idx < len(correct) else None
        for rank, grid in enumerate(sorted(uniques, key=lambda g: counts.get(str(g), 0), reverse=True)[:5]):
            votes = counts.get(str(grid), 0)
            q = _quality_label(grid, correct) if correct is not None else "unknown"
            # Compact grid (highlight errors); limit size for HTML
            grid_html = ""
            if isinstance(grid, list) and grid:
                for r_i, row in enumerate(grid[:8]):
                    if isinstance(row, list):
                        row_html = []
                        for c_i, cell in enumerate(row[:12]):
                            err = (
                                correct and r_i < len(correct) and c_i < len(correct[r_i]) and correct[r_i][c_i] != cell
                            )
                            style = "background:#ffebee;color:#c62828;font-weight:700;padding:1px 2px;" if err else ""
                            row_html.append(f"<span style='{style}'>{cell}</span>")
                        grid_html += " ".join(row_html)
                        if r_i < len(grid) - 1 and r_i < 7:
                            grid_html += "<br>"
                if len(grid) > 8:
                    grid_html += "<br>..."
            else:
                s = str(grid)
                grid_html = (s[:200] + "...") if len(s) > 200 else s
            blocks.append(
                "<div class='prediction-item'>"
                f"<div class='prediction-header'><span>Item {idx} - Rank {rank + 1} ({q})</span>"
                f"<span class='prediction-votes'>{votes} votes</span></div>"
                f"<div class='prediction-grid-display'>{grid_html}</div></div>"
            )
    if not blocks:
        return ""
    return "".join(blocks) + "</div></div></div>"  # close .prediction-grid, .vote-details, .vote-counts


def _task_card(rank: int, tr: dict, plot_output_dir: str, vote_metadata: dict, solutions_data: dict) -> str:
    task_id = tr["task_id"]
    pix2 = tr["pixel_accuracy_top2"]
    pix1 = tr["pixel_accuracy_top1"]
    top1 = tr["top1_score"]
    top2 = tr["top2_score"]
    attempted = tr["attempted"]
    score_class = "score-high" if top2 >= 0.8 else "score-medium" if top2 >= 0.4 else "score-low"
    attempted_cls = "attempted" if attempted else "not-attempted"
    image_filename = f"rank_{rank}_task_{task_id}.png"
    image_path = os.path.join(plot_output_dir, image_filename)
    stats_html = (
        "<div class='task-stats'>"
        f"<div class='stat'><span class='label'>Pixel Accuracy (Top-2):</span><span class='value {score_class}'>{pix2:.3f}</span></div>"
        f"<div class='stat'><span class='label'>Pixel Accuracy (Top-1):</span><span class='value'>{pix1:.3f}</span></div>"
        f"<div class='stat'><span class='label'>Top-1 Score:</span><span class='value'>{top1:.3f}</span></div>"
        f"<div class='stat'><span class='label'>Top-2 Score:</span><span class='value {score_class}'>{top2:.3f}</span></div>"
        "</div>"
    )
    votes_block = ""
    if vote_metadata:
        start = _vote_badges(task_id, vote_metadata, solutions_data)
        details = _vote_prediction_cards(task_id, vote_metadata, solutions_data)
        votes_block = (start + details) if details else ""
    img_html = (
        f"<a href='{image_filename}' target='_blank'><img src='{image_filename}' alt='Task {task_id} visualization' title='Click to view full size'></a>"
        "<p style='margin-top:10px;font-size:.9em;color:#666;'>Click image to view full size</p>"
        if os.path.exists(image_path)
        else "<p style='color:#e74c3c;font-style:italic;'>⚠️ Visualization not available for this task</p>"
    )
    return (
        f"<div class='task-card' id='task-{task_id}'>"
        f"<div class='task-header {attempted_cls}'>Rank {rank}: {task_id}"
        f"<span style='float:right;'>{'✓ Attempted' if attempted else '✗ Not Attempted'}</span></div>"
        f"{stats_html}{votes_block}<div class='task-image'>{img_html}</div></div>"
    )


# --------------------------------- REPORT ------------------------------------


def generate_html_report(
    scoring_results: dict,
    data_path: str,
    submission_path: str,
    plot_output_dir: str,
    top_n: int = 10,
    run_log_data: dict | None = None,
) -> str | None:
    if "error" in scoring_results:
        print(f"❌ Cannot generate HTML report due to scoring error: {scoring_results['error']}")
        return None

    stats = scoring_results["overall_stats"]
    position_stats = scoring_results.get("position_stats", {})
    task_results = scoring_results["task_results"]
    vote_metadata = scoring_results.get("vote_metadata", {})

    # Solutions for vote-quality classification
    solutions_data = {}
    sol_path = get_solutions_path(data_path)
    if sol_path:
        try:
            with open(sol_path) as f:
                solutions_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load solutions for quality classification: {e}")

    sorted_tasks = sorted(task_results, key=lambda x: (not x["attempted"], -x["pixel_accuracy_top2"]))

    header = _page_header(config.RUN_TITLE, "ARC Prize 2025 - Scoring Results")
    nav = _nav_links()
    run_summary = _run_summary_table(run_log_data, scoring_results)
    overall = _stats_section_overall(stats)
    augmentation = _augmentation_section(scoring_results.get("augmentation_stats", {}))
    pos_stats = _position_stats_section(position_stats)

    intro = (
        "<div class='summary-section'>"
        f"<h2 id='task-details'>🎨 Top {min(top_n, len(sorted_tasks))} Tasks (Attempted First, Then By Pixel Accuracy)</h2>"
        "<p>Click on any image to view it in full size. Tasks are sorted by attempted status first, then by pixel accuracy.</p>"
        "</div>"
    )
    cards = "".join(
        _task_card(i + 1, tr, plot_output_dir, vote_metadata, solutions_data)
        for i, tr in enumerate(sorted_tasks[:top_n])
    )
    tasks_grid = f"<div class='task-grid'>{cards}</div>"

    body = (
        header
        + nav
        + (run_summary or "")
        + overall
        + augmentation
        + pos_stats
        + intro
        + tasks_grid
        + _footer(
            f"<p><strong>{config.RUN_TITLE}</strong><br><em>ARC Prize 2025 Solution</em><br>"
            f"Data Path: {data_path}<br>Submission Path: {submission_path}<br>"
            f"Plot Directory: {os.path.abspath(plot_output_dir)}</p>"
        )
    )
    html = _html_shell("ARC Prize 2025 - Scoring Results", _wrap_container(body))
    out_path = os.path.join(plot_output_dir, "scoring_report.html")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n📄 HTML report generated: {os.path.abspath(out_path)}")
        try:
            from .config import get_all_run_logs

            run_logs = get_all_run_logs()
            if run_logs:
                generate_leaderboard_html(run_logs, plot_output_dir)
            else:
                print("📊 No run logs found for leaderboard generation")
        except Exception as e:
            print(f"⚠️ Failed to generate leaderboard: {e}")
        return out_path
    except Exception as e:
        print(f"\n⚠️ Failed to generate HTML report: {e}")
        return None
    if not _MATPLOTLIB_AVAILABLE:
        print("   ⚠️ matplotlib unavailable; skipping task visualization.")
        return
    if not _MATPLOTLIB_AVAILABLE:
        print("⚠️ matplotlib unavailable; skipping task visualization.")
        return
