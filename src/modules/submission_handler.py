"""
Submission handling module for ARC Prize 2025 solution
Creates and manages submission files
"""

import json
import sys

from .grid_utils import is_valid_prediction


def extract_vote_metadata(predictions: dict[str, list], all_task_keys: list[str]) -> dict:
    """Extract vote metadata from predictions without creating submission file.

    Args:
        predictions: Dictionary of task predictions
        all_task_keys: List of all task keys

    Returns:
        Dictionary containing vote metadata for true Top-Max scoring
    """
    vote_metadata = {}

    for task_key in all_task_keys:
        preds_for_task_key = predictions.get(task_key, [])

        # Extract grids from predictions (handle both raw grids and dict format)
        task_grids = []
        for pred_item in preds_for_task_key:
            if isinstance(pred_item, dict) and "final_decoded_grid" in pred_item:
                grid = pred_item["final_decoded_grid"]
            else:
                grid = pred_item

            # Use validation that excludes fallback patterns
            if is_valid_prediction(grid):
                task_grids.append(grid)

        # Simple deduplication and counting
        task_grids_unique = [p for i, p in enumerate(task_grids) if task_grids.index(p) == i]

        # Collect vote metadata for true Top-Max scoring
        vote_counts = {}
        for grid in task_grids_unique:
            vote_counts[str(grid)] = task_grids.count(grid)

        vote_metadata[task_key] = {
            "total_predictions": len(task_grids),
            "unique_predictions": len(task_grids_unique),
            "vote_counts": vote_counts,
            "all_unique_grids": task_grids_unique,
        }

    return vote_metadata


def keysplit(key: str) -> tuple[str, int]:
    """Splits key like '0000000b_0' into ('0000000b', 0)."""
    parts = key.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        challenge_id_part = "_".join(parts[:-1])
        test_index_part = int(parts[-1])
        return challenge_id_part, test_index_part
    return key, 0


def make_submission(predictions: dict[str, list], all_task_keys: list[str], output_path: str) -> dict:
    """Creates and saves submission file in the specified JSON format."""
    submission_dict, _ = make_submission_with_vote_metadata(predictions, all_task_keys, output_path)
    return submission_dict


def make_submission_with_vote_metadata(
    predictions: dict[str, list], all_task_keys: list[str], output_path: str
) -> tuple[dict, dict]:
    """Creates and saves submission file in the specified JSON format.

    Returns:
        Tuple of (submission_dict, vote_metadata) where vote_metadata contains
        vote counts for all unique predictions per task for true Top-Max scoring.
    """

    # Group task keys by their base challenge ID
    challenge_id_to_task_keys = {}
    for task_key in all_task_keys:
        challenge_id, _ = keysplit(task_key)
        if challenge_id not in challenge_id_to_task_keys:
            challenge_id_to_task_keys[challenge_id] = []
        challenge_id_to_task_keys[challenge_id].append(task_key)

    # Sort challenge IDs for consistent submission order
    sorted_challenge_ids = sorted(challenge_id_to_task_keys.keys())

    final_submission_dict = {}
    vote_metadata = {}

    for cid in sorted_challenge_ids:
        # Sort task keys for this challenge by their test index
        task_keys_for_cid = sorted(challenge_id_to_task_keys[cid], key=lambda tk: keysplit(tk)[1])

        final_submission_dict[cid] = []

        for task_key in task_keys_for_cid:
            preds_for_task_key = predictions.get(task_key, [])

            # Extract grids from predictions (handle both raw grids and dict format)
            task_grids = []
            for pred_item in preds_for_task_key:
                if isinstance(pred_item, dict) and "final_decoded_grid" in pred_item:
                    grid = pred_item["final_decoded_grid"]
                else:
                    grid = pred_item

                # Use validation that excludes fallback patterns
                if is_valid_prediction(grid):
                    task_grids.append(grid)

            # Simple deduplication and counting (like original solution)
            task_grids_unique = [p for i, p in enumerate(task_grids) if task_grids.index(p) == i]
            attempts = sorted(task_grids_unique, key=lambda x: -task_grids.count(x))[:2]

            # Collect vote metadata for true Top-Max scoring
            # Store all unique predictions with their vote counts
            vote_counts = {}
            for grid in task_grids_unique:
                vote_counts[str(grid)] = task_grids.count(grid)

            vote_metadata[task_key] = {
                "total_predictions": len(task_grids),
                "unique_predictions": len(task_grids_unique),
                "vote_counts": vote_counts,
                "all_unique_grids": task_grids_unique,
            }

            # Ensure we have 2 attempts, pad with [[0]] if needed
            while len(attempts) < 2:
                attempts.append([[0]])

            # Select top 2 predictions for attempt_1 and attempt_2
            attempt1_grid = attempts[0]
            attempt2_grid = attempts[1]

            final_submission_dict[cid].append({"attempt_1": attempt1_grid, "attempt_2": attempt2_grid})

    # Save submission to file
    print(f"Saving submission with {len(final_submission_dict)} challenge IDs to {output_path}...")
    try:
        with open(output_path, "w") as f:
            json.dump(final_submission_dict, f, indent=4)
        print("Submission saved successfully.")
    except OSError as e:
        print(f"Error saving submission file to {output_path}: {e}", file=sys.stderr)

    return final_submission_dict, vote_metadata
