"""
Data loading module for ARC Prize 2025 solution
Handles loading and preprocessing of challenge data
"""

import json
from random import sample, seed
import sys

from .grid_utils import is_grid


def load_data(
    data_path: str, limit: int | None = None, task_ids: list | None = None, random_seed: int | None = None
) -> dict[str, dict]:
    """Loads and preprocesses data from JSON file."""
    print(f"Loading data from {data_path}...")

    try:
        with open(data_path) as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Exiting.", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {data_path}: {e}. Exiting.", file=sys.stderr)
        return {}

    # First, determine which challenges to include
    all_challenge_ids = sorted(raw_data.keys())

    if task_ids:
        # Support base id (e.g., 'task_24') to select all test cases, and full id (e.g., 'task_24_0') for specific
        base_ids = set()
        full_ids = set()
        for tid in task_ids:
            if tid in all_challenge_ids:
                base_ids.add(tid)
            elif tid in [f"{cid}_{i}" for cid in all_challenge_ids for i in range(100)]:
                full_ids.add(tid)
            elif "_" in tid:
                base_ids.add(tid.split("_")[0])
            else:
                base_ids.add(tid)
        sampled_challenge_ids = [cid for cid in base_ids if cid in all_challenge_ids]
        print(f"Limiting to {len(sampled_challenge_ids)} challenges based on {len(task_ids)} specified task IDs.")
    elif limit is not None and limit > 0:
        # Set random seed for reproducible sampling if provided
        if random_seed is not None:
            seed(random_seed)
        # Sample at challenge level, not task level
        num_challenges_to_sample = min(limit, len(all_challenge_ids))
        sampled_challenge_ids = sample(all_challenge_ids, num_challenges_to_sample)
        print(
            "Limiting to "
            f"{num_challenges_to_sample} randomly sampled challenges from {len(all_challenge_ids)} available."
        )
    else:
        sampled_challenge_ids = all_challenge_ids

    # Process all test cases for sampled challenges
    processed_tasks = {}

    for challenge_id in sampled_challenge_ids:
        challenge_content = raw_data[challenge_id]
        if not isinstance(challenge_content, dict):
            continue

        train_examples = challenge_content.get("train", [])
        test_examples = challenge_content.get("test", [])

        if not isinstance(train_examples, list):
            train_examples = []
        if not isinstance(test_examples, list):
            test_examples = []

        # Process ALL test cases for this challenge
        for test_idx, test_ex_data in enumerate(test_examples):
            if not (isinstance(test_ex_data, dict) and "input" in test_ex_data and is_grid(test_ex_data["input"])):
                continue

            task_key = f"{challenge_id}_{test_idx}"

            task_dict_for_item = {"train": train_examples, "test": [{"input": test_ex_data["input"]}]}
            processed_tasks[task_key] = task_dict_for_item

    # If specific task_ids are provided, filter the processed_tasks dict
    if task_ids:
        filtered_tasks = {}
        for tid in task_ids:
            # If base id, include all matching keys
            if tid in sampled_challenge_ids:
                for k in processed_tasks:
                    if k.startswith(f"{tid}_") or k == tid:
                        filtered_tasks[k] = processed_tasks[k]
            # If full id, include only that key
            elif tid in processed_tasks:
                filtered_tasks[tid] = processed_tasks[tid]
        print(
            "Filtered to "
            f"{len(filtered_tasks)} specific tasks from the initially processed {len(processed_tasks)} tasks."
        )
        processed_tasks = filtered_tasks

    print(
        f"Data loading complete. {len(processed_tasks)} tasks processed from {len(sampled_challenge_ids)} challenges."
    )
    return processed_tasks
