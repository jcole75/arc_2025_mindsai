"""
Dataset generation module for ARC Prize 2025 solution.

This module is responsible for creating two primary types of datasets:
1.  TTT (Test-Time Training) Dataset:
    This dataset is designed for "meta-learning" or learning to transform.
    It works as follows:
    - For each original task, one augmentation (variant) is created.
    - From this augmented task's training examples, a leave-one-out strategy is applied.
    - One example is held out (its input becomes the test input, its output the target).
    - The remaining examples from the augmented task are formatted into a prompt.
    - The final TTT item consists of this prompt and the expected output string for the
      held-out example.
    - The goal is to train the model to infer the transformation rule from a few
      examples and apply it to a new input.

2.  Inference Dataset:
    This dataset consists of multiple augmentations (variations) of the original
    training tasks if AIRV is enabled. If AIRV is disabled (0-shot), it consists
    of the original, un-augmented tasks.
    - For each original task:
        - If AIRV enabled: a specified number of augmentations are generated.
          The distribution aims to meet a target total while ensuring
          each task receives a minimum number of variations.
        - If AIRV disabled: the original task itself is used.
    - Each item (augmented task or original task) becomes an entry in the
      inference dataset, along with its corresponding decoder description
      (identity for 0-shot).

The module also includes functionality to convert the generated TTT items into
HuggingFace `Dataset` objects, suitable for training and evaluation pipelines.
"""

from typing import Any
from random import shuffle  # Added sample

import tqdm
from transformers import AutoTokenizer

from . import config
from .augmentation_framework import get_n_augs_flexible
from .config import SYMBOL_ENCODING, debug_print, verbose_print
from .grid_utils import grid_to_string, is_grid, makeprompt, output_prefix


def get_token_count_estimate(text: str, tokenizer=None) -> int:
    """Get rough token count estimate. Use actual tokenizer if provided, else word count * 1.3"""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback: rough estimate (words * 1.3 to account for subword tokenization)
    return max(1, int(len(text.split()) * 1.3))


def make_datasets(
    train_data: dict,
    keys: list[str],
    target_ttt: int,
    target_inference: int,
    max_input_tokens: int = 2048,
    max_target_tokens: int = 512,
    model_path: str | None = None,
    airv_enabled_for_model: bool = True,
    enable_token_filtering: bool = True,
) -> tuple[list, dict, dict]:
    """Generates TTT and inference datasets with token length filtering."""
    ttt_items = []
    inference_data_dict = {key: {"tasks": [], "decs": []} for key in keys}

    # Report symbol encoding status
    try:
        enc_enabled = bool(SYMBOL_ENCODING.get("enabled"))
        enc_scheme = SYMBOL_ENCODING.get("scheme", "letters") if enc_enabled else "disabled"
        if enc_enabled:
            print(f"  Symbol Encoding: ENABLED ({enc_scheme})")
        else:
            print("  Symbol Encoding: DISABLED")
    except Exception:
        pass

    # Initialize tokenizer for accurate token counting (if needed for filtering)
    tokenizer = None

    prompt_style = "legacy"
    if model_path:
        try:
            model_settings = getattr(config, "MODEL_SETTINGS", {}).get(model_path, {}) or {}
            general_settings = model_settings.get("general", {}) or {}
            prompt_style = str(general_settings.get("prompt_format") or "legacy").lower()
        except Exception:
            prompt_style = "legacy"

    def _format_prompt(raw: str) -> str:
        formatted = str(raw or "").rstrip()
        if prompt_style == "arc_diffusion":
            return formatted
        return formatted + " "

    if enable_token_filtering and model_path and not model_path.startswith("local-dummy"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            debug_print(f"Loaded tokenizer for token filtering: {model_path}")
        except Exception as e:
            debug_print(f"Could not load tokenizer for {model_path}: {e}. Using word-count estimation.")

    # Track filtering statistics
    ttt_total_generated = 0
    ttt_filtered_prompt = 0
    ttt_filtered_answer = 0
    inference_total_generated = 0
    inference_filtered_prompt = 0

    if not enable_token_filtering:
        debug_print("Token filtering completely disabled for this model.")

    # Track augmentation statistics
    augmentation_stats = {
        "total_augmentations_generated": 0,
        "successful_augmentations": 0,
        "failed_augmentations": 0,
        "augmentation_type_breakdown": {},
    }
    def _next_task_batch() -> list[str]:
        if not keys:
            return []
        shuffled = list(keys)
        shuffle(shuffled)
        return shuffled

    # Generate TTT dataset
    if target_ttt > 0:
        print(f"Generating TTT dataset (target: {target_ttt} items)...")
        with tqdm.tqdm(total=target_ttt, desc="TTT Generation", unit="item") as pbar:
            max_passes = max(10, (target_ttt // max(1, len(keys) if keys else 1)) + 20)
            passes_done = 0

            while len(ttt_items) < target_ttt and passes_done < max_passes:
                passes_done += 1
                items_generated_this_pass = 0

                task_order = _next_task_batch()

                for _key_idx, task_key in enumerate(task_order):
                    if len(ttt_items) >= target_ttt:
                        break

                    original_task = train_data.get(task_key)
                    if original_task is None:
                        continue

                    augmented_tasks_list, aug_decoder_descriptions, ttt_aug_stats = get_n_augs_flexible(
                        original_task, 1, "ttt", list(train_data.values())
                    )

                    # Update augmentation statistics for TTT
                    augmentation_stats["total_augmentations_generated"] += ttt_aug_stats.get("total_requested", 0)
                    augmentation_stats["successful_augmentations"] += ttt_aug_stats.get("successful", 0)
                    augmentation_stats["failed_augmentations"] += ttt_aug_stats.get("failed", 0)

                    # Update type breakdown
                    for aug_type, type_stats in ttt_aug_stats.get("by_type", {}).items():
                        if aug_type not in augmentation_stats["augmentation_type_breakdown"]:
                            augmentation_stats["augmentation_type_breakdown"][aug_type] = {
                                "count": 0,
                                "success_rate": 0,
                            }
                        count = type_stats.get("generated", 0) + type_stats.get("failed", 0)
                        success_rate = (type_stats.get("generated", 0) / count * 100) if count > 0 else 0
                        augmentation_stats["augmentation_type_breakdown"][aug_type]["count"] += count
                        augmentation_stats["augmentation_type_breakdown"][aug_type]["success_rate"] = success_rate
                    if not augmented_tasks_list:
                        continue

                    current_augmented_task = augmented_tasks_list[0]
                    augmented_train_examples = current_augmented_task.get("train", [])
                    if not augmented_train_examples:
                        continue

                    indices_for_loo = list(range(len(augmented_train_examples)))
                    shuffle(indices_for_loo)

                    for loo_idx in indices_for_loo:
                        if len(ttt_items) >= target_ttt:
                            break

                        held_out_example = augmented_train_examples[loo_idx]
                        if not (is_grid(held_out_example.get("input")) and is_grid(held_out_example.get("output"))):
                            continue

                        remaining_examples_for_prompt = [
                            ex for i, ex in enumerate(augmented_train_examples) if i != loo_idx
                        ]

                        prompt_gen_task_dict = {
                            "train": remaining_examples_for_prompt,
                            "test": [{"input": held_out_example["input"]}],
                        }

                        prompt_str = makeprompt(prompt_gen_task_dict, style=prompt_style)

                        # Skip TTT items with invalid prompts
                        if (
                            "[prompt_error]" in prompt_str
                            or "[missing_input]" in prompt_str
                            or "[no_valid_train_examples]" in prompt_str
                            or "[no_test_data]" in prompt_str
                        ):
                            debug_print(
                                "TTT generation: Skipping item due to prompt error for task "
                                f"{task_key} (LOO idx {loo_idx}). Prompt: {prompt_str[:100]}..."
                            )
                            continue

                        if prompt_style == "arc_diffusion":
                            rows = [
                                row
                                for row in grid_to_string(held_out_example["output"]).split(" ")
                                if row
                            ]
                            target_str = "\n".join(rows)
                        else:
                            target_str = (
                                " "
                                + output_prefix(held_out_example["output"])
                                + grid_to_string(held_out_example["output"])
                                + "."
                            )

                        # Ensure prompt ends with space and answer starts with space
                        prompt_formatted = _format_prompt(prompt_str)  # Remove trailing whitespace, add single space

                        ttt_total_generated += 1

                        # Token length filtering for TTT items when early filtering is enabled
                        if enable_token_filtering:
                            prompt_tokens = get_token_count_estimate(prompt_formatted, tokenizer)
                            answer_tokens = get_token_count_estimate(target_str, tokenizer)

                            if prompt_tokens > max_input_tokens:
                                ttt_filtered_prompt += 1
                                debug_print(
                                    "TTT: Filtering prompt too long "
                                    f"({prompt_tokens} > {max_input_tokens}) for task {task_key}"
                                )
                                continue

                            if answer_tokens > max_target_tokens:
                                ttt_filtered_answer += 1
                                debug_print(
                                    "TTT: Filtering answer too long "
                                    f"({answer_tokens} > {max_target_tokens}) for task {task_key}"
                                )
                                continue

                        # Get augmentation info for this TTT item
                        aug_info = aug_decoder_descriptions[0] if aug_decoder_descriptions else {}

                        # Convert any integer keys to strings for PyArrow compatibility
                        def convert_int_keys_to_strings(obj):
                            if isinstance(obj, dict):
                                return {str(k): convert_int_keys_to_strings(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_int_keys_to_strings(item) for item in obj]
                            else:
                                return obj

                        aug_info_cleaned = convert_int_keys_to_strings(aug_info)

                        ttt_items.append(
                            {
                                "prompt": prompt_formatted,
                                "target": target_str,
                                "task_key": task_key,
                                "augmentation": aug_info_cleaned,
                            }
                        )
                        items_generated_this_pass += 1
                        pbar.update(1)

                if items_generated_this_pass == 0 and len(keys) > 0:
                    print(f"TTT generation: Pass {passes_done} added 0 items. Stopping early.")
                    break

    # Generate inference dataset
    if keys:  # Always try to generate inference data if keys exist
        print(
            "Generating inference dataset "
            f"(Configured AIRV target: {target_inference}) for {len(keys)} tasks..."
        )
        print(f"  AIRV (Augmentation pipeline) enabled for this model: {airv_enabled_for_model}")

        if not airv_enabled_for_model:  # 0-shot inference path
            print("  Generating 0-shot inference prompts (original tasks only).")
            items_added_0_shot = 0
            inference_order = _next_task_batch()
            for task_key in tqdm.tqdm(inference_order, desc="0-Shot Inference Prep", unit="task"):
                original_task = train_data.get(task_key)
                if original_task is None:
                    continue

                # Create the prompt string for filtering
                prompt_str_for_filtering = _format_prompt(makeprompt(original_task, style=prompt_style))
                inference_total_generated += 1  # Count before filtering

                # Token length filtering (only if token filtering is enabled and early filtering is True)
                if enable_token_filtering:
                    prompt_tokens = get_token_count_estimate(prompt_str_for_filtering, tokenizer)
                    if prompt_tokens > max_input_tokens:
                        inference_filtered_prompt += 1
                        debug_print(
                            "0-Shot Inference: Filtering prompt too long "
                            f"({prompt_tokens} > {max_input_tokens}) for task {task_key}"
                        )
                        continue  # inference_data_dict[task_key] will remain with empty tasks/decs

                # For 0-shot, the 'task' is the original task, decoder is identity
                inference_data_dict[task_key]["tasks"].append(original_task)
                inference_data_dict[task_key]["decs"].append({"geom_name": "identity", "colmap_inv": {}})
                items_added_0_shot += 1

            print(f"  Generated {items_added_0_shot} 0-shot inference prompts for {len(keys)} tasks (after filtering).")

        elif target_inference > 0:  # AIRV enabled AND target_inference is positive
            print(f"  AIRV enabled: Generating augmented inference data (target: {target_inference} augmentations)...")
            num_tasks = len(keys)
            if num_tasks == 0:
                print("  No tasks available for AIRV augmentation.")
                return ttt_items, inference_data_dict, augmentation_stats

            base_augs_per_task = target_inference // num_tasks
            extra_augs_needed = target_inference % num_tasks

            print(f"  Augmentation distribution: {base_augs_per_task} base augs per task")

            debug_print(
                f"Distribution plan: {base_augs_per_task} base augs per task, {extra_augs_needed} tasks get +1 extra"
            )

            distribution_plan = {}
            keys_shuffled = _next_task_batch()

            for i, task_key in enumerate(keys_shuffled):
                distribution_plan[task_key] = base_augs_per_task + (1 if i < extra_augs_needed else 0)

            total_augs_generated_airv = 0
            failed_tasks_airv = []

            estimated_total_airv = sum(distribution_plan.values()) if distribution_plan else 0
            with tqdm.tqdm(total=estimated_total_airv, desc="AIRV Augmentation Generation", unit="aug") as pbar:
                for task_key in keys_shuffled:
                    if task_key not in distribution_plan:
                        debug_print(f"Skipping task {task_key} - not in distribution plan (this should not happen).")
                        continue

                    augs_for_this_task = distribution_plan[task_key]
                    if augs_for_this_task == 0:  # if still 0, skip
                        continue

                    original_task = train_data.get(task_key)
                    if original_task is None:
                        continue
                    # Provide a pool with task keys so mixup_combine can record source IDs
                    task_pool = [
                        {"task_key": k, "task": train_data[k]}
                        for k in keys
                        if k in train_data
                    ]
                    augmented_task_dicts_list, decoder_descs_list, inf_aug_stats = get_n_augs_flexible(
                        original_task, augs_for_this_task, "inference", task_pool=task_pool
                    )

                    # Update augmentation statistics for inference
                    augmentation_stats["total_augmentations_generated"] += inf_aug_stats.get("total_requested", 0)
                    augmentation_stats["successful_augmentations"] += inf_aug_stats.get("successful", 0)
                    augmentation_stats["failed_augmentations"] += inf_aug_stats.get("failed", 0)

                    # Update type breakdown
                    for aug_type, type_stats in inf_aug_stats.get("by_type", {}).items():
                        if aug_type not in augmentation_stats["augmentation_type_breakdown"]:
                            augmentation_stats["augmentation_type_breakdown"][aug_type] = {
                                "count": 0,
                                "success_rate": 0,
                            }
                        count = type_stats.get("generated", 0) + type_stats.get("failed", 0)
                        success_rate = (type_stats.get("generated", 0) / count * 100) if count > 0 else 0
                        prev_count = augmentation_stats["augmentation_type_breakdown"][aug_type]["count"]
                        prev_rate = augmentation_stats["augmentation_type_breakdown"][aug_type]["success_rate"]

                        # Weighted average for success rate
                        total_count = prev_count + count
                        if total_count > 0:
                            augmentation_stats["augmentation_type_breakdown"][aug_type]["success_rate"] = (
                                prev_rate * prev_count + success_rate * count
                            ) / total_count
                        augmentation_stats["augmentation_type_breakdown"][aug_type]["count"] = total_count

                    if augmented_task_dicts_list:
                        valid_aug_tasks, valid_decs = [], []
                        for aug_idx_loop, (aug, dec) in enumerate(
                            zip(augmented_task_dicts_list, decoder_descs_list, strict=False)
                        ):
                            pr = _format_prompt(makeprompt(aug, style=prompt_style))
                            inference_total_generated += 1

                            if (
                                "[prompt_error]" in pr
                                or "[missing_input]" in pr
                                or "[no_valid_train_examples]" in pr
                                or "[no_test_data]" in pr
                            ):
                                debug_print(
                                "Skipping invalid augmentation for "
                                f"{task_key} (aug {aug_idx_loop}): prompt error. "
                                f"Prompt: {pr[:100]}..."
                            )
                                continue

                            # Token length filtering (only if token filtering is enabled and early filtering is True)
                            if enable_token_filtering:
                                prompt_tokens = get_token_count_estimate(pr, tokenizer)
                                if prompt_tokens > max_input_tokens:
                                    inference_filtered_prompt += 1
                                    debug_print(
                                        "Inference: Filtering prompt too long "
                                        f"({prompt_tokens} > {max_input_tokens}) for task {task_key} "
                                        f"aug {aug_idx_loop}"
                                    )
                                    continue

                            valid_aug_tasks.append(aug)
                            valid_decs.append(dec)

                        if not valid_aug_tasks and augmented_task_dicts_list:
                            message = (
                                f"All {len(augmented_task_dicts_list)} augmentations for task {task_key} "
                                "were filtered out. No inference items for this task from this batch of augmentations."
                            )
                            debug_print(message)

                        inference_data_dict[task_key]["tasks"].extend(valid_aug_tasks)
                        inference_data_dict[task_key]["decs"].extend(valid_decs)

                        num_added = len(valid_aug_tasks)
                        total_augs_generated_airv += num_added
                        pbar.update(num_added)

                        verbose_print(f"Added {num_added}/{augs_for_this_task} planned tasks to {task_key}")

                        if num_added < augs_for_this_task:
                            debug_print(
                                "WARNING: "
                                f"{task_key} only got {num_added}/{augs_for_this_task} planned augmentations "
                                "(likely due to filtering)"
                            )
                    else:
                        debug_print(
                            "CRITICAL: get_n_augs returned empty list for "
                            f"{task_key} when {augs_for_this_task} were requested."
                        )
                        failed_tasks_airv.append(task_key)

            if failed_tasks_airv:
                sample = failed_tasks_airv[:5]
                suffix = "..." if len(failed_tasks_airv) > 5 else ""
                print(
                    "WARNING: "
                    f"{len(failed_tasks_airv)} tasks failed to generate any augmentations: {sample}{suffix}"
                )
            print(f"Generated {total_augs_generated_airv} total AIRV augmentations (target was {target_inference})")
        else:  # AIRV enabled but target_inference is 0 or less
            print(
                "  AIRV enabled but target_inference "
                f"({target_inference}) is not positive. No AIRV inference items generated."
            )
            # inference_data_dict is already initialized with empty lists.

    if ttt_items:
        shuffle(ttt_items)

    num_inference_augs = sum(len(d["tasks"]) for d in inference_data_dict.values())
    print(f"Generated {len(ttt_items)} TTT items and {num_inference_augs} inference items (prompts).")

    if ttt_total_generated > 0:
        ttt_kept_perc = (len(ttt_items) / ttt_total_generated * 100) if ttt_total_generated > 0 else 0
        if enable_token_filtering:
            print(
                "TTT Token Filtering: "
                f"{ttt_total_generated} generated, {ttt_filtered_prompt} filtered (prompt), "
                f"{ttt_filtered_answer} filtered (answer)"
            )
            print(f"TTT Final: {len(ttt_items)}/{ttt_total_generated} items kept ({ttt_kept_perc:.1f}%)")
        else:
            print(f"TTT Generation: {len(ttt_items)} items generated (token filtering disabled)")

    if inference_total_generated > 0:
        inf_kept_perc = (num_inference_augs / inference_total_generated * 100) if inference_total_generated > 0 else 0
        if enable_token_filtering:
            print(
                "Inference Token Filtering: "
                f"{inference_total_generated} generated, {inference_filtered_prompt} filtered (prompt)"
            )
            print(
                f"Inference Final: {num_inference_augs}/{inference_total_generated} items kept ({inf_kept_perc:.1f}%)"
            )
        else:
            print(f"Inference Generation: {num_inference_augs} items generated (token filtering disabled)")

    from .config import DEBUG_MODE  # Re-import to ensure it's up-to-date

    if DEBUG_MODE and ttt_items:
        import os

        import pandas as pd

        # Handle both old tuple format and new dict format for debug output
        if ttt_items and isinstance(ttt_items[0], dict):
            ttt_data = [
                {"key": f"ttt_item_{i}", "prompt": item["prompt"], "correct_answer": item["target"]}
                for i, item in enumerate(ttt_items)
            ]
        else:
            ttt_data = [
                {"key": f"ttt_item_{i}", "prompt": p, "correct_answer": a} for i, (p, a) in enumerate(ttt_items)
            ]
        ttt_df = pd.DataFrame(ttt_data)
        ttt_csv_path = "/kaggle/working/debug_ttt_dataset.csv" if os.path.exists("/kaggle") else "debug_ttt_dataset.csv"
        ttt_df.to_csv(ttt_csv_path, index=False)
        print(f"[DEBUG] TTT dataset written to {ttt_csv_path} ({len(ttt_data)} items)")

    if DEBUG_MODE and inference_data_dict:
        import os

        import pandas as pd

        inference_debug_data = []
        for task_key, data in inference_data_dict.items():
            for i, (aug_task, decoder) in enumerate(zip(data.get("tasks", []), data.get("decs", []), strict=False)):
                try:
                    prompt = _format_prompt(makeprompt(aug_task, style=prompt_style))
                    inference_debug_data.append(
                        {"key": f"{task_key}_aug_{i}", "prompt": prompt, "decoder": str(decoder)}
                    )
                except Exception as e:
                    inference_debug_data.append(
                        {"key": f"{task_key}_aug_{i}", "prompt": f"[ERROR_PROMPT]: {e}", "decoder": str(decoder)}
                    )

        if inference_debug_data:
            inference_df = pd.DataFrame(inference_debug_data)
            inf_csv_path = (
                "/kaggle/working/debug_inference_dataset.csv"
                if os.path.exists("/kaggle")
                else "debug_inference_dataset.csv"
            )
            inference_df.to_csv(inf_csv_path, index=False)
            print(f"[DEBUG] Inference dataset written to {inf_csv_path} ({len(inference_debug_data)} items)")

    debug_print("Final inference_data_dict structure check:")
    for key_check, data_check in list(inference_data_dict.items())[:2]:  # Check first 2 tasks
        task_count = len(data_check.get("tasks", []))
        decoder_count = len(data_check.get("decs", []))
        debug_print(f"  Task {key_check}: {task_count} tasks, {decoder_count} decoders")
        if task_count == 0 and key_check in train_data:  # If it's a valid key but no tasks
            original_task_debug = train_data.get(key_check, {})
            train_ex_count = len(original_task_debug.get("train", []))
            test_ex_count = len(original_task_debug.get("test", []))
            debug_print(
                f"    Original task has {train_ex_count} train examples, {test_ex_count} test examples. Filtered out?"
            )

    return ttt_items, inference_data_dict, augmentation_stats


def prep_ttt_dataset(ttt_items: list[dict], test_size: float):
    """Converts TTT items to HuggingFace datasets."""
    # Import Dataset locally to avoid JAX initialization at module level
    from datasets import Dataset

    if not ttt_items:
        empty_ds_dict = {"prompt": [], "correct_answer": []}
        empty_dataset = Dataset.from_dict(empty_ds_dict)
        return empty_dataset, empty_dataset

    # Handle both old tuple format and new dict format for backward compatibility
    if ttt_items and isinstance(ttt_items[0], dict):
        dataset_dict = {
            "prompt": [item["prompt"] for item in ttt_items],
            "correct_answer": [item["target"] for item in ttt_items],
        }
        if any("task_key" in item for item in ttt_items):
            dataset_dict["task_key"] = [item.get("task_key", "") for item in ttt_items]
        if any("augmentation" in item for item in ttt_items):
            dataset_dict["augmentation"] = [item.get("augmentation") for item in ttt_items]
    else:
        # Legacy tuple format
        dataset_dict = {"prompt": [item[0] for item in ttt_items], "correct_answer": [item[1] for item in ttt_items]}
    full_dataset = Dataset.from_dict(dataset_dict)

    if len(full_dataset) > 1 and 0 < test_size < 1:
        try:
            split_datasets = full_dataset.train_test_split(test_size=test_size, seed=42)
            return split_datasets["train"], split_datasets["test"]
        except Exception as e:
            print(f"Warning: Dataset split failed: {e}. Using full dataset for training, empty for testing.")
            empty_ds_dict = {"prompt": [], "correct_answer": []}
            return full_dataset, Dataset.from_dict(empty_ds_dict)

    empty_ds_dict = {"prompt": [], "correct_answer": []}
    return full_dataset, Dataset.from_dict(empty_ds_dict)
