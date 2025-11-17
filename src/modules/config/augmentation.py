from __future__ import annotations

from typing import Any

# AUGMENTATION_CONFIG: dict[str, Any] = {
#     "duplicate": {
#         "enabled": True,
#         "reversible": False,
#         "use_for_ttt": True,
#         "use_for_inference": True,
#         "weight": 1.0,
#         "description": "Duplicate the item",
#     },
#     # "color": {
#     #     "enabled": [True, False],
#     #     "reversible": True,
#     #     "use_for_ttt": True,
#     #     "use_for_inference": True,
#     #     "weight": 1.0,
#     #     "description": "Combined geometric + color augmentation",
#     # },    
# }


AUGMENTATION_CONFIG: dict[str, Any] = {
    "geometric_color": {
        "enabled": True,
        "reversible": True,
        "use_for_ttt": True,
        "use_for_inference": True,
        "weight": 0.6,
        # Optional specialized weights per use-case (fallback to 'weight')
        # 'weight_ttt': 1.0,
        # 'weight_inference': 1.0,
        # Optional: apply another augmentation before this one (e.g., 'color')
        # 'aug_before_with': 'color',
        "description": "Combined geometric + color augmentation",
    },
    "geometric": {
        "enabled": False,
        "reversible": True,
        "use_for_ttt": True,
        "use_for_inference": True,
        "weight": 0.4,
        "description": "Pure geometric transforms without color remapping",
    },
    "color": {
        "enabled": False,
        "reversible": True,
        "use_for_ttt": True,
        "use_for_inference": True,
        "weight": 0.4,
        "description": "Color permutation augmentation only",
    },
    "mixup": {
        "enabled": True,
        "reversible": False,
        "use_for_ttt": True,
        "use_for_inference": False,
        "weight": 0.2,
        # 'weight_ttt': 1.0,
        "aug_before_with": "geometric_color",
        # Overlay mode: 'replace_nonzero' or 'add_mod'
        "overlay_mode": "add_mod",
        # Optional color shift applied to the smaller (foreground) task before overlay
        "color_shift_increment": 1,
        "description": "Mixup augmentation with aligned overlay",
    },
    "input_output_swap": {
        "enabled": True,
        "reversible": False,
        "use_for_ttt": True,
        "use_for_inference": False,
        "weight": 0.1,
        # 'weight_ttt': 0.2,
        "aug_before_with": "geometric_color",
        "swap_probability": 0.3,
        "description": "Input-output swap augmentation",
    },
    "combine": {
        "enabled": True,  # Enable/disable via env_mod.COMBINE_AUGMENTATION
        "reversible": True,
        "use_for_ttt": True,
        "use_for_inference": False,
        # 'weight': 0.1,
        "weight_ttt": 0.1,
        "weight_inference": 0.1,
        "aug_before_with": "geometric_color",
        # Allow up to 4 to enable 2x2 grid combinations
        "max_tasks_to_combine": 2,
        "test_board_combination_methods": [
            "horizontal",
            "vertical",
            "color_separator_horizontal",
            "color_separator_vertical",
            # Placeholder; resolved to 'grid_RxC' at runtime (e.g., grid_2x2)
            "grid",
        ],
        # Ensure each subsequent combined board is color-shifted by +1 mod 10
        # relative to the previous board to keep a consistent pattern.
        "enforce_sequential_color_shift": True,
        "color_shift_increment": 1,
        "augment_train_pairs": True,
        "description": "Combine with augmented boards of the same task",
    },
    "mixup_combine": {
        "enabled": True,
        "reversible": True,
        "use_for_ttt": True,
        "use_for_inference": False,
        "weight_ttt": 0.05,
        "weight_inference": 0.05,
        # Optionally apply a pre-augmentation to the base task before mixing
        "aug_before_with": "geometric_color",
        # Select how many distinct tasks to concatenate (>=2)
        "max_tasks_to_combine": 2,
        "test_board_combination_methods": [
            "horizontal",
            "vertical",
            "color_separator_horizontal",
            "color_separator_vertical",
            # Placeholder; resolved to 'grid_RxC' at runtime (e.g., grid_2x2)
            "grid",
        ],
        # Require truly distinct tasks (by test input signature). If not enough
        # distinct tasks exist in the pool, skip generating this variant.
        "require_distinct_tasks": True,
        "allow_duplicates_if_insufficient": False,
        # Maintain reversibility by ensuring each subsequent task is color-shifted.
        "enforce_sequential_color_shift": True,
        "color_shift_increment": 1,
        # Build combined training pairs across selected tasks
        "augment_train_pairs": True,
        "description": "Combine multiple different tasks side-by-side (reversible)",
    },
}


# GRID_AUGMENTATION_CONDITIONS: list[dict[str, Any]] = [
#     {
#         "label": "current_settings_plus_input_output_swap",
#         "description": "Baseline config (geometric_color + mixup + combine + mixup_combine) with swap enabled.",
#         "toggles": {
#             "geometric_color": True,
#             "geometric": False,
#             "color": False,
#             "mixup": True,
#             "combine": True,
#             "mixup_combine": True,
#             "input_output_swap": True,
#         },
#     },
#     {
#         "label": "geometric_color_only",
#         "description": "Only the combined geometric+color augmentation is active.",
#         "toggles": {
#             "geometric_color": True,
#         },
#     },
#     {
#         "label": "geometric_only",
#         "description": "Only pure geometric transforms are active.",
#         "toggles": {
#             "geometric": True,
#         },
#     },
#     {
#         "label": "color_only",
#         "description": "Only pure color remapping is active.",
#         "toggles": {
#             "color": True,
#         },
#     },
#     {
#         "label": "geometric_color_plus_mixup",
#         "description": "Combined geometric+color plus mixup overlay.",
#         "toggles": {
#             "geometric_color": True,
#             "mixup": True,
#         },
#     },
#     {
#         "label": "geometric_color_plus_combine",
#         "description": "Combined geometric+color plus task combination.",
#         "toggles": {
#             "geometric_color": True,
#             "combine": True,
#         },
#     },
#     {
#         "label": "geometric_color_plus_mixup_combine",
#         "description": "Combined geometric+color plus mixup_combine augmentation.",
#         "toggles": {
#             "geometric_color": True,
#             "mixup_combine": True,
#         },
#     },
# ]

