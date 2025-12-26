from __future__ import annotations

import copy

from .salesforce_codet5_large import DEFAULT_ACTIVE as BASE_DEFAULT_ACTIVE
from .salesforce_codet5_large import MODEL_CONFIG as BASE_MODEL_CONFIG
from .salesforce_codet5_large import MODEL_PATH as BASE_MODEL_PATH


MODEL_PATH = BASE_MODEL_PATH
DEFAULT_ACTIVE = BASE_DEFAULT_ACTIVE

MODEL_CONFIG = copy.deepcopy(BASE_MODEL_CONFIG)

hf_pretrain_cfg = {
    "enabled": True,
    "dataset_name": "mindware/arc-agi-mega",
    "split": "train",
    "data_root": "data/train",
    "arc_source": "arc_mindsai",  # switch to 'arc_json' to use canonical JSON puzzles
    "non_arc_fraction": 0.25,  # 25% of samples from non-ARC mindsai partition by default
    "streaming": True,
    "shuffle_buffer_size": 4096,
    "arc_mindsai": {
        "data_dir": "data/train/arc_mindsai",
    },
    "non_arc_mindsai": {
        "data_dir": "data/train/non_arc_mindsai",
    },
    "arc_json": {
        "data_dir": "data/train/arc_json",
        "apply_augs": False,
        "variants_per_task": 1,
        "enable_color_shift": True,
        "prompt_format": "legacy",
    },
}

MODEL_CONFIG.setdefault("training", {})
MODEL_CONFIG["training"]["hf_pretraining"] = hf_pretrain_cfg
