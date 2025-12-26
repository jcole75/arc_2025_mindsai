from __future__ import annotations

import os
from typing import Any

from .. import env as env_mod
from ..model_common import (
    MIN_VOTES_AMBIGUOUS,
    MIN_VOTES_SINGLE_PRED,
    Z_SCORE_THRESHOLD,
)


MODEL_PATH = "Salesforce/codet5-large"
DEFAULT_ACTIVE = False

MODEL_CONFIG: dict[str, Any] = {
    "general": {
        "model_type": "seq2seq",
        "airv_enabled": True,
        "ttt_enabled": True,
        "max_input_length": 2900,
        "max_target_length": 600,
        "max_generation_length": 600,
        "enable_token_filtering": True,
        "tokenizer_dropout": {
            "train": {"enabled": True, "rate": 0.2, "apply_to_labels": False},
            "inference": {"enabled": True, "rate": 0.1},
        },
    },
    "training": {
        "self_ensemble_count": env_mod.SE,
        "task_group_size": env_mod.GROUP_SIZE,
        "model_averaging": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.03,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "use_bf16": True,
        "use_fp16": False,
        "train_batch_size": 1 if not env_mod.USE_FLAX else 3,
        "mini_lr_grid": {
            "enabled": True,  # Disable in TEST_MODE if sweep is too slow.
            "num_trials": 4,
            "min": 2e-6,
            "max": 5e-4,
            "spacing": "log",
            "metric": ["accuracy_pct_max", "loss_train_min", "loss_train_final"],
            "metric_mode": "max",
            "ttt_items": 800,
            "max_steps": 15,
            "max_steps_test_mode": 15,
            "interpolation": "quadratic",  # None
            "interpolation_verify": True,
            "interpolation_min_rel_distance": 0.05,
            "finalize_full_run": True,
        },
        "lora": {
            "enabled": False,
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "task_type": "seq2seq_lm",
            "target_modules": ["q", "v"],
            "modules_to_save": ["lm_head"],
        },
        "target_total_ttt_items": env_mod.TEST_MODE_TTT_ITEMS if env_mod.TEST_MODE else env_mod.GLOBAL_TTT_SAMPLES,
        "logging_steps": 10,
        "warmup_ratio": 0.0,
        "dataloader_num_workers": 0,
        "use_torch_compile": False,
        "label_smoothing": 0.0,
        # Pretraining controls (added for parity)
        "pretrain_file": None,
        "pretrain_dir": os.environ.get("PRETRAINING_DATA_DIR"),
        "pretraining_examples": env_mod.PRETRAINING_EXAMPLES_OVERRIDE,
        "pretraining_steps": env_mod.PRETRAINING_STEPS_OVERRIDE,
        "early_stopping": {
            "enabled": False,
            "accuracy_pct_target": 99.0,
        },
    },
    "inference": {
        "target_total_inference_items": env_mod.TEST_MODE_INFERENCE_ITEMS
        if env_mod.TEST_MODE
        else env_mod.GLOBAL_INF_SAMPLES,
        "eval_batch_size": 5 if not env_mod.USE_FLAX else 9,
        "num_beams": 4,
        "num_return_sequences": 4,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.0,
        "z_score_filtering_threshold": Z_SCORE_THRESHOLD,
        "min_votes_for_single_pred_filter": MIN_VOTES_SINGLE_PRED,
        "min_votes_for_ambiguous_filter": MIN_VOTES_AMBIGUOUS,
        "use_mixed_precision_inference": True,
        "preview_interval_batches": 5,
        "preview_max_samples": 1,
        "preview_show_terminal_grid": True,
        "preview_snippet_chars": 160,
        "oom_reduction_factor": 0.2,
        "min_batch_size": 1,
        "max_oom_retries": 5,
    },
    "ensembling": {
        "enable_self_ensemble": True,
        "enable_model_ensemble": True,
        "enable_run_tracking": True,
        "iterative_ensemble_mode": False,
        "airv_last_cycle_only": False,
    },
}
