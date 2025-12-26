from __future__ import annotations

from typing import Any

from .. import env as env_mod
from ..model_common import (
    MIN_VOTES_AMBIGUOUS,
    MIN_VOTES_SINGLE_PRED,
    Z_SCORE_THRESHOLD,
)


MODEL_PATH = "mindware/arc-codet5-small"
DEFAULT_ACTIVE = True

_LIMITED_TTT = int(env_mod.GLOBAL_TTT_SAMPLES or 512)
_LIMITED_INF = int(env_mod.GLOBAL_INF_SAMPLES or 256)


MODEL_CONFIG: dict[str, Any] = {
    "general": {
        "model_type": "seq2seq",
        "airv_enabled": True,
        "ttt_enabled": True,
        "max_input_length": 2500,
        "max_target_length": 600,
        "max_generation_length": 600,
        "enable_token_filtering": True,
        "prompt_settings": {
            "general_prefix": "",
        },
    },
    "training": {
        "self_ensemble_count": max(1, int(env_mod.SE or 1)),
        "task_group_size": None,
        "model_averaging": False,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,
        "use_bf16": True,
        "use_fp16": False,
        "train_batch_size": 5,
        "mini_lr_grid": {
            "enabled": False,
        },
        "target_total_ttt_items": _LIMITED_TTT,
        "logging_steps": 5,
        "warmup_ratio": 0.1,
        "dataloader_num_workers": 0,
        "use_torch_compile": False,
        "label_smoothing": 0.0,
        "pretrain_file": None,
        "pretrain_dir": None,
        "pretraining_examples": env_mod.PRETRAINING_EXAMPLES_OVERRIDE,
        "pretraining_steps": env_mod.PRETRAINING_STEPS_OVERRIDE,
        "lora": {
            "enabled": False,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "task_type": "SEQ_CLS",
        },
        "early_stopping": {
            "enabled": False,
            "accuracy_pct_target": 99.0,
        },
    },
    "inference": {
        "target_total_inference_items": _LIMITED_INF,
        "eval_batch_size": 8,
        "num_beams": 2,
        "num_return_sequences": 2,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.0,
        "z_score_filtering_threshold": Z_SCORE_THRESHOLD,
        "min_votes_for_single_pred_filter": MIN_VOTES_SINGLE_PRED,
        "min_votes_for_ambiguous_filter": MIN_VOTES_AMBIGUOUS,
        "use_mixed_precision_inference": False,
        "preview_interval_batches": 10,
        "preview_max_samples": 1,
        "preview_show_terminal_grid": True,
        "preview_snippet_chars": 160,
        "oom_reduction_factor": 0.5,
        "min_batch_size": 1,
        "max_oom_retries": 2,
    },
    "ensembling": {
        "enable_self_ensemble": True,
        "enable_model_ensemble": False,
        "enable_run_tracking": True,
        "iterative_ensemble_mode": False,
        "airv_last_cycle_only": False,
    },
}
