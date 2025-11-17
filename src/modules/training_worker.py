# File: ./src/modules/training_worker.py
# !/usr/bin/env python3
"""Compact training worker: same features, fewer lines, clearer flow."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterable
from contextlib import nullcontext, suppress
import glob
import inspect
import json
import logging
import math
import os
import random
import signal
import sys
import time
import traceback
from typing import Any
import types
import warnings


sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=r"Trainer\\.tokenizer is now deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*assign=True.*", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import statistics  # noqa: E402

try:
    import numpy as np  # noqa: E402
except Exception:  # pragma: no cover - optional numpy dependency
    np = None  # type: ignore
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_flax_pytorch_utils import (  # noqa: E402
    load_flax_checkpoint_in_pytorch_model,
    load_flax_weights_in_pytorch_model,
)
from transformers.optimization import Adafactor  # noqa: E402
from transformers.trainer_callback import TrainerCallback  # noqa: E402

try:  # noqa: E402 - optional dependency for quantized fine-tuning
    import bitsandbytes as bnb  # type: ignore
except Exception:  # pragma: no cover - optional
    bnb = None  # type: ignore


_TRAINER_SUPPORTS_PROCESSING = "processing_class" in inspect.signature(Trainer.__init__).parameters
_SEQ2SEQ_TRAINER_SUPPORTS_PROCESSING = "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters

# ---------- local imports (module or script mode) ----------
try:
    from . import config
    from .config import env as config_env
    from .quantization_utils import build_bitsandbytes_kwargs
except Exception:
    from importlib import import_module
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    package_root = module_dir.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    package_name = module_dir.name

    config = import_module(f"{package_name}.config")
    config_env = config.env

    build_bitsandbytes_kwargs = import_module(f"{package_name}.quantization_utils").build_bitsandbytes_kwargs


_DEFAULT_OUTPUT_LABEL = "Output:"


def set_default_output_label(label: str | None) -> None:
    global _DEFAULT_OUTPUT_LABEL
    if label is None:
        _DEFAULT_OUTPUT_LABEL = "Output:"
    else:
        text = str(label).strip()
        _DEFAULT_OUTPUT_LABEL = text if text else "Output:"


def prepare_causal_prompt_with_target(
    prompt: str | None,
    target: str | None,
    *,
    output_label: str | None = None,
) -> tuple[str, str]:
    prompt_text = "" if prompt is None else str(prompt)
    target_text = "" if target is None else str(target)
    label = output_label if output_label is not None else _DEFAULT_OUTPUT_LABEL
    label = label if label else ""
    prefix = f"{prompt_text} {label}".strip()
    combined = f"{prefix} {target_text}".strip()
    return combined, prefix


class DreamDataCollator:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Dream training has been removed.")


def build_dream_training_features(*args, **kwargs):
    raise RuntimeError("Dream training has been removed.")


def compute_dream_loss(*args, **kwargs):
    raise RuntimeError("Dream training has been removed.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_SIGNAL_SECONDS = (1 << 31) - 1

# ---------- env helpers ----------
_ENV: dict[str, Any] = {}
if hasattr(config_env, "_ENV") and isinstance(config_env._ENV, dict):
    _ENV = {str(k): v for k, v in config_env._ENV.items()}


def _env_str(k: str, d: str | None = None) -> str | None:
    v = os.environ.get(k, _ENV.get(k))
    v = None if v is None else str(v).strip()
    return v if v else d


def _env_float(k: str) -> float | None:
    v = os.environ.get(k, _ENV.get(k))
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except Exception:
        logger.warning("Invalid float for %s: %s", k, v)
        return None


def _env_int(k: str, d: int | None = None) -> int | None:
    v = os.environ.get(k, _ENV.get(k))
    if v is None:
        return d
    if isinstance(v, str):
        txt = v.strip()
        if not txt:
            return d
        if txt.lower() in {"none", "null", "nan"}:
            return None
        try:
            return int(txt)
        except Exception:
            logger.warning("Invalid int for %s: %s", k, v)
            return d
    try:
        return int(v)
    except Exception:
        logger.warning("Invalid int for %s: %s", k, v)
        return d


def _cli_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return False
    return text not in {"0", "false", "no", "off"}


def _local_rank(default: int = 0) -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", default))
    except Exception:
        return default


# ---------- Muon helpers ----------
_MUON_EXCLUDE_DEFAULT = {"bias", "layernorm", "norm", "embedding", "embeddings", "lm_head"}


def _normalize_muon_exclusions(ex: Iterable[str] | str | None) -> set[str]:
    if ex is None:
        return set(_MUON_EXCLUDE_DEFAULT)
    toks = [
        t.strip().lower()
        for t in (ex.replace(";", ",").split(",") if isinstance(ex, str) else map(str, ex))
        if t.strip()
    ]
    return set([] if "none" in toks else toks).union(_MUON_EXCLUDE_DEFAULT)


def _is_muon_param(name: str, p: torch.nn.Parameter, excl: set[str]) -> bool:
    if p is None or not p.requires_grad or p.ndim < 2:
        return False
    if p.shape[-1] <= 1 or p.shape[-2] <= 1:
        return False
    ln = name.lower()
    return not any(k in ln for k in excl)


def _split_param_groups(
    trainer, use_muon: bool, excl: set[str]
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay_names = trainer.get_decay_parameter_names(trainer.model)
    muon, dec, nodec = [], [], []
    for n, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if use_muon and _is_muon_param(n, p, excl):
            muon.append(p)
            continue
        (dec if n in decay_names else nodec).append(p)
    return muon, dec, nodec


def _ensure_generation_config_compat(model) -> None:
    """Ensure generation configs tolerate HF strict validation during save."""

    def _patch_config(gen_cfg) -> None:
        if gen_cfg is None:
            return
        cls = gen_cfg.__class__
        if getattr(cls, "_arc_validates_strict", False):
            return
        validate = getattr(cls, "validate", None)
        if not callable(validate):
            return
        try:
            sig = inspect.signature(validate)
        except Exception:
            return
        if "strict" in sig.parameters:
            return

        original_validate = validate

        def _wrapped(self, *args, **kwargs):
            kwargs.pop("strict", None)
            return original_validate(self, *args, **kwargs)

        cls.validate = _wrapped
        cls._arc_validates_strict = True

    visited: set[int] = set()

    def _visit(obj) -> None:
        if obj is None:
            return
        ident = id(obj)
        if ident in visited:
            return
        visited.add(ident)
        _patch_config(getattr(obj, "generation_config", None))
        for attr in ("base_model", "model"):
            _visit(getattr(obj, attr, None))

    _visit(model)


def _mk_adamw(trainer, dec, nodec):
    groups = []
    if dec:
        groups.append({"params": list(dec), "weight_decay": trainer.args.weight_decay})
    if nodec:
        groups.append({"params": list(nodec), "weight_decay": 0.0})
    if not groups:
        groups = [
            {
                "params": [p for p in trainer.model.parameters() if p.requires_grad],
                "weight_decay": trainer.args.weight_decay,
            }
        ]

    use_bnb = bool(getattr(trainer.args, "bitsandbytes_is_4bit", False) or getattr(trainer.args, "bitsandbytes_is_8bit", False))
    if use_bnb and bnb is not None:
        try:
            if getattr(trainer.args, "bitsandbytes_is_4bit", False):
                OptimCls = getattr(bnb.optim, "PagedAdamW32bit")
            else:
                OptimCls = getattr(bnb.optim, "AdamW8bit")
        except AttributeError:
            OptimCls = None
        if OptimCls is not None:
            return OptimCls(
                groups,
                lr=trainer.args.learning_rate,
                betas=(0.9, 0.999),
                eps=trainer.args.adam_epsilon,
                weight_decay=trainer.args.weight_decay,
            )

    return AdamW(
        groups,
        lr=trainer.args.learning_rate,
        betas=(0.9, 0.999),  # Standard AdamW betas
        eps=trainer.args.adam_epsilon,
        weight_decay=trainer.args.weight_decay,
    )


def _mk_adafactor(trainer, dec, nodec):
    groups = []
    if dec:
        groups.append({"params": list(dec), "weight_decay": trainer.args.weight_decay})
    if nodec:
        groups.append({"params": list(nodec), "weight_decay": 0.0})
    if not groups:
        groups = [
            {
                "params": [p for p in trainer.model.parameters() if p.requires_grad],
                "weight_decay": trainer.args.weight_decay,
            }
        ]
    return Adafactor(
        groups, lr=trainer.args.learning_rate, relative_step=False, scale_parameter=False, warmup_init=False
    )


def _mk_muon(trainer, muon_params, dec, nodec, muon_lr: float, muon_mom: float):
    if not muon_params:
        raise ValueError("Muon requested but no eligible parameters found.")
    try:
        import muon as _muon_mod
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    except Exception as e:
        raise ImportError("Install muon-optimizer: `pip install muon-optimizer`.") from e
    betas = (0.9, 0.999)
    groups = []
    if dec:
        groups.append(
            {
                "params": list(dec),
                "lr": trainer.args.learning_rate,
                "betas": betas,
                "eps": trainer.args.adam_epsilon,
                "weight_decay": trainer.args.weight_decay,
                "use_muon": False,
            }
        )
    if nodec:
        groups.append(
            {
                "params": list(nodec),
                "lr": trainer.args.learning_rate,
                "betas": betas,
                "eps": trainer.args.adam_epsilon,
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    groups.append(
        {
            "params": list(muon_params),
            "lr": float(muon_lr),
            "momentum": float(muon_mom),
            "weight_decay": trainer.args.weight_decay,
            "use_muon": True,
        }
    )
    dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    base_cls = MuonWithAuxAdam if dist else SingleDeviceMuonWithAuxAdam

    class _AccelerateFriendlyMuon(base_cls):
        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            dist_ready = (
                base_cls is MuonWithAuxAdam and torch.distributed.is_available() and torch.distributed.is_initialized()
            )

            # muon<=0.1.0 has incorrect padding logic for distributed all_gather when the
            # parameter count is not divisible by world size; fall back to the original
            # step if torch.distributed is unavailable or we are in single-device mode.
            if not dist_ready:
                res = super().step()
                return res if res is not None else loss

            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            for group in self.param_groups:
                if group.get("use_muon", False):
                    params = group["params"]
                    if not params:
                        continue
                    pad_count = (-len(params)) % world_size
                    pad_tensor = params[-1].detach()
                    params_pad = list(params)
                    if pad_count:
                        params_pad.extend(pad_tensor.new_empty(pad_tensor.shape) for _ in range(pad_count))
                    for base_i in range(0, len(params_pad), world_size):
                        if base_i + rank < len(params):
                            p = params[base_i + rank]
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)
                            update = _muon_mod.muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(update, alpha=-group["lr"])
                        torch.distributed.all_gather(
                            params_pad[base_i : base_i + world_size],
                            params_pad[base_i + rank],
                        )
                else:
                    _beta1, _beta2 = group["betas"]
                    for p in group["params"]:
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = _muon_mod.adam_update(
                            p.grad,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["step"],
                            group["betas"],
                            group["eps"],
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])

            return loss

    return _AccelerateFriendlyMuon(groups)


# ---------- loss helpers ----------
def _token_ce_loss(logits, labels, label_smoothing=0.0):
    V = logits.size(-1)
    f = nn.CrossEntropyLoss(ignore_index=-100, reduction="none", label_smoothing=label_smoothing)
    return f(logits.view(-1, V), labels.view(-1)).view_as(labels)


class LearningRateResetCallback(TrainerCallback):
    """Periodically reset optimizer LR and scheduler based on processed examples."""

    def __init__(self, reset_examples: int):
        self.reset_examples = max(0, int(reset_examples or 0))
        self.examples_since_reset = 0
        self.initial_lrs: list[float] | None = None
        self.scheduler_state: dict[str, Any] | None = None

    def _ensure_initial_state(self, trainer) -> None:
        if self.initial_lrs is None and getattr(trainer, "optimizer", None) is not None:
            self.initial_lrs = [group.get("lr", 0.0) for group in trainer.optimizer.param_groups]
        if self.scheduler_state is None and getattr(trainer, "lr_scheduler", None) is not None:
            self.scheduler_state = copy.deepcopy(trainer.lr_scheduler.state_dict())

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: D401, ANN001
        if self.reset_examples <= 0:
            return
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._ensure_initial_state(trainer)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401, ANN001
        if self.reset_examples <= 0:
            return
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        self._ensure_initial_state(trainer)
        if self.initial_lrs is None:
            return
        try:
            examples_per_step = int(getattr(trainer.args, "train_batch_size", 0))
        except Exception:
            per_device = getattr(trainer.args, "per_device_train_batch_size", 1)
            grad_accum = getattr(trainer.args, "gradient_accumulation_steps", 1)
            world = getattr(trainer.args, "world_size", 1)
            examples_per_step = int(per_device) * int(grad_accum) * max(1, int(world))
        examples_per_step = max(1, examples_per_step)
        self.examples_since_reset += examples_per_step
        if self.examples_since_reset < self.reset_examples:
            return
        for group, base_lr in zip(trainer.optimizer.param_groups, self.initial_lrs):
            group["lr"] = float(base_lr)
            if "initial_lr" in group:
                group["initial_lr"] = float(base_lr)
        if self.scheduler_state is not None and getattr(trainer, "lr_scheduler", None) is not None:
            trainer.lr_scheduler.load_state_dict(copy.deepcopy(self.scheduler_state))
        logger.info(
            "Learning rate reset after %d examples; current lr=%s",
            self.reset_examples,
            [round(group.get("lr", 0.0), 8) for group in trainer.optimizer.param_groups],
        )
        self.examples_since_reset = 0


# ---------- Seq2Seq trainer ----------
class TokenLoggingMixin:
    """Track token throughput so Trainer logs surface accurate counters."""

    def _init_token_logging(self) -> None:
        if not hasattr(self, "_token_log_state"):
            self._token_log_state = {
                "input_tokens": 0.0,
                "target_tokens": 0.0,
                "examples": 0.0,
                "start_time": None,
                "last_update": None,
            }

    def _get_world_size_for_logging(self) -> int:
        try:
            return max(1, int(getattr(self.args, "world_size", 1)))  # type: ignore[attr-defined]
        except Exception:
            try:
                return max(1, int(os.environ.get("WORLD_SIZE", "1")))
            except Exception:
                return 1

    def _count_examples(self, inputs: dict[str, Any]) -> float:
        ids = inputs.get("input_ids")
        if isinstance(ids, torch.Tensor):
            return float(ids.size(0))
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            return float(labels.size(0))
        return 0.0

    def _count_input_tokens(self, inputs: dict[str, Any]) -> float:
        mask = inputs.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            return float(mask.sum().item())
        ids = inputs.get("input_ids")
        if isinstance(ids, torch.Tensor):
            pad_id = None
            tok = getattr(self, "tokenizer", None)
            if tok is not None:
                pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is None:
                cfg = getattr(getattr(self, "model", None), "config", None)
                pad_id = getattr(cfg, "pad_token_id", None)
            if pad_id is not None:
                return float(ids.ne(pad_id).sum().item())
            return float(ids.numel())
        return 0.0

    def _count_target_tokens(self, inputs: dict[str, Any]) -> float:
        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            return float(labels.ne(-100).sum().item())
        return 0.0

    def _update_token_counters(self, inputs: dict[str, Any]) -> None:
        try:
            self._init_token_logging()
            state = self._token_log_state  # type: ignore[attr-defined]
            now = time.time()
            if state["start_time"] is None:
                state["start_time"] = now
            state["last_update"] = now
            state["input_tokens"] += self._count_input_tokens(inputs)
            state["target_tokens"] += self._count_target_tokens(inputs)
            state["examples"] += self._count_examples(inputs)
        except Exception:
            pass

    def get_token_statistics(self) -> dict[str, float] | None:
        state = getattr(self, "_token_log_state", None)
        if not state:
            return None
        world = float(self._get_world_size_for_logging())
        return {
            "input_tokens": float(state.get("input_tokens", 0.0)) * world,
            "target_tokens": float(state.get("target_tokens", 0.0)) * world,
            "examples": float(state.get("examples", 0.0)) * world,
            "start_time": state.get("start_time"),
            "last_update": state.get("last_update"),
        }

    def _augment_logs_with_token_metrics(self, logs: dict[str, Any]) -> dict[str, Any]:
        stats = self.get_token_statistics()
        if not stats:
            return logs
        updates: dict[str, Any] = {}
        inputs_total = stats.get("input_tokens")
        targets_total = stats.get("target_tokens")
        examples_total = stats.get("examples")
        if inputs_total and inputs_total > 0:
            updates["num_input_tokens_seen"] = int(inputs_total)
        if targets_total and targets_total > 0:
            updates["num_target_tokens_seen"] = int(targets_total)
        if examples_total and examples_total > 0:
            updates["train_examples_seen"] = int(examples_total)
        start, end = stats.get("start_time"), stats.get("last_update")
        if inputs_total and inputs_total > 0 and start and end and end > start:
            updates["train_tokens_per_second"] = float(inputs_total) / max(end - start, 1e-6)
        if updates:
            logs.update(updates)
        return logs


class EnhancedSeq2SeqTrainer(TokenLoggingMixin, Seq2SeqTrainer):
    def __init__(
        self,
        label_smoothing=0.0,
        max_grad_norm=1.0,
        optimizer_choice="adamw",
        muon_learning_rate: float | None = None,
        muon_momentum=0.95,
        muon_exclude_keywords: Iterable[str] | str | None = None,
        **kw,
    ):
        super().__init__(**kw)
        self.label_smoothing = float(label_smoothing)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer_choice = (optimizer_choice or "adamw").lower()
        self.muon_learning_rate = muon_learning_rate
        self.muon_momentum = float(muon_momentum)
        self.muon_exclude_keywords = _normalize_muon_exclusions(muon_exclude_keywords)
        self._init_token_logging()

    def compute_loss(self, model, inputs, return_outputs=False, *a, **k):
        keep = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "labels", "decoder_input_ids"}}
        labels = keep.get("labels")
        if "decoder_input_ids" not in keep and labels is not None:
            base_model = getattr(model, "module", model)
            prepare_ids = getattr(base_model, "prepare_decoder_input_ids_from_labels", None)
            if callable(prepare_ids):
                with torch.no_grad():
                    keep["decoder_input_ids"] = prepare_ids(labels=labels)
            else:
                shift = getattr(base_model, "_shift_right", None)
                if callable(shift):
                    keep["decoder_input_ids"] = shift(labels)
        out = model(**{k: v for k, v in keep.items() if k != "labels"})
        logits = out.logits
        if labels is None:
            return super().compute_loss(model, keep, return_outputs, *a, **k)
        per_tok = _token_ce_loss(logits, labels, self.label_smoothing)
        valid_mask = labels.ne(-100)
        valid_weights = valid_mask.to(per_tok.dtype)
        valid_count = valid_weights.sum()
        if bool(valid_mask.any()):
            loss = (per_tok * valid_weights).sum() / valid_count.clamp(min=1.0)
        else:
            loss = per_tok.mean()
        if return_outputs:
            out.loss = loss
            return (loss, out)
        return loss

    def log(self, logs: dict, *a, **k):  # type: ignore[override]
        L = dict(logs or {})
        L = self._augment_logs_with_token_metrics(L)
        last_top2 = getattr(self, "last_top2_score", None)
        if last_top2 is not None:
            L.setdefault("prev_top2_score", float(last_top2))
            L.setdefault("prev_top2_pct", float(last_top2) * 100.0)
        super().log(L, *a, **k)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        use_muon = self.optimizer_choice in {"muon", "adamuon"}
        muon, dec, nodec = _split_param_groups(self, use_muon, self.muon_exclude_keywords)
        if use_muon:
            lr = self.muon_learning_rate if self.muon_learning_rate is not None else self.args.learning_rate
            self.optimizer = _mk_muon(self, muon, dec, nodec, lr, self.muon_momentum)
        elif self.optimizer_choice == "adafactor":
            self.optimizer = _mk_adafactor(self, dec, nodec)
        else:
            self.optimizer = _mk_adamw(self, dec, nodec)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            ws = int(num_training_steps * self.args.warmup_ratio)
            self.lr_scheduler = get_linear_schedule_with_warmup(optimizer or self.optimizer, ws, num_training_steps)
        return self.lr_scheduler

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        self._update_token_counters(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        if self.max_grad_norm > 0:
            # Apply gradient clipping
            if hasattr(self.accelerator, "clip_grad_norm_"):
                self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        return loss.detach() / self.args.gradient_accumulation_steps


def _retie_embeddings(m):
    try:
        if hasattr(m, "tie_weights"):
            m.tie_weights()
        shared = getattr(m, "shared", None)
        for emb in [
            getattr(getattr(m, "encoder", None), "embed_tokens", None),
            getattr(getattr(m, "decoder", None), "embed_tokens", None),
        ]:
            if not emb or getattr(getattr(emb, "weight", None), "is_meta", False):
                continue
            if (
                shared is not None
                and not getattr(shared.weight, "is_meta", False)
                and emb.weight.data_ptr() != shared.weight.data_ptr()
            ):
                emb.weight = shared.weight
    except Exception as e:
        print(f"Retie embeddings warn: {e}", flush=True)


_MODEL_FILE_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model-*-of-*.bin",
    "pytorch_model.safetensors",
    "pytorch_model-*.safetensors",
    "model.safetensors",
    "model-*.safetensors",
    "model.bin",
    "flax_model.msgpack",
)

_TOKENIZER_FILE_CANDIDATES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


def _directory_has_model_weights(directory: str | None) -> bool:
    if not directory or not os.path.isdir(directory):
        return False
    return any(glob.glob(os.path.join(directory, pattern)) for pattern in _MODEL_FILE_PATTERNS)


def _directory_has_tokenizer(directory: str | None) -> bool:
    if not directory or not os.path.isdir(directory):
        return False
    return any(os.path.exists(os.path.join(directory, candidate)) for candidate in _TOKENIZER_FILE_CANDIDATES)


def _resolve_peft_adapter_source(path: str) -> tuple[str | None, dict[str, Any] | None]:
    if not path or not os.path.isdir(path):
        return None, None
    adapter_cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(adapter_cfg_path):
        return None, None
    if _directory_has_model_weights(path):
        return None, None
    has_adapter_weights = any(
        os.path.exists(os.path.join(path, candidate)) for candidate in ("adapter_model.bin", "adapter_model.safetensors")
    )
    if not has_adapter_weights:
        return None, None
    try:
        with open(adapter_cfg_path, "r") as f:
            adapter_cfg = json.load(f)
    except Exception as exc:
        print(f"Adapter config read warn ({path}): {exc}", flush=True)
        return None, None
    base_model = adapter_cfg.get("base_model_name_or_path")
    if not base_model or not str(base_model).strip():
        print(f"Adapter config missing base_model_name_or_path at {adapter_cfg_path}", flush=True)
        return None, None
    return str(base_model).strip(), adapter_cfg


def _should_from_flax(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    torch_files = ("pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors", "model.bin")
    has_torch = any(os.path.exists(os.path.join(path, f)) for f in torch_files)
    return (not has_torch) and os.path.exists(os.path.join(path, "flax_model.msgpack"))


def _load_flax_params_into(model, model_id: str, load_fn):
    os.environ.setdefault("ACCELERATE_DISABLE_DEVICE_MAP", "1")

    flax_checkpoint = None
    if os.path.isdir(model_id):
        candidate = os.path.join(model_id, "flax_model.msgpack")
        if os.path.exists(candidate):
            flax_checkpoint = candidate

    if flax_checkpoint is None:
        try:
            from transformers.utils.hub import cached_file

            flax_checkpoint = cached_file(model_id, "flax_model.msgpack")
        except Exception:
            flax_checkpoint = None

    if flax_checkpoint and os.path.exists(flax_checkpoint):
        load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint)
        return model

    flax_model = load_fn(model_id)
    try:
        flax_params = getattr(flax_model, "params", None)
        if flax_params is None:
            raise RuntimeError("Flax model params unavailable for manual conversion")
        payload = flax_params if (hasattr(flax_params, "keys") and "params" in flax_params) else {"params": flax_params}
        load_flax_weights_in_pytorch_model(model, payload)
    finally:
        del flax_model

    return model


def _normalize_lora_list(items) -> list[str] | None:
    if not items:
        return None
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized or None


def _normalize_lora_layers(items) -> list[int] | None:
    if not items:
        return None
    layers: list[int] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            layers.append(int(text))
        except Exception:
            raise ValueError(f"LoRA layer value '{text}' is not an integer.")
    return layers or None


def _parse_lora_rank_pattern(spec: str | None) -> dict[str, int] | None:
    if not spec:
        return None
    try:
        payload = json.loads(spec)
    except Exception as exc:
        raise ValueError(f"Failed to parse LoRA rank_pattern JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LoRA rank_pattern must decode to an object/dict.")
    pattern: dict[str, int] = {}
    for key, value in payload.items():
        if value is None:
            continue
        try:
            pattern[str(key)] = int(value)
        except Exception as exc:
            raise ValueError(f"LoRA rank for '{key}' must be an integer: {exc}") from exc
    return pattern or None


def _apply_lora_if_enabled(model, args):
    if getattr(args, "_lora_loaded_from_adapter", False):
        return model
    if not getattr(args, "use_lora", False):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    except ImportError as exc:
        raise ImportError("LoRA requested but the 'peft' package is not installed. Install via `pip install peft`.") from exc

    if getattr(args, "bitsandbytes_is_4bit", False):
        gc_flag = bool(getattr(args, "use_gradient_checkpointing", False))
        gc_kwargs = {"use_reentrant": False} if gc_flag else None
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gc_flag,
            gradient_checkpointing_kwargs=gc_kwargs,
        )

    target_modules = _normalize_lora_list(getattr(args, "lora_target_modules", None))
    modules_to_save = _normalize_lora_list(getattr(args, "lora_modules_to_save", None))
    layers_to_transform = _normalize_lora_layers(getattr(args, "lora_layers_to_transform", None))
    rank_pattern = _parse_lora_rank_pattern(getattr(args, "lora_rank_pattern", None))
    layers_pattern = getattr(args, "lora_layers_pattern", None)
    scaling = getattr(args, "lora_scaling", None)

    init_spec = getattr(args, "lora_init_lora_weights", None)
    init_lora_weights: bool | str | None
    if init_spec is None:
        init_lora_weights = None
    else:
        txt = str(init_spec).strip().lower()
        if txt in {"true", "1", "yes", "y"}:
            init_lora_weights = True
        elif txt in {"false", "0", "no", "n"}:
            init_lora_weights = False
        else:
            init_lora_weights = str(init_spec)

    if getattr(args, "lora_task_type", None):
        raw = str(args.lora_task_type).strip().upper()
        raw = raw.replace("-", "_")
        raw = raw.replace("SEQ2SEQ", "SEQ_2_SEQ")
        if not raw.endswith("_LM") and raw in {"SEQ_2_SEQ", "SEQ2SEQ"}:
            raw = "SEQ_2_SEQ_LM"
        try:
            task_type = TaskType[raw]
        except KeyError as exc:
            valid = ", ".join(sorted(t.name for t in TaskType))
            raise ValueError(f"Unsupported LoRA task_type '{args.lora_task_type}'. Valid options: {valid}") from exc
    else:
        task_type = TaskType.CAUSAL_LM if getattr(args, "is_causal_lm", False) else TaskType.SEQ_2_SEQ_LM

    lora_kwargs: dict[str, Any] = {
        "r": int(getattr(args, "lora_r", 8)),
        "lora_alpha": float(getattr(args, "lora_alpha", 16)),
        "lora_dropout": float(getattr(args, "lora_dropout", 0.0)),
        "bias": str(getattr(args, "lora_bias", "none")),
        "task_type": task_type,
    }
    if target_modules:
        lora_kwargs["target_modules"] = target_modules
    if modules_to_save:
        lora_kwargs["modules_to_save"] = modules_to_save
    if init_lora_weights is not None:
        lora_kwargs["init_lora_weights"] = init_lora_weights
    if rank_pattern:
        lora_kwargs["rank_pattern"] = rank_pattern
    if scaling is not None:
        lora_kwargs["scaling"] = scaling
    if layers_to_transform:
        lora_kwargs["layers_to_transform"] = layers_to_transform
    if layers_pattern:
        lora_kwargs["layers_pattern"] = layers_pattern
    if bool(getattr(args, "lora_use_dora", False)):
        lora_kwargs["use_dora"] = True

    def _apply_with_config(config_kwargs, *, fallback=False):
        cfg = LoraConfig(**config_kwargs)
        print(
            ("Retrying LoRA with automatic target detection: " if fallback else "Applying LoRA adapters: ")
            + f"r={cfg.r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}, "
            f"target_modules={config_kwargs.get('target_modules') or 'auto'}",
            flush=True,
        )
        return cfg

    lora_config = _apply_with_config(lora_kwargs)

    if not hasattr(model, "prepare_inputs_for_generation"):
        def _dream_prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
            prepared = {"input_ids": input_ids}
            if past_key_values is not None:
                prepared["past_key_values"] = past_key_values
            if attention_mask is not None:
                attn = attention_mask
                if isinstance(attn, torch.Tensor) and attn.dim() == 2:
                    attn_bool = attn.to(torch.bool)
                    attn = torch.logical_and(
                        attn_bool.unsqueeze(1).unsqueeze(-2),
                        attn_bool.unsqueeze(1).unsqueeze(-1),
                    )
                prepared["attention_mask"] = attn
            prepared.update(kwargs)
            return prepared

        model.prepare_inputs_for_generation = types.MethodType(_dream_prepare_inputs_for_generation, model)

    try:
        model = get_peft_model(model, lora_config)
    except ValueError as err:
        message = str(err)
        missing_targets = target_modules and "No modules were targeted for adaptation" in message
        if not missing_targets:
            raise
        print(
            "LoRA warning: target modules "
            f"{target_modules} not found in model. Falling back to automatic selection.",
            flush=True,
        )
        lora_kwargs.pop("target_modules", None)
        lora_config = _apply_with_config(lora_kwargs, fallback=True)
        model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad and isinstance(name, str) and "lora_" in name
        )
        preserved_params = trainable_params - lora_params
        pct = (trainable_params / total_params * 100.0) if total_params else 0.0
        print(
            "LoRA trainable breakdown → "
            f"adapters: {lora_params:,} | preserved modules: {preserved_params:,} | "
            f"total trainable: {trainable_params:,} ({pct:.4f}%)",
            flush=True,
        )
    except Exception as count_err:
        print(f"LoRA param accounting warn: {count_err}", flush=True)
    return model


def _load_model_and_tokenizer(args, model_id: str):
    ms = getattr(config, "MODEL_SETTINGS", {}).get(model_id, {})
    general_settings = ms.get("general") or {}

    adapter_path: str | None = None
    adapter_cfg: dict[str, Any] | None = None
    base_model_override: str | None = None
    if os.path.isdir(model_id):
        base_candidate, cfg_candidate = _resolve_peft_adapter_source(model_id)
        if base_candidate:
            adapter_path = model_id
            adapter_cfg = cfg_candidate or {}
            base_model_override = base_candidate
            print(
                f"Detected PEFT adapter checkpoint at {adapter_path}; base model → {base_model_override}",
                flush=True,
            )
    model_load_path = base_model_override or model_id
    tokenizer_preferred_path = adapter_path or model_load_path
    setattr(args, "_base_model_path", model_load_path)
    if adapter_path:
        setattr(args, "_adapter_model_path", adapter_path)
        if adapter_cfg:
            setattr(args, "_adapter_config", adapter_cfg)

    # attn impl passthrough from config
    attn_impl = general_settings.get("attn_implementation")
    coda_enabled_cfg = bool(general_settings.get("coda_enabled"))
    dream_enabled_cfg = False

    prompt_settings = general_settings.get("prompt_settings") or {}
    prompt_style = str(general_settings.get("prompt_format") or "legacy").lower()
    set_default_output_label(general_settings.get("output_label"))
    setattr(args, "prompt_style", prompt_style)
    prompt_special_tokens: list[str] = []

    model_kwargs = {"trust_remote_code": True}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    device_map_used = False
    mp_cfg = general_settings.get("model_parallel")
    if isinstance(mp_cfg, dict):
        enabled = bool(mp_cfg.get("enabled"))
        allow_multi = bool(mp_cfg.get("allow_multi_process"))
        mp_kwargs = {
            k: v for k, v in mp_cfg.items() if k not in {"enabled", "notes", "allow_multi_process"} and v is not None
        }
        if enabled and mp_kwargs:
            try:
                world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
            except Exception:
                world_size = 1
            if world_size > 1 and not allow_multi:
                print(
                    "Model-parallel settings detected but WORLD_SIZE>1. "
                    "Skipping device_map because allow_multi_process is False.",
                    flush=True,
                )
            else:
                model_kwargs.update(mp_kwargs)
                device_map_used = "device_map" in mp_kwargs and mp_kwargs["device_map"] is not None
                print(f"Using model-parallel settings: {mp_kwargs}", flush=True)
    elif general_settings.get("device_map"):
        model_kwargs["device_map"] = general_settings["device_map"]
        device_map_used = True
        print(f"Using device_map from config: {model_kwargs['device_map']}", flush=True)

    quant_cfg = general_settings.get("bitsandbytes")
    quant_kwargs, quant_info = build_bitsandbytes_kwargs(
        quant_cfg,
        rank=_local_rank(),
        for_training=True,
    )
    args.bitsandbytes_config = quant_cfg if quant_info.get("enabled") else None
    args.bitsandbytes_is_4bit = bool(
        quant_info.get("enabled") and quant_cfg and quant_cfg.get("load_in_4bit") and not quant_cfg.get("load_in_8bit")
    )
    args.bitsandbytes_is_8bit = bool(
        quant_info.get("enabled") and quant_cfg and quant_cfg.get("load_in_8bit") and not quant_cfg.get("load_in_4bit")
    )
    if quant_info.get("enabled"):
        model_kwargs.update(quant_kwargs)
        mode = "4bit" if quant_cfg and quant_cfg.get("load_in_4bit") else "8bit"
        print(f"[Quantization] Training load for {model_id} using bitsandbytes ({mode}).", flush=True)
    elif quant_cfg and quant_cfg.get("enabled") and quant_info.get("message"):
        print(f"[Quantization] {quant_info['message']}. Training will proceed without quantization.", flush=True)

    # Autoconfig to detect enc/dec vs causal
    base_cfg = None
    is_coda = coda_enabled_cfg
    is_dream = False
    try:
        base_cfg = AutoConfig.from_pretrained(model_load_path, trust_remote_code=True)
        model_kwargs.setdefault("config", base_cfg)
        is_encdec = bool(getattr(base_cfg, "is_encoder_decoder", False))
        args.model_type = "seq2seq" if is_encdec else "causal_lm"
        args.is_causal_lm = not is_encdec
        model_type_name = str(getattr(base_cfg, "model_type", "") or "").lower()
        is_coda = is_coda or (model_type_name == "coda")
    except Exception as e:
        print(f"AutoConfig load failed ({model_load_path}): {e}", flush=True)

    if is_coda and not model_kwargs.get("attn_implementation"):
        model_kwargs["attn_implementation"] = "eager"
        print("Using attention implementation: eager", flush=True)
    elif attn_impl:
        print(f"Using attention implementation: {attn_impl}", flush=True)

    args.is_coda_model = bool(is_coda)
    if args.is_coda_model:
        args.is_causal_lm = True
        args.model_type = "causal_lm"
    args.is_dream_model = False

    # Flax hint if only Flax weights present
    if _should_from_flax(model_load_path):
        model_kwargs["from_flax"] = True
        model_kwargs.setdefault("low_cpu_mem_usage", False)
        print("No PyTorch weights; trying from_flax=True", flush=True)

    # Load model w/ fallback to manual Flax→PT
    torch_files = ("pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors", "model.bin")
    weight_search_path = None
    for candidate in (model_load_path, adapter_path, model_id):
        if candidate and os.path.isdir(candidate):
            weight_search_path = candidate
            break
    has_torch = False
    if weight_search_path:
        has_torch = any(os.path.exists(os.path.join(weight_search_path, f)) for f in torch_files)
    need_retie = bool(model_kwargs.get("from_flax")) and not args.is_coda_model

    def _manual_flax_seq2seq():
        from transformers import FlaxAutoModelForSeq2SeqLM

        cfg = base_cfg or AutoConfig.from_pretrained(model_load_path, trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_config(cfg)
        _load_flax_params_into(
            mdl,
            model_load_path,
            lambda path: FlaxAutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True),
        )
        _retie_embeddings(mdl)
        return mdl

    def _manual_flax_causal():
        from transformers import FlaxAutoModelForCausalLM

        cfg = base_cfg or AutoConfig.from_pretrained(model_load_path, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_config(cfg)
        _load_flax_params_into(
            mdl,
            model_load_path,
            lambda path: FlaxAutoModelForCausalLM.from_pretrained(path, trust_remote_code=True),
        )
        _retie_embeddings(mdl)
        return mdl

    if args.is_coda_model:
        try:
            model = AutoModel.from_pretrained(model_load_path, **model_kwargs)
        except Exception as e:
            print(f"CoDA AutoModel load failed: {e}", flush=True)
            raise
    elif args.is_causal_lm:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs)
        except Exception as e:
            model = (
                _manual_flax_causal() if (model_kwargs.get("from_flax") and not has_torch) else (_ for _ in ()).throw(e)
            )
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path, **model_kwargs)
        except Exception as e:
            model = (
                _manual_flax_seq2seq()
                if (model_kwargs.get("from_flax") and not has_torch)
                else (_ for _ in ()).throw(e)
            )
    if need_retie:
        _retie_embeddings(model)

    if hasattr(model, "config") and getattr(model.config, "use_cache", None):
        try:
            model.config.use_cache = False
            print("Disabled use_cache for training.", flush=True)
        except Exception:
            pass

    if adapter_path:
        try:
            from peft import PeftModel  # type: ignore

            is_trainable = bool(getattr(args, "use_lora", False))
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
            setattr(args, "_lora_loaded_from_adapter", True)
            print(
                f"Loaded LoRA adapters from {adapter_path} (trainable={is_trainable}).",
                flush=True,
            )
        except Exception as adapter_err:
            print(f"Adapter load failed ({adapter_path}): {adapter_err}", flush=True)
            raise

    model = _apply_lora_if_enabled(model, args)

    _ensure_generation_config_compat(model)

    # Tokenizers: optional SP dropout for encoder side; deterministic copy for labels if needed
    sp = (
        {"enable_sampling": True, "alpha": args.tokenizer_dropout_rate}
        if (args.tokenizer_dropout_enabled and args.tokenizer_dropout_rate > 0)
        else {}
    )
    tokenizer_source = tokenizer_preferred_path
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, **sp)
    except Exception as tok_err:
        if adapter_path:
            print(
                f"Tokenizer load warn ({tokenizer_source}): {tok_err}. Falling back to {model_load_path}.",
                flush=True,
            )
            tokenizer_source = model_load_path
            tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, **sp)
        else:
            raise
    label_tok = tok
    
    # If tokenizer dropout is enabled and we're not doing causal LM, decide whether to use dropout for labels
    if args.tokenizer_dropout_enabled and args.tokenizer_dropout_rate > 0 and not args.is_causal_lm:
        if args.tokenizer_dropout_apply_to_labels:
            # Use the same tokenizer with dropout for labels
            print("Tokenizer dropout: using dropout for both inputs and labels.", flush=True)
            label_tok = tok
        else:
            # Create a separate deterministic tokenizer for labels
            try:
                label_tok = AutoTokenizer.from_pretrained(
                    tokenizer_source, trust_remote_code=True, use_fast=getattr(tok, "is_fast", False)
                )
                label_tok.padding_side = getattr(tok, "padding_side", label_tok.padding_side)
                if getattr(tok, "pad_token", None) and getattr(label_tok, "pad_token", None) is None:
                    label_tok.pad_token = tok.pad_token
                print("Tokenizer dropout: using deterministic label tokenizer.", flush=True)
            except Exception as e:
                print(f"Label tokenizer fallback: {e}", flush=True)
                label_tok = tok

    tokens_added = 0
    if prompt_special_tokens:
        try:
            tokens_to_add = [tokn for tokn in prompt_special_tokens if tokn not in tok.get_vocab()]
            if tokens_to_add:
                tokens_added = tok.add_special_tokens({"additional_special_tokens": tokens_to_add})
        except Exception as add_err:
            print(f"Prompt token add warn: {add_err}", flush=True)
            tokens_added = 0
        if label_tok is not tok:
            try:
                tokens_to_add_label = [tokn for tokn in prompt_special_tokens if tokn not in label_tok.get_vocab()]
                if tokens_to_add_label:
                    label_tok.add_special_tokens({"additional_special_tokens": tokens_to_add_label})
            except Exception as add_label_err:
                print(f"Prompt token (label) warn: {add_label_err}", flush=True)

    if tokens_added:
        try:
            model.resize_token_embeddings(len(tok))
        except Exception as resize_err:
            print(f"Token embedding resize warn: {resize_err}", flush=True)

    # Causal LM padding defaults
    if args.is_causal_lm and not getattr(args, "is_coda_model", False) and tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("Set pad_token=eos_token for causal LM", flush=True)
    if getattr(args, "is_coda_model", False):
        tok.padding_side = "right"
        print("padding_side='right' for CoDA diffusion LM", flush=True)
    elif args.is_causal_lm:
        tok.padding_side = "left"
        print("padding_side='left' for decoder-only", flush=True)

    device_map_used = model_kwargs.get("device_map") is not None
    if device_map_used:
        print(f"Model loaded with device_map={model_kwargs['device_map']}", flush=True)

    # Device placement
    if not device_map_used and torch.cuda.is_available():
        dev = f"cuda:{os.getenv('LOCAL_RANK', '0')}"
        try:
            model.to(dev)
        except NotImplementedError as e:
            if "meta tensor" in str(e):
                model = model.to_empty(device=dev)
            else:
                raise

    return model, tok, label_tok


def _build_training_metrics(args, trainer, train_result, train_ds, token_train):
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    world_size = 1
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or 1))
    except Exception:
        world_size = 1
    steps_total = int(getattr(trainer.state, "global_step", 0) or 0)

    def _safe_float(val):
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    train_runtime = _safe_float(metrics.get("train_runtime"))
    try:
        train_example_count = len(train_ds) if train_ds is not None else None
    except Exception:
        train_example_count = None

    param_count_total = param_count_trainable = param_count_embeddings = 0
    try:
        for name, param in trainer.model.named_parameters():
            count = param.numel()
            param_count_total += count
            if param.requires_grad:
                param_count_trainable += count
            if "embed" in name.lower():
                param_count_embeddings += count
    except Exception:
        pass
    param_count_non_embedding = param_count_total - param_count_embeddings

    tokens_total_dataset = None
    try:
        input_ids_column = token_train["input_ids"]
        if hasattr(input_ids_column, "to_pylist"):
            input_ids_column = input_ids_column.to_pylist()
        tokens_total_dataset = int(sum(len(ids) for ids in input_ids_column))
    except Exception:
        tokens_total_dataset = None

    token_stats = None
    if hasattr(trainer, "get_token_statistics"):
        try:
            token_stats = trainer.get_token_statistics()
        except Exception:
            token_stats = None

    tokens_seen = None
    target_tokens_seen = None
    examples_seen = None
    token_wall_time = None
    if token_stats:
        itok = token_stats.get("input_tokens")
        ttok = token_stats.get("target_tokens")
        ex_seen = token_stats.get("examples")
        if itok and itok > 0:
            tokens_seen = round(itok)
        if ttok and ttok > 0:
            target_tokens_seen = round(ttok)
        if ex_seen and ex_seen > 0:
            examples_seen = round(ex_seen)
        start_t, end_t = token_stats.get("start_time"), token_stats.get("last_update")
        if start_t and end_t and end_t > start_t:
            token_wall_time = end_t - start_t

    tokens_total = tokens_seen if tokens_seen is not None else tokens_total_dataset
    examples_effective = examples_seen if examples_seen is not None else train_example_count
    runtime_for_throughput = train_runtime if train_runtime else token_wall_time

    tokens_per_step = (tokens_total / steps_total) if tokens_total is not None and steps_total else None
    throughput_tokens_per_s = (
        (tokens_total / runtime_for_throughput) if tokens_total is not None and runtime_for_throughput else None
    )
    examples_per_s = (
        (examples_effective / runtime_for_throughput) if (examples_effective and runtime_for_throughput) else None
    )
    seq_len_effective_mean = (
        (tokens_total / examples_effective) if (tokens_total is not None and examples_effective) else None
    )
    context_length = None
    try:
        context_length = int(args.max_length if getattr(args, "is_causal_lm", False) else args.max_input_length)
    except Exception:
        context_length = None

    loss_history = []
    grad_norms = []
    step_loss_pairs = []
    eval_loss_pairs = []
    grad_clip_hits = 0
    grad_clip_den = 0
    accuracy_history: list[float] = []
    accuracy_steps: list[int | None] = []
    base_accuracy_history: list[float] = []
    try:
        max_grad_norm = float(args.max_grad_norm) if getattr(args, "max_grad_norm", None) else None
    except Exception:
        max_grad_norm = None

    try:
        for entry in trainer.state.log_history:
            if not isinstance(entry, dict):
                continue
            if "loss" in entry:
                try:
                    val = float(entry["loss"])
                    loss_history.append(val)
                    step_val = entry.get("step")
                    if step_val is not None:
                        step_loss_pairs.append((int(step_val), val))
                except Exception:
                    pass
            if "grad_norm" in entry:
                try:
                    g = float(entry["grad_norm"])
                    grad_norms.append(g)
                    grad_clip_den += 1
                    if max_grad_norm is not None and g >= max_grad_norm:
                        grad_clip_hits += 1
                except Exception:
                    pass
            if "eval_loss" in entry:
                try:
                    eval_val = float(entry["eval_loss"])
                    step_val = entry.get("step")
                    if step_val is not None:
                        eval_loss_pairs.append((int(step_val), eval_val))
                except Exception:
                    pass
            if "accuracy_pct" in entry:
                try:
                    acc_val = float(entry["accuracy_pct"])
                except Exception:
                    pass
                else:
                    accuracy_history.append(acc_val)
                    step_val = entry.get("step")
                    if step_val is not None:
                        try:
                            accuracy_steps.append(int(step_val))
                        except Exception:
                            accuracy_steps.append(None)
                    else:
                        accuracy_steps.append(None)
            if "base_accuracy_pct" in entry:
                try:
                    base_acc = float(entry["base_accuracy_pct"])
                except Exception:
                    pass
                else:
                    base_accuracy_history.append(base_acc)
    except Exception:
        loss_history = []
        grad_norms = []
        step_loss_pairs = []
        eval_loss_pairs = []
        grad_clip_hits = 0
        grad_clip_den = 0
        accuracy_history = []
        accuracy_steps = []
        base_accuracy_history = []

    loss_train_min = min(loss_history) if loss_history else None
    loss_train_first = loss_history[0] if loss_history else None
    loss_train_final = _safe_float(metrics.get("train_loss"))
    if loss_train_final is None and loss_history:
        loss_train_final = loss_history[-1]

    accuracy_pct_max = max(accuracy_history) if accuracy_history else None
    accuracy_pct_final = accuracy_history[-1] if accuracy_history else None
    accuracy_pct_max_step: int | None = None
    if accuracy_history and accuracy_steps:
        try:
            best_idx = max(range(len(accuracy_history)), key=lambda idx: accuracy_history[idx])
            step_value = accuracy_steps[best_idx]
            if step_value is not None:
                accuracy_pct_max_step = int(step_value)
        except Exception:
            accuracy_pct_max_step = None
    base_accuracy_pct_max = max(base_accuracy_history) if base_accuracy_history else None

    grad_norm_mean = statistics.mean(grad_norms) if grad_norms else None
    grad_norm_std = statistics.stdev(grad_norms) if len(grad_norms) > 1 else None
    grad_clip_hit_rate = None
    if grad_clip_den > 0:
        grad_clip_hit_rate = grad_clip_hits / max(grad_clip_den, 1)

    smin_estimate = None
    if step_loss_pairs:
        try:
            smin_estimate = min(step_loss_pairs, key=lambda x: x[1])[0]
        except Exception:
            smin_estimate = None

    eval_loss_min = None
    loss_eval_at_min_step = None
    eval_steps = None
    if eval_loss_pairs:
        try:
            eval_loss_min = min(val for _, val in eval_loss_pairs)
        except Exception:
            eval_loss_min = None
        eval_steps = len(eval_loss_pairs)
        if smin_estimate is not None:
            for step_val, eval_val in eval_loss_pairs:
                if step_val == smin_estimate:
                    loss_eval_at_min_step = eval_val
                    break

    global_batch = int(args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size)
    device_hours = (train_runtime / 3600.0 * world_size) if train_runtime else None

    compute_c_raw = None
    tflops_per_device = None
    flops_total_est = None
    if param_count_trainable and global_batch and steps_total:
        compute_c_raw = 6.0 * param_count_trainable * global_batch * steps_total
        if train_runtime and world_size:
            tflops_per_device = compute_c_raw / (train_runtime * world_size * 1e12)
        flops_total_est = compute_c_raw

    grad_noise_scale = None
    bcrit_estimate = None
    if grad_norm_mean and grad_norm_mean > 0 and grad_norm_std is not None and global_batch:
        grad_noise_scale = (grad_norm_std**2) / max(grad_norm_mean**2, 1e-12)
        bcrit_estimate = max(grad_noise_scale * global_batch, 1e-3)

    b_over_bcrit = None
    if bcrit_estimate and bcrit_estimate > 0:
        b_over_bcrit = global_batch / bcrit_estimate

    compute_cmin = None
    if compute_c_raw and bcrit_estimate and bcrit_estimate > 0:
        compute_cmin = compute_c_raw / (1.0 + (global_batch / bcrit_estimate))

    emin_estimate = None
    if tokens_per_step and smin_estimate is not None:
        emin_estimate = tokens_per_step * smin_estimate

    convergence_factor = None
    if loss_train_final is not None and loss_train_min and loss_train_min > 0:
        convergence_factor = (loss_train_final / loss_train_min) - 1.0

    compute_columns_used = None
    if compute_c_raw is not None:
        compute_columns_used = "C=6*N_trainable*B*steps;Cmin=C/(1+B/Bcrit_est)"

    train_samples_per_second = (
        examples_per_s if examples_per_s is not None else _safe_float(metrics.get("train_samples_per_second"))
    )
    train_tokens_per_second = (
        throughput_tokens_per_s
        if throughput_tokens_per_s is not None
        else _safe_float(metrics.get("train_tokens_per_second"))
    )

    metrics_payload = {
        "steps_total": steps_total if steps_total else None,
        "microbatches": int(args.gradient_accumulation_steps),
        "global_batch_size_effective": global_batch,
        "tokens_total": tokens_total,
        "tokens_per_step": tokens_per_step,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "examples_per_s": examples_per_s,
        "seq_len_effective_mean": seq_len_effective_mean,
        "context_length": context_length,
        "loss_train_final": loss_train_final,
        "loss_train_min": loss_train_min,
        "train_loss_first": loss_train_first,
        "ttft_train_loss_delta": (loss_train_final - loss_train_first)
        if (loss_train_final is not None and loss_train_first is not None)
        else None,
        "loss_eval_min": eval_loss_min,
        "loss_eval_at_min_step": loss_eval_at_min_step,
        "eval_steps": eval_steps,
        "train_runtime": train_runtime,
        "train_samples_per_second": train_samples_per_second,
        "train_steps_per_second": _safe_float(metrics.get("train_steps_per_second")),
        "train_tokens_per_second": train_tokens_per_second,
        "ttft_batch_size": int(args.per_device_train_batch_size),
        "ttft_lr": _safe_float(args.learning_rate),
        "ttft_time_s": train_runtime,
        "ttft_tokens": tokens_total,
        "ttft_steps": steps_total if steps_total else None,
        "ttft_dataset_examples": train_example_count,
        "train_examples_seen": examples_seen,
        "num_input_tokens_seen": tokens_seen,
        "num_target_tokens_seen": target_tokens_seen,
        "device_hours": device_hours,
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_std": grad_norm_std,
        "grad_clip_hit_rate": grad_clip_hit_rate,
        "param_count_total": param_count_total or None,
        "param_count_trainable": param_count_trainable or None,
        "param_count_embeddings": param_count_embeddings or None,
        "param_count_non_embedding": param_count_non_embedding or None,
        "compute_c_raw": compute_c_raw,
        "compute_cmin": compute_cmin,
        "compute_columns_used": compute_columns_used,
        "tflops_per_device_est": tflops_per_device,
        "flops_total_est": flops_total_est,
        "Bcrit_estimate": bcrit_estimate,
        "B_over_Bcrit": b_over_bcrit,
        "Smin_estimate": smin_estimate,
        "Emin_estimate": emin_estimate,
        "convergence_factor": convergence_factor,
        "grad_noise_scale": grad_noise_scale,
        "accuracy_pct_max": accuracy_pct_max,
        "accuracy_pct_final": accuracy_pct_final,
        "accuracy_pct_max_step": accuracy_pct_max_step,
        "base_accuracy_pct_max": base_accuracy_pct_max,
    }
    early_stop_cb = getattr(trainer, "_arc_early_stop_cb", None)
    if early_stop_cb is not None:
        triggered = bool(getattr(early_stop_cb, "stop_reason", None))
        metrics_payload["early_stop_triggered"] = triggered
        if triggered:
            metrics_payload["early_stop_reason"] = early_stop_cb.stop_reason
        metric_name = getattr(early_stop_cb, "metric_used", None)
        if metric_name:
            metrics_payload["early_stop_metric"] = metric_name
        best_val = getattr(early_stop_cb, "best_value", None)
        if best_val is not None:
            metrics_payload["early_stop_best"] = best_val
        best_step = getattr(early_stop_cb, "best_step", None)
        if best_step is not None:
            metrics_payload["early_stop_step"] = best_step
        last_val = getattr(early_stop_cb, "_last_value", None)
        if last_val is not None:
            metrics_payload["early_stop_last_value"] = last_val
    try:
        if isinstance(getattr(train_result, "metrics", None), dict):
            if tokens_seen is not None:
                train_result.metrics["num_input_tokens_seen"] = tokens_seen
            if target_tokens_seen is not None:
                train_result.metrics["num_target_tokens_seen"] = target_tokens_seen
            if train_tokens_per_second is not None:
                train_result.metrics["train_tokens_per_second"] = train_tokens_per_second
            if train_samples_per_second is not None:
                train_result.metrics["train_samples_per_second"] = train_samples_per_second
            if examples_seen is not None:
                train_result.metrics["train_examples_seen"] = examples_seen
    except Exception:
        pass
    return metrics_payload


# ---------- preprocessors ----------
def _prep_seq2seq(args, tokenizer, label_tokenizer):
    def fn(ex):
        inp, tgt = ex["prompt"], ex["correct_answer"]
        # optional length filtering (late)
        if args.enable_token_filtering:
            tinp = tokenizer(inp, truncation=False, padding=False)
            ttgt = label_tokenizer(tgt, truncation=False, padding=False)
            L = [
                (len(a) <= args.max_input_length and len(b) <= args.max_target_length)
                for a, b in zip(tinp["input_ids"], ttgt["input_ids"], strict=False)
            ]
            if not any(L):
                return {"input_ids": [], "attention_mask": [], "labels": []}
            inp = [x for x, m in zip(inp, L, strict=False) if m]
            tgt = [x for x, m in zip(tgt, L, strict=False) if m]
        model_in = tokenizer(inp, max_length=args.max_input_length, truncation=True, padding=False)
        # Legacy helper comment retained for downstream tests/tools: with tokenizer.as_target_tokenizer()
        ctx = (
            label_tokenizer.as_target_tokenizer() if hasattr(label_tokenizer, "as_target_tokenizer") else nullcontext()
        )
        with ctx:
            lbl_ids = label_tokenizer(tgt, max_length=args.max_target_length, truncation=True, padding=False)[
                "input_ids"
            ]
        pad_id = label_tokenizer.pad_token_id if label_tokenizer.pad_token_id is not None else tokenizer.pad_token_id
        labels = [
            [(t if (pad_id is None or t != pad_id) else -100) for t in row]
            for row in lbl_ids
        ]  # Use -100 for proper loss computation
        model_in["labels"] = labels
        return model_in

    return fn


def _prep_ul2(args, tokenizer):
    def fn(ex):
        prompts = ex.get("prompt", [])
        answers = ex.get("correct_answer", [])
        combined: list[str] = []
        for idx, prompt in enumerate(prompts):
            answer = answers[idx] if idx < len(answers) else None
            pieces: list[str] = []
            if isinstance(prompt, str) and prompt.strip():
                pieces.append(prompt.strip())
            if isinstance(answer, str) and answer.strip():
                pieces.append(answer.strip())
            text = " ".join(pieces).strip()
            if not text:
                if isinstance(prompt, str) and prompt.strip():
                    text = prompt.strip()
                elif isinstance(answer, str) and answer.strip():
                    text = str(answer).strip()
                else:
                    text = ""
            combined.append(text)
        return {"text": combined}

    return fn


def _prep_coda(args, tokenizer):
    def fn(ex):
        prompts, targets = ex["prompt"], ex["correct_answer"]
        combined_texts: list[str] = []
        prefix_lengths: list[int] = []
        for prompt, target in zip(prompts, targets, strict=False):
            combo, prefix = prepare_causal_prompt_with_target(prompt, target)
            combined_texts.append(combo)
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            prefix_lengths.append(prefix_len)

        tokenized = tokenizer(
            combined_texts,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        src_masks: list[list[int]] = []
        for ids, pref_len in zip(tokenized["input_ids"], prefix_lengths, strict=False):
            keep = min(pref_len, len(ids))
            mask = [1] * keep + [0] * (len(ids) - keep)
            src_masks.append(mask)
        tokenized["src_mask"] = src_masks
        return tokenized

    return fn


def _prep_dream(args, tokenizer):
    def fn(ex):
        prompts, targets = ex["prompt"], ex["correct_answer"]
        return build_dream_training_features(
            prompts,
            targets,
            tokenizer,
            max_length=args.max_length,
        )

    return fn


def _prep_causal(args, tokenizer):
    def fn(ex):
        inp, tgt = ex["prompt"], ex["correct_answer"]
        texts, plen = [], []
        for p, t in zip(inp, tgt, strict=False):
            combo, pfx = prepare_causal_prompt_with_target(p, t)
            texts.append(combo)
            plen.append(len(tokenizer(pfx, add_special_tokens=False)["input_ids"]))
        model_in = tokenizer(texts, max_length=args.max_length, truncation=True, padding=False)
        labels = []
        for ids, L in zip(model_in["input_ids"], plen, strict=False):
            lab = ids.copy()
            for j in range(min(L, len(lab))):
                lab[j] = -100
            labels.append(lab)
        model_in["labels"] = labels
        return model_in

    return fn


# ---------- UL2 helpers ----------
UL2_OBJECTIVE_ID_MAP: dict[str, int] = {"r": 0, "x": 1, "s": 2}
_DEFAULT_UL2_COMPONENTS = {
    "r": {"weight": 0.5, "noise_density": 0.15, "mean_span_length": 3.0},
    "x": {"weight": 0.25, "noise_density": 0.5, "mean_span_length": 10.0},
    "s": {"weight": 0.25, "min_prefix_ratio": 0.25, "max_prefix_ratio": 0.75},
}


def _random_segmentation(length: int, num_segments: int, rng: random.Random) -> list[int]:
    """Partition `length` items into at most `num_segments` segments (allow zero-length)."""
    num_segments = max(1, int(num_segments or 1))
    if length <= 0:
        return [0] * num_segments
    num_segments = max(1, num_segments)
    # When more segments than tokens, allow zero-length segments by padding later.
    sample_segments = min(num_segments - 1, max(0, length - 1))
    if sample_segments > 0:
        cut_points = sorted(rng.sample(range(1, length), sample_segments))
    else:
        cut_points = []
    segments: list[int] = []
    prev = 0
    for cp in cut_points:
        segments.append(cp - prev)
        prev = cp
    segments.append(length - prev)
    # pad with zero-length segments if required
    while len(segments) < num_segments:
        segments.append(0)
    return segments


def _random_spans_noise_mask(
    length: int,
    noise_density: float,
    mean_span_length: float,
    rng: random.Random,
) -> list[int]:
    """Produce a mask of 0/1 values denoting noise positions for span corruption."""
    if length <= 0:
        return []
    if length == 1:
        return [0]
    noise_density = float(max(0.0, min(noise_density, 1.0)))
    mean_span_length = max(1.0, float(mean_span_length or 1.0))
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(num_noise_tokens / mean_span_length))
    num_noise_spans = max(1, num_noise_spans)
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng)
    num_non_noise_tokens = length - num_noise_tokens
    non_noise_spans = len(noise_span_lengths) + 1
    non_noise_span_lengths = _random_segmentation(num_non_noise_tokens, non_noise_spans, rng)

    mask: list[int] = []
    total_segments = max(len(noise_span_lengths), len(non_noise_span_lengths) - 1)
    for idx in range(total_segments):
        non_noise_len = non_noise_span_lengths[idx] if idx < len(non_noise_span_lengths) - 1 else 0
        if non_noise_len:
            mask.extend([0] * non_noise_len)
        noise_len = noise_span_lengths[idx] if idx < len(noise_span_lengths) else 0
        if noise_len:
            mask.extend([1] * noise_len)
    if non_noise_span_lengths:
        mask.extend([0] * non_noise_span_lengths[-1])
    if len(mask) < length:
        mask.extend([0] * (length - len(mask)))
    elif len(mask) > length:
        mask = mask[:length]
    return mask


def _normalize_ul2_mixture(settings: dict[str, Any] | None) -> list[dict[str, Any]]:
    base = copy.deepcopy(_DEFAULT_UL2_COMPONENTS)
    if isinstance(settings, dict):
        mixture = settings.get("mixture")
        if isinstance(mixture, dict):
            settings = mixture
        for key, override in settings.items():
            k = str(key).lower()
            if k in base and isinstance(override, dict):
                for ok, ov in override.items():
                    if ov is not None:
                        base[k][ok] = ov
    components: list[dict[str, Any]] = []
    total_weight = 0.0
    for name, cfg in base.items():
        try:
            weight = float(cfg.get("weight", 0.0))
        except Exception:
            weight = 0.0
        if weight <= 0:
            continue
        comp = dict(cfg)
        comp["name"] = name
        components.append(comp)
        total_weight += weight
    if not components or total_weight <= 0:
        components = [
            {**dict(v), "name": k, "weight": float(dict(v).get("weight", 0.0))}
            for k, v in _DEFAULT_UL2_COMPONENTS.items()
        ]
        total_weight = sum(c["weight"] for c in components)
    total_weight = total_weight or 1.0
    for comp in components:
        comp["weight"] = float(comp.get("weight", 0.0)) / total_weight
    return components


class DataCollatorForUL2Objective:
    """Mixture-of-denoisers collator implementing an UL2-style objective."""

    def __init__(
        self,
        tokenizer,
        *,
        pad_to_multiple_of: int | None = None,
        return_tensors: str = "pt",
        max_length: int | None = None,
        settings: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.max_length = max_length
        self.settings = settings or {}
        self.mixture = _normalize_ul2_mixture(self.settings if isinstance(self.settings, dict) else None)
        rng_seed = seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
        self.rng = random.Random(rng_seed)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.unk_token_id = getattr(tokenizer, "unk_token_id", None)
        self.sentinel_ids = self._collect_sentinel_ids(limit=int(self.settings.get("max_sentinels", 100) or 100))
        self._objective_counts: dict[str, int] = {comp["name"]: 0 for comp in self.mixture}
        self._total_examples = 0

    def _collect_sentinel_ids(self, limit: int = 100) -> list[int]:
        sentinel_tokens: list[str] = []
        specials = getattr(self.tokenizer, "additional_special_tokens", []) or []
        for tok in specials:
            if isinstance(tok, str) and tok.startswith("<extra_id_"):
                sentinel_tokens.append(tok)
        if not sentinel_tokens:
            for idx in range(limit):
                tok = f"<extra_id_{idx}>"
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                if tok_id is None:
                    continue
                if self.unk_token_id is not None and tok_id == self.unk_token_id:
                    continue
                sentinel_tokens.append(tok)
        sentinel_ids: list[int] = []
        for tok in sentinel_tokens:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id is None:
                continue
            if tok_id not in sentinel_ids:
                sentinel_ids.append(int(tok_id))
            if len(sentinel_ids) >= limit:
                break
        if not sentinel_ids:
            fallback = self.pad_token_id
            if fallback is None:
                fallback = self.unk_token_id if self.unk_token_id is not None else 0
            sentinel_ids = [int(fallback)]
        return sentinel_ids

    def _sample_objective(self) -> tuple[str, dict[str, Any]]:
        roll = self.rng.random()
        cumulative = 0.0
        for comp in self.mixture:
            cumulative += float(comp.get("weight", 0.0))
            if roll <= cumulative:
                return comp["name"], comp
        return self.mixture[-1]["name"], self.mixture[-1]

    def _ensure_length_limit(self, token_ids: list[int]) -> list[int]:
        if self.max_length is None or self.max_length <= 0:
            return token_ids
        if len(token_ids) <= self.max_length:
            return token_ids
        return token_ids[: self.max_length]

    def _apply_span_corruption(
        self,
        token_ids: list[int],
        noise_density: float,
        mean_span_length: float,
    ) -> tuple[list[int], list[int]]:
        mask = _random_spans_noise_mask(len(token_ids), noise_density, mean_span_length, self.rng)
        inputs: list[int] = []
        labels: list[int] = []
        sentinel_idx = 0
        idx = 0
        while idx < len(token_ids):
            if mask[idx]:
                sentinel_id = self.sentinel_ids[min(sentinel_idx, len(self.sentinel_ids) - 1)]
                sentinel_idx += 1
                inputs.append(sentinel_id)
                labels.append(sentinel_id)
                while idx < len(token_ids) and mask[idx]:
                    labels.append(token_ids[idx])
                    idx += 1
            else:
                inputs.append(token_ids[idx])
                idx += 1
        if self.eos_token_id is not None and (not labels or labels[-1] != self.eos_token_id):
            labels.append(self.eos_token_id)
        return inputs, labels

    def _apply_prefix_lm(self, token_ids: list[int], params: dict[str, Any]) -> tuple[list[int], list[int]]:
        if len(token_ids) <= 1:
            fallback_inputs = token_ids[:]
            fallback_labels = token_ids[:]
            if self.eos_token_id is not None and fallback_labels[-1] != self.eos_token_id:
                fallback_labels.append(self.eos_token_id)
            return fallback_inputs, fallback_labels
        min_ratio = float(params.get("min_prefix_ratio", 0.25) or 0.25)
        max_ratio = float(params.get("max_prefix_ratio", 0.75) or 0.75)
        min_ratio = max(0.05, min(min_ratio, 0.95))
        max_ratio = max(min_ratio + 1e-3, min(max_ratio, 0.98))
        ratio = self.rng.uniform(min_ratio, max_ratio)
        prefix_len = int(round(len(token_ids) * ratio))
        prefix_len = max(1, min(prefix_len, len(token_ids) - 1))
        sentinel_id = self.sentinel_ids[0]
        inputs = token_ids[:prefix_len] + [sentinel_id]
        suffix = token_ids[prefix_len:]
        labels = [sentinel_id] + suffix
        if self.eos_token_id is not None and labels[-1] != self.eos_token_id:
            labels.append(self.eos_token_id)
        return inputs, labels

    def _pad_labels(self, sequences: list[list[int]]) -> list[list[int]]:
        if not sequences:
            return sequences
        max_len = max(len(seq) for seq in sequences)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 0:
            multiple = self.pad_to_multiple_of
            if max_len % multiple != 0:
                max_len = ((max_len // multiple) + 1) * multiple
        padded: list[list[int]] = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded.append(seq + [-100] * pad_len)
            else:
                padded.append(seq)
        return padded

    def mixture_summary(self) -> str:
        parts = []
        for comp in self.mixture:
            name = comp.get("name", "?").upper()
            weight = comp.get("weight", 0.0)
            if name in {"R", "X"}:
                parts.append(
                    f"{name}:{weight:.2f} (noise={comp.get('noise_density', '?')}, span={comp.get('mean_span_length', '?')})"
                )
            else:
                parts.append(
                    f"{name}:{weight:.2f} (prefix=[{comp.get('min_prefix_ratio', '?')}, {comp.get('max_prefix_ratio', '?')}])"
                )
        return "; ".join(parts)

    def objective_statistics(self) -> dict[str, Any]:
        if self._total_examples <= 0:
            return {name.upper(): 0.0 for name in self._objective_counts}
        return {
            name.upper(): self._objective_counts.get(name, 0) / float(self._total_examples)
            for name in self._objective_counts
        }

    def _extract_text(self, feature: dict[str, Any]) -> str | None:
        text = feature.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        prompt = feature.get("prompt")
        answer = feature.get("correct_answer")
        pieces: list[str] = []
        if isinstance(prompt, str) and prompt.strip():
            pieces.append(prompt.strip())
        if isinstance(answer, str) and answer.strip():
            pieces.append(answer.strip())
        combined = " ".join(pieces).strip()
        return combined or None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_sequences: list[list[int]] = []
        label_sequences: list[list[int]] = []
        objective_ids: list[int] = []
        for feature in features:
            text = self._extract_text(feature)
            if not text:
                continue
            tokenized = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            token_ids = list((tokenized or {}).get("input_ids") or [])
            if not token_ids:
                continue
            token_ids = self._ensure_length_limit(token_ids)
            objective_name, params = self._sample_objective()
            name_lc = objective_name.lower()
            if name_lc in {"r", "x"}:
                inputs, labels = self._apply_span_corruption(
                    token_ids,
                    params.get("noise_density", 0.15),
                    params.get("mean_span_length", 3.0),
                )
            elif name_lc == "s":
                inputs, labels = self._apply_prefix_lm(token_ids, params)
            else:
                inputs = token_ids[:]
                labels = token_ids[:]
                if self.eos_token_id is not None and labels[-1] != self.eos_token_id:
                    labels.append(self.eos_token_id)
            if not inputs or not labels:
                continue
            inputs = self._ensure_length_limit(inputs)
            input_sequences.append(inputs)
            label_sequences.append(labels)
            objective_id = UL2_OBJECTIVE_ID_MAP.get(name_lc, -1)
            objective_ids.append(objective_id)
            self._objective_counts[name_lc] = self._objective_counts.get(name_lc, 0) + 1
            self._total_examples += 1

        if not input_sequences:
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            input_sequences = [[pad_id]]
            label_sequences = [[-100]]
            objective_ids = [-1]

        batch = self.tokenizer.pad(
            {"input_ids": input_sequences},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        if self.return_tensors == "pt":
            max_len = max(len(seq) for seq in label_sequences) if label_sequences else 0
            if self.pad_to_multiple_of and self.pad_to_multiple_of > 0 and max_len > 0:
                multiple = self.pad_to_multiple_of
                if max_len % multiple != 0:
                    max_len = ((max_len // multiple) + 1) * multiple
            labels_tensor = torch.full((len(label_sequences), max_len), -100, dtype=torch.long)
            for idx, seq in enumerate(label_sequences):
                clipped = seq[:max_len]
                if clipped:
                    labels_tensor[idx, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            objectives_tensor = torch.tensor(objective_ids, dtype=torch.int64)
        elif self.return_tensors == "np":
            if np is None:
                raise ImportError("numpy is required for return_tensors='np' in UL2 collator.")
            max_len = max(len(seq) for seq in label_sequences) if label_sequences else 0
            if self.pad_to_multiple_of and self.pad_to_multiple_of > 0 and max_len > 0:
                multiple = self.pad_to_multiple_of
                if max_len % multiple != 0:
                    max_len = ((max_len // multiple) + 1) * multiple
            labels_tensor = np.full((len(label_sequences), max_len), -100, dtype=np.int64)
            for idx, seq in enumerate(label_sequences):
                clipped = seq[:max_len]
                if clipped:
                    labels_tensor[idx, : len(clipped)] = np.asarray(clipped, dtype=np.int64)
            objectives_tensor = np.asarray(objective_ids, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported return_tensors value: {self.return_tensors!r}")

        batch["labels"] = labels_tensor
        batch["ul2_objective_id"] = objectives_tensor
        return batch


# ---------- collator for causal LM ----------
class DataCollatorForCausalLM:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, feats: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [f["input_ids"] for f in feats]
        labels = [f["labels"] for f in feats]
        batch = self.tokenizer.pad(
            {"input_ids": ids},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        L = batch["input_ids"].shape[1]
        lab = [row + [-100] * (L - len(row)) for row in labels]
        batch["labels"] = torch.tensor(lab, dtype=torch.long)
        return batch


class DataCollatorForCoDA:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, feats: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        pad_id = pad_id if pad_id is not None else 0

        input_ids = [f["input_ids"] for f in feats]
        attention_mask = [f.get("attention_mask") or [1] * len(f["input_ids"]) for f in feats]
        src_masks = [f.get("src_mask") or [1] * len(f["input_ids"]) for f in feats]

        max_len = max(len(ids) for ids in input_ids) if input_ids else 0
        if self.pad_to_multiple_of and max_len:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        padded_ids, padded_attn, padded_src = [], [], []
        for ids, attn, src in zip(input_ids, attention_mask, src_masks, strict=False):
            pad_len = max(0, max_len - len(ids))
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_attn.append(attn + [0] * pad_len)
            padded_src.append(src + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.float32),
            "src_mask": torch.tensor(padded_src, dtype=torch.bool),
        }


# ---------- timeout ----------
# Public timeout helpers (mirrors inference worker naming)
def timeout_handler(*_):
    print(f"[Rank {os.getenv('LOCAL_RANK', 'N/A')}] Training timeout, exiting...", flush=True)
    sys.exit(1)


def setup_timeout(seconds: int | float | None):
    signal.signal(signal.SIGALRM, timeout_handler)

    if seconds in (None, "", False):
        signal.alarm(0)
        return

    try:
        requested = float(seconds)
    except (TypeError, ValueError):
        logger.warning("Invalid timeout value %r; disabling alarm.", seconds)
        signal.alarm(0)
        return

    if not math.isfinite(requested) or requested <= 0:
        signal.alarm(0)
        return

    capped = min(requested, float(MAX_SIGNAL_SECONDS))
    if capped != requested:
        logger.warning(
            "Requested timeout %.0f exceeds max %d; capping to supported range.",
            requested,
            MAX_SIGNAL_SECONDS,
        )

    seconds_int = int(capped)
    if seconds_int <= 0:
        seconds_int = 1

    signal.alarm(seconds_int)


# ---------- early stopping callback ----------
class AccuracyThresholdEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        *,
        model_name: str | None,
        metric_name: str,
        target: float,
        warmup_steps: int = 0,
    ):
        self.model_name = model_name or ""
        self.metric_name = metric_name
        try:
            self.target = float(target)
        except Exception:
            raise ValueError(f"Invalid target for early stopping: {target!r}")
        try:
            self.warmup_steps = max(0, int(warmup_steps))
        except Exception:
            self.warmup_steps = 0
        self.metric_used = metric_name
        self.stop_reason: str | None = None
        self.best_value: float | None = None
        self.best_step: int | None = None
        self._last_value: float | None = None
        self._triggered = False

    def _emit(self, message: str) -> None:
        prefix = f"[EarlyStop]['{self.model_name}'] " if self.model_name else "[EarlyStop] "
        print(prefix + message, flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if self._triggered or not logs:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if step < self.warmup_steps:
            return
        if self.metric_name not in logs:
            return
        try:
            value = float(logs[self.metric_name])
        except Exception:
            return
        self._last_value = value
        if self.best_value is None or value > self.best_value:
            self.best_value = value
            self.best_step = step
        if value >= self.target:
            self.stop_reason = (
                f"{self.metric_name}={value:.4f} reached target {self.target:.4f} at step {step}"
            )
            self._emit(self.stop_reason)
            self._triggered = True
            control.should_training_stop = True
            control.should_save = True


class MetricEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        *,
        model_name: str | None,
        metrics: list[str],
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.0,
        warmup_steps: int = 0,
        target: float | None = None,
    ):
        self.model_name = model_name or ""
        self.metrics = [m for m in metrics if isinstance(m, str) and m.strip()]
        self.mode = (mode or "max").strip().lower()
        self.greater_is_better = self.mode != "min"
        self.patience = max(0, int(patience))
        try:
            self.min_delta = float(min_delta)
        except Exception:
            self.min_delta = 0.0
        self.min_delta = max(0.0, self.min_delta)
        self.warmup_steps = max(0, int(warmup_steps))
        try:
            self.target = float(target) if target is not None else None
        except Exception:
            self.target = None

        self.metric_used: str | None = None
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.stop_reason: str | None = None
        self._bad_counts: int = 0
        self._triggered = False
        self._last_value: float | None = None

    def _extract_metric(self, logs: dict[str, Any] | None) -> tuple[float | None, str | None]:
        if not logs:
            return None, None
        for name in self.metrics:
            if name not in logs:
                continue
            value = logs.get(name)
            try:
                return float(value), name
            except Exception:
                continue
        return None, None

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.greater_is_better:
            return value > (self.best_value + self.min_delta)
        return value < (self.best_value - self.min_delta)

    def _meets_target(self, value: float) -> bool:
        if self.target is None:
            return False
        if self.greater_is_better:
            return value >= self.target
        return value <= self.target

    def _format_metric(self, name: str | None, value: float | None) -> str:
        if value is None:
            return f"{name or 'metric'}=n/a"
        return f"{name or 'metric'}={value:.4f}"

    def _emit(self, message: str) -> None:
        prefix = f"[EarlyStop]['{self.model_name}'] " if self.model_name else "[EarlyStop] "
        print(prefix + message, flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if self._triggered or not self.metrics:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if step < self.warmup_steps:
            return

        value, metric_name = self._extract_metric(logs)
        if value is None:
            return
        self.metric_used = self.metric_used or metric_name
        self._last_value = value

        if self._meets_target(value):
            self.stop_reason = (
                f"{self._format_metric(metric_name, value)} reached target {self.target:.4f} at step {step}"
            )
            self._emit(self.stop_reason)
            self._triggered = True
            control.should_training_stop = True
            control.should_save = True
            return

        if self._is_improvement(value):
            self.best_value = value
            self.best_step = step
            self._bad_counts = 0
            return

        self._bad_counts += 1
        if self._bad_counts >= self.patience:
            self.stop_reason = (
                f"{self._format_metric(metric_name, value)} plateaued "
                f"(best {self._format_metric(metric_name, self.best_value)} at step {self.best_step})"
            )
            self._emit(self.stop_reason)
            self._triggered = True
            control.should_training_stop = True
            control.should_save = True


# ---------- CSV callback ----------
class CSVLoggingCallback(TrainerCallback):
    def __init__(self, out_dir: str | None, model_name: str, lr: float, ctx: int, run_id: str | None):
        self.enabled = bool(out_dir)
        if not self.enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = model_name.replace("/", "_")
        self.step_log_path = os.path.join(out_dir, f"training_steps_{run_id or ts}_{safe}.csv")
        self.step_jsonl_path = os.path.join(out_dir, f"training_steps_{run_id or ts}_{safe}.jsonl")
        self.overall_path = os.path.join(out_dir, "training_overall.csv")
        self.meta = {
            "model_name": model_name,
            "config_learning_rate": lr,
            "context_size": ctx,
            "run_id": run_id or ts,
        }

    @staticmethod
    def _coerce(value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [CSVLoggingCallback._coerce(v) for v in value]
        if isinstance(value, dict):
            return {k: CSVLoggingCallback._coerce(v) for k, v in value.items()}
        try:
            return float(value)
        except Exception:
            pass
        try:
            return int(value)
        except Exception:
            pass
        return str(value)

    def on_log(self, args, state, control, logs=None, **kw):
        if not self.enabled or not logs:
            return
        row = {
            "timestamp": time.time(),
            "step": state.global_step,
            "loss": logs.get("loss", logs.get("train_loss")),
            "learning_rate": logs.get("learning_rate"),
            "epoch": logs.get("epoch"),
            **self.meta,
        }
        try:
            import csv
            import os

            write_header = not os.path.exists(self.step_log_path)
            with open(self.step_log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"CSV step log warn: {e}", flush=True)

        payload = {
            "timestamp": time.time(),
            "global_step": state.global_step,
            **self.meta,
        }
        payload.update({k: self._coerce(v) for k, v in logs.items()})
        try:
            with open(self.step_jsonl_path, "a") as jf:
                jf.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"JSON step log warn: {e}", flush=True)


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_data_path", required=True)
    p.add_argument("--reload_path", required=True)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=3)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    env_opt = (_env_str("ARC_TORCH_OPTIMIZER") or _env_str("ARC_OPTIMIZER") or "adamw").lower()
    env_opt = env_opt if env_opt in {"adamw", "muon", "adamuon", "adafactor"} else "adamw"
    p.add_argument("--optimizer", type=str, default=env_opt, choices=["adamw", "muon", "adamuon", "adafactor"])
    p.add_argument("--muon_learning_rate", type=float, default=_env_float("ARC_TORCH_MUON_LR"))
    p.add_argument("--muon_momentum", type=float, default=_env_float("ARC_TORCH_MUON_MOMENTUM") or 0.95)
    p.add_argument(
        "--muon_exclude_keywords",
        type=str,
        default=_env_str("ARC_TORCH_MUON_EXCLUDE", "bias,layernorm,norm,embedding,embeddings,lm_head"),
    )
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--save_steps", type=int, default=_env_int("SAVE_STEPS", 500))
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--max_target_length", type=int, default=600)
    p.add_argument("--enable_token_filtering", action="store_true")
    p.add_argument("--use_mixed_precision", action="store_true")
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--tokenizer_dropout_enabled", action="store_true")
    p.add_argument("--tokenizer_dropout_rate", type=float, default=0.1)
    p.add_argument("--tokenizer_dropout_apply_to_labels", action="store_true", help="Apply tokenizer dropout to labels as well (when enabled)")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_strategy", type=str, default="no", choices=["steps", "epoch", "no"])
    # Legacy alias retained for documentation/tests: evaluation_strategy
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_strategy", type=str, default="no", choices=["steps", "epoch", "no"])
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_total_limit", type=int, default=_env_int("SAVE_TOTAL_LIMIT", 2))
    p.add_argument("--save_optimizer", type=_cli_bool, default=True)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--metrics_output", type=str, default=None)
    p.add_argument("--is_causal_lm", action="store_true")
    p.add_argument("--model_type", type=str, default="seq2seq", choices=["seq2seq", "causal_lm"])
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument(
        "--lr_reset_examples",
        type=int,
        default=0,
        help="Reset learning rate scheduler after this many training examples (0 disables resets).",
    )
    p.add_argument("--last_top2_score", type=float, default=None)
    p.add_argument("--pretraining", action="store_true")
    p.add_argument("--pretraining_objective", type=str, default="standard")
    p.add_argument("--pretraining_objective_settings", type=str, default=None)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_task_type", type=str, default=None)
    p.add_argument("--lora_target_modules", action="append", default=None)
    p.add_argument("--lora_modules_to_save", action="append", default=None)
    p.add_argument("--lora_init_lora_weights", type=str, default=None)
    p.add_argument("--lora_rank_pattern", type=str, default=None)
    p.add_argument("--lora_scaling", type=float, default=None)
    p.add_argument("--lora_use_dora", action="store_true")
    p.add_argument("--lora_layers_to_transform", action="append", default=None)
    p.add_argument("--lora_layers_pattern", type=str, default=None)
    args = p.parse_args()
    if not hasattr(args, "is_coda_model"):
        args.is_coda_model = False
    if not hasattr(args, "is_dream_model"):
        args.is_dream_model = False

    if isinstance(args.pretraining_objective, str):
        args.pretraining_objective = args.pretraining_objective.strip().lower() or "standard"
    else:
        args.pretraining_objective = "standard"
    if args.pretraining_objective_settings not in (None, ""):
        try:
            parsed = json.loads(args.pretraining_objective_settings)
            args.pretraining_objective_settings = parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            print(f"[UL2] Warning: could not parse objective settings JSON: {exc}", flush=True)
            args.pretraining_objective_settings = {}
    else:
        args.pretraining_objective_settings = {}

    if args.save_steps is not None:
        try:
            args.save_steps = int(args.save_steps)
        except Exception:
            args.save_steps = None
    if args.save_steps is not None and args.save_steps <= 0:
        args.save_steps = None
    if args.save_strategy == "no" and args.save_steps:
        print(f"Enabling periodic checkpoints every {args.save_steps} steps.", flush=True)
        args.save_strategy = "steps"

    setup_timeout(args.timeout)
    try:
        print(f"LOSS_SETTINGS: label_smoothing={args.label_smoothing}", flush=True)
        if args.pretraining and args.pretraining_objective not in {"", "standard"}:
            print(f"[Pretraining] Objective={args.pretraining_objective}", flush=True)
        print(f"Loading model from {args.model_path}...", flush=True)

        model, tokenizer, label_tokenizer = _load_model_and_tokenizer(args, args.model_path)
        print(f"Tokenizer pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id}", flush=True)
        if not args.is_causal_lm:
            print(f"Decoder start id: {getattr(model.config, 'decoder_start_token_id', None)}", flush=True)

        model_settings = getattr(config, "MODEL_SETTINGS", {}).get(args.model_path, {}) or {}
        training_block = model_settings.get("training") or {}
        early_stop_cfg = training_block.get("early_stopping") if isinstance(training_block, dict) else {}
        if args.pretraining and isinstance(early_stop_cfg, dict):
            if early_stop_cfg.get("enabled"):
                print("[Pretraining] Early stopping disabled for pretraining run.", flush=True)
            early_stop_cfg = {}
        coda_train_cfg = training_block.get("coda") or {}
        dream_train_cfg = training_block.get("dream") or {}
        args.dream_train_config = dream_train_cfg

        # gradient checkpointing (non-reentrant for DDP)
        if args.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "encoder"):
                model.encoder.gradient_checkpointing = True
            if hasattr(model, "decoder"):
                model.decoder.gradient_checkpointing = True
            print(f"[Rank {os.getenv('LOCAL_RANK', 'N/A')}] Enabled non-reentrant gradient checkpointing.", flush=True)

        # datasets
        from datasets import load_from_disk

        ds = load_from_disk(args.train_data_path)
        if "train" in ds:
            train_ds, eval_ds = ds["train"], ds.get("test", None)
        else:
            train_ds, eval_ds = ds, None
        if args.eval_strategy != "no" and (eval_ds is None or len(eval_ds) == 0):
            n = max(50, min(int(0.1 * len(train_ds)), 500))
            eval_ds = train_ds.select(range(min(n, len(train_ds))))
            train_ds = train_ds.select(range(min(n, len(train_ds)), len(train_ds)))
        args.log_dir = (
            args.log_dir
            or os.environ.get("ARC_TRAINING_LOG_DIR")
            or (os.environ.get("ARC_LOG_DIR") and os.path.join(os.environ["ARC_LOG_DIR"], "training_logs"))
            or "training_logs"
        )
        print(f"Train={len(train_ds)} Eval={0 if eval_ds is None else len(eval_ds)}", flush=True)
        if args.last_top2_score is not None:
            try:
                last_score_val = float(args.last_top2_score)
                print(
                    f"📈 Previous ensemble top-2 score: {last_score_val:.3f} ({last_score_val * 100:.2f}%)", flush=True
                )
            except Exception:
                print(f"📈 Previous ensemble top-2 score: {args.last_top2_score}", flush=True)

        ul2_objective = str(getattr(args, "pretraining_objective", "standard") or "standard").lower()
        ul2_active = (ul2_objective == "ul2") and not getattr(args, "is_coda_model", False) and not args.is_causal_lm

        if getattr(args, "is_coda_model", False):
            preprocess = _prep_coda(args, tokenizer)
        elif args.is_causal_lm:
            preprocess = _prep_causal(args, tokenizer)
        elif ul2_active:
            preprocess = _prep_ul2(args, tokenizer)
        else:
            preprocess = _prep_seq2seq(args, tokenizer, label_tokenizer)
        rm_cols = [c for c in ("prompt", "correct_answer") if c in train_ds.column_names]
        token_train = train_ds.map(preprocess, batched=True, remove_columns=rm_cols, desc="Preprocess Train")
        token_eval = (
            None
            if eval_ds is None
            else eval_ds.map(preprocess, batched=True, remove_columns=rm_cols, desc="Preprocess Eval")
        )

        bf16_ok = args.use_bf16 or (args.use_mixed_precision and torch.cuda.is_bf16_supported())
        fp16_ok = args.use_mixed_precision and not bf16_ok
        if getattr(args, "is_coda_model", False):
            collator = DataCollatorForCoDA(tokenizer, pad_to_multiple_of=8 if (bf16_ok or fp16_ok) else None)
        elif args.is_causal_lm:
            collator = DataCollatorForCausalLM(tokenizer, pad_to_multiple_of=8 if (bf16_ok or fp16_ok) else None)
        elif ul2_active:
            settings = args.pretraining_objective_settings if isinstance(args.pretraining_objective_settings, dict) else {}
            collator = DataCollatorForUL2Objective(
                tokenizer=tokenizer,
                pad_to_multiple_of=8 if (bf16_ok or fp16_ok) else None,
                return_tensors="pt",
                max_length=getattr(args, "max_input_length", None),
                settings=settings,
                seed=getattr(args, "seed", None),
            )
            try:
                summary = collator.mixture_summary()
                if summary:
                    print(f"[UL2] Objective mixture: {summary}", flush=True)
            except Exception:
                pass
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8 if (bf16_ok or fp16_ok) else None,
                return_tensors="pt",
            )

        # TrainingArguments
        gc_kwargs = {"use_reentrant": False} if args.use_gradient_checkpointing else None
        targs = dict(
            output_dir=args.reload_path,
            num_train_epochs=args.num_train_epochs,
            max_steps=(args.max_steps if args.max_steps and args.max_steps > 0 else -1),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.use_gradient_checkpointing,
            fp16=fp16_ok,
            bf16=bf16_ok,
            ddp_find_unused_parameters=True if args.use_gradient_checkpointing else None,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=(args.eval_steps if args.eval_strategy == "steps" else None),
            save_strategy=args.save_strategy,
            save_steps=(args.save_steps if args.save_strategy == "steps" else None),
            save_total_limit=args.save_total_limit,
            save_safetensors=True,
            load_best_model_at_end=(args.eval_strategy != "no" and args.save_strategy != "no"),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            prediction_loss_only=True,
        )
        if gc_kwargs:
            targs["gradient_checkpointing_kwargs"] = gc_kwargs

        TArgsCls = TrainingArguments if args.is_causal_lm else Seq2SeqTrainingArguments
        training_args = TArgsCls(**targs)
        setattr(training_args, "save_optimizer", bool(getattr(args, "save_optimizer", True)))
        setattr(training_args, "bitsandbytes_is_4bit", bool(getattr(args, "bitsandbytes_is_4bit", False)))
        setattr(training_args, "bitsandbytes_is_8bit", bool(getattr(args, "bitsandbytes_is_8bit", False)))

        # Build trainer (causal variant reuses standard CE logic via small subclass)
        if args.is_causal_lm:

            class EnhancedCausalLMTrainer(TokenLoggingMixin, Trainer):
                def __init__(self, *, coda_settings=None, dream_settings=None, **kw):
                    self.coda_settings = coda_settings or {}
                    super().__init__(**kw)
                    self.max_grad_norm = args.max_grad_norm
                    self.label_smoothing = args.label_smoothing
                    self.optimizer_choice = args.optimizer
                    self.muon_learning_rate = args.muon_learning_rate
                    self.muon_momentum = args.muon_momentum
                    self.muon_exclude_keywords = _normalize_muon_exclusions(args.muon_exclude_keywords)
                    self.coda_training_mode = self.coda_settings.get("training_mode", "sft")
                    self.coda_masking_schedule = self.coda_settings.get("masking_schedule")
                    self._init_token_logging()

                def create_optimizer(self):
                    if self.optimizer is not None:
                        return self.optimizer
                    use_muon = self.optimizer_choice == "muon"
                    muon, dec, nodec = _split_param_groups(self, use_muon, self.muon_exclude_keywords)
                    self.optimizer = (
                        _mk_muon(
                            self,
                            muon,
                            dec,
                            nodec,
                            (self.muon_learning_rate or self.args.learning_rate),
                            self.muon_momentum,
                        )
                        if use_muon
                        else _mk_adamw(self, dec, nodec)
                    )
                    return self.optimizer

                def compute_loss(self, model, inputs, return_outputs=False):
                    inputs = self._prepare_inputs(inputs)
                    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    if args.is_coda_model:
                        model_inputs.setdefault("training_mode", self.coda_training_mode)
                        if self.coda_masking_schedule is not None:
                            model_inputs.setdefault("masking_schedule", self.coda_masking_schedule)
                        if "epoch" not in model_inputs or model_inputs.get("epoch") is None:
                            epoch_val = getattr(self.state, "epoch", None)
                            model_inputs["epoch"] = int(epoch_val) if epoch_val is not None else None
                    out = model(**model_inputs)
                    if args.is_coda_model:
                        loss = None
                        if isinstance(out, tuple):
                            if len(out) > 1 and torch.is_tensor(out[1]) and out[1].ndim == 0:
                                loss = out[1]
                            elif len(out) > 0 and torch.is_tensor(out[0]) and out[0].ndim == 0:
                                loss = out[0]
                        else:
                            loss = getattr(out, "loss", None)
                        if loss is None:
                            raise ValueError("CoDA model did not return a loss value.")
                        if return_outputs:
                            return (loss, out)
                        return loss
                    logits = out.logits
                    labels = inputs.get("labels")
                    if labels is None:
                        return super().compute_loss(model, inputs, return_outputs)
                    per = _token_ce_loss(logits, labels, self.label_smoothing)
                    valid_mask = labels.ne(-100)
                    valid_weights = valid_mask.to(per.dtype)
                    valid_count = valid_weights.sum()
                    if bool(valid_mask.any()):
                        loss = (per * valid_weights).sum() / valid_count.clamp(min=1.0)
                    else:
                        loss = per.mean()
                    if return_outputs:
                        out.loss = loss
                        return (loss, out)
                    return loss

                def training_step(self, model, inputs, num_items_in_batch=None):
                    model.train()
                    inputs = self._prepare_inputs(inputs)
                    self._update_token_counters(inputs)
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, inputs)
                    if self.args.n_gpu > 1:
                        loss = loss.mean()
                    if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                        loss = loss / self.args.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    if self.max_grad_norm > 0:
                        if hasattr(self.accelerator, "clip_grad_norm_"):
                            self.accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    return loss.detach() / self.args.gradient_accumulation_steps

                def log(self, logs: dict, start_time=None) -> None:  # type: ignore[override]
                    L = dict(logs or {})
                    L = self._augment_logs_with_token_metrics(L)
                    last_top2 = getattr(self, "last_top2_score", None)
                    if last_top2 is not None:
                        L.setdefault("prev_top2_score", float(last_top2))
                        L.setdefault("prev_top2_pct", float(last_top2) * 100.0)
                    super().log(L)

            trainer = EnhancedCausalLMTrainer(
                model=model,
                args=training_args,
                train_dataset=token_train,
                eval_dataset=token_eval,
                data_collator=collator,
                processing_class=tokenizer if _TRAINER_SUPPORTS_PROCESSING else None,
                coda_settings=coda_train_cfg if getattr(args, "is_coda_model", False) else None,
            )
            if getattr(trainer, "processing_class", None) is None:
                trainer.processing_class = tokenizer
        else:
            trainer = EnhancedSeq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=token_train,
                eval_dataset=token_eval,
                data_collator=collator,
                label_smoothing=args.label_smoothing,
                max_grad_norm=args.max_grad_norm,
                optimizer_choice=args.optimizer,
                muon_learning_rate=args.muon_learning_rate,
                muon_momentum=args.muon_momentum,
                muon_exclude_keywords=args.muon_exclude_keywords,
                processing_class=tokenizer if _SEQ2SEQ_TRAINER_SUPPORTS_PROCESSING else None,
            )
            if getattr(trainer, "processing_class", None) is None:
                trainer.processing_class = tokenizer

        lr_reset_examples = max(0, int(getattr(args, "lr_reset_examples", 0) or 0))
        if lr_reset_examples > 0:
            trainer.add_callback(LearningRateResetCallback(lr_reset_examples))

        trainer.last_top2_score = None
        if getattr(args, "last_top2_score", None) is not None:
            try:
                trainer.last_top2_score = float(args.last_top2_score)
            except Exception:
                trainer.last_top2_score = None

        # Seeds + logging callback
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        ctx_size = args.max_length if args.is_causal_lm else args.max_input_length
        cb = CSVLoggingCallback(
            args.log_dir, os.path.basename(args.model_path), float(args.learning_rate), int(ctx_size), args.run_id
        )
        if getattr(cb, "enabled", False):
            trainer.add_callback(cb)

        if isinstance(early_stop_cfg, dict) and early_stop_cfg.get("enabled"):
            configured = False
            metric_name = None
            target_value = None
            if "accuracy_pct_target" in early_stop_cfg:
                metric_name = "accuracy_pct"
                target_value = early_stop_cfg.get("accuracy_pct_target")
            elif "base_accuracy_pct_target" in early_stop_cfg:
                metric_name = "base_accuracy_pct"
                target_value = early_stop_cfg.get("base_accuracy_pct_target")
            if metric_name and target_value is not None:
                try:
                    target_f = float(target_value)
                except Exception:
                    target_f = None
                if target_f is not None:
                    warmup_steps = early_stop_cfg.get("warmup_steps", 0)
                    try:
                        warmup_i = int(warmup_steps)
                    except Exception:
                        warmup_i = 0
                    threshold_cb = AccuracyThresholdEarlyStoppingCallback(
                        model_name=os.path.basename(args.model_path),
                        metric_name=metric_name,
                        target=target_f,
                        warmup_steps=warmup_i,
                    )
                    trainer.add_callback(threshold_cb)
                    trainer._arc_early_stop_cb = threshold_cb
                    configured = True
            if not configured:
                metrics_cfg = early_stop_cfg.get("metrics")
                if isinstance(metrics_cfg, str):
                    metrics = [metrics_cfg]
                elif isinstance(metrics_cfg, (list, tuple)):
                    metrics = [str(m) for m in metrics_cfg if isinstance(m, (str, bytes))]
                else:
                    metrics = []
                fallback_metric = early_stop_cfg.get("metric")
                if fallback_metric:
                    metrics.append(str(fallback_metric))
                metrics = [m.strip() for m in metrics if m and m.strip()]
                if metrics:
                    metrics = list(dict.fromkeys(metrics))
                    mode = str(early_stop_cfg.get("mode", "max") or "max")
                    patience = early_stop_cfg.get("patience", 5)
                    warmup_steps = early_stop_cfg.get("warmup_steps", 0)
                    min_delta = early_stop_cfg.get("min_delta", 0.0)
                    target = early_stop_cfg.get("target")
                    try:
                        patience_i = int(patience)
                    except Exception:
                        patience_i = 5
                    try:
                        warmup_i = int(warmup_steps)
                    except Exception:
                        warmup_i = 0
                    try:
                        min_delta_f = float(min_delta)
                    except Exception:
                        min_delta_f = 0.0
                    try:
                        target_f = float(target) if target is not None else None
                    except Exception:
                        target_f = None
                    early_stop_cb = MetricEarlyStoppingCallback(
                        model_name=os.path.basename(args.model_path),
                        metrics=metrics,
                        mode=mode,
                        patience=patience_i,
                        min_delta=min_delta_f,
                        warmup_steps=warmup_i,
                        target=target_f,
                    )
                    trainer.add_callback(early_stop_cb)
                    trainer._arc_early_stop_cb = early_stop_cb

        # Train
        # Print one or two raw prompt/answer examples for debugging formatting
        try:
            for idx in range(2):
                sample = train_ds[idx] if idx < len(train_ds) else None
                if not sample:
                    break
                model_kind = ("causal" if args.is_causal_lm or args.model_type == "causal_lm" else "seq2seq")
                prompt_val = sample.get("prompt") if isinstance(sample, dict) else None
                answer_val = sample.get("correct_answer") if isinstance(sample, dict) else None
                if model_kind == "causal" and prompt_val is not None and answer_val is not None:
                    combo, prefix = prepare_causal_prompt_with_target(prompt_val, answer_val)
                    print(f"[TTT Sample {idx}][CAUSAL]\nPrompt:\n{prompt_val}\n---\nAnswer (raw):\n{answer_val}\n---\nCombined:\n{combo}\n", flush=True)
                elif model_kind == "seq2seq":
                    if prompt_val is not None or answer_val is not None:
                        print(f"[TTT Sample {idx}][SEQ2SEQ]\nPrompt:\n{prompt_val}\n---\nAnswer:\n{answer_val}\n", flush=True)
                    elif "text" in sample:
                        print(f"[TTT Sample {idx}] text field:\n{sample['text']}", flush=True)
                else:
                    # Fallback generic display
                    printable = {k: v for k, v in sample.items() if k in {"prompt", "correct_answer", "text"}}
                    print(f"[TTT Sample {idx}] {printable}", flush=True)
        except Exception as sample_err:  # keep training resilient
            print(f"[TTT Sample Print Warn] {sample_err}", flush=True)
        logger.info("Starting training...")
        train_result = trainer.train()
        if hasattr(train_result, "metrics"):
            for k, v in train_result.metrics.items():
                logger.info(f"{k}: {v}")
        early_stop_cb = getattr(trainer, "_arc_early_stop_cb", None)
        if early_stop_cb and getattr(early_stop_cb, "stop_reason", None):
            trainer._arc_early_stop_reason = early_stop_cb.stop_reason
            logger.info(f"Early stopping triggered: {early_stop_cb.stop_reason}")

        metrics_payload = None
        try:
            metrics_payload = _build_training_metrics(args, trainer, train_result, train_ds, token_train)
        except Exception as metrics_err:
            print(f"Training metrics build warn: {metrics_err}", flush=True)
            metrics_payload = None
        if args.metrics_output and metrics_payload:
            try:
                with open(args.metrics_output, "w") as mf:
                    json.dump(metrics_payload, mf)
            except Exception as metrics_err:
                print(f"Training metrics write warn: {metrics_err}", flush=True)

        # Save (rank 0 handled by HF)
        print(f"Saving final model to {args.reload_path} ...", flush=True)
        if getattr(args, "use_lora", False):
            try:
                from peft import PeftModel  # type: ignore

                if isinstance(trainer.model, PeftModel):
                    print("Merging LoRA adapters into base model for export…", flush=True)
                    merged_model = trainer.model.merge_and_unload()
                    trainer.model = merged_model
                    model = merged_model
                    _ensure_generation_config_compat(trainer.model)
            except Exception as merge_err:
                print(f"LoRA merge warn: {merge_err}", flush=True)
        trainer.save_model()
        if getattr(trainer, "is_world_process_zero", lambda: True)():
            tokenizer.save_pretrained(args.reload_path)
            print("Tokenizer saved.", flush=True)
        # cleanup checkpoints if not preserving
        try:
            from . import config as _cfg
        except Exception:
            import config as _cfg
        if not getattr(_cfg, "PRESERVE_FINETUNED_MODEL", False):
            for ck in glob.glob(os.path.join(args.reload_path, "checkpoint-*")):
                try:
                    import shutil

                    shutil.rmtree(ck)
                    print(f"Cleaned {ck}", flush=True)
                except Exception as e:
                    print(f"Cleanup warn {ck}: {e}", flush=True)

        # overall CSV
        if args.log_dir:
            try:
                import csv

                os.makedirs(args.log_dir, exist_ok=True)
                overall = os.path.join(args.log_dir, "training_overall.csv")
                metrics = dict(getattr(train_result, "metrics", {}))
                row = {
                    "timestamp": time.time(),
                    "model_name": os.path.basename(args.model_path),
                    "run_id": args.run_id or "",
                    "learning_rate": args.learning_rate,
                    "context_size": ctx_size,
                    "global_steps": getattr(trainer.state, "global_step", None),
                    **{k: metrics.get(k) for k in sorted(metrics.keys())},
                }
                with open(overall, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerow(row)
            except Exception as e:
                print(f"Overall CSV warn: {e}", flush=True)

        print("Training completed successfully!", flush=True)

    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        logger.error("Training failed: %s", e)
        traceback.print_exc(file=sys.stderr)  # traceback.print_exc()
        sys.exit(1)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
