#!/usr/bin/env python3
from contextlib import suppress
from dataclasses import dataclass, replace
import copy
import glob
import logging
import math
import os
import re
import signal
import sys
import textwrap
import time
from typing import Any
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*assign=True.*", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch  # noqa: E402
from torch.cuda import empty_cache  # noqa: E402
import tqdm  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402
from transformers.modeling_flax_pytorch_utils import (  # noqa: E402
    load_flax_checkpoint_in_pytorch_model,
    load_flax_weights_in_pytorch_model,
)


try:
    from .hf_compat import apply_t5gemma_config_patch
except Exception:  # pragma: no cover - script mode fallback
    from hf_compat import apply_t5gemma_config_patch  # type: ignore

apply_t5gemma_config_patch("inference_worker")

try:
    from . import config
    from .grid_utils import grid_to_string, is_grid, makeprompt, output_prefix, output_to_grid
    from .transformations import mix_augms
    from .quantization_utils import build_bitsandbytes_kwargs
except Exception:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from grid_utils import grid_to_string, is_grid, makeprompt, output_prefix, output_to_grid
    from transformations import mix_augms
    from quantization_utils import build_bitsandbytes_kwargs

try:
    from .terminal_vis import draw_grid_terminal  # type: ignore
except Exception:  # pragma: no cover - optional import
    try:
        from terminal_vis import draw_grid_terminal  # type: ignore
    except Exception:
        draw_grid_terminal = None

logger = logging.getLogger(__name__)

_GRAPH_NN_HELPERS: dict[str, Any] = {}

MAX_SIGNAL_SECONDS = (1 << 31) - 1

# Track models that have encountered mixed-precision failures so we can auto-disable
# MP for the remainder of the worker lifetime.
_MP_DISABLED_MODELS: set[int] = set()


_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")

def _sanitize_model_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _SPECIAL_TOKEN_PATTERN.sub(" ", str(text))
    cleaned = re.sub(r"<im_(start|end)>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    cleaned = re.sub(r"[\t\r]", " ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()

@dataclass
class GenerationConfig:
    num_beams: int = 1
    num_return_sequences: int = 1
    top_k: int | None = None
    top_p: float | None = None
    temperature: float = 1.0
    max_generation_length: int = 600
    max_input_length: int = 2048
    use_mixed_precision: bool = False
    model_type: str = "seq2seq"
    is_causal_lm: bool = False
    max_length: int = 4096
    generation_mode: str = "standard"
    diffusion_steps: int = 1
    diffusion_eps: float = 1e-4
    diffusion_alg: str | None = None
    diffusion_alg_temp: float = 1.0
    coda_quality_profile: str | None = None
    coda_quality_presets: dict[str, Any] | None = None


@dataclass
class AdaptiveConfig:
    min_batch_size: int = 1
    max_oom_retries: int = 3
    oom_reduction_factor: float = 0.5


@dataclass
class PreviewController:
    enabled: bool
    interval: int
    max_samples: int
    rank: int
    batches: int = 0
    total: int = 0
    non_empty: int = 0
    sample_offset: int = 0

    def process_batch(
        self,
        payloads: list[dict[str, Any]],
        texts_batch: list[list[str]],
        worker_settings: dict[str, Any],
    ) -> None:
        if not self.enabled or not payloads or not texts_batch:
            return
        self.batches += 1
        batch_total = len(texts_batch)
        self.total += batch_total
        self.non_empty += sum(1 for seqs in texts_batch if seqs and _sanitize_model_text(seqs[0]))
        if self.interval <= 0 or (self.batches % self.interval) != 0:
            return
        pct = (self.non_empty / self.total) * 100.0 if self.total else 0.0
        print(
            f"\n[Preview][Rank {self.rank}] Batch {self.batches}: "
            f"processed {self.total} prompts; non-empty {self.non_empty} ({pct:.1f}%)"
        )
        sample_span = min(self.max_samples, len(payloads))
        if sample_span <= 0:
            return
        for offset in range(sample_span):
            idx = (self.sample_offset + offset) % len(payloads)
            seqs = texts_batch[idx] if idx < len(texts_batch) else None
            raw_text = seqs[0] if seqs else ""
            payload = payloads[idx]
            label = payload.get("display") or payload.get("base_key") or f"sample_{idx}"
            _emit_preview_sample(
                label,
                payload.get("prompt", ""),
                raw_text,
                payload.get("expected_text"),
                payload.get("expected_grid"),
                worker_settings,
            )
        self.sample_offset = (self.sample_offset + sample_span) % max(1, len(payloads))
        sys.stdout.flush()


def timeout_handler(*_):
    print(f"[Rank {os.getenv('LOCAL_RANK', 'N/A')}] Inference timeout, exiting.")
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


def _maybe_autocast(enabled: bool):
    from contextlib import nullcontext

    if enabled and torch.cuda.is_available():
        dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _truthy(v) -> bool:
    return v if isinstance(v, bool) else (str(v).strip().lower() in ("1", "true", "yes", "on"))


# =============================== Model load ===============================
def _coerce_generation_config(cfg) -> None:
    """Ensure HF configs expose attributes expected by modern generation helpers."""
    if cfg is None:
        return
    try:
        if getattr(cfg, "is_encoder_decoder", False):
            if not hasattr(cfg, "encoder_layers") and hasattr(cfg, "num_layers"):
                cfg.encoder_layers = cfg.num_layers
            if not hasattr(cfg, "decoder_layers") and hasattr(cfg, "num_decoder_layers"):
                cfg.decoder_layers = cfg.num_decoder_layers
        if not hasattr(cfg, "num_hidden_layers"):
            if hasattr(cfg, "num_layers"):
                cfg.num_hidden_layers = cfg.num_layers
            elif hasattr(cfg, "encoder_layers"):
                cfg.num_hidden_layers = cfg.encoder_layers
        if not hasattr(cfg, "num_attention_heads") and hasattr(cfg, "num_heads"):
            cfg.num_attention_heads = cfg.num_heads
    except Exception as err:
        logger.debug("Config patch skipped: %s", err)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_pct(value, default_pct: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default_pct
    if val <= 1.0:
        return val * 100.0
    return val


def _indent_lines(text: str, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _count_grid_differences(grid_a: list[list[int]], grid_b: list[list[int]]) -> int | None:
    if not (is_grid(grid_a) and is_grid(grid_b)):
        return None
    rows = max(len(grid_a), len(grid_b))
    cols = max(len(grid_a[0]), len(grid_b[0])) if rows else 0
    diff = 0
    for r in range(rows):
        for c in range(cols):
            va = grid_a[r][c] if r < len(grid_a) and c < len(grid_a[r]) else None
            vb = grid_b[r][c] if r < len(grid_b) and c < len(grid_b[r]) else None
            if va != vb:
                diff += 1
    return diff


def _emit_preview_sample(
    label: str,
    prompt: str,
    raw_text: str,
    expected_text: str | None,
    expected_grid: list[list[int]] | None,
    worker_settings: dict[str, Any],
) -> None:
    snippet_width = _safe_int(worker_settings.get("preview_snippet_chars", 160), 160)
    prompt_width = _safe_int(worker_settings.get("preview_prompt_chars", 160), 160)
    include_prompt = _truthy(worker_settings.get("preview_include_prompt_snippet", False))
    show_grid = _truthy(worker_settings.get("preview_show_terminal_grid", False))

    cleaned_text = _sanitize_model_text(raw_text)
    summary_source = cleaned_text.replace("\n", " ")
    summary = textwrap.shorten(summary_source, width=snippet_width, placeholder="…") if summary_source else "<empty>"
    print(f"  • {label}: {summary}")

    cleaned_expected = _sanitize_model_text(expected_text)
    expected_line = cleaned_expected.replace("\n", " ") if cleaned_expected else ""
    if expected_line:
        target_preview = textwrap.shorten(expected_line, width=snippet_width, placeholder="…")
        print(f"    Target: {target_preview}")

    if include_prompt:
        prompt_s = " ".join((prompt or "").split())
        prompt_preview = textwrap.shorten(prompt_s, width=prompt_width, placeholder="…") if prompt_s else ""
        if prompt_preview:
            print(f"    Prompt: {prompt_preview}")
    if show_grid:
        if not draw_grid_terminal:
            print("    [Preview] Terminal visualization unavailable in this environment.")
        else:
            refined_grid = output_to_grid(cleaned_text) if cleaned_text else None
            if refined_grid is not None:
                rendered = draw_grid_terminal(refined_grid, title=f"Output grid ({label})")
                print(_indent_lines(rendered))
            elif raw_text:
                print("    [Preview] Output grid unavailable (output_to_grid failed).")
                print(f"    Output text: {repr(cleaned_text or raw_text)}")

            if expected_grid:
                rendered_target = draw_grid_terminal(expected_grid, title=f"Target grid ({label})")
                print(_indent_lines(rendered_target))


def get_model(
    model_path,
    rank,
    use_torch_compile=False,
    model_type="seq2seq",
    is_causal_lm=False,
    tokenizer_dropout_enabled=False,
    tokenizer_dropout_rate=0.1,
    attn_implementation=None,
):
    loader = _ModelLoader(
        model_path=model_path,
        rank=rank,
        use_torch_compile=use_torch_compile,
        model_type=model_type,
        is_causal_lm=is_causal_lm,
        tokenizer_dropout_enabled=tokenizer_dropout_enabled,
        tokenizer_dropout_rate=tokenizer_dropout_rate,
        attn_implementation=attn_implementation,
    )
    return loader.load()


class _ModelLoader:
    def __init__(
        self,
        *,
        model_path: str,
        rank: int,
        use_torch_compile: bool,
        model_type: str,
        is_causal_lm: bool,
        tokenizer_dropout_enabled: bool,
        tokenizer_dropout_rate: float,
        attn_implementation: str | None,
    ) -> None:
        self.model_path = model_path
        self.rank = rank
        self.use_torch_compile = use_torch_compile
        self.requested_model_type = model_type
        self.is_causal_lm = is_causal_lm
        self.tokenizer_dropout_enabled = tokenizer_dropout_enabled
        self.tokenizer_dropout_rate = tokenizer_dropout_rate
        self.attn_implementation = attn_implementation
        self.settings = getattr(config, "MODEL_SETTINGS", {}).get(model_path, {}) or {}
        self.general_settings = self.settings.get("general", {}) or {}
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        prompt_settings = self.general_settings.get("prompt_settings") or {}
        self.prompt_general_prefix = str(prompt_settings.get("general_prefix") or "")
        self.prompt_format = str(self.general_settings.get("prompt_format") or "legacy").lower()
        self._prompt_tokens_added = False
        self._quantization_info: dict[str, Any] | None = None
        bb_cfg = self.general_settings.get("bitsandbytes") if isinstance(self.general_settings, dict) else None
        self._quantization_requested = bool(isinstance(bb_cfg, dict) and bb_cfg.get("enabled"))

    def load(self):
        load_path = self._resolve_load_path()
        tokenizer = self._initialize_tokenizer(load_path)
        model_kwargs = self._build_model_kwargs(load_path)
        model = self._create_model(load_path, model_kwargs, tokenizer)
        self._resize_token_embeddings_if_needed(model, tokenizer)
        if self._quantization_requested and self._quantization_info and not self._quantization_info.get("enabled"):
            message = self._quantization_info.get("message")
            if message:
                print(
                    f"[Quantization] {message}. Falling back to standard precision for {self.model_path}.",
                    flush=True,
                )
        model = self._finalize_model(model_kwargs, model)
        return model, tokenizer, self.device

    # ---- path & tokenizer -------------------------------------------------

    def _resolve_load_path(self) -> str:
        return self.model_path

    def _initialize_tokenizer(self, load_path: str):
        sampling = self.tokenizer_dropout_enabled and self.tokenizer_dropout_rate > 0
        extra = {"enable_sampling": True, "alpha": float(self.tokenizer_dropout_rate)} if sampling else {}
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, **extra)
        special_tokens = self._collect_prompt_special_tokens()
        if special_tokens:
            tokens_to_add = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
            if tokens_to_add:
                with suppress(Exception):
                    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
                    self._prompt_tokens_added = True
        return tokenizer

    def _collect_prompt_special_tokens(self) -> list[str]:
        trimmed = str(self.prompt_general_prefix or "").strip()
        if not trimmed:
            return []
        token = trimmed.split()[0]
        if token and token.startswith("<") and token.endswith(">"):
            return [token]
        return []

    # ---- model kwargs -----------------------------------------------------

    def _build_model_kwargs(self, load_path: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        quant_cfg = self.general_settings.get("bitsandbytes")
        quant_kwargs, quant_info = build_bitsandbytes_kwargs(
            quant_cfg,
            rank=self.rank,
            for_training=False,
        )
        self._quantization_info = quant_info
        if quant_info.get("enabled"):
            kwargs.update(quant_kwargs)
            mode = "4bit" if quant_cfg and quant_cfg.get("load_in_4bit") else "8bit"
            print(
                f"[Quantization] Loading {self.model_path} with bitsandbytes ({mode}).",
                flush=True,
            )
        # Explicitly prevent device_map to ensure model stays on single device unless quantization overrode it
        # This prevents accelerate from automatically distributing the model across GPUs
        if "device_map" not in kwargs:
            kwargs["device_map"] = None
        # Also disable low_cpu_mem_usage when device_map is None to ensure standard loading
        if "low_cpu_mem_usage" not in kwargs:
            kwargs["low_cpu_mem_usage"] = False
        self._maybe_enable_flax_bridge(load_path, kwargs)
        config = self._safe_load_config(load_path)
        if config is not None:
            kwargs.setdefault("config", config)
            self.is_causal_lm = not bool(getattr(config, "is_encoder_decoder", False))
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs

    def _maybe_enable_flax_bridge(self, load_path: str, kwargs: dict[str, Any]) -> None:
        if not os.path.isdir(load_path):
            return
        has_direct, has_sharded = self._weight_presence_flags(load_path)
        if load_path == self.model_path and (has_direct or has_sharded):
            return
        kwargs["from_flax"] = True
        kwargs.setdefault("low_cpu_mem_usage", False)

    def _weight_presence_flags(self, load_path: str) -> tuple[bool, bool]:
        direct_files = (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.safetensors",
            "model.bin",
        )
        has_direct = any(os.path.exists(os.path.join(load_path, name)) for name in direct_files)
        has_sharded = bool(glob.glob(os.path.join(load_path, "model-*.safetensors")))
        has_sharded |= os.path.exists(os.path.join(load_path, "model.safetensors.index.json"))
        has_sharded |= bool(glob.glob(os.path.join(load_path, "pytorch_model-*.bin")))
        has_sharded |= os.path.exists(os.path.join(load_path, "pytorch_model.bin.index.json"))
        return has_direct, has_sharded

    def _safe_load_config(self, load_path: str):
        with suppress(Exception):
            return AutoConfig.from_pretrained(load_path, trust_remote_code=True)
        return None

    # ---- model creation ---------------------------------------------------

    def _create_model(self, load_path: str, model_kwargs: dict[str, Any], tokenizer):
        if self.is_causal_lm:
            model = self._load_causal_model(load_path, model_kwargs)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            diffusion_enabled = bool(
                self.general_settings.get("coda_enabled") or self.general_settings.get("dream_enabled")
            )
            tokenizer.padding_side = "right" if diffusion_enabled else "left"
            return model
        return self._load_seq2seq_model(load_path, model_kwargs)

    def _resize_token_embeddings_if_needed(self, model, tokenizer) -> None:
        if not getattr(self, "_prompt_tokens_added", False):
            return
        if not hasattr(model, "resize_token_embeddings"):
            return
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    def _load_causal_model(self, load_path: str, model_kwargs: dict[str, Any]):
        config = model_kwargs.get("config")
        config_class_name = getattr(config, "__class__", None).__name__ if config else ""
        model_type_name = str(getattr(config, "model_type", "") or "").lower()
        is_coda = bool(config) and ("coda" in model_type_name or "coda" in config_class_name.lower())
        is_dream = bool(config) and ("dream" in model_type_name or "dream" in config_class_name.lower())
        if is_coda or is_dream:
            from transformers import AutoModel

            model_kwargs_with_trust = {**model_kwargs, "trust_remote_code": True}
            try:
                return AutoModel.from_pretrained(load_path, **model_kwargs_with_trust)
            except Exception as base_exc:
                if is_dream:
                    raise
                logger.info("AutoModel load failed for %s; falling back to causal loader. Error: %s", load_path, base_exc)
        try:
            return AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs)
        except Exception as exc:
            logger.info(
                "PyTorch checkpoint load failed for %s; attempting Flax fallback. Error: %s",
                load_path,
                exc,
            )
            # Check if this is a CoDA model - needs AutoModel instead of AutoModelForCausalLM
            if config and ("CoDA" in config_class_name or "coda" in config_class_name.lower()):
                from transformers import AutoModel

                model_kwargs_with_trust = {**model_kwargs, "trust_remote_code": True}
                try:
                    return AutoModel.from_pretrained(load_path, **model_kwargs_with_trust)
                except Exception:
                    pass  # Fall through to manual_from_flax
            if config is None:
                raise
            return self._manual_from_flax(
                load_fn=lambda p: __import__("transformers").FlaxAutoModelForCausalLM.from_pretrained(p),
                build_fn=lambda cfg: AutoModelForCausalLM.from_config(cfg),
                load_path=load_path,
                config=config,
            )

    def _load_seq2seq_model(self, load_path: str, model_kwargs: dict[str, Any]):
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(load_path, **model_kwargs)
        except Exception as exc:
            logger.info(
                "PyTorch checkpoint load failed for %s; attempting Flax fallback. Error: %s",
                load_path,
                exc,
            )
            config = model_kwargs.get("config")
            if getattr(config, "__class__", None).__name__ == "T5GemmaConfig":
                from transformers import T5GemmaForConditionalGeneration

                return T5GemmaForConditionalGeneration.from_pretrained(load_path, **model_kwargs)
            if config is None:
                raise
            try:
                return self._manual_from_flax(
                    load_fn=lambda p: __import__("transformers").FlaxAutoModelForSeq2SeqLM.from_pretrained(p),
                    build_fn=lambda cfg: AutoModelForSeq2SeqLM.from_config(cfg),
                    load_path=load_path,
                    config=config,
                )
            except Exception as manual_err:
                raise exc from manual_err

    def _manual_from_flax(self, *, load_fn, build_fn, load_path: str, config):
        os.environ.setdefault("ACCELERATE_DISABLE_DEVICE_MAP", "1")
        model = build_fn(config)
        flax_checkpoint = self._discover_flax_checkpoint(load_path)
        if flax_checkpoint:
            load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint)
        else:
            flax_model = load_fn(load_path)
            try:
                flax_params = getattr(flax_model, "params", None)
                if flax_params is None:
                    raise RuntimeError("Flax model params unavailable for manual conversion")
                if hasattr(flax_params, "keys") and "params" in flax_params:
                    payload = flax_params
                else:
                    payload = {"params": flax_params}
                load_flax_weights_in_pytorch_model(model, payload)
            finally:
                del flax_model
        if hasattr(model, "tie_weights"):
            with suppress(Exception):
                model.tie_weights()
        return model

    def _discover_flax_checkpoint(self, load_path: str) -> str | None:
        if os.path.isdir(load_path):
            candidate = os.path.join(load_path, "flax_model.msgpack")
            if os.path.exists(candidate):
                return candidate
        with suppress(Exception):
            from transformers.utils.hub import cached_file

            return cached_file(load_path, "flax_model.msgpack")
        return None

    # ---- final touches ----------------------------------------------------

    def _finalize_model(self, model_kwargs: dict[str, Any], model):
        model = self._place_model_on_device(model_kwargs, model)
        cfg = getattr(model, "config", None)
        _coerce_generation_config(cfg)
        if self.use_torch_compile and hasattr(torch, "compile"):
            with suppress(Exception):
                compiled = torch.compile(model, mode="reduce-overhead")
                if compiled is not None:
                    model = compiled
        return model

    def _place_model_on_device(self, model_kwargs: dict[str, Any], model):
        # Skip manual device placement if device_map is set to a truthy non-None value
        # (e.g., "auto", "balanced", "sequential", etc.)
        device_map_value = model_kwargs.get("device_map")
        # device_map_value can be None (manual placement), a string like "auto", or a dict
        # Only skip if it's a truthy value (not None, not empty dict)
        if not torch.cuda.is_available():
            return model
        if device_map_value is not None and device_map_value:
            # device_map is explicitly set to something like "auto"
            return model
        # Manually place model on the device assigned to this worker (use self.rank, not LOCAL_RANK)
        dev = f"cuda:{self.rank}"
        try:
            model.to(dev)
        except NotImplementedError as exc:
            if "meta tensor" in str(exc):
                model = model.to_empty(device=dev)
            else:
                raise
        return model


# =============================== Generation ===============================
def _build_coda_generation_config(model, tokenizer, cfg: GenerationConfig, input_ids, attention_mask):
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        try:
            generation_config = copy.deepcopy(generation_config)
        except Exception:
            # Fall back to using the existing config instance but avoid mutating shared state.
            generation_config = generation_config.clone() if hasattr(generation_config, "clone") else generation_config
    else:
        generation_config = None
    if generation_config is None:
        # Attempt to load from the model package
        DLMGenerationConfig = None
        try:
            module = __import__(model.__module__.rsplit(".", 1)[0] + ".generation_utils", fromlist=["DLMGenerationConfig"])
            DLMGenerationConfig = getattr(module, "DLMGenerationConfig", None)
        except Exception:
            DLMGenerationConfig = None
        if DLMGenerationConfig is not None:
            generation_config = DLMGenerationConfig()
        else:
            # Minimal fallback object with attribute assignment support
            class _FallbackConfig:
                pass

            generation_config = _FallbackConfig()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mask_token_id = getattr(model.config, "mask_token_id", getattr(tokenizer, "mask_token_id", None))

    prompt_lengths = attention_mask.sum(dim=1, dtype=torch.int64)
    prompt_len_max = int(prompt_lengths.max().item()) if prompt_lengths.numel() else input_ids.shape[-1]
    max_length = min(
        cfg.max_length or (prompt_len_max + cfg.max_generation_length),
        int(getattr(model.config, "max_position_embeddings", prompt_len_max + cfg.max_generation_length)),
    )

    setattr(generation_config, "max_new_tokens", int(cfg.max_generation_length))
    setattr(generation_config, "temperature", float(max(cfg.temperature or 0.0, 0.0)))
    setattr(generation_config, "top_p", None if cfg.top_p is None else float(cfg.top_p))
    setattr(generation_config, "top_k", None if cfg.top_k is None else int(cfg.top_k))
    setattr(generation_config, "steps", int(cfg.diffusion_steps or 128))
    setattr(generation_config, "eps", float(cfg.diffusion_eps or getattr(generation_config, "eps", 1e-3)))
    if cfg.diffusion_alg:
        setattr(generation_config, "alg", str(cfg.diffusion_alg))
    if cfg.diffusion_alg_temp is not None:
        setattr(generation_config, "alg_temp", float(cfg.diffusion_alg_temp))
    setattr(generation_config, "num_return_sequences", int(cfg.num_return_sequences))
    setattr(generation_config, "return_dict_in_generate", True)
    setattr(generation_config, "output_history", False)
    if pad_token_id is not None:
        setattr(generation_config, "pad_token_id", int(pad_token_id))
    if mask_token_id is not None:
        setattr(generation_config, "mask_token_id", int(mask_token_id))
    if tokenizer.eos_token_id is not None:
        setattr(generation_config, "eos_token_id", int(tokenizer.eos_token_id))
    if getattr(tokenizer, "bos_token_id", None) is not None:
        setattr(generation_config, "bos_token_id", int(tokenizer.bos_token_id))
    setattr(generation_config, "max_length", int(max_length))
    return generation_config


def multipredict_coda(
    model,
    tokenizer,
    device,
    prompts_for_batch,
    *,
    generation: GenerationConfig | None = None,
    payloads_for_batch: list[dict[str, Any]] | None = None,
):
    if not prompts_for_batch:
        return []
    cfg = generation or GenerationConfig()
    if not hasattr(model, "diffusion_generate"):
        logger.warning("CoDA generation requested but model lacks diffusion_generate; falling back to causal LM path.")
        return multipredict_causal_lm(
            model,
            tokenizer,
            device,
            prompts_for_batch,
            generation=cfg,
            payloads_for_batch=payloads_for_batch,
        )
    max_prompt_length = cfg.max_input_length or cfg.max_length or 4096
    try:
        tokenized = tokenizer(
            prompts_for_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device) if "attention_mask" in tokenized else None
    except Exception as exc:
        logger.error(f"Tokenize fail (CoDA): {exc}")
        empty = [""] * int(cfg.num_return_sequences)
        return [empty for _ in prompts_for_batch]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_lengths = attention_mask.sum(dim=1, dtype=torch.int64)

    gen_config = _build_coda_generation_config(model, tokenizer, cfg, input_ids, attention_mask)

    with torch.no_grad(), _maybe_autocast(cfg.use_mixed_precision):
        outputs = model.diffusion_generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )

    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    if sequences is None:
        logger.error("CoDA diffusion_generate returned no sequences.")
        empty = [""] * int(cfg.num_return_sequences)
        return [empty for _ in prompts_for_batch]
    sequences = sequences.to("cpu")
    prompt_lengths = prompt_lengths.to("cpu")

    batch_size = len(prompts_for_batch)
    nrs = int(cfg.num_return_sequences)
    try:
        sequences = sequences.view(batch_size, nrs, -1)
    except Exception as exc:
        logger.error(f"CoDA reshape failure: {exc}")
        sequences = sequences.reshape(batch_size, nrs, -1)

    pad_token_id = getattr(gen_config, "pad_token_id", tokenizer.pad_token_id)
    mask_token_id = getattr(gen_config, "mask_token_id", getattr(model.config, "mask_token_id", None))

    decoded_batches: list[list[str]] = []
    for idx in range(batch_size):
        prompt_len = int(prompt_lengths[idx].item())
        samples: list[str] = []
        for j in range(nrs):
            tokens = sequences[idx, j]
            tokens = tokens[prompt_len:]
            if pad_token_id is not None:
                tokens = tokens[tokens != pad_token_id]
            if mask_token_id is not None:
                tokens = tokens[tokens != mask_token_id]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if not text:
                # Fallback to original prompt slice if decoding produced empty output
                text = tokenizer.decode(tokens, skip_special_tokens=False).strip()
            samples.append(text)
        decoded_batches.append(samples)
    return decoded_batches


def multipredict_causal_lm(
    model,
    tokenizer,
    device,
    prompts_for_batch,
    *,
    generation: GenerationConfig | None = None,
    payloads_for_batch: list[dict[str, Any]] | None = None,
):
    if not prompts_for_batch:
        return []
    cfg = generation or GenerationConfig()
    max_in_length = (
        max(1, cfg.max_length - cfg.max_generation_length)
        if cfg.max_generation_length < cfg.max_length
        else cfg.max_length // 2
    )
    try:
        tokenized = tokenizer(
            prompts_for_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_in_length,
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
    except Exception as exc:
        logger.error(f"Tokenize fail (causal): {exc}")
        empty = [""] * int(cfg.num_return_sequences)
        return [empty for _ in prompts_for_batch]

    gen_kwargs = {
        "max_new_tokens": cfg.max_generation_length,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": int(cfg.num_return_sequences),
    }
    if cfg.num_beams > 1:
        gen_kwargs.update(num_beams=int(cfg.num_beams), do_sample=False, early_stopping=True)
    else:
        gen_kwargs["do_sample"] = True
        if cfg.temperature and cfg.temperature > 0:
            gen_kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_k and cfg.top_k > 0:
            gen_kwargs["top_k"] = int(cfg.top_k)
        if cfg.top_p and 0.0 < cfg.top_p < 1.0:
            gen_kwargs["top_p"] = float(cfg.top_p)

    with torch.no_grad(), _maybe_autocast(cfg.use_mixed_precision):
        sequences = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    prompt_length = input_ids.shape[1]
    generated_only = [seq[prompt_length:] for seq in sequences]
    decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
    nrs = int(cfg.num_return_sequences)
    return [decoded[i * nrs : (i + 1) * nrs] for i in range(len(prompts_for_batch))]


def multipredict(
    model,
    tokenizer,
    device,
    prompts_for_batch,
    *,
    generation: GenerationConfig | None = None,
    payloads_for_batch: list[dict[str, Any]] | None = None,
):
    if not prompts_for_batch:
        return []
    cfg = generation or GenerationConfig()
    if cfg.generation_mode == "diffusion":
        return multipredict_coda(
            model,
            tokenizer,
            device,
            prompts_for_batch,
            generation=cfg,
            payloads_for_batch=payloads_for_batch,
        )
    if cfg.is_causal_lm or cfg.model_type == "causal_lm":
        return multipredict_causal_lm(
            model,
            tokenizer,
            device,
            prompts_for_batch,
            generation=cfg,
            payloads_for_batch=payloads_for_batch,
        )

    try:
        tokenized = tokenizer(
            prompts_for_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_input_length,
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
    except Exception as exc:
        logger.error(f"Tokenize fail: {exc}")
        empty = [""] * int(cfg.num_return_sequences)
        return [empty for _ in prompts_for_batch]

    max_prompt_length = cfg.max_input_length

    gen_kwargs = {
        "max_new_tokens": cfg.max_generation_length,
        "num_beams": int(cfg.num_beams),
        "num_return_sequences": int(cfg.num_return_sequences),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": False,
    }
    gen_kwargs["do_sample"] = cfg.num_beams <= 1
    if gen_kwargs["do_sample"]:
        if cfg.temperature and cfg.temperature > 0:
            gen_kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_k and cfg.top_k > 0:
            gen_kwargs["top_k"] = int(cfg.top_k)
        if cfg.top_p and 0.0 < cfg.top_p < 1.0:
            gen_kwargs["top_p"] = float(cfg.top_p)
    if gen_kwargs["num_beams"] > 1:
        gen_kwargs["early_stopping"] = True

    with torch.no_grad(), _maybe_autocast(cfg.use_mixed_precision):
        sequences = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        if hasattr(sequences, "sequences"):
            sequences = sequences.sequences
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    nrs = int(cfg.num_return_sequences)
    return [decoded[i * nrs : (i + 1) * nrs] for i in range(len(prompts_for_batch))]


def multipredict_adaptive(
    model,
    tokenizer,
    device,
    prompts_for_batch,
    *,
    generation: GenerationConfig,
    adaptive: AdaptiveConfig | None = None,
    payloads_for_batch: list[dict[str, Any]] | None = None,
):
    if not prompts_for_batch:
        return []
    adaptive = adaptive or AdaptiveConfig()
    current_generation = generation
    cur_bs = len(prompts_for_batch)
    nrs = min(int(current_generation.num_return_sequences), int(current_generation.num_beams or 1))
    model_id = id(model)

    if current_generation.use_mixed_precision and model_id in _MP_DISABLED_MODELS:
        current_generation = replace(current_generation, use_mixed_precision=False)

    if cur_bs <= adaptive.min_batch_size:
        return multipredict(model, tokenizer, device, prompts_for_batch, generation=current_generation)

    outputs: list[list[str]] = []
    index, retries, total = 0, 0, len(prompts_for_batch)
    max_retries = adaptive.max_oom_retries
    mp_fallback_used = False

    while index < total:
        size = min(cur_bs, total - index)
        subset = prompts_for_batch[index : index + size]
        payload_subset = (
            payloads_for_batch[index : index + size]
            if payloads_for_batch is not None
            else None
        )
        try:
            result = multipredict(
                model,
                tokenizer,
                device,
                subset,
                generation=current_generation,
                payloads_for_batch=payload_subset,
            )
            outputs.extend(result)
            index += size
            retries = 0
            continue
        except Exception as exc:
            cur_bs, index, retries, handled = _handle_oom_generation_error(
                exc,
                adaptive_cfg=adaptive,
                outputs=outputs,
                nrs=nrs,
                size=size,
                cur_bs=cur_bs,
                index=index,
                retries=retries,
                max_retries=max_retries,
            )
            if handled:
                continue

            current_generation, mp_fallback_used, index, retries, handled = _handle_non_oom_generation_error(
                exc,
                current_generation=current_generation,
                model=model,
                mp_fallback_used=mp_fallback_used,
                outputs=outputs,
                nrs=nrs,
                size=size,
                index=index,
                retries=retries,
                max_retries=max_retries,
                subset_prompts=subset,
            )
            if handled:
                continue
            raise
    return outputs


def _handle_oom_generation_error(
    exc: Exception,
    *,
    adaptive_cfg: AdaptiveConfig,
    outputs: list[list[str]],
    nrs: int,
    size: int,
    cur_bs: int,
    index: int,
    retries: int,
    max_retries: int,
):
    if not is_gpu_oom_error(exc):
        return cur_bs, index, retries, False
    if cur_bs <= adaptive_cfg.min_batch_size:
        outputs.extend([["GPU_OOM_ERROR"] * nrs for _ in range(size)])
        index += size
        retries = 0
    else:
        cur_bs = calculate_reduced_batch_size_gpu(
            cur_bs,
            adaptive_cfg.oom_reduction_factor,
            adaptive_cfg.min_batch_size,
        )
        retries += 1
        if retries <= max_retries:
            wait_and_clear_gpu_memory(retries - 1)
        else:
            outputs.extend([["MAX_RETRIES_EXCEEDED"] * nrs for _ in range(size)])
            index += size
            retries = 0
    return cur_bs, index, retries, True


def _handle_non_oom_generation_error(
    exc: Exception,
    *,
    current_generation: GenerationConfig,
    model,
    mp_fallback_used: bool,
    outputs: list[list[str]],
    nrs: int,
    size: int,
    index: int,
    retries: int,
    max_retries: int,
    subset_prompts: list[str] | None = None,
):
    model_identifier = id(model)
    if current_generation.use_mixed_precision and not mp_fallback_used:
        mp_fallback_used = True
        _MP_DISABLED_MODELS.add(model_identifier)
        current_generation = replace(current_generation, use_mixed_precision=False)
        logger.warning(
            "Mixed-precision generation failed for model %s; retrying without MP. Error: %s",
            getattr(model, "name_or_path", model.__class__.__name__),
            exc,
            exc_info=True,
        )
        retries = 0
        return current_generation, mp_fallback_used, index, retries, True

    retries += 1
    if retries <= max_retries:
        logger.warning(
            "Generation retry %d/%d for model %s after error: %s",
            retries,
            max_retries,
            getattr(model, "name_or_path", model.__class__.__name__),
            exc,
        )
        wait_and_clear_gpu_memory(retries - 1)
        return current_generation, mp_fallback_used, index, retries, True

    logger.error(
        "Generation failed after %d retries for model %s: %s",
        retries,
        getattr(model, "name_or_path", model.__class__.__name__),
        exc,
        exc_info=True,
    )
    if subset_prompts:
        logger.debug("Failed prompts sample: %s", subset_prompts[:min(3, len(subset_prompts))])
    outputs.extend([["ERROR_GENERATING"] * nrs for _ in range(size)])
    index += size
    retries = 0
    return current_generation, mp_fallback_used, index, retries, True


# =============================== Batching ===============================
def dynamic_batch_worker(items_with_prompts, max_tokens_per_batch: int):
    """Greedy pack by rough token estimate (space-split) while preserving order."""
    batches, cur, cur_toks = [], [], 0
    for payload, prompt in items_with_prompts:
        est = len(prompt.split())
        if cur and cur_toks + est > max_tokens_per_batch:
            batches.append(cur)
            cur, cur_toks = [(payload, prompt)], est
        else:
            cur.append((payload, prompt))
            cur_toks += est
    if cur:
        batches.append(cur)
    return batches


# =============================== Refine ===============================


# =============================== Worker ===============================
def _build_batches_global(
    all_items: list[tuple[dict, str]], use_dynamic: bool, max_tokens_per_batch: int, fixed_bs: int
):
    """all_items: list[(payload_dict, prompt_str)] -> batches of same tuples"""
    if use_dynamic:
        return dynamic_batch_worker(all_items, max_tokens_per_batch)
    # fixed sized
    return [all_items[i : i + fixed_bs] for i in range(0, len(all_items), fixed_bs)]


def _build_generation_configs(worker_settings) -> tuple[GenerationConfig, AdaptiveConfig]:
    top_k_raw = worker_settings.get("top_k", 50)
    top_p_raw = worker_settings.get("top_p", 0.95)
    temperature = _safe_float(worker_settings.get("temperature", 0.1), 0.1)
    generation_mode = str(worker_settings.get("generation_mode", "standard") or "standard").strip().lower()
    presets_raw = worker_settings.get("coda_quality_presets")
    presets_copy = copy.deepcopy(presets_raw) if isinstance(presets_raw, dict) else None
    coda_profile = worker_settings.get("coda_quality_profile")
    coda_profile = str(coda_profile).strip().lower() if coda_profile else None
    generation = GenerationConfig(
        num_beams=_safe_int(worker_settings.get("num_beams", 1), 1),
        num_return_sequences=_safe_int(worker_settings.get("num_return_sequences", 1), 1),
        top_k=None if top_k_raw is None else _safe_int(top_k_raw, 0),
        top_p=None if top_p_raw is None else _safe_float(top_p_raw, 0.0),
        temperature=temperature,
        max_generation_length=_safe_int(worker_settings.get("max_generation_length", 600), 600),
        max_input_length=_safe_int(worker_settings.get("max_input_length", 2048), 2048),
        use_mixed_precision=bool(worker_settings.get("use_mixed_precision_inference", False)),
        model_type=str(worker_settings.get("model_type", "seq2seq")),
        is_causal_lm=bool(worker_settings.get("is_causal_lm", False)),
        max_length=_safe_int(worker_settings.get("max_length", 4096), 4096),
        generation_mode=generation_mode,
        diffusion_steps=_safe_int(worker_settings.get("diffusion_steps"), None),
        diffusion_eps=_safe_float(worker_settings.get("diffusion_eps"), None),
        diffusion_alg=(str(worker_settings.get("diffusion_alg")).strip().lower() if worker_settings.get("diffusion_alg") else None),
        diffusion_alg_temp=_safe_float(worker_settings.get("diffusion_alg_temp"), None),
        coda_quality_profile=coda_profile,
        coda_quality_presets=presets_copy,
    )

    if generation.generation_mode == "diffusion":
        presets = generation.coda_quality_presets or {}
        profile = generation.coda_quality_profile or worker_settings.get("coda_quality_profile")
        if profile and isinstance(presets, dict):
            preset_vals = presets.get(profile)
            if isinstance(preset_vals, dict):
                if preset_vals.get("diffusion_steps") is not None:
                    generation.diffusion_steps = _safe_int(preset_vals.get("diffusion_steps"), generation.diffusion_steps)
                if preset_vals.get("temperature") is not None:
                    generation.temperature = _safe_float(preset_vals.get("temperature"), generation.temperature)
                if preset_vals.get("top_p") is not None:
                    generation.top_p = _safe_float(preset_vals.get("top_p"), generation.top_p)
                if preset_vals.get("top_k") is not None:
                    generation.top_k = _safe_int(preset_vals.get("top_k"), generation.top_k or 0)
        if generation.diffusion_steps is None:
            generation.diffusion_steps = 128
        if generation.diffusion_eps is None:
            generation.diffusion_eps = 1e-3
        if not generation.diffusion_alg:
            generation.diffusion_alg = "origin"

    adaptive = AdaptiveConfig(
        min_batch_size=_safe_int(worker_settings.get("min_batch_size", 1), 1),
        max_oom_retries=_safe_int(worker_settings.get("max_oom_retries", 3), 3),
        oom_reduction_factor=_safe_float(worker_settings.get("oom_reduction_factor", 0.5), 0.5),
    )
    return generation, adaptive


def _build_preview_controller(worker_settings: dict[str, Any], rank: int) -> PreviewController:
    max_samples = _safe_int(worker_settings.get("preview_max_samples", 0), 0)
    interval = _safe_int(worker_settings.get("preview_interval_batches", 0), 0)
    preview_all = bool(worker_settings.get("preview_all_ranks", False))
    enabled = max_samples > 0 and interval > 0 and (preview_all or rank == 0)
    return PreviewController(
        enabled=enabled,
        interval=interval if interval > 0 else 0,
        max_samples=max_samples if max_samples > 0 else 0,
        rank=rank,
    )


def _generation_config_to_kwargs(cfg: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_beams": int(cfg.num_beams),
        "num_return_sequences": 1,
        "early_stopping": False,
    }
    do_sample = cfg.num_beams <= 1
    kwargs["do_sample"] = do_sample
    if do_sample:
        if cfg.temperature and cfg.temperature > 0:
            kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_k and cfg.top_k > 0:
            kwargs["top_k"] = int(cfg.top_k)
        if cfg.top_p and 0.0 < cfg.top_p < 1.0:
            kwargs["top_p"] = float(cfg.top_p)
    else:
        kwargs["early_stopping"] = True
    return kwargs


def _execute_inference_batches(
    batches: list[list[tuple[dict[str, Any], str]]],
    model,
    tokenizer,
    device,
    generation_cfg: GenerationConfig,
    adaptive_cfg: AdaptiveConfig,
    preview_controller: PreviewController,
    worker_settings: dict[str, Any],
    rank: int,
) -> dict[tuple[str, int], list[str]]:
    texts_map: dict[tuple[str, int], list[str]] = {}
    iterator = tqdm.tqdm(batches, desc="Global batches", unit="batch") if rank == 0 else batches
    for batch in iterator:
        payloads = [payload for (payload, _) in batch]
        prompts = [prompt for (_, prompt) in batch]
        texts_batch = multipredict_adaptive(
            model,
            tokenizer,
            device,
            prompts,
            generation=generation_cfg,
            adaptive=adaptive_cfg,
            payloads_for_batch=payloads,
        )
        preview_controller.process_batch(payloads, texts_batch, worker_settings)
        for idx, payload in enumerate(payloads):
            key = (payload["base_key"], payload["out_index"])
            texts_map[key] = texts_batch[idx] if idx < len(texts_batch) else [""]
    return texts_map


def _assemble_predictions(
    registry: list[dict[str, Any]],
    texts_map: dict[tuple[str, int], list[str]],
) -> dict[str, list[dict[str, Any]]]:
    worker_predictions_map: dict[str, list[dict[str, Any]]] = {}
    for entry in registry:
        base_key = entry["base_key"]
        out_index = entry["out_index"]
        texts = texts_map.get((base_key, out_index), [""])
        worker_predictions_map.setdefault(base_key, []).append(
            {
                "prompt": entry["prompt"],
                "texts": texts,
                "decoder": entry["decoder"],
            }
        )
    return worker_predictions_map


def _build_worker_failure_payload(exc: Exception, rank: int) -> dict[str, Any]:
    return {
        "__worker_failure__": True,
        "__error_message__": str(exc),
        "__error_type__": type(exc).__name__,
        "__rank__": rank,
        "__gpu_count__": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }


def predict_worker(rank, world_size, data_subsets_for_spawn, model_path_for_worker, worker_settings, return_dict_mp):
    setup_timeout(int(worker_settings.get("timeout", 3600)))
    try:
        _configure_cuda_device(rank)
        attn_impl = _resolve_attention_impl(model_path_for_worker)

        model, tokenizer, device = get_model(
            model_path_for_worker,
            rank,
            worker_settings.get("use_torch_compile", False),
            worker_settings.get("model_type", "seq2seq"),
            worker_settings.get("is_causal_lm", False),
            worker_settings.get("tokenizer_dropout_enabled", False),
            worker_settings.get("tokenizer_dropout_rate", 0.1),
            attn_impl,
        )
        _update_worker_model_flags(model, worker_settings)

        infer_data = _extract_infer_data(data_subsets_for_spawn, rank)
        if not infer_data:
            logger.info(f"[Rank {rank}] No tasks assigned.")
            return_dict_mp[rank] = {}
            return

        generation_cfg, adaptive_cfg = _build_generation_configs(worker_settings)
        preview_controller = _build_preview_controller(worker_settings, rank)

        eval_bs = int(worker_settings.get("eval_batch_size", 16))
        use_dynamic = bool(worker_settings.get("use_dynamic_batching", False))
        max_tok_batch = int(worker_settings.get("max_tokens_per_batch", 8192))

        registry, prompts_global = _collect_inference_payloads(infer_data, worker_settings, rank)
        if not prompts_global:
            logger.warning(f"[Rank {rank}] No valid prompts.")
            return_dict_mp[rank] = {}
            return

        all_items = list(zip(registry, prompts_global, strict=False))
        batches = _build_batches_global(all_items, use_dynamic, max_tok_batch, eval_bs)

        texts_map = _execute_inference_batches(
            batches,
            model,
            tokenizer,
            device,
            generation_cfg,
            adaptive_cfg,
            preview_controller,
            worker_settings,
            rank,
        )

        worker_predictions_map = _assemble_predictions(registry, texts_map)
        logger.info(f"[Rank {rank}] Inference complete for {len(worker_predictions_map)} task keys.")
        return_dict_mp[rank] = worker_predictions_map

    except Exception as exc:
        logger.error(f"[Rank {rank}] CRITICAL: {exc}", exc_info=True)
        return_dict_mp[rank] = _build_worker_failure_payload(exc, rank)
    finally:
        signal.alarm(0)
        if torch.cuda.is_available():
            empty_cache()

def _configure_cuda_device(rank: int) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
    except Exception as exc:
        logger.warning('[Rank %s] set_device failed: %s', rank, exc)


def _resolve_attention_impl(model_path_for_worker: str | None):
    if not model_path_for_worker:
        return None
    with suppress(Exception):
        model_settings = getattr(config, "MODEL_SETTINGS", {}).get(model_path_for_worker, {})
        general_settings = model_settings.get("general") or {}
        return general_settings.get("attn_implementation")
    return None


def _update_worker_model_flags(model, worker_settings: dict[str, Any]) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))
    worker_settings["model_type"] = "seq2seq" if is_encoder_decoder else "causal_lm"
    worker_settings["is_causal_lm"] = not is_encoder_decoder
    coda_flag = bool(worker_settings.get("coda_enabled"))
    dream_flag = bool(worker_settings.get("dream_enabled"))
    model_type_name = str(getattr(cfg, "model_type", "") or "").lower()
    if not coda_flag:
        coda_flag = model_type_name == "coda"
    if not dream_flag:
        dream_flag = model_type_name == "dream"
    worker_settings["is_coda_model"] = coda_flag
    worker_settings["is_dream_model"] = dream_flag
    diffusion_active = coda_flag or dream_flag or str(worker_settings.get("generation_mode", "")).lower() == "diffusion"
    if diffusion_active and worker_settings.get("generation_mode") is None:
        worker_settings["generation_mode"] = "diffusion"
    if diffusion_active:
        worker_settings["is_causal_lm"] = True


def _extract_infer_data(data_subsets_for_spawn, rank: int) -> dict:
    infer_data = data_subsets_for_spawn.get(rank, {}) or {}
    return infer_data if isinstance(infer_data, dict) else {}


def _extract_expected_output(task: dict, prompt_style: str = "legacy") -> tuple[str | None, list[list[int]] | None]:
    try:
        test_examples = task.get("test", [])
        if not isinstance(test_examples, list):
            return None, None
        for example in test_examples:
            if not isinstance(example, dict) or "output" not in example:
                continue
            target = example["output"]
            if is_grid(target):
                if prompt_style == "arc_diffusion":
                    rows = [row for row in grid_to_string(target).split(" ") if row]
                    text_repr = "\n".join(rows) if rows else None
                else:
                    text_repr = f"{output_prefix(target)}{grid_to_string(target)}"
                return text_repr, target
            if isinstance(target, str):
                cleaned = target.strip()
                return (cleaned or None), None
            if isinstance(target, list):
                flattened = " ".join(str(item) for item in target if item is not None)
                cleaned = flattened.strip()
                return (cleaned or None), None
        return None, None
    except Exception:
        return None, None


def _collect_inference_payloads(
    infer_data: dict,
    worker_settings: dict[str, Any],
    rank: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    registry: list[dict[str, Any]] = []
    prompts_global: list[str] = []
    prompt_prefix = str(
        worker_settings.get("prompt_general_prefix")
        or worker_settings.get("prompt_prefix")
        or ""
    )
    prompt_style = str(worker_settings.get("prompt_format") or "legacy").lower()
    items = infer_data.items()
    iterator = tqdm.tqdm(items, desc="Rank 0 Collect", unit="task") if rank == 0 else items
    for task_key, task_data in iterator:
        base_key = task_data.get("__original_task_key__", task_key)
        shard_idx = task_data.get("__shard_index__")
        display_key = (
            f"{base_key}#shard{shard_idx}" if shard_idx is not None and base_key != task_key else task_key
        )
        aug_tasks, decs = task_data.get("tasks", []), task_data.get("decs", [])
        mixed_tasks, mixed_decs = mix_augms(
            aug_tasks,
            decs,
            worker_settings.get("expand_factor", 0.0),
            worker_settings.get("prune_factor", 0.0),
        )
        if not mixed_tasks:
            continue
        for index, task in enumerate(mixed_tasks):
            prompt_raw = makeprompt(task, style=prompt_style)
            prompt_prefixed = ensure_prompt_prefix(prompt_raw, prompt_prefix).rstrip()
            prompt = prompt_prefixed if prompt_style == "arc_diffusion" else f"{prompt_prefixed} "
            if not prompt or "[prompt_error]" in prompt:
                continue
            expected_text, expected_grid = _extract_expected_output(task, prompt_style=prompt_style)
            registry.append(
                {
                    "base_key": base_key,
                    "out_index": index,
                    "decoder": mixed_decs[index],
                    "prompt": prompt,
                    "display": display_key,
                    "expected_text": expected_text,
                    "expected_grid": expected_grid,
                }
            )
            prompts_global.append(prompt)
    return registry, prompts_global
def ensure_prompt_prefix(prompt: str | None, prefix: str) -> str:
    text = "" if prompt is None else str(prompt)
    prefix = str(prefix or "")
    if prefix and not text.startswith(prefix):
        return f"{prefix}{text}"
    return text
