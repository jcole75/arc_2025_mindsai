import ast
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Any

import pandas as pd

from . import config
from .config import BUFFER_TIME, START_TIME, TOTAL_TIME, TRAIN_DATA_DISK_PATH


# Conditional imports to avoid JAX initialization conflicts
if not config.USE_FLAX:
    try:
        from datasets import Dataset
        import torch
        import torch.multiprocessing as mp
    except ImportError:
        torch = None
        mp = None
        Dataset = None
else:
    torch = None
    mp = None
    Dataset = None

import traceback

from .grid_utils import makeprompt


def _print_inference_runtime_overview(settings: dict[str, Any], num_workers: int) -> None:
    if not isinstance(settings, dict):
        return

    def _coerce_int(value, default: int | None = None) -> int | None:
        try:
            as_int = int(value)
            return as_int
        except (TypeError, ValueError):
            return default

    def _coerce_float(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    eval_bs = _coerce_int(settings.get("eval_batch_size"), 16)
    if eval_bs is None or eval_bs <= 0:
        eval_bs = 16

    grad_key = settings.get("gradient_accumulation_steps_inference", settings.get("gradient_accumulation_steps", 1))
    grad_accum = _coerce_int(grad_key, 1)
    if grad_accum is None or grad_accum <= 0:
        grad_accum = 1

    world = max(1, int(num_workers or 1))
    global_batch = eval_bs * grad_accum * world

    mixed_precision = bool(settings.get("use_mixed_precision_inference", False))
    precision_desc = "float32"
    if mixed_precision:
        dtype = "float16"
        if torch is not None:
            try:
                if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                    dtype = "bfloat16"
            except Exception:
                pass
        precision_desc = f"autocast {dtype}"

    use_dynamic = bool(settings.get("use_dynamic_batching", False))
    max_tokens = _coerce_int(settings.get("max_tokens_per_batch"))

    beams = _coerce_int(settings.get("num_beams"))
    num_return = _coerce_int(settings.get("num_return_sequences"))
    max_len = _coerce_int(settings.get("max_generation_length") or settings.get("max_length"))
    temperature = _coerce_float(settings.get("temperature"))
    top_p = _coerce_float(settings.get("top_p"))

    print("Inference runtime configuration:")
    print(f"  • Per-device batch size: {eval_bs}")
    print(f"  • Gradient accumulation: {grad_accum}")
    print(f"  • Global batch size: {global_batch} (workers={world})")
    print(f"  • Precision: {precision_desc}")
    if use_dynamic:
        token_desc = str(max_tokens) if max_tokens is not None else "auto"
        print(f"  • Dynamic batching: on (max_tokens_per_batch={token_desc})")
    else:
        print("  • Dynamic batching: off")

    generation_parts: list[str] = []
    if beams is not None:
        generation_parts.append(f"beams={beams}")
    if num_return is not None:
        generation_parts.append(f"returns={num_return}")
    if max_len is not None:
        generation_parts.append(f"max_len={max_len}")
    if temperature is not None:
        generation_parts.append(f"temperature={temperature}")
    if top_p is not None:
        generation_parts.append(f"top_p={top_p}")
    if generation_parts:
        print(f"  • Generation: {'; '.join(generation_parts)}")


def train_distributed(model_path: str, dataset, reload_path: str, train_args: dict) -> dict[str, Any] | None:
    """Launches distributed training using accelerate or JAX/Flax TPU."""
    # Detect model type and adjust args
    model_type = config.get_model_type(model_path)
    is_causal_lm = model_type == "causal_lm"
    train_args.update({"model_type": model_type, "is_causal_lm": is_causal_lm})
    if is_causal_lm:
        train_args["max_length"] = train_args.get("max_length", train_args.get("max_input_length", 4096))
        print(f"Detected causal LM: {model_path}, max_length: {train_args['max_length']}")
    else:
        print(f"Detected seq2seq: {model_path}")

    if config.USE_FLAX:
        target_device = "tpu" if config.USE_TPU else "gpu"
        return train_distributed_flax(model_path, dataset, reload_path, train_args, target_device)
    else:
        return train_distributed_gpu(model_path, dataset, reload_path, train_args)


def train_distributed_flax(
    model_path: str, dataset, reload_path: str, train_args: dict, target_device: str
) -> dict[str, Any] | None:
    device_label = target_device.upper()
    print(f"Launching Flax training on {device_label} for {os.path.basename(model_path)}...")
    elapsed = time.time() - START_TIME
    timeout = max(300, TOTAL_TIME - elapsed - BUFFER_TIME - 60)

    # Convert to CSV
    train_csv = os.path.join("/kaggle/working", "tpu_train_data.csv")
    try:
        df = pd.DataFrame(dataset)
        if "prompt" not in df or "correct_answer" not in df:
            raise ValueError("Dataset must have 'prompt' and 'correct_answer' columns")
        df[["prompt", "correct_answer"]].to_csv(train_csv, index=False)
        print(f"Converted {len(df)} examples to {train_csv}")
    except Exception as e:
        raise RuntimeError(f"Dataset conversion failed: {e}") from e

    # Prepare output dir
    if config.PRESERVE_FINETUNED_MODEL:
        os.makedirs(reload_path, exist_ok=True)
        print(f"Preserving {reload_path} (overwrite files)")
    else:
        if os.path.exists(reload_path):
            shutil.rmtree(reload_path, ignore_errors=True)
        os.makedirs(reload_path, exist_ok=True)

    # Flax config
    is_causal = train_args.get("is_causal_lm", False)
    max_len = train_args.get("max_length", 4096) if is_causal else None
    warmup_steps = train_args.get("warmup_steps")
    if warmup_steps is None:
        warmup_ratio = train_args.get("warmup_ratio", 0.1)
        try:
            warmup_steps = int(float(warmup_ratio) * len(dataset))
        except (TypeError, ValueError):
            warmup_steps = 0
    else:
        warmup_steps = int(warmup_steps)
    warmup_steps = max(warmup_steps, 0)
    flax_config = {
        "model_args": {
            "model_name_or_path": model_path,
            "config_name": model_path,
            "tokenizer_name": model_path,
            "cache_dir": None,
            "dtype": "bfloat16" if train_args.get("use_bf16") else "float32",
            "tokenizer_dropout_enabled": train_args.get("tokenizer_dropout_enabled", False),
            "tokenizer_dropout_rate": train_args.get("tokenizer_dropout_rate", 0.1),
            "model_type": train_args.get("model_type", "seq2seq"),
            "is_causal_lm": is_causal,
        },
        "data_args": {
            "train_file": train_csv,
            "max_source_length": train_args.get("max_input_length", 2200) if not is_causal else max_len,
            "max_target_length": train_args.get("max_target_length", 600) if not is_causal else max_len,
            "max_length": max_len,
            "preprocessing_num_workers": None,
            "overwrite_cache": False,
        },
        "training_args": {
            "output_dir": reload_path,
            "per_device_train_batch_size": train_args.get("per_device_train_batch_size", 8),
            "gradient_accumulation_steps": 1,
            "learning_rate": train_args.get("learning_rate", 3e-5),
            "weight_decay": train_args.get("weight_decay", 0.0),
            "num_train_epochs": train_args.get("num_train_epochs", 1),
            "warmup_steps": warmup_steps,
            "logging_steps": train_args.get("logging_steps", 10),
            "save_steps": train_args.get("save_steps", 500),
            "gradient_checkpointing": train_args.get("use_gradient_checkpointing", False),
            "filter_datasets_for_max_tokens": True,
            "shuffle_training_data": True,
        },
    }

    # Allow optional GPU-specific batch size override while respecting per-model defaults
    if target_device == "gpu":
        original_bs = int(flax_config["training_args"].get("per_device_train_batch_size", 8) or 1)
        env_override_raw = os.environ.get("FLAX_GPU_PER_DEVICE_BATCH", "").strip()
        config_override = train_args.get("flax_gpu_per_device_batch_size")
        override_bs = None
        if config_override is not None:
            try:
                override_bs = int(config_override)
            except (TypeError, ValueError):
                print(f"[Flax][GPU] Ignoring invalid config override for per-device batch size: {config_override}")
                override_bs = None
        if override_bs is None and env_override_raw:
            try:
                override_bs = int(env_override_raw)
            except ValueError:
                print(f"[Flax][GPU] Ignoring invalid FLAX_GPU_PER_DEVICE_BATCH value: {env_override_raw}")
                override_bs = None
        if override_bs is not None:
            if override_bs <= 0:
                print(f"[Flax][GPU] Override batch size must be positive. Received {override_bs}. Skipping override.")
            elif override_bs != original_bs:
                print(f"[Flax][GPU] Adjusting per-device batch size from {original_bs} to {override_bs} (override).")
                flax_config["training_args"]["per_device_train_batch_size"] = override_bs
                train_args["per_device_train_batch_size"] = override_bs

    config_path = os.path.join(reload_path, "flax_config.json")
    with open(config_path, "w") as f:
        json.dump(flax_config, f, indent=2)

    # Run subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flax_trainer.py")
    cmd = [sys.executable, script, config_path]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "FLAX_TARGET_DEVICE": target_device,
        }
    )

    if target_device == "tpu":
        env.update(
            {
                "JAX_PLATFORMS": "tpu,cpu",
                "PJRT_DEVICE": "TPU",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = ":".join(p for p in env["LD_LIBRARY_PATH"].split(":") if "torch_xla" not in p)
    else:
        env.update(
            {
                "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda,cpu"),
            }
        )
        env.pop("PJRT_DEVICE", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("TPU_LIBRARY_PATH", None)
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        extra_flags = os.environ.get(
            "FLAX_GPU_XLA_FLAGS",
            (
                "--xla_gpu_autotune_level=1 --xla_gpu_enable_latency_hiding_scheduler=false "
                "--xla_gpu_force_compilation_parallelism=1"
            ),
        ).strip()
        if extra_flags:
            existing_flags = env.get("XLA_FLAGS", "").strip()
            if extra_flags not in existing_flags:
                combined_flags = f"{existing_flags} {extra_flags}".strip()
                env["XLA_FLAGS"] = combined_flags

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/kaggle/working",
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
        process.wait(timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError(f"Flax training on {device_label} failed with code {process.returncode}")
        print(f"Flax training on {device_label} completed!")
    except subprocess.TimeoutExpired as err:
        print(f"Flax training on {device_label} timed out after {timeout}s.")
        if process:
            process.kill()
            process.wait()
        raise RuntimeError(f"Flax training on {device_label} timed out after {timeout}s.") from err
    except Exception as e:
        print(f"Flax training on {device_label} failed: {e}")
        if process and process.returncode is None:
            process.kill()
            process.wait()
        raise RuntimeError(f"Flax training on {device_label} failed: {e}") from e
    finally:
        for f in [train_csv, config_path]:
            if os.path.exists(f):
                os.remove(f) if os.path.isfile(f) else shutil.rmtree(f, ignore_errors=True)
    return None


def train_distributed_gpu(model_path: str, dataset, reload_path: str, train_args: dict) -> None:
    world_size = torch.cuda.device_count() if torch else 0
    num_procs = train_args.get("num_processes", world_size or 1)
    print(f"Launching training on {num_procs} process(es) for {os.path.basename(model_path)}...")
    elapsed = time.time() - START_TIME
    timeout = max(300, TOTAL_TIME - elapsed - BUFFER_TIME - 120)
    train_args["timeout"] = int(max(240, timeout - 60))

    if os.path.exists(TRAIN_DATA_DISK_PATH):
        shutil.rmtree(TRAIN_DATA_DISK_PATH, ignore_errors=True)
    dataset.save_to_disk(TRAIN_DATA_DISK_PATH)

    if config.PRESERVE_FINETUNED_MODEL:
        os.makedirs(reload_path, exist_ok=True)
        print(f"Preserving {reload_path} (overwrite files)")
    else:
        if os.path.exists(reload_path):
            shutil.rmtree(reload_path, ignore_errors=True)
        os.makedirs(reload_path, exist_ok=True)

    metrics_path = train_args.get("metrics_output")
    if not metrics_path:
        metrics_path = os.path.join(reload_path, "training_metrics.json")
        train_args["metrics_output"] = metrics_path
    else:
        metrics_path = str(metrics_path)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_worker.py")
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_procs),
        "--num_machines",
        "1",
        "--machine_rank",
        "0",
    ]

    if train_args.get("use_mixed_precision"):
        mp_setting = "bf16" if (train_args.get("use_bf16") or torch.cuda.is_bf16_supported()) else "fp16"
        cmd.extend(["--mixed_precision", mp_setting])

    cmd.extend(
        [script, "--model_path", model_path, "--train_data_path", TRAIN_DATA_DISK_PATH, "--reload_path", reload_path]
    )

    bool_keys = [
        "use_gradient_checkpointing",
        "use_mixed_precision",
        "use_bf16",
        "use_fp16",
        "use_torch_compile",
        "tokenizer_dropout_enabled",
        "tokenizer_dropout_apply_to_labels",
        "enable_token_filtering",
        "is_causal_lm",
        "pretraining",
        "use_lora",
        "lora_use_dora",
    ]
    for key in bool_keys:
        if train_args.get(key):
            cmd.append(f"--{key}")

    value_keys = [
        "timeout",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "save_steps",
        "save_strategy",
        "save_total_limit",
        "save_optimizer",
        "eval_steps",
        "logging_steps",
        "max_input_length",
        "max_target_length",
        "max_length",
        "dataloader_num_workers",
        "seed",
        "tokenizer_dropout_rate",
        "model_type",
        "label_smoothing",
        "metrics_output",
        "last_top2_score",
        "pretraining_objective",
        "pretraining_objective_settings",
        "lr_reset_examples",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "lora_task_type",
        "lora_init_lora_weights",
        "lora_rank_pattern",
        "lora_scaling",
        "lora_layers_pattern",
    ]
    for key in value_keys:
        val = train_args.get(key)
        if val is not None:
            cmd.extend([f"--{key}", str(val)])

    if train_args.get("trainable_layer_patterns"):
        for pat in train_args["trainable_layer_patterns"]:
            cmd.extend(["--trainable_layer_patterns", pat])

    for key, flag in [
        ("lora_target_modules", "--lora_target_modules"),
        ("lora_modules_to_save", "--lora_modules_to_save"),
        ("lora_layers_to_transform", "--lora_layers_to_transform"),
    ]:
        values = train_args.get(key)
        if values:
            for value in values:
                cmd.extend([flag, str(value)])

    # Optional logging directory and run_id for CSV logs
    if train_args.get("log_dir"):
        cmd.extend(["--log_dir", str(train_args["log_dir"])])
    if train_args.get("run_id"):
        cmd.extend(["--run_id", str(train_args["run_id"])])

    env = os.environ.copy()
    env.update(
        {"OMP_NUM_THREADS": "1", "NCCL_DEBUG": "INFO", "PYTHONUNBUFFERED": "1", "TORCH_DISTRIBUTED_DEBUG": "DETAIL"}
    )

    process = None
    metrics_data: dict[str, Any] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
        process.wait(timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError(f"Training failed with code {process.returncode}")
        print("Training completed.")
    except subprocess.TimeoutExpired as err:
        print(f"Training timed out after {timeout}s.")
        if process:
            process.kill()
            process.wait()
        raise RuntimeError(f"Training timed out after {timeout}s.") from err
    except Exception as e:
        print(f"Training failed: {e}")
        if process and process.returncode is None:
            process.kill()
            process.wait()
        raise RuntimeError(f"Training failed: {e}") from e
    finally:
        if process and process.stdout and not process.stdout.closed:
            process.stdout.close()
        if os.path.exists(TRAIN_DATA_DISK_PATH):
            shutil.rmtree(TRAIN_DATA_DISK_PATH, ignore_errors=True)
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path) as mf:
                metrics_data = json.load(mf)
        except Exception as metrics_err:
            print(f"Training metrics read warn: {metrics_err}")
    return metrics_data


def predict_distributed(
    model_path: str, infer_data: dict, keys: list[str], settings: dict
) -> dict[str, list[dict[str, Any]]]:
    model_type = config.get_model_type(model_path)
    is_causal_lm = model_type == "causal_lm"
    settings.update({"model_type": model_type, "is_causal_lm": is_causal_lm})
    if is_causal_lm:
        settings["max_length"] = settings.get("max_length", settings.get("max_input_length", 4096))
        settings["num_beams"] = 1  # Default for causal
        print(f"Detected causal LM for inference: {model_path}, max_length: {settings['max_length']}")
    else:
        print(f"Detected seq2seq for inference: {model_path}")

    if config.USE_FLAX:
        target_device = "tpu" if config.USE_TPU else "gpu"
        return predict_distributed_flax(model_path, infer_data, keys, settings, target_device)
    else:
        return predict_distributed_gpu(model_path, infer_data, keys, settings)


def predict_distributed_flax(
    model_path: str, infer_data: dict, keys: list[str], settings: dict, target_device: str
) -> dict[str, list[dict[str, Any]]]:
    device_label = target_device.upper()
    print(f"Launching Flax inference on {device_label} for {os.path.basename(model_path)}... Total keys: {len(keys)}")
    elapsed = time.time() - START_TIME
    timeout = max(120, TOTAL_TIME - elapsed - BUFFER_TIME - 60)
    input_csv = "/kaggle/working/tpu_inference_input.csv"
    output_csv = "/kaggle/working/tpu_inference_output.csv"

    try:
        data = []
        prompt_format = str(
            getattr(config, "MODEL_SETTINGS", {}).get(model_path, {}).get("general", {}).get("prompt_format") or "legacy"
        ).lower()
        for key in keys:
            if key in infer_data:
                tasks = infer_data[key].get("tasks", [])
                decs = infer_data[key].get("decs", [])
                for i, task in enumerate(tasks):
                    prompt_raw = makeprompt(task, style=prompt_format)
                    prompt_clean = prompt_raw.rstrip()
                    prompt = prompt_clean if prompt_format == "arc_diffusion" else f"{prompt_clean} "
                    dec = str(decs[i] if i < len(decs) else {})
                    data.append({"prompt": prompt, "task_key": key, "decoder": dec})
        if not data:
            print("No inference prompts. Returning empty.")
            return {k: [] for k in keys}
        pd.DataFrame(data).to_csv(input_csv, index=False)
        print(f"Converted {len(data)} items to {input_csv}")
    except Exception as e:
        raise RuntimeError(f"Inference data conversion failed: {e}") from e

    # Dtype selection
    train_set = config.MODEL_SETTINGS.get(model_path, {}).get("training", {})
    dtype = (
        "bfloat16"
        if train_set.get("use_bf16", False)
        else "float16"
        if train_set.get("use_fp16", False)
        else "bfloat16"
    )
    print(f"Using dtype '{dtype}' for Flax inference on {device_label}.")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flax_tpu_inference.py")
    cmd = [
        sys.executable,
        script,
        "--model_path",
        model_path,
        "--input_file",
        input_csv,
        "--output_file",
        output_csv,
        "--batch_size",
        str(settings.get("eval_batch_size", 32)),
        "--dtype",
        dtype,
        "--num_return_sequences",
        str(settings.get("num_return_sequences", 2)),
        "--num_beams",
        str(settings.get("num_beams", 2)),
        "--temperature",
        str(settings.get("temperature", 1.0)),
        "--max_input_length",
        str(
            settings.get("max_input_length", 3000)
            if not settings.get("is_causal_lm")
            else settings.get("max_length", 4096)
        ),
        "--max_output_length",
        str(settings.get("max_generation_length", 600)),
        "--model_type",
        str(settings.get("model_type", "seq2seq")),
    ]
    if settings.get("is_causal_lm"):
        cmd.append("--is_causal_lm")
        if "max_length" in settings:
            cmd.extend(["--max_length", str(settings["max_length"])])
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "FLAX_TARGET_DEVICE": target_device,
        }
    )

    if target_device == "tpu":
        env.update(
            {
                "JAX_PLATFORMS": "",
                "PJRT_DEVICE": "TPU",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = ":".join(p for p in env["LD_LIBRARY_PATH"].split(":") if "torch_xla" not in p)
    else:
        env.update(
            {
                "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda,cpu"),
            }
        )
        env.pop("PJRT_DEVICE", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("TPU_LIBRARY_PATH", None)
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        extra_flags = os.environ.get(
            "FLAX_GPU_XLA_FLAGS",
            (
                "--xla_gpu_autotune_level=1 --xla_gpu_enable_latency_hiding_scheduler=false "
                "--xla_gpu_force_compilation_parallelism=1"
            ),
        ).strip()
        if extra_flags:
            existing_flags = env.get("XLA_FLAGS", "").strip()
            if extra_flags not in existing_flags:
                combined_flags = f"{existing_flags} {extra_flags}".strip()
                env["XLA_FLAGS"] = combined_flags

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/kaggle/working",
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
        process.wait(timeout=timeout)
        if process.returncode != 0:
            raise RuntimeError(f"Flax inference on {device_label} failed with code {process.returncode}")

        df = pd.read_csv(output_csv)
        results = {k: [] for k in keys}
        pred_cols = [c for c in df.columns if c.startswith("raw_prediction_text_")]
        for _, row in df.iterrows():
            key = row["task_key"]
            try:
                dec = ast.literal_eval(row["decoder"]) if row["decoder"] and row["decoder"] not in ["[]", "{}"] else {}
            except (ValueError, SyntaxError):
                print(f"Warning: Failed parsing decoder '{row['decoder']}'. Using empty.")
                dec = {}
            texts = [str(row.get(c, "")).strip() for c in pred_cols if pd.notna(row.get(c))]
            prompt = row.get("prompt", "PROMPT_NOT_AVAILABLE_IN_CSV")
            results[key].append({"prompt": prompt, "texts": texts, "decoder": dec})
        total = sum(len(v) for v in results.values())
        print(f"Flax inference on {device_label} complete. Total results: {total} across {len(keys)} tasks")
        return results
    except subprocess.TimeoutExpired as err:
        print(f"Flax inference on {device_label} timed out after {timeout}s.")
        if process:
            process.kill()
            process.wait()
        raise RuntimeError(f"Flax inference on {device_label} timed out after {timeout}s.") from err
    except FileNotFoundError as e:
        if str(e).endswith(output_csv):
            print(f"Output not found: {output_csv}. Returning empty.")
            return {k: [] for k in keys}
        raise
    except Exception as e:
        raise RuntimeError(f"Flax inference on {device_label} failed: {e}") from e
    finally:
        for f in [input_csv, output_csv]:
            if os.path.exists(f):
                os.remove(f)


def predict_distributed_gpu(
    model_path: str, infer_data: dict, keys: list[str], settings: dict
) -> dict[str, list[dict[str, Any]]]:
    world_size = torch.cuda.device_count() if torch else 0
    num_workers = max(1, world_size)
    print(f"Detected up to {num_workers} GPU worker(s) for {os.path.basename(model_path)}. Total keys: {len(keys)}")
    _print_inference_runtime_overview(settings, num_workers)
    elapsed = time.time() - START_TIME
    timeout = max(120, TOTAL_TIME - elapsed - BUFFER_TIME - 60)
    settings["timeout"] = settings.get("timeout", int(timeout))

    valid_keys = [k for k in keys if k in infer_data]
    if not valid_keys:
        print("No inference keys matched provided data. Returning empty results.")
        return {k: [] for k in keys}

    # Assign keys greedily based on an estimated work cost per task.
    def _estimate_task_cost(task_key: str) -> int:
        data = infer_data.get(task_key) or {}
        tasks = data.get("tasks") or []
        decs = data.get("decs") or []
        stats = data.get("stats") or {}

        cost_candidates = []
        if isinstance(stats, dict):
            for field in ("estimated_prompt_count", "num_prompts", "mix_total", "num_batches"):
                try:
                    val = int(stats.get(field, 0) or 0)
                except (TypeError, ValueError):
                    val = 0
                if val > 0:
                    cost_candidates.append(val)

        if tasks:
            cost_candidates.append(len(tasks))
        if decs:
            cost_candidates.append(len(decs))

        # Guarantee at least a unit of work so idle GPUs still participate in scheduling.
        return max(cost_candidates) if cost_candidates else 1

    work_items = []
    for key in valid_keys:
        base = infer_data[key]
        base_cost = max(1, _estimate_task_cost(key))
        tasks_list = list(base.get("tasks") or [])
        decs_list = list(base.get("decs") or [])

        if num_workers > 1 and len(tasks_list) > 1:
            shard_count = min(num_workers, len(tasks_list))
            shard_size = max(1, math.ceil(len(tasks_list) / shard_count))
            for shard_idx in range(shard_count):
                start = shard_idx * shard_size
                end = min(len(tasks_list), start + shard_size)
                if start >= end:
                    break
                shard_entry = dict(base)
                shard_entry["tasks"] = tasks_list[start:end]
                shard_entry["decs"] = decs_list[start:end] if decs_list else []
                shard_entry["__original_task_key__"] = key
                shard_entry["__shard_index__"] = shard_idx
                shard_key = f"{key}__shard_{shard_idx}"
                shard_ratio = (end - start) / max(1, len(tasks_list))
                shard_cost = max(1, int(base_cost * shard_ratio))
                work_items.append((shard_key, shard_cost, shard_entry))
        else:
            work_items.append((key, base_cost, dict(base)))

    # Greedy load balancing keeps large tasks from piling onto a single worker.
    work_items.sort(key=lambda item: (-item[1], item[0]))
    data_per_rank = {i: {} for i in range(num_workers)}
    load_per_rank = [0 for _ in range(num_workers)]
    worker_heap = [(0, idx) for idx in range(num_workers)]
    heapq.heapify(worker_heap)

    for assignment_key, cost, payload in work_items:
        load, target_rank = heapq.heappop(worker_heap)
        data_per_rank[target_rank][assignment_key] = payload
        load += cost
        load_per_rank[target_rank] = load
        heapq.heappush(worker_heap, (load, target_rank))

    assigned_workers = sum(1 for data in data_per_rank.values() if data)
    standby_workers = num_workers - assigned_workers
    print(
        f"Launching {num_workers} worker(s) after load balancing"
        f" (assigned: {assigned_workers}, standby: {standby_workers})"
    )
    for i in range(num_workers):
        print(f"Rank {i}: {len(data_per_rank[i])} tasks (est load {load_per_rank[i]})")

    try:
        from . import inference_worker

        with mp.Manager() as manager:
            shared_dict = manager.dict()
            mp.spawn(
                inference_worker.predict_worker,
                args=(num_workers, data_per_rank, model_path, settings, shared_dict),
                nprocs=num_workers,
                join=True,
                daemon=False,
            )

            results = {}
            failed = 0
            empty = 0
            details = []
            for i in range(num_workers):
                out = shared_dict.get(i)
                if out is None:
                    failed += 1
                    details.append(f"Worker {i}: Complete failure")
                    print(f"🔴 Worker {i} failed completely")
                    continue
                if not isinstance(out, dict):
                    failed += 1
                    details.append(f"Worker {i}: Invalid type {type(out)}")
                    print(f"🔴 Worker {i} invalid type: {type(out)}")
                    continue
                if "__worker_failure__" in out:
                    failed += 1
                    err_type = out.get("__error_type__", "Unknown")
                    msg = out.get("__error_message__", "Unknown")
                    details.append(f"Worker {i}: {err_type} - {msg}")
                    print(f"🔴 Worker {i} failure: {err_type} - {msg}")
                    continue
                if not out:
                    empty += 1
                    details.append(f"Worker {i}: Empty dict")
                    print(f"⚠️ Worker {i} empty dict")
                    continue

                task_count = 0
                empty_task = 0
                for k, v in out.items():
                    if isinstance(v, list):
                        results.setdefault(k, []).extend(v)
                        task_count += 1
                        if not v:
                            empty_task += 1
                    else:
                        print(f"🔴 Worker {i} task {k}: Expected list, got {type(v)}")
                if empty_task:
                    details.append(f"Worker {i}: {empty_task}/{task_count} empty tasks")
                    print(f"⚠️ Worker {i}: {empty_task}/{task_count} empty tasks")

            success = num_workers - failed - empty
            print("\n📊 SUMMARY:")
            print(f"Workers: {num_workers}, Successful: {success}, Failed: {failed}, Empty: {empty}")
            if details:
                print("🔍 DETAILS:")
                for d in details:
                    print(f" {d}")
            if failed or empty > num_workers // 2:
                print("🚨 Degraded processing!")

            for k in keys:
                results.setdefault(k, [])
            total = sum(len(v) for v in results.values())
            print(f"GPU inference complete. Total results: {total} across {len(results)} tasks")
            return results
    except Exception as e:
        print(f"GPU inference error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {k: [] for k in keys}
