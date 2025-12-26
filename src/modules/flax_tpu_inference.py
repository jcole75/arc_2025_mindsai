#!/usr/bin/env python3
"""
flax_tpu_inference.py - Run inference using Flax model on TPU/GPU (Optimized Version)
"""

import argparse
import logging
import math
import os

import numpy as np
import pandas as pd


# Set environment variables BEFORE importing JAX
target_device = os.environ.get("FLAX_TARGET_DEVICE", "tpu").strip().lower()
if target_device not in {"tpu", "gpu"}:
    target_device = "tpu"

if target_device == "tpu":
    os.environ.setdefault("JAX_PLATFORMS", "")
    os.environ.setdefault("TPU_LIBRARY_PATH", "/home/jcole75/.local/lib/python3.8/site-packages/libtpu/libtpu.so")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.pop("TPU_LIBRARY_PATH", None)
    os.environ.pop("PJRT_DEVICE", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

if target_device == "gpu":
    import importlib.util

    gpu_backend_spec = importlib.util.find_spec("jaxlib.xla_extension_gpu")
    if gpu_backend_spec is None:
        print("CUDA-enabled JAX not detected. Falling back to CPU execution for inference.")
        target_device = "cpu"
        os.environ["FLAX_TARGET_DEVICE"] = "cpu"
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")  # Suppress TensorFlow warnings

# Block torch_xla from being loaded and force JAX to use correct libtpu
import sys


# Remove torch_xla from sys.modules if it's already loaded
modules_to_remove = [key for key in sys.modules if key.startswith("torch_xla")]
for module in modules_to_remove:
    del sys.modules[module]

# Now import JAX and related libraries
from datasets import Dataset
from flax import jax_utils
from flax.training.common_utils import shard
import jax
import jax.numpy as jnp
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM


logger = logging.getLogger(__name__)

# Optional compilation cache setup to speed up repeated runs
try:
    from jax.experimental import compilation_cache

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir and target_device == "gpu":
        cache_dir = "/kaggle/temp/jax_cache"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_init = getattr(compilation_cache, "initialize_cache", None) or getattr(
            compilation_cache, "initialize_compilation_cache", None
        )
        if cache_init:
            cache_init(cache_dir)
            logger.info(f"Using JAX compilation cache directory: {cache_dir}")
        else:
            logger.info("JAX compilation cache API not available; proceeding without cache.")
except Exception as cache_exc:
    logger.warning(f"Failed to initialize JAX compilation cache: {cache_exc}")

try:
    devices = jax.devices()
    backend = jax.default_backend()
except RuntimeError as backend_err:
    if target_device == "gpu":
        logger.warning(f"Failed to initialize CUDA backend: {backend_err}")
        logger.info(
            "Falling back to CPU execution. Set USE_FLAX_GPU=false or install CUDA-enabled JAX for GPU support."
        )
        target_device = "cpu"
        os.environ["FLAX_TARGET_DEVICE"] = "cpu"
        os.environ.pop("JAX_PLATFORMS", None)
        devices = jax.devices("cpu")
        backend = "cpu"
    else:
        raise

logger.info(f"JAX devices: {devices}")
logger.info(f"JAX backend: {backend}")
expected_backend = "tpu" if target_device == "tpu" else ("gpu" if target_device == "gpu" else "cpu")
if backend != expected_backend:
    logger.warning(f"Expected backend '{expected_backend}' but got '{backend}'. Proceeding regardless.")


def data_loader(rng, dataset, batch_size):
    """A simple and fast data loader for a pre-formatted dataset."""
    steps_per_epoch = math.ceil(len(dataset) / batch_size)

    indices = np.arange(len(dataset))

    for i in range(steps_per_epoch):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        yield dataset[batch_indices]


def run_inference(
    model,
    tokenizer,
    dataset_file,
    batch_size=8,
    num_return_sequences=3,
    num_beams=3,
    temperature=1.0,
    max_input_length=3000,
    max_output_length=600,
):
    def preprocess_function(examples):
        return tokenizer(
            examples["prompt"], max_length=max_input_length, padding="max_length", truncation=True, return_tensors="np"
        )

    original_df = pd.read_csv(dataset_file)
    dataset = Dataset.from_pandas(original_df)

    encoded_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "attention_mask"]],
    )
    encoded_dataset.set_format(type="numpy", columns=["input_ids", "attention_mask"])

    gen_kwargs = {
        "max_length": max_output_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
    }

    if temperature > 0.0 and num_beams == 1:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        gen_kwargs.update({"do_sample": False})

    def generate_step(params, batch):
        return model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs).sequences

    use_pmap = target_device != "cpu" and jax.local_device_count() > 1
    num_devices = jax.local_device_count() if use_pmap else 1

    if use_pmap:
        replicated_params = jax_utils.replicate(model.params)
        p_generate_step = jax.pmap(generate_step, "batch")
    else:
        compiled_generate = jax.jit(
            lambda params, batch: model.generate(
                batch["input_ids"], attention_mask=batch["attention_mask"], params=params, **gen_kwargs
            ).sequences
        )

    total_batch_size = batch_size * num_devices

    pred_loader = data_loader(jax.random.PRNGKey(0), encoded_dataset, total_batch_size)

    all_predictions = []
    pbar_total = math.ceil(len(encoded_dataset) / total_batch_size)
    pbar = tqdm(total=pbar_total, desc=f"{target_device.upper()} Predicting")

    for batch in pred_loader:
        batch_size_unpadded = batch["input_ids"].shape[0]
        pad_len = -batch_size_unpadded % num_devices

        if use_pmap:
            if pad_len > 0:
                padded_batch = jax.tree_map(
                    lambda x, pad_amount=pad_len: np.pad(
                        x,
                        ((0, pad_amount), (0, 0)),
                        mode="constant",
                        constant_values=tokenizer.pad_token_id if x.dtype == np.int32 else 0,
                    ),
                    batch,
                )
            else:
                padded_batch = batch

            pred_ids = p_generate_step(replicated_params, shard(padded_batch))
            decoded_ids = pred_ids.reshape(-1, *pred_ids.shape[2:])
            if pad_len > 0:
                decoded_ids = decoded_ids[:-pad_len]
        else:
            if pad_len > 0:
                batch = jax.tree_map(
                    lambda x, pad_amount=pad_len: np.pad(
                        x,
                        ((0, pad_amount), (0, 0)),
                        mode="constant",
                        constant_values=tokenizer.pad_token_id if x.dtype == np.int32 else 0,
                    ),
                    batch,
                )
            batch_jnp = {k: jnp.array(v) for k, v in batch.items()}
            decoded_ids = compiled_generate(model.params, batch_jnp)
            decoded_ids = np.array(decoded_ids)
            if pad_len > 0:
                decoded_ids = decoded_ids[:-pad_len]

        decoded_texts = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)

        for i in range(0, len(decoded_texts), num_return_sequences):
            all_predictions.append(decoded_texts[i : i + num_return_sequences])

        pbar.update(1)
    pbar.close()

    pred_cols = [f"raw_prediction_text_{i + 1}" for i in range(num_return_sequences)]
    predictions_df = pd.DataFrame(all_predictions, columns=pred_cols)

    final_df = pd.concat([original_df.iloc[: len(predictions_df)], predictions_df], axis=1)
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Run inference using Flax model on TPU/GPU")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size *per device*")
    # <--- KEY FIX: Add dtype argument --->
    parser.add_argument("--dtype", type=str, default="float32", help="Model dtype (e.g., bfloat16, float32)")
    parser.add_argument("--num_return_sequences", type=int, default=3)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_input_length", type=int, default=3000)
    parser.add_argument("--max_output_length", type=int, default=600)
    parser.add_argument("--model_type", type=str, default="seq2seq", help="Model type (seq2seq or causal_lm)")
    parser.add_argument("--tokenizer_dropout_enabled", action="store_true", help="Enable tokenizer BPE dropout.")
    parser.add_argument("--tokenizer_dropout_rate", type=float, default=0.1, help="Rate for tokenizer BPE dropout.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Target device: {target_device.upper()}")
    logger.info(f"JAX devices: {jax.devices()}")
    backend = jax.default_backend()
    logger.info(f"JAX backend: {backend}")
    expected_backend = "tpu" if target_device == "tpu" else "gpu"
    if backend != expected_backend:
        logger.warning(f"Expected backend '{expected_backend}' but got '{backend}'. Performance may be impacted.")

    # <--- KEY FIX: Use the dtype argument when loading the model --->
    logger.info(f"Loading model from {args.model_path} with dtype: {args.dtype}")

    logger.info("Loading standard T5 model")
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(args.model_path, dtype=getattr(jnp, args.dtype))

    # Load tokenizer with optional dropout
    sp_model_kwargs = {}
    if args.tokenizer_dropout_enabled and args.tokenizer_dropout_rate > 0.0:
        sp_model_kwargs = {"enable_sampling": True, "alpha": args.tokenizer_dropout_rate}
        logger.info(
            "Tokenizer BPE dropout enabled with rate (alpha): "
            f"{args.tokenizer_dropout_rate}. Passing sp_model_kwargs: {sp_model_kwargs}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **(sp_model_kwargs if sp_model_kwargs else {}))

    results_df = run_inference(
        model=model,
        tokenizer=tokenizer,
        dataset_file=args.input_file,
        batch_size=args.batch_size,
        num_return_sequences=args.num_return_sequences,
        num_beams=args.num_beams,
        temperature=args.temperature,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )

    logger.info(f"Saving {len(results_df)} results to {args.output_file}")
    results_df.to_csv(args.output_file, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
