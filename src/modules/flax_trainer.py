print("****************** flax_trainer.py -- Training script for Flax.")
import os


target_device = os.environ.get("FLAX_TARGET_DEVICE", "tpu").strip().lower()
if target_device not in {"tpu", "gpu"}:
    target_device = "tpu"

# Enforce platform preference before importing JAX
if target_device == "tpu":
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TPU_LIBRARY_PATH", "/home/jcole75/.local/lib/python3.8/site-packages/libtpu/libtpu.so")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.pop("PJRT_DEVICE", None)
    os.environ.pop("TPU_LIBRARY_PATH", None)

# If GPU backend isn't actually available, fall back to CPU before importing jax
if target_device == "gpu":
    import importlib.util

    gpu_backend_spec = importlib.util.find_spec("jaxlib.xla_extension_gpu")
    if gpu_backend_spec is None:
        print("INFO: CUDA-enabled JAX not found. Falling back to CPU execution.")
        target_device = "cpu"
        os.environ["FLAX_TARGET_DEVICE"] = "cpu"
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

print(f"DEBUG: Target device: {target_device.upper()}")
print(f"DEBUG: JAX_PLATFORMS env var at start: {os.environ.get('JAX_PLATFORMS', 'NOT_SET')}")

print("DEBUG: About to import JAX FIRST")
# Block torch_xla from being loaded and force JAX to use correct backend
import sys


# Remove torch_xla from sys.modules if it's already loaded
modules_to_remove = [key for key in sys.modules if key.startswith("torch_xla")]
for module in modules_to_remove:
    del sys.modules[module]

# Import JAX FIRST before anything else
import jax


print("DEBUG: JAX imported successfully")

try:
    devices = jax.devices()
    backend = jax.default_backend()
except RuntimeError as backend_err:
    if target_device == "gpu":
        print(f"WARNING: Failed to initialize CUDA backend: {backend_err}")
        print(
            "INFO: Falling back to CPU execution. Set USE_FLAX_GPU=false or install "
            "CUDA-enabled JAX if you prefer GPU acceleration."
        )
        target_device = "cpu"
        os.environ["FLAX_TARGET_DEVICE"] = "cpu"
        os.environ.pop("JAX_PLATFORMS", None)
        devices = jax.devices("cpu")
        backend = "cpu"
    else:
        raise

print(f"DEBUG: JAX devices after import: {devices}")
print(f"DEBUG: JAX backend: {backend}")
expected_backend = "tpu" if target_device == "tpu" else ("gpu" if target_device == "gpu" else "cpu")
if backend != expected_backend:
    print(
        "WARNING: JAX did not initialize with expected backend "
        f"'{expected_backend}'. Using '{backend}' instead; performance may differ."
    )

# Initialize optional compilation cache to amortize XLA build times across runs
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
            print(f"DEBUG: JAX compilation cache directory: {cache_dir}")
        else:
            print("WARNING: JAX compilation cache API not available on this version.")
except Exception as cache_exc:
    print(f"WARNING: Failed to initialize JAX compilation cache: {cache_exc}")

import jax.numpy as jnp


# Decide execution mode (pmap vs jit)
gpu_pmap_env = os.environ.get("FLAX_GPU_USE_PMAP", "").strip().lower()
use_pmap = target_device == "tpu"
auto_pmap = False
if target_device == "gpu":
    if gpu_pmap_env in ("1", "true", "yes", "on"):
        use_pmap = True
    elif gpu_pmap_env in ("0", "false", "no", "off"):
        use_pmap = False
    else:
        auto_pmap = True
        use_pmap = True  # provisional, may change after inspecting dataset size
elif target_device == "cpu":
    use_pmap = False
print(f"DEBUG: Training execution mode: {'pmap' if use_pmap else 'jit'}")

print("DEBUG: Now importing other libraries")
# Now import other libraries
from dataclasses import dataclass
from functools import partial
import json
import logging
import sys

from datasets import Dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import shard
import numpy as np
import optax
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    HfArgumentParser,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    output_dir: str
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    warmup_steps: int = 0
    logging_steps: int = 500
    save_steps: int = 500
    gradient_checkpointing: bool = False
    filter_datasets_for_max_tokens: bool = False
    shuffle_training_data: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: str
    config_name: str | None = None
    tokenizer_name: str | None = None
    cache_dir: str | None = None
    dtype: str = "float32"
    tokenizer_dropout_enabled: bool = False
    tokenizer_dropout_rate: float = 0.1
    model_type: str = "seq2seq"
    is_causal_lm: bool = False
    pass


@dataclass
class DataTrainingArguments:
    train_file: str
    max_source_length: int = 1024
    max_target_length: int = 128
    preprocessing_num_workers: int | None = None
    overwrite_cache: bool = False
    max_length: int | None = None


def setup_logging(log_level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=log_level
    )


def load_and_process_dataset(file_name, tokenizer, data_args, config, shift_tokens_right_fn):
    def normalize_string(text):
        return " ".join(str(text).strip().split())

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["prompt"], max_length=data_args.max_source_length, padding="max_length", truncation=True
        )
        labels = tokenizer(
            text_target=examples["correct_answer"],
            max_length=data_args.max_target_length,
            padding="max_length",
            truncation=True,
        )
        # Shift before mutating labels for ignore index
        shifted_decoder_inputs = shift_tokens_right_fn(
            np.array(labels["input_ids"]), config.pad_token_id, config.decoder_start_token_id
        )
        # Replace padding tokens in labels with -100 so they are ignored in loss
        pad_id = config.pad_token_id if config.pad_token_id is not None else 0
        labels_ids = labels["input_ids"]
        for i, seq in enumerate(labels_ids):
            labels_ids[i] = [(-100 if tok == pad_id else tok) for tok in seq]
        model_inputs["labels"] = labels_ids
        model_inputs["decoder_input_ids"] = shifted_decoder_inputs
        return model_inputs

    df = pd.read_csv(file_name)
    df = df[["prompt", "correct_answer"]]
    df["prompt"] = df["prompt"].apply(normalize_string)
    df["correct_answer"] = df["correct_answer"].apply(normalize_string)
    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    dataset.set_format(type="numpy", columns=["input_ids", "attention_mask", "decoder_input_ids", "labels"])
    return dataset


def create_learning_rate_fn(train_ds_size, train_batch_size, num_train_epochs, num_warmup_steps, learning_rate):
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    return optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])


def create_optimizer(learning_rate_fn, weight_decay):
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        return traverse_util.unflatten_dict({path: (path[-1] != "bias") for path in flat_params})

    # Add global norm gradient clipping + AdamW
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_fn, weight_decay=weight_decay, mask=decay_mask_fn),
    )


def train_step(state, batch, apply_fn, learning_rate_fn, axis_name=None):
    step_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    dropout_rng = step_rng

    def compute_loss(params):
        labels = batch["labels"]
        valid_mask = labels != -100
        model_batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "decoder_input_ids")}
        outputs = apply_fn(**model_batch, params=params, dropout_rng=dropout_rng, train=True)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).astype(jnp.float32)
        loss = (per_token_loss * valid_mask).sum() / valid_mask.sum()
        return loss

    grad_fn = jax.value_and_grad(compute_loss, has_aux=False)
    loss, grad = grad_fn(state.params)
    if axis_name:
        grad = jax.lax.psum(grad, axis_name)

    new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)
    current_lr = learning_rate_fn(state.step)
    metrics = {
        "loss": jax.lax.psum(loss, axis_name) if axis_name else loss,
        "learning_rate": current_lr,
    }
    return new_state, metrics


def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size
    if steps_per_epoch == 0:
        return
    indices = jax.random.permutation(rng, len(dataset)) if shuffle else np.arange(len(dataset))
    for i in range(0, steps_per_epoch * batch_size, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield dataset[batch_indices]


def main():
    _parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    config_file = sys.argv[1]
    with open(config_file) as f:
        config_dict = json.load(f)

    model_args = ModelArguments(**config_dict.get("model_args", {}))
    data_args = DataTrainingArguments(**config_dict.get("data_args", {}))
    training_args = TrainingArguments(**config_dict.get("training_args", {}))

    setup_logging(logging.INFO)
    logger.info(f"JAX devices: {jax.devices()}")

    # Resolve model path: allow absolute Kaggle dataset path fallback to HF repo id
    resolved_model_id = model_args.model_name_or_path
    if resolved_model_id.startswith("/kaggle/input/") and not os.path.exists(resolved_model_id):
        base_name = os.path.basename(resolved_model_id)
        kaggle_to_hf_map = {"codet5-large": "Salesforce/codet5-large", "codet5-small": "Salesforce/codet5-small"}
        if base_name in kaggle_to_hf_map:
            print(
                "[MODEL_PATH_FALLBACK][flax_trainer] Path "
                f"'{resolved_model_id}' not found. Using '{kaggle_to_hf_map[base_name]}'"
            )
            resolved_model_id = kaggle_to_hf_map[base_name]
        else:
            print(
                "[MODEL_PATH_FALLBACK][flax_trainer] Path "
                f"'{resolved_model_id}' not found. Trying basename '{base_name}' as repo id"
            )
            resolved_model_id = base_name

    config = AutoConfig.from_pretrained(model_args.config_name or resolved_model_id, cache_dir=model_args.cache_dir)
    # Load tokenizer with optional dropout
    sp_model_kwargs = {}
    if model_args.tokenizer_dropout_enabled and model_args.tokenizer_dropout_rate > 0.0:
        sp_model_kwargs = {"enable_sampling": True, "alpha": model_args.tokenizer_dropout_rate}
        logger.info(
            "Tokenizer BPE dropout enabled with rate (alpha): "
            f"{model_args.tokenizer_dropout_rate}. Passing sp_model_kwargs: {sp_model_kwargs}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or resolved_model_id,
        cache_dir=model_args.cache_dir,
        **(sp_model_kwargs if sp_model_kwargs else {}),
    )

    flax_model_path = os.path.join(model_args.model_name_or_path, "flax_model.msgpack")

    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
        resolved_model_id,
        config=config,
        seed=42,
        dtype=getattr(jnp, model_args.dtype),
        from_pt=not os.path.exists(flax_model_path),
        cache_dir=model_args.cache_dir,
    )

    if training_args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    model_module = __import__(model.__module__, fromlist=["shift_tokens_right"])
    shift_tokens_right_fn = model_module.shift_tokens_right

    train_dataset = load_and_process_dataset(data_args.train_file, tokenizer, data_args, config, shift_tokens_right_fn)

    num_epochs = int(training_args.num_train_epochs)
    available_devices = len(devices)
    use_pmap_local = use_pmap
    device_count = available_devices if use_pmap_local else 1
    train_batch_size = int(training_args.per_device_train_batch_size) * device_count
    steps_per_epoch = len(train_dataset) // max(train_batch_size, 1)

    if auto_pmap:
        min_steps = int(os.environ.get("FLAX_GPU_PMAP_MIN_STEPS", "80"))
        min_examples = int(os.environ.get("FLAX_GPU_PMAP_MIN_EXAMPLES", "4096"))
        if available_devices <= 1:
            use_pmap_local = False
            device_count = 1
        elif len(train_dataset) < min_examples or steps_per_epoch < min_steps:
            use_pmap_local = False
            device_count = 1
            reason = "dataset too small" if len(train_dataset) < min_examples else "too few steps"
            logger.info(
                "[Flax][GPU] Auto-disabling multi-device pmap "
                f"({reason}, steps={steps_per_epoch}, examples={len(train_dataset)}). "
                "Override with FLAX_GPU_USE_PMAP=true."
            )
        else:
            use_pmap_local = True
        if not use_pmap_local:
            train_batch_size = int(training_args.per_device_train_batch_size)
            steps_per_epoch = len(train_dataset) // max(train_batch_size, 1)

    device_count = available_devices if use_pmap_local else 1
    train_batch_size = int(training_args.per_device_train_batch_size) * device_count
    steps_per_epoch = len(train_dataset) // max(train_batch_size, 1)

    if steps_per_epoch == 0:
        logger.warning("Dataset too small. No training steps. Saving original model.")
        if jax.process_index() == 0:
            model.save_pretrained(training_args.output_dir, params=model.params)
            tokenizer.save_pretrained(training_args.output_dir)
        return

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset), train_batch_size, num_epochs, training_args.warmup_steps, training_args.learning_rate
    )

    optimizer = create_optimizer(learning_rate_fn, training_args.weight_decay)

    # Track dropout rng in state
    class TrainStateWithRng(train_state.TrainState):
        dropout_rng: jax.random.PRNGKey

    rng_for_state = jax.random.PRNGKey(42)
    state = TrainStateWithRng.create(
        apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=rng_for_state
    )

    if use_pmap_local:
        p_train_step = jax.pmap(
            partial(
                train_step,
                apply_fn=model.__call__,
                learning_rate_fn=learning_rate_fn,
                axis_name="batch",
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
        state = jax_utils.replicate(state)
    else:
        train_step_fn = jax.jit(
            partial(
                train_step,
                apply_fn=model.__call__,
                learning_rate_fn=learning_rate_fn,
                axis_name=None,
            )
        )

    rng = jax.random.PRNGKey(42)

    # <--- Enhanced Training Logging Block --->
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Active devices = {device_count}")
    logger.info(f"  Batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {train_batch_size}")
    logger.info(f"  Total optimization steps = {steps_per_epoch * num_epochs}")

    # <------------------------------>

    for epoch in tqdm(range(num_epochs), desc="Epoch ...", position=0):
        rng, input_rng = jax.random.split(rng)
        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)

        with tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False) as train_progress:
            for _ in train_progress:
                batch = next(train_loader)
                if use_pmap_local:
                    state, train_metric = p_train_step(state, shard(batch))
                    train_metric = unreplicate(train_metric)
                    loss_value = float(train_metric["loss"])
                else:
                    batch = {k: jnp.array(v) for k, v in batch.items()}
                    state, train_metric = train_step_fn(state, batch)
                    loss_value = float(train_metric["loss"])
                train_progress.set_postfix(loss=f"{loss_value:.4f}")

        final_loss = float(train_metric["loss"])
        final_lr = float(train_metric["learning_rate"])
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Loss: {final_loss:.4f}, LR: {final_lr:.6f}")
        if jax.process_index() == 0:
            if use_pmap_local:
                params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            else:
                params = jax.device_get(state.params)
            model.save_pretrained(training_args.output_dir, params=params)
            tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
