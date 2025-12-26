"""
Utility functions module for ARC Prize 2025 solution
Contains helper functions and setup utilities
"""

import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

# Import config to determine which libraries to import
from . import config

KAGGLE_CODE_DATASET_DEFAULT = "arc2025-solution-public"


def get_kaggle_code_dataset_slug() -> str:
    """Return Kaggle dataset slug (overridable via ARC_KAGGLE_CODE_DATASET)."""
    return os.getenv("ARC_KAGGLE_CODE_DATASET", KAGGLE_CODE_DATASET_DEFAULT)


# Conditional imports based on framework setting
if not config.USE_FLAX:
    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError:
        torch = None
        mp = None
        print("Warning: torch not available for PyTorch model averaging")
else:
    torch = None
    mp = None
    # JAX/Flax imports are moved to function-level to avoid early initialization
    jax = None
    jnp = None


def setup_multiprocessing():
    """Setup multiprocessing start method."""
    if mp is None:
        print("Note: Multiprocessing not available (torch.multiprocessing not imported). Proceeding.", file=sys.stderr)
        return
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
    except RuntimeError as e_mp:
        print(f"Note: Multiprocessing context issue: {e_mp}. Proceeding.", file=sys.stderr)
        pass


def cleanup_directories(paths: list[str]):
    """Clean up specified directories.

    Honors the PRESERVE_FINETUNED_MODEL flag by skipping deletion of the
    fine-tuned model directory (reload path) and any directory whose basename
    contains 'model_fine_tuned'. This prevents accidental loss of the saved
    fine-tuned weights on Kaggle/TPU runs where the user wants to reuse them.
    """
    preserve_enabled = getattr(config, "PRESERVE_FINETUNED_MODEL", False)
    reload_path = getattr(config, "RELOAD_PATH", None)
    for path in paths:
        if not os.path.exists(path):
            continue

        # Determine if this path should be preserved
        should_preserve = False
        if preserve_enabled:
            try:
                abs_path = os.path.abspath(path)
                if (reload_path and os.path.abspath(reload_path) == abs_path) or "model_fine_tuned" in os.path.basename(
                    abs_path
                ):
                    should_preserve = True
            except Exception:
                # Fail safe: if any issue determining, don't delete
                should_preserve = True

        if should_preserve:
            print(f"🔒 Preserving directory (PRESERVE_FINETUNED_MODEL=True): {path}")
            continue

        try:
            shutil.rmtree(path)
            print(f"Cleaned up directory: {path}")
        except OSError as e:
            print(f"Warning: Failed to remove directory {path}: {e}", file=sys.stderr)


def setup_kaggle_environment() -> tuple[bool, bool]:
    """
    Setup code from Kaggle dataset and display environment information.
    Returns (code_setup_success, is_kaggle)
    """
    is_kaggle = bool(getattr(config, "IS_KAGGLE", False))

    print(f"🌍 Environment: {'Kaggle' if is_kaggle else 'Local'}")

    # Display GPU information
    if is_kaggle:
        def _print_gpu_info():
            try:
                gpu_info = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                print(f"  - GPUs: {gpu_info.stdout.strip() if gpu_info.returncode == 0 else 'None detected'}")
            except FileNotFoundError:
                print("  - nvidia-smi not available")

        # Only touch JAX when TPU mode is requested. Otherwise a missing CUDA-enabled
        # jaxlib (common on local runs) raises runtime errors even for PyTorch flows.
        if config.USE_FLAX:
            # When Flax is enabled, avoid importing JAX prematurely but surface accelerator info if available
            mode_desc = "TPU" if config.USE_TPU else "Flax GPU"
            print(f"  - {mode_desc} mode enabled - skipping early device detection to avoid JAX initialization")
            try:
                _print_gpu_info()
            except Exception:
                print("  - Unable to query GPU info (nvidia-smi).")
        elif config.USE_TPU:
            try:
                import jax

                tpu_devices = len(jax.devices())
                if tpu_devices > 0:
                    print(f"  - TPU devices: {tpu_devices}")
                else:
                    _print_gpu_info()
            except (ImportError, RuntimeError) as exc:
                print(f"  - TPU detection skipped (JAX unavailable: {exc}).")
                _print_gpu_info()
        else:
            _print_gpu_info()

    print(f"  - Working directory: {os.getcwd()}")

    if not is_kaggle:
        return False, is_kaggle

    # Setup code from dataset (Kaggle only)
    code_setup_success = copy_code_from_dataset()

    # Add src directory to Python path
    src_path = Path("/kaggle/working/src")
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✓ Added src path: {src_path}")

    # Verify files are available
    main_py_path = Path("/kaggle/working/src/main.py")
    modules_dir = Path("/kaggle/working/src/modules")

    if main_py_path.exists():
        print(f"✓ Main script found: {main_py_path}")
    else:
        print(f"❌ Main script not found: {main_py_path}")

    if modules_dir.exists():
        module_files = list(modules_dir.glob("*.py"))
        print(f"✓ Modules directory found with {len(module_files)} Python files")
    else:
        print(f"❌ Modules directory not found: {modules_dir}")

    return code_setup_success, is_kaggle


def copy_code_from_dataset() -> bool:
    """Copy code files from Kaggle dataset to working directory"""
    dataset_slug = get_kaggle_code_dataset_slug()
    dataset_path = Path("/kaggle/input") / dataset_slug
    working_path = Path("/kaggle/working")

    if not dataset_path.exists():
        print(f"⚠️  Dataset '{dataset_slug}' not found at {dataset_path}. Using local files if available.")
        print("   Override with ARC_KAGGLE_CODE_DATASET if you mounted a different dataset.")
        return False

    print(f"\n📦 Copying code from dataset: {dataset_path}")

    # Files to copy directly
    direct_files = ["requirements.txt", "README.md", ".gitignore"]

    copied_count = 0

    # Copy direct files
    for item_name in direct_files:
        src_path = dataset_path / item_name
        dst_path = working_path / item_name

        try:
            if src_path.is_file():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ Copied file: {item_name}")
                copied_count += 1
        except Exception as e:
            print(f"  ❌ Failed to copy {item_name}: {e}")

    # Handle src.zip (created by --dir-mode zip)
    src_zip_path = dataset_path / "src.zip"
    if src_zip_path.exists():
        try:
            print("  📦 Extracting src.zip...")
            with zipfile.ZipFile(src_zip_path, "r") as zip_ref:
                zip_ref.extractall(working_path)

            # Verify extraction
            src_dir = working_path / "src"
            if src_dir.exists():
                file_count = len([f for f in src_dir.rglob("*") if f.is_file()])
                print(f"  ✓ Extracted src/ directory ({file_count} files)")
                copied_count += 1
            else:
                print("  ❌ src/ directory not found after extraction")

        except Exception as e:
            print(f"  ❌ Failed to extract src.zip: {e}")

    # Fallback: Check for direct src directory
    elif (dataset_path / "src").exists():
        try:
            src_path = dataset_path / "src"
            dst_path = working_path / "src"

            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            file_count = len([f for f in dst_path.rglob("*") if f.is_file()])
            print(f"  ✓ Copied directory: src/ ({file_count} files)")
            copied_count += 1
        except Exception as e:
            print(f"  ❌ Failed to copy src directory: {e}")
    else:
        print("  ❌ No src.zip or src/ directory found in dataset")

    print(f"📋 Setup complete: {copied_count}/{len(direct_files) + 1} items from dataset")
    return copied_count > 0


def display_working_directory_contents():
    """Display contents of working directory for debugging"""
    working_path = Path("/kaggle/working")
    if not working_path.exists():
        working_path = Path(".")

    print("\n📁 Working directory contents:")
    working_files = list(working_path.glob("*"))
    for f in sorted(working_files):
        if f.is_dir():
            file_count = len([x for x in f.rglob("*") if x.is_file()])
            print(f"  📂 {f.name}/ ({file_count} files)")
        else:
            print(f"  📄 {f.name}")


# =============================================================================
# MODEL WEIGHT AVERAGING UTILITIES
# =============================================================================


def average_model_weights(model_paths: list[str], output_path: str, device: str = "cpu") -> bool:
    """
    Average the weights of multiple fine-tuned models and save the averaged model.
    Routes to either PyTorch or JAX/Flax implementation based on USE_FLAX setting.

    Args:
        model_paths: List of paths to fine-tuned model directories
        output_path: Path where the averaged model should be saved
        device: Device to load models on ('cpu' or 'cuda') - ignored for JAX/Flax

    Returns:
        bool: True if averaging was successful, False otherwise
    """
    if len(model_paths) < 2:
        print(f"Warning: Need at least 2 models to average, got {len(model_paths)}")
        return False

    if config.USE_FLAX:
        return average_model_weights_flax(model_paths, output_path)
    else:
        return average_model_weights_pytorch(model_paths, output_path, device)


def average_model_weights_pytorch(model_paths: list[str], output_path: str, device: str = "cpu") -> bool:
    """
    Average the weights of multiple fine-tuned PyTorch models and save the averaged model.

    Args:
        model_paths: List of paths to fine-tuned model directories
        output_path: Path where the averaged model should be saved
        device: Device to load models on ('cpu' or 'cuda')

    Returns:
        bool: True if averaging was successful, False otherwise
    """
    if torch is None:
        print("Error: PyTorch not available for model averaging")
        return False

    print(f"🔄 Averaging weights from {len(model_paths)} PyTorch models...")

    try:
        # Load the first model to get the structure
        first_model_path = model_paths[0]
        print(f"  Loading structure from: {first_model_path}")

        # Find and load model state dict
        model_files_to_check = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors", "model.bin"]

        first_model_file = None
        for model_file in model_files_to_check:
            model_file_path = os.path.join(first_model_path, model_file)
            if os.path.exists(model_file_path):
                first_model_file = model_file_path
                break

        if first_model_file is None:
            print(f"Error: No model file found in {first_model_path}")
            return False

        print(f"  Loading structure from: {os.path.basename(first_model_file)}")

        # Load model state dict (handle both .bin and .safetensors)
        if first_model_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file

                model_state_dict = load_file(first_model_file, device=device)
            except ImportError:
                print("Error: safetensors library not available. Cannot load .safetensors files.")
                return False
        else:
            model_state_dict = torch.load(first_model_file, map_location=device)

        # Convert to float for averaging
        averaged_state_dict = {}
        for key, value in model_state_dict.items():
            if isinstance(value, torch.Tensor):
                averaged_state_dict[key] = value.float()
            else:
                averaged_state_dict[key] = value

        # Add weights from remaining models
        for i, model_path in enumerate(model_paths[1:], 1):
            print(f"  Adding weights from model {i + 1}/{len(model_paths)}: {model_path}")

            # Find model file for this path
            current_model_file = None
            for model_file in model_files_to_check:
                model_file_path = os.path.join(model_path, model_file)
                if os.path.exists(model_file_path):
                    current_model_file = model_file_path
                    break

            if current_model_file is None:
                print(f"  Warning: No model file found in {model_path}, skipping")
                continue

            # Load model state dict (handle both .bin and .safetensors)
            if current_model_file.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file

                    current_state_dict = load_file(current_model_file, device=device)
                except ImportError:
                    print(f"  Warning: Cannot load .safetensors file {current_model_file}, skipping")
                    continue
            else:
                current_state_dict = torch.load(current_model_file, map_location=device)

            for key, value in current_state_dict.items():
                if isinstance(value, torch.Tensor) and key in averaged_state_dict:
                    averaged_state_dict[key] += value.float()
                elif key not in averaged_state_dict:
                    print(f"  Warning: Key '{key}' not found in first model, skipping")

            # Clear memory
            del current_state_dict
            gc.collect()

        # Average the weights
        print("  Computing average weights...")
        for key, value in averaged_state_dict.items():
            if isinstance(value, torch.Tensor):
                averaged_state_dict[key] = value / len(model_paths)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save averaged model (use standard transformers filenames)
        # Note: transformers library expects specific filenames
        output_format = (
            "model.safetensors" if first_model_file.endswith(".safetensors") else "pytorch_model.bin"
        )

        averaged_model_path = os.path.join(output_path, output_format)

        # Save in the appropriate format
        if output_format.endswith(".safetensors"):
            try:
                from safetensors.torch import save_file

                save_file(averaged_state_dict, averaged_model_path)
                print(f"  ✓ Saved averaged model (safetensors) to: {averaged_model_path}")
            except ImportError:
                # Fallback to pytorch format with standard filename
                output_format = "pytorch_model.bin"
                averaged_model_path = os.path.join(output_path, output_format)
                torch.save(averaged_state_dict, averaged_model_path)
                print(f"  ✓ Saved averaged model (pytorch fallback) to: {averaged_model_path}")
        else:
            torch.save(averaged_state_dict, averaged_model_path)
            print(f"  ✓ Saved averaged model (pytorch) to: {averaged_model_path}")

        # Copy other necessary files from the first model
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "added_tokens.json",
        ]

        for filename in files_to_copy:
            src_file = os.path.join(first_model_path, filename)
            dst_file = os.path.join(output_path, filename)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ Copied {filename}")

        # Clear memory
        del averaged_state_dict
        gc.collect()

        print("✅ Model averaging completed successfully")
        return True

    except Exception as e:
        print(f"❌ Model averaging failed: {e}")
        return False


def average_model_weights_flax(model_paths: list[str], output_path: str) -> bool:
    """
    Average the weights of multiple fine-tuned JAX/Flax models and save the averaged model.

    Args:
        model_paths: List of paths to fine-tuned model directories
        output_path: Path where the averaged model should be saved

    Returns:
        bool: True if averaging was successful, False otherwise
    """
    # Import JAX/Flax locally to avoid early initialization at module level
    try:
        from flax.serialization import from_bytes, to_bytes
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("Error: JAX/Flax not available for model averaging")
        return False

    print(f"🔄 Averaging weights from {len(model_paths)} JAX/Flax models...")

    try:
        # Load the first model to get the structure
        first_model_path = model_paths[0]
        print(f"  Loading structure from: {first_model_path}")

        # Look for Flax model file
        flax_model_file = os.path.join(first_model_path, "flax_model.msgpack")

        if not os.path.exists(flax_model_file):
            print(f"Error: No flax_model.msgpack found in {first_model_path}")
            return False

        print("  Loading structure from: flax_model.msgpack")

        # Load model parameters
        with open(flax_model_file, "rb") as f:
            first_params = from_bytes(None, f.read())

        # Initialize averaged parameters
        averaged_params = jax.tree_map(lambda x: x.astype(jnp.float32), first_params)

        # Add weights from remaining models
        for i, model_path in enumerate(model_paths[1:], 1):
            print(f"  Adding weights from model {i + 1}/{len(model_paths)}: {model_path}")

            current_flax_file = os.path.join(model_path, "flax_model.msgpack")

            if not os.path.exists(current_flax_file):
                print(f"  Warning: No flax_model.msgpack found in {model_path}, skipping")
                continue

            # Load current model parameters
            with open(current_flax_file, "rb") as f:
                current_params = from_bytes(None, f.read())

            # Add to averaged parameters
            averaged_params = jax.tree_map(
                lambda avg, curr: avg + curr.astype(jnp.float32), averaged_params, current_params
            )

            # Clear memory
            del current_params
            gc.collect()

        # Average the weights
        print("  Computing average weights...")
        averaged_params = jax.tree_map(lambda x: x / len(model_paths), averaged_params)

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save averaged model
        averaged_model_path = os.path.join(output_path, "flax_model.msgpack")
        with open(averaged_model_path, "wb") as f:
            f.write(to_bytes(averaged_params))

        print(f"  ✓ Saved averaged Flax model to: {averaged_model_path}")

        # Copy other necessary files from the first model
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "added_tokens.json",
        ]

        for filename in files_to_copy:
            src_file = os.path.join(first_model_path, filename)
            dst_file = os.path.join(output_path, filename)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ Copied {filename}")

        # Clear memory
        del averaged_params
        gc.collect()

        print("✅ Flax model averaging completed successfully")
        return True

    except Exception as e:
        print(f"❌ Flax model averaging failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_models_for_averaging(model_paths: list[str]) -> bool:
    """
    Validate that all models exist and are compatible for averaging.
    Routes to either PyTorch or JAX/Flax validation based on USE_FLAX setting.

    Args:
        model_paths: List of paths to model directories

    Returns:
        bool: True if all models are valid for averaging
    """
    if len(model_paths) < 2:
        print(f"Warning: Need at least 2 models to average, got {len(model_paths)}")
        return False

    if config.USE_FLAX:
        return validate_models_for_averaging_flax(model_paths)
    else:
        return validate_models_for_averaging_pytorch(model_paths)


def validate_models_for_averaging_pytorch(model_paths: list[str]) -> bool:
    """
    Validate that all PyTorch models exist and are compatible for averaging.

    Args:
        model_paths: List of paths to model directories

    Returns:
        bool: True if all models are valid for averaging
    """
    print(f"🔍 Validating {len(model_paths)} PyTorch models for averaging...")

    # Check all models exist
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"❌ Model {i + 1} not found: {model_path}")
            return False

        # Check for model file (any of the supported formats)
        model_files_to_check = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors", "model.bin"]

        model_file_found = False
        for model_file in model_files_to_check:
            if os.path.exists(os.path.join(model_path, model_file)):
                model_file_found = True
                break

        if not model_file_found:
            print(f"❌ No PyTorch model file found in model {i + 1}: {model_path}")
            print(f"    Expected one of: {model_files_to_check}")
            return False

        # Check for config file
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            print(f"❌ Required file missing in model {i + 1}: config.json")
            return False

        print(f"  ✓ PyTorch Model {i + 1} validated: {model_path}")

    # Check model compatibility (same architecture)
    try:
        first_config_path = os.path.join(model_paths[0], "config.json")
        with open(first_config_path) as f:
            import json

            first_config = json.load(f)

        for i, model_path in enumerate(model_paths[1:], 1):
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                current_config = json.load(f)

            # Check critical config parameters
            critical_params = ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]
            for param in critical_params:
                if (
                    param in first_config
                    and param in current_config
                    and first_config[param] != current_config[param]
                ):
                    print(
                        "❌ Config mismatch in model "
                        f"{i + 1}: {param} = {current_config[param]} vs {first_config[param]}"
                    )
                    return False

        print("  ✓ All models have compatible configurations")

    except Exception as e:
        print(f"❌ Error validating model configurations: {e}")
        return False

    print("✅ All PyTorch models validated for averaging")
    return True


def validate_models_for_averaging_flax(model_paths: list[str]) -> bool:
    """
    Validate that all JAX/Flax models exist and are compatible for averaging.

    Args:
        model_paths: List of paths to model directories

    Returns:
        bool: True if all models are valid for averaging
    """
    print(f"🔍 Validating {len(model_paths)} JAX/Flax models for averaging...")

    # Check all models exist
    for i, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"❌ Model {i + 1} not found: {model_path}")
            return False

        # Check for Flax model file
        flax_model_file = os.path.join(model_path, "flax_model.msgpack")
        if not os.path.exists(flax_model_file):
            print(f"❌ No flax_model.msgpack found in model {i + 1}: {model_path}")
            return False

        # Check for config file
        config_file = os.path.join(model_path, "config.json")
        if not os.path.exists(config_file):
            print(f"❌ Required file missing in model {i + 1}: config.json")
            return False

        print(f"  ✓ JAX/Flax Model {i + 1} validated: {model_path}")

    # Check model compatibility (same architecture)
    try:
        first_config_path = os.path.join(model_paths[0], "config.json")
        with open(first_config_path) as f:
            import json

            first_config = json.load(f)

        for i, model_path in enumerate(model_paths[1:], 1):
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                current_config = json.load(f)

            # Check critical config parameters
            critical_params = ["model_type", "hidden_size", "num_hidden_layers", "vocab_size"]
            for param in critical_params:
                if (
                    param in first_config
                    and param in current_config
                    and first_config[param] != current_config[param]
                ):
                    print(
                        "❌ Config mismatch in model "
                        f"{i + 1}: {param} = {current_config[param]} vs {first_config[param]}"
                    )
                    return False

        print("  ✓ All JAX/Flax models have compatible configurations")

    except Exception as e:
        print(f"❌ Error validating JAX/Flax model configurations: {e}")
        return False

    print("✅ All JAX/Flax models validated for averaging")
    return True


def get_model_ensemble_paths(base_path: str, num_models: int) -> list[str]:
    """
    Generate paths for ensemble models based on a base path and number of models.

    Args:
        base_path: Base path for fine-tuned models
        num_models: Number of ensemble models

    Returns:
        List of paths for ensemble models
    """
    if num_models == 1:
        return [base_path]

    ensemble_paths = []
    for i in range(num_models):
        ensemble_path = f"{base_path}_ensemble_{i + 1}"
        ensemble_paths.append(ensemble_path)

    return ensemble_paths


def cleanup_ensemble_models(ensemble_paths: list[str], keep_averaged: bool = True):
    """
    Clean up ensemble model directories to save disk space.

    Args:
        ensemble_paths: List of ensemble model paths to clean up
        keep_averaged: Whether to keep the averaged model directory
    """
    from . import config

    print(f"🧹 Cleaning up {len(ensemble_paths)} ensemble model directories...")

    for i, path in enumerate(ensemble_paths):
        preserve_condition = config.PRESERVE_FINETUNED_MODEL and (
            (
                hasattr(config, "RELOAD_PATH")
                and os.path.abspath(path) == os.path.abspath(config.RELOAD_PATH)
            )
            or "model_fine_tuned" in os.path.basename(path)
        )
        if preserve_condition:
            print(f"  🔒 Preserving fine-tuned model (PRESERVE_FINETUNED_MODEL=True): {path}")

        should_preserve = preserve_condition

        if not should_preserve and os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✓ Cleaned up ensemble model {i + 1}: {path}")
            except Exception as e:
                print(f"  ❌ Failed to clean up {path}: {e}")
        elif not should_preserve:
            print(f"  ⚠️  Ensemble model {i + 1} not found: {path}")

    print("✅ Ensemble model cleanup completed")
