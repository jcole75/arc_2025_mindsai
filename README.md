# MindsAI ARC 2025

MindsAI ARC Prize 2025 solution:
- **Pretraining** via HuggingFace `mindware/arc-agi-mega`
- **Test-Time Training (TTT)** dataset generation
- **Augmented Inference (AIRV)**

## Repository Layout

```
arc_2025_mindsai/
├── data/                     # raw + cleaned tasks (create as needed)
├── notebooks/                # demo notebooks (optional)
├── src/                      # solution code
├── tpu/                      # tpu training scripts, hydra config
├── prepare_data.py           # entry point: data cleaning/copy
├── train.py                  # entry point: pretraining + fine-tuning
├── predict.py                # entry point: inference
├── entry_points.md           # summary of the three commands
├── Makefile                  # convenience wrappers
└── SETTINGS.example.json     # template for environment configuration
```

## Quickstart

1. **Create a SETTINGS file**

   ```bash
   cp SETTINGS.example.json SETTINGS.json
   ```

   - `SETTINGS.json` (default) runs evaluation-only TTFT/AIRV on `arc-agi-2_evaluation_challenges.json` using the `codet5_660m,codet5_660m_scr` configs (no ARC Mega pretraining). Set to Kaggle solution settings by default.
   - `SETTINGS.pretraining.example.json` is pre-wired for full ARC Mega pretraining + TTFT/AIRV (`MODEL_CONFIG_MODULES`: `codet5_660m_arcmega`). Copy it over when you want to stream `mindware/arc-agi-mega`.
   - `SETTINGS.smoke_test.json` points at `data/sample_tasks.json` and uses `codet5_small` so you can validate the pipeline in a couple of minutes.
   Adjust paths as needed (e.g., change `TRAIN_DATA_CLEAN_PATH` / `TEST_DATA_CLEAN_PATH` to another ARC split).

2. **Install dependencies**

   ```bash
   python3 -m venv .venv  # install python3-venv or run: python3 -m virtualenv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > **Note:** On Debian/Ubuntu, `python -m venv` may fail without `ensurepip`. Install `python3-venv`
   > (`sudo apt install python3-venv`) or run `python3 -m virtualenv .venv` (after `pip install --break-system-packages virtualenv`)
   > before installing requirements.

3. **Run the stages** (or use `make prepare-data`, `make train`, `make predict`)

   ```bash
   python3 prepare_data.py --settings SETTINGS.json
   python3 train.py --settings SETTINGS.json
   python3 predict.py --settings SETTINGS.json
   ```

Each script accepts `--extra ...` to forward experimental flags to `src.main`.

> ✅ `python3 prepare_data.py --settings SETTINGS.json` copies the requested ARC split (e.g., `arc-agi-2_evaluation_challenges.json`)
> into `data/clean/…`. Training and inference then operate on that file. `codet5_small` finishes in minutes; `codet5_660m_arcmega`
> streams the `mindware/arc-agi-mega` dataset from HuggingFace before training and should be run on a GPU machine with ample space.
> `predict.py` reruns per-task TTT followed by AIRV inference using the latest checkpoint in `MODEL_DIR`, matching the Kaggle evaluation flow.

### Full pipeline: ARC Mega pretraining → ARC-AGI-2 evaluation

1. Set `"MODEL_CONFIG_MODULES": "salesforce_codet5_large_arcmega"` in `SETTINGS.json`.
2. If needed, authenticate with HuggingFace (`huggingface-cli login`) so the trainer can stream `mindware/arc-agi-mega`.
3. Stage the evaluation split:

   ```bash
   python3 prepare_data.py --settings SETTINGS.json
   ```

4. Launch pretraining + TTT/AIRV fine-tuning:

   ```bash
   python3 train.py --settings SETTINGS.json
   ```

5. Generate predictions/submission on the cleaned `arc-agi-2` file:

   ```bash
   python3 predict.py --settings SETTINGS.json
   ```

The resulting `submission.json` (and scoring visuals under `scoring_visualizations/`) correspond to `arc-agi-2_evaluation_challenges.json`.

## Configuration Notes

- The pipeline ships with a small, fast model config (`codet5_small`). Edit `src/modules/config/model_configs/` (or override `MODEL_CONFIG_MODULES`) to point at larger variants if needed.
- Pretraining uses the HuggingFace ARC Mega dataset via the `hf_pretraining` block. Populate `PRETRAIN_DIR` with CSV shards or point at your own source.
- Default data fallback is `data/sample_tasks.json`, matching the smoke-test fixture invoked when `RAW_DATA_DIR` / `CLEAN_DATA_DIR` are empty; point `TRAIN_DATA_CLEAN_PATH` at any of the bundled ARC JSON files (e.g., `data/arc-agi-2_evaluation_challenges.json`) for real runs.
- `entry_points.md` documents the three commands expected by the ARC handover guide and mirrors the `Makefile` shortcuts.
- Settings summary:
  - `SETTINGS.json`: evaluation-only TTFT/AIRV on `arc-agi-2` using `codet5_660m,codet5_660m_scr` per Kaggle solution settings.
  - `SETTINGS.pretraining.example.json`: ARC Mega pretraining + TTFT/AIRV via `salesforce_codet5_large_arcmega`.
  - `SETTINGS.smoke_test.json`: fast smoke test on `sample_tasks.json` with `codet5_small`.

### Dataset Formats

- **ARC tasks (`train.json`, `arc-agi-2_evaluation_challenges.json`, etc.)**  
  Each file is a JSON object where keys are task IDs (e.g., `"00d62c1b"`). Values have:
  - `"train"`: list of examples, each with `"input"` and `"output"` grids (2‑D lists of ints 0‑9).
  - `"test"`: list with `"input"` grids (outputs withheld for evaluation).
  `prepare_data.py` simply copies these files from `RAW_DATA_DIR` into `CLEAN_DATA_DIR` so downstream stages work off the cleaned copy.

- **Pretraining datasets**  
  When `training.pretrain_file` (or `hf_pretraining`) is enabled, the loader expects rows with at least two columns:
  - `prompt`: serialized ARC prompt text (e.g., `solve: train input1 … output1 … test tinput1 …`). all output boards follow: `{total_chars} {height} {width} {symbols} …`. symbols are the unique symbols in the order they are encountered in the grid.
  - `correct_answer`: serialized target string (`{total_chars} {height} {width} {symbols} …`).
  The hosted dataset `mindware/arc-agi-mega` already exposes these columns, so setting `"MODEL_CONFIG_MODULES": "codet5_660m_arcmega"` automatically streams prompts from HuggingFace before TTFT/AIRV training.
  
### Extra Datasets

- Extra datasets are used by the TPU trainer and not included here due to size.  They were uploaded to a kaggle dataset.
