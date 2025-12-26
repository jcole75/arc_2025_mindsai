# Entry Points

Use these commands (wrapped in the Makefile as `make prepare-data`, `make train`, `make predict`) to run the full pipeline on the provided ARC datasets:

1. **Data preparation**
   ```bash
   python3 prepare_data.py --settings SETTINGS.json
   ```
   - Copies raw ARC JSON files from `RAW_DATA_DIR` (e.g., `data/arc-agi-2_evaluation_challenges.json`) into `CLEAN_DATA_DIR`.
   - Populates `TRAIN_DATA_CLEAN_PATH` / `TEST_DATA_CLEAN_PATH` so subsequent stages read from the cleaned copy.

2. **Model training (TTT/AIRV + optional ARC Mega pretraining)**
   ```bash
   python3 train.py --settings SETTINGS.json
   ```
   - Uses the config referenced by `MODEL_CONFIG_MODULES`:
     - `codet5_660m` → evaluation-only TTT/AIRV on ARC-AGI-2 (no pretraining). Note to train from scratch, use: `salesforce_codet5_large` and `salesforce_codet5_arcmega`.
     - `codet5_660m_arcmega` → streams `mindware/arc-agi-mega` first, then runs TTT/AIRV.
   - Writes checkpoints to `MODEL_DIR` / `CHECKPOINT_DIR`.
   - Pass `--extra ...` if you need to override runtime flags (they are forwarded to `src.main`).

3. **Prediction / Submission creation (TTT + AIRV inference)**
   ```bash
   python3 predict.py --settings SETTINGS.json
   ```
   - By default, runs with the models and settings used with the Kaggle solution using `MODEL_CONFIG_MODULES` of `codet5_660m,codet5_660m_scr`. 
   - Runs the full TTT/AIRV pipeline: full test dataset TTT followed by AIRV inference on `TEST_DATA_CLEAN_PATH`.
   - Writes predictions + `submission.json` to `SUBMISSION_DIR`.
