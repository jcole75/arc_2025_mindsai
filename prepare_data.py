from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def _bootstrap_target(clean_dir: Path, target_path: Path, pattern: str) -> None:
    if target_path.exists():
        return
    candidates = sorted(clean_dir.glob(pattern))
    if not candidates:
        return
    _copy_path(candidates[0], target_path)

def prepare(settings_path: Path) -> None:
    with settings_path.open() as fh:
        settings = json.load(fh)

    raw_dir = Path(settings["RAW_DATA_DIR"]).expanduser().resolve()
    clean_dir = Path(settings["CLEAN_DATA_DIR"]).expanduser().resolve()
    clean_dir.mkdir(parents=True, exist_ok=True)

    if raw_dir.exists():
        for entry in raw_dir.iterdir():
            entry_resolved = entry.resolve()
            # Avoid copying the clean directory into itself (which previously caused clean/clean/... chains)
            if entry_resolved == clean_dir or clean_dir in entry_resolved.parents:
                continue
            target = clean_dir / entry.name
            _copy_path(entry, target)
    else:
        print(f"RAW_DATA_DIR {raw_dir} does not exist; skipping copy.")

    train_clean = Path(settings["TRAIN_DATA_CLEAN_PATH"]).expanduser().resolve()
    test_clean = Path(settings["TEST_DATA_CLEAN_PATH"]).expanduser().resolve()
    _bootstrap_target(clean_dir, train_clean, "*train*.json*")
    _bootstrap_target(clean_dir, test_clean, "*test*.json*")
    sample_tasks = Path("data/sample_tasks.json").resolve()
    if sample_tasks.exists():
        if not train_clean.exists():
            _copy_path(sample_tasks, train_clean)
        if not test_clean.exists():
            _copy_path(sample_tasks, test_clean)

    for key in ("PRETRAIN_DIR", "MODEL_DIR", "CHECKPOINT_DIR", "SUBMISSION_DIR"):
        path_str = settings.get(key, "")
        if not path_str:
            continue
        path = Path(path_str).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)

    print("Data preparation complete.")
    print(f"Clean data directory: {clean_dir}")
    print(f"Train file: {train_clean}")
    print(f"Test file: {test_clean}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and clean ARC datasets")
    parser.add_argument("--settings", default="SETTINGS.json", help="Path to SETTINGS.json")
    args = parser.parse_args()
    prepare(Path(args.settings))


if __name__ == "__main__":
    main()
