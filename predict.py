from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from train import _apply_env, _load_settings  # reuse helpers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with MindsAI ARC model")
    parser.add_argument("--settings", default="SETTINGS.json", help="Path to SETTINGS.json")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args to pass to src.main")
    args = parser.parse_args()

    settings_path = Path(args.settings)
    settings = _load_settings(settings_path)
    _apply_env(settings, mode="predict")

    cmd = [sys.executable, "-m", "src.main"]
    if args.extra:
        cmd.extend(args.extra)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
