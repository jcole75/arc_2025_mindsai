from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from . import env as env_mod

try:
    from ..git_utils import create_changes_archive
except ImportError:
    import sys
    from pathlib import Path

    _THIS_FILE = Path(__file__).resolve()
    _MODULE_ROOT = _THIS_FILE.parents[1]
    if str(_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODULE_ROOT))
    from git_utils import create_changes_archive  # type: ignore  # noqa: F401


def _run_logs_dir() -> str:
    def _contains_unresolved_template(value: str) -> bool:
        """Check if a string contains unresolved template placeholders."""
        if not value:
            return False
        return ("{{" in value and "}}" in value) or ("${" in value and "}" in value)

    base = os.environ.get("ARC_LOG_DIR") or os.environ.get("RUN_LOGS_DIR")
    if base and not _contains_unresolved_template(base):
        d = Path(base) / "run_logs"
    else:
        # Fallback: try to find the actual task directory in /mnt/logs
        task_dir = _find_real_task_dir()
        d = task_dir / "run_logs" if task_dir else Path("run_logs")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _find_real_task_dir() -> Path | None:
    """Find the real task directory when environment variables fail to resolve."""
    logs_base = Path("/mnt/logs")
    if not logs_base.exists():
        return None

    # Look for sky-managed directories (most recent first)
    sky_dirs = []
    for item in logs_base.iterdir():
        if item.is_dir() and item.name.startswith("sky-managed-"):
            sky_dirs.append(item)

    if sky_dirs:
        # Sort by modification time, newest first
        sky_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return sky_dirs[0]

    # Fallback to SKYPILOT_TASK_ID if it's a valid directory name
    task_id = os.environ.get("SKYPILOT_TASK_ID")
    if task_id and not ("{{" in task_id or "${" in task_id):
        candidate = logs_base / task_id
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def get_run_logs_dir() -> str:
    return _run_logs_dir()


def get_all_run_logs(max_file_size_mb: int = 50) -> list[dict[str, Any]]:
    logs_dir = _run_logs_dir()
    run_logs: list[dict[str, Any]] = []
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    for p in sorted(Path(logs_dir).glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            if p.stat().st_size > max_file_size_bytes:
                continue
            with p.open("r") as f:
                run_logs.append(json.load(f))
        except Exception:
            continue
    return run_logs


def _new_run_id() -> str:
    import uuid

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{str(uuid.uuid4())[:8]}"


class RunLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or _new_run_id()
        self.logs_dir = _run_logs_dir()
        self.file = Path(self.logs_dir) / f"{self.run_id}.json"
        self.run_log_file = str(self.file)
        self.artifacts_dir = Path(self.logs_dir) / self.run_id
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.data: dict[str, Any] = {
            "run_id": self.run_id,
            "run_title": None,  # set by caller via config
            "start_time": self.start_time,
            "start_time_formatted": datetime.fromtimestamp(self.start_time).isoformat(),
            "environment": "kaggle",
            "test_mode": env_mod.TEST_MODE,
            "config": {
                "test_mode_num_tasks": env_mod.TEST_MODE_NUM_TASKS,
                "test_mode_ttt_items": env_mod.TEST_MODE_TTT_ITEMS,
                "test_mode_inference_items": env_mod.TEST_MODE_INFERENCE_ITEMS,
                "global_ttt_enabled": env_mod.GLOBAL_TTT_ENABLED,
                "expand_factor": env_mod.EXPAND_FACTOR,
                "prune_factor": env_mod.PRUNE_FACTOR,
                "total_time": env_mod.TOTAL_TIME,
                "buffer_time": env_mod.BUFFER_TIME,
                "debug_mode": env_mod.DEBUG_MODE,
                "verbose_logging": env_mod.VERBOSE_LOGGING,
            },
            "models": [],
            "task_stats": {},
            "overall_stats": {},
            "scoring_results": {},
            "session": {},
            "repo_state": {},
            "artifacts": {},
            "end_time": None,
            "end_time_formatted": None,
            "total_duration": None,
            "status": "running",
        }
        self._save()

    def _save(self):
        try:
            with self.file.open("w") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save run log: {e}")

    def _normalize_path(self, value: Any) -> str:
        try:
            return str(Path(value))
        except Exception:
            return str(value)

    def _update_artifacts(self, updates: dict[str, Any]) -> bool:
        if not updates:
            return False
        store = self.data.setdefault("artifacts", {})
        changed = False
        for key, value in updates.items():
            if value is None:
                continue
            normalized = self._normalize_path(value)
            if store.get(key) != normalized:
                store[key] = normalized
                changed = True
        return changed

    def register_session_artifacts(self, artifacts: dict[str, Any]) -> None:
        if not artifacts:
            return
        session_store = self.data.setdefault("session", {})
        artifact_updates: dict[str, Any] = {}
        changed = False
        for key, value in artifacts.items():
            if value is None:
                continue
            normalized = self._normalize_path(value)
            if session_store.get(key) != normalized:
                session_store[key] = normalized
                changed = True
            artifact_updates[key] = normalized
        if self._update_artifacts(artifact_updates) or changed:
            self._save()

    def record_repo_state(self, repo_state: dict[str, Any]) -> None:
        if not repo_state:
            return
        snapshot_dir = self.artifacts_dir / "git_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        repo_store = self.data.setdefault("repo_state", {})
        artifact_updates: dict[str, Any] = {}
        changed = False
        repo_store["captured_at"] = time.time()

        for key in ("commit_sha", "short_sha", "git_branch", "git_dirty"):
            value = repo_state.get(key)
            if value is None:
                continue
            if repo_store.get(key) != value:
                repo_store[key] = value
                changed = True

        for list_key in ("modified_files", "untracked_files", "staged_files"):
            value = repo_state.get(list_key)
            if value is None:
                continue
            if repo_store.get(list_key) != value:
                repo_store[list_key] = list(value)
                changed = True

        status_text = repo_state.get("status_summary") or repo_state.get("status_porcelain")
        if status_text:
            status_path = snapshot_dir / "git_status.txt"
            status_path.write_text(status_text)
            repo_store["status_path"] = str(status_path)
            repo_store["status_preview"] = status_text.strip()
            artifact_updates["git_status"] = status_path

        diff_head = repo_state.get("diff_head")
        if diff_head:
            worktree_patch = snapshot_dir / "diff_worktree.patch"
            worktree_patch.write_text(diff_head)
            repo_store["diff_head_path"] = str(worktree_patch)
            artifact_updates["git_diff_worktree"] = worktree_patch

        diff_cached = repo_state.get("diff_cached")
        if diff_cached:
            index_patch = snapshot_dir / "diff_index.patch"
            index_patch.write_text(diff_cached)
            repo_store["diff_cached_path"] = str(index_patch)
            artifact_updates["git_diff_index"] = index_patch

        modified_files = repo_state.get("modified_files") or []
        untracked_files = repo_state.get("untracked_files") or []
        if modified_files or untracked_files:
            archive_path = snapshot_dir / "uncommitted_changes.tar.gz"
            if create_changes_archive(archive_path, list(dict.fromkeys(modified_files)), list(dict.fromkeys(untracked_files))):
                repo_store["uncommitted_archive_path"] = str(archive_path)
                artifact_updates["git_changes_archive"] = archive_path
        else:
            archive_path = snapshot_dir / "uncommitted_changes.tar.gz"
            if archive_path.exists():
                archive_path.unlink()

        instructions_lines = [
            f"Run ID: {self.run_id}",
            f"Commit: {repo_store.get('commit_sha', 'unknown')}",
            f"Branch: {repo_store.get('git_branch', 'unknown')}",
            "",
            "Restore working tree snapshot:",
            f"  git checkout {repo_store.get('commit_sha', '<commit>')}",
        ]
        archive_ref = repo_store.get("uncommitted_archive_path")
        if archive_ref:
            archive_name = Path(archive_ref).name
            instructions_lines.append(f"  tar -xzf {archive_name} -C <repo_root>")
        db_patch = repo_store.get("diff_head_path")
        if db_patch:
            patch_name = Path(db_patch).name
            instructions_lines.append(f"  git apply {patch_name}")
        instructions_lines.extend(
            [
                "",
                "Notes:",
                "  - Archive and patch files are stored alongside this log under run_logs.",
                "  - Paths recorded reflect the repository root at runtime.",
            ]
        )
        instructions_path = snapshot_dir / "RESTORE_INSTRUCTIONS.txt"
        instructions_path.write_text("\n".join(instructions_lines))
        repo_store["instructions_path"] = str(instructions_path)
        artifact_updates["git_restore_instructions"] = instructions_path

        if self._update_artifacts(artifact_updates) or changed:
            self._save()

    def log_model_start(self, model_path: str, model_settings: dict[str, Any]) -> int:
        self.data["models"].append(
            {
                "model_path": model_path,
                "model_basename": Path(model_path).name,
                "start_time": time.time(),
                "settings": model_settings,
                "runs": [],
                "status": "running",
            }
        )
        self._save()
        return len(self.data["models"]) - 1

    def log_model_run(self, model_idx: int, run_num: int, run_stats: dict[str, Any]):
        if model_idx < len(self.data["models"]):
            entry = {
                "run_number": run_num,
                **{
                    k: run_stats.get(k)
                    for k in (
                        "start_time",
                        "duration",
                        "ttt_items",
                        "inference_items",
                        "active_tasks_before",
                        "active_tasks_after",
                        "raw_outputs",
                        "parsed_outputs",
                        "valid_outputs",
                    )
                    if run_stats.get(k) is not None
                },
            }

            # Persist scoring metrics when available so downstream reports capture progress accurately
            for extra_key in ("score", "solved_tasks", "total_tasks", "final_score", "final_solved"):
                if run_stats.get(extra_key) is not None:
                    entry[extra_key] = run_stats.get(extra_key)

            self.data["models"][model_idx]["runs"].append(entry)
            self._save()

    def log_model_end(self, model_idx: int):
        if model_idx < len(self.data["models"]):
            self.data["models"][model_idx]["end_time"] = time.time()
            self.data["models"][model_idx]["status"] = "completed"
            self._save()

    def log_task_stats(self, task_stats: dict[str, Any]):
        self.data["task_stats"] = task_stats
        self._save()

    def log_overall_stats(self, overall_stats: dict[str, Any]):
        self.data["overall_stats"] = overall_stats
        self._save()

    def log_scoring_results(self, scoring_results: dict[str, Any]):
        self.data["scoring_results"] = scoring_results
        self._save()

    def finalize_run(self, status: str = "completed"):
        self.data["end_time"] = time.time()
        self.data["end_time_formatted"] = datetime.fromtimestamp(self.data["end_time"]).isoformat()
        self.data["total_duration"] = self.data["end_time"] - self.data["start_time"]
        self.data["status"] = status
        self._save()

    # Backward-compat shim for older callsites expecting `run_data`
    @property
    def run_data(self) -> dict[str, Any]:
        return self.data


current_run_logger: RunLogger | None = None
