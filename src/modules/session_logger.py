"""
Session logger for ARC Prize 2025 solution
Tracks execution metrics, git state, and creates comprehensive session logs
"""

from contextlib import suppress
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

from .git_utils import create_changes_archive, get_git_commit_info, get_uncommitted_changes, is_git_repository


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


class _TeeStream:
    """Mirror writes to the original stream and a log file."""

    def __init__(self, primary_stream, log_file):
        self._primary = primary_stream
        self._log_file = log_file

    def write(self, data):
        text = data if isinstance(data, str) else str(data)
        try:
            self._primary.write(text)
        except Exception:
            pass
        try:
            self._log_file.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        for stream in (self._primary, self._log_file):
            with suppress(Exception):
                stream.flush()

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self._primary, "fileno", lambda: -1)()

    def __getattr__(self, name):
        return getattr(self._primary, name)


class SessionLogger:
    """
    Comprehensive session logger that tracks:
    - Git commit information and uncommitted changes
    - Execution metrics (scores, run counts, models used)
    - Session logs and timing information
    """

    def __init__(self, session_name: str | None = None):
        self.session_name = session_name or f"arc_session_{int(time.time())}"
        self.start_time = time.time()
        self.session_log_path: Path | None = None
        self.session_json_path: Path | None = None
        self.console_log_path: Path | None = None
        self._console_capture_active = False
        self._console_file_handle = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._stdout_tee: _TeeStream | None = None
        self._stderr_tee: _TeeStream | None = None
        self.session_data = {
            "session_name": self.session_name,
            "start_time": datetime.now().isoformat(),
            "git_info": {},
            "models_used": [],
            "run_count": 0,
            "scores": [],
            "top_score": 0.0,
            "logs": [],
            "timing": {},
            "environment": {},
            "empty_output_samples": [],
            "empty_output_stats": {
                "total": 0,
                "by_task": {},
                "by_model": {},
                "mixed_precision": {
                    "with_mp": 0,
                    "without_mp": 0,
                },
            },
        }

        # Create logs directory (optionally under ARC_LOG_DIR)
        base_dir = os.environ.get("ARC_LOG_DIR")

        def _contains_unresolved_template(value: str) -> bool:
            """Check if a string contains unresolved template placeholders."""
            if not value:
                return False
            return ("{{" in value and "}}" in value) or ("${" in value and "}" in value)

        if base_dir and not _contains_unresolved_template(base_dir):
            self.logs_dir = Path(base_dir) / "session_logs"
        else:
            # Fallback: try to find the actual task directory in /mnt/logs
            task_dir = _find_real_task_dir()
            if task_dir:
                self.logs_dir = task_dir / "session_logs"
            else:
                self.logs_dir = Path("session_logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_path = self.logs_dir / f"{self.session_name}.log"
        self.session_json_path = self.logs_dir / f"{self.session_name}.json"

        # Set up session-specific logger
        self.logger = self._setup_session_logger()

        # Initialize session
        self._initialize_session()

    def start_console_capture(self, capture_stdout: bool = True, capture_stderr: bool = True) -> Path | None:
        """Mirror stdout/stderr into a session-scoped console log."""
        if self._console_capture_active:
            return self.console_log_path

        console_path = self.logs_dir / f"{self.session_name}_console.log"
        self.console_log_path = console_path
        try:
            console_file = open(console_path, "a", buffering=1)
        except Exception as exc:
            self.logger.error(f"Failed to open console log file: {exc}")
            return None

        self._console_file_handle = console_file
        if capture_stdout:
            self._orig_stdout = sys.stdout
            self._stdout_tee = _TeeStream(sys.stdout, console_file)
            sys.stdout = self._stdout_tee
        if capture_stderr:
            self._orig_stderr = sys.stderr
            self._stderr_tee = _TeeStream(sys.stderr, console_file)
            sys.stderr = self._stderr_tee

        self._console_capture_active = True
        artifacts = self.session_data.setdefault("artifacts", {})
        artifacts["console_log"] = str(console_path)
        artifacts["session_log"] = str(self.session_log_path) if self.session_log_path else None
        artifacts["session_json"] = str(self.session_json_path) if self.session_json_path else None
        self._save_session_data()
        return console_path

    def stop_console_capture(self) -> None:
        """Restore stdout/stderr if console capture is active."""
        if not self._console_capture_active:
            return

        for tee, original in ((self._stdout_tee, self._orig_stdout), (self._stderr_tee, self._orig_stderr)):
            if original is not None:
                if tee is self._stdout_tee:
                    sys.stdout = original
                else:
                    sys.stderr = original
        self._orig_stdout = None
        self._orig_stderr = None

        for tee in (self._stdout_tee, self._stderr_tee):
            if tee is not None:
                with suppress(Exception):
                    tee.flush()
        self._stdout_tee = None
        self._stderr_tee = None

        if self._console_file_handle is not None:
            with suppress(Exception):
                self._console_file_handle.flush()
                self._console_file_handle.close()
        self._console_file_handle = None
        self._console_capture_active = False

    def get_artifact_manifest(self) -> dict[str, Any]:
        """Return paths to session artifacts for downstream logging."""
        artifacts = self.session_data.setdefault("artifacts", {})
        if self.session_log_path:
            artifacts.setdefault("session_log", str(self.session_log_path))
        if self.session_json_path:
            artifacts.setdefault("session_json", str(self.session_json_path))
        if self.console_log_path:
            artifacts.setdefault("console_log", str(self.console_log_path))
        return dict(artifacts)

    def _setup_session_logger(self) -> logging.Logger:
        """Set up a dedicated logger for this session."""
        logger = logging.getLogger(f"session_{self.session_name}")
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        logger.handlers.clear()

        # Create session log file
        log_path = self.session_log_path or (self.logs_dir / f"{self.session_name}.log")
        self.session_log_path = log_path
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def _initialize_session(self):
        """Initialize session with git and environment information."""
        self.logger.info(f"Starting session: {self.session_name}")

        # Capture git information
        if is_git_repository():
            git_info = get_git_commit_info()
            self.session_data["git_info"] = git_info
            self.logger.info(f"Git commit: {git_info['short_hash']} on branch {git_info['branch']}")
            self.logger.info(f"Commit message: {git_info['message']}")

            # Check for uncommitted changes
            modified_files, untracked_files = get_uncommitted_changes()
            self.session_data["git_info"]["modified_files"] = modified_files
            self.session_data["git_info"]["untracked_files"] = untracked_files

            if modified_files or untracked_files:
                self.logger.warning(
                    f"Uncommitted changes detected: {len(modified_files)} modified, {len(untracked_files)} untracked"
                )
        else:
            self.logger.warning("Not in a git repository")

        # Capture environment information
        self.session_data["environment"] = {
            "python_path": os.sys.executable,
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                key: value for key, value in os.environ.items() if key.startswith(("CUDA", "HF_", "KAGGLE", "TPU"))
            },
        }

        # Log environment info
        self.logger.info(f"Working directory: {self.session_data['environment']['working_directory']}")

        artifacts = self.session_data.setdefault("artifacts", {})
        if self.session_log_path:
            artifacts["session_log"] = str(self.session_log_path)
        if self.session_json_path:
            artifacts["session_json"] = str(self.session_json_path)

        # Save initial session data
        self._save_session_data()

    def log_model_usage(self, model_name: str, model_config: dict[str, Any]):
        """Log that a model is being used in this session."""
        model_info = {"name": model_name, "config": model_config, "timestamp": datetime.now().isoformat()}

        self.session_data["models_used"].append(model_info)
        self.logger.info(f"Using model: {model_name}")
        self._save_session_data()

    def log_run_completion(self, score: float, additional_metrics: dict | None = None):
        """Log completion of a run with score and metrics."""
        self.session_data["run_count"] += 1
        self.session_data["scores"].append(score)
        self.session_data["top_score"] = max(self.session_data["top_score"], score)

        run_info = {
            "run_number": self.session_data["run_count"],
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metrics": additional_metrics or {},
        }

        self.session_data["logs"].append(run_info)
        self.logger.info(
            f"Run {self.session_data['run_count']} completed - Score: {score:.4f} "
            f"(Best: {self.session_data['top_score']:.4f})"
        )

        self._save_session_data()

    def log_custom_event(self, event_type: str, message: str, data: dict | None = None):
        """Log a custom event with optional data."""
        event_info = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        }

        self.session_data["logs"].append(event_info)
        self.logger.info(f"{event_type}: {message}")

        if data:
            self.logger.debug(f"Event data: {json.dumps(data, indent=2)}")

        self._save_session_data()

    def log_empty_output_sample(
        self,
        *,
        task_id: str,
        augmentation_index: int,
        text_index: int,
        model_name: str,
        used_mixed_precision: bool,
        raw_text: str,
        prompt_snippet: str,
        context: str = "",
    ):
        """Record an inference attempt that produced an empty output."""
        samples = self.session_data.setdefault("empty_output_samples", [])
        stats = self.session_data.setdefault(
            "empty_output_stats",
            {
                "total": 0,
                "by_task": {},
                "by_model": {},
                "mixed_precision": {"with_mp": 0, "without_mp": 0},
            },
        )

        sample = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "augmentation_index": augmentation_index,
            "text_index": text_index,
            "model_name": model_name,
            "used_mixed_precision": used_mixed_precision,
            "raw_text_preview": raw_text[:500],
            "prompt_snippet": prompt_snippet[:500],
            "context": context,
        }
        samples.append(sample)

        stats["total"] = stats.get("total", 0) + 1
        stats.setdefault("by_task", {})[task_id] = stats["by_task"].get(task_id, 0) + 1
        stats.setdefault("by_model", {})[model_name] = stats["by_model"].get(model_name, 0) + 1
        mp_key = "with_mp" if used_mixed_precision else "without_mp"
        stats.setdefault("mixed_precision", {}).setdefault(mp_key, 0)
        stats["mixed_precision"][mp_key] += 1

        self.logger.warning(
            "Empty model output detected for %s (aug=%s text=%s, model=%s, mp=%s)",
            task_id,
            augmentation_index,
            text_index,
            model_name,
            used_mixed_precision,
        )
        if context:
            self.logger.info("Empty output context: %s", context)

        self._save_session_data()

    def _save_session_data(self):
        """Save current session data to JSON file."""
        try:
            if self.session_json_path is None:
                self.session_json_path = self.logs_dir / f"{self.session_name}.json"
            with open(self.session_json_path, "w") as json_file:
                json.dump(self.session_data, json_file, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save session data: {exc}")

    def finalize_session(self) -> str | None:
        """
        Finalize the session and create a comprehensive zip file.

        Returns:
            Path to the created zip file, or None if creation failed
        """
        end_time = time.time()
        session_duration = end_time - self.start_time

        self.session_data["end_time"] = datetime.now().isoformat()
        self.session_data["duration_seconds"] = session_duration
        self.session_data["timing"]["total_duration"] = session_duration

        self.logger.info(f"Session completed - Duration: {session_duration:.2f}s")
        self.logger.info(
            f"Final stats - Runs: {self.session_data['run_count']}, Top Score: {self.session_data['top_score']:.4f}"
        )

        # Save final session data
        self._save_session_data()

        # Create comprehensive zip file
        return self._create_session_zip()

    def _create_session_zip(self) -> str | None:
        """Create a comprehensive zip file with all session information."""
        try:
            if self._console_file_handle is not None:
                with suppress(Exception):
                    self._console_file_handle.flush()
            # Generate zip filename with key metrics
            models_str = "_".join(
                [m["name"].replace("/", "_") for m in self.session_data["models_used"][:3]]
            )  # First 3 models
            if len(models_str) > 50:  # Truncate if too long
                models_str = models_str[:47] + "..."

            git_info = self.session_data["git_info"]
            commit_str = git_info.get("short_hash", "unknown")

            zip_filename = (
                f"{self.session_name}_"
                f"score{self.session_data['top_score']:.3f}_"
                f"runs{self.session_data['run_count']}_"
                f"{commit_str}_"
                f"{models_str}.zip"
            )

            # Clean filename (remove invalid characters)
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                zip_filename = zip_filename.replace(char, "_")

            zip_path = self.logs_dir / zip_filename

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add session log file
                if self.session_log_path and self.session_log_path.exists():
                    zipf.write(self.session_log_path, f"logs/{self.session_log_path.name}")

                # Add console capture if available
                if self.console_log_path and self.console_log_path.exists():
                    zipf.write(self.console_log_path, f"logs/{self.console_log_path.name}")

                # Add session data JSON
                if self.session_json_path and self.session_json_path.exists():
                    zipf.write(self.session_json_path, "session_data.json")

                # Add git commit info as separate file
                git_info_content = json.dumps(self.session_data["git_info"], indent=2)
                zipf.writestr("git_info.json", git_info_content)

                # Add uncommitted changes archive if available
                if is_git_repository():
                    modified_files = self.session_data["git_info"].get("modified_files", [])
                    untracked_files = self.session_data["git_info"].get("untracked_files", [])

                    if modified_files or untracked_files:
                        changes_archive_path = self.logs_dir / f"{self.session_name}_changes.tar.gz"
                        if create_changes_archive(changes_archive_path, modified_files, untracked_files):
                            zipf.write(changes_archive_path, "uncommitted_changes.tar.gz")
                            # Clean up temporary archive
                            changes_archive_path.unlink()

                # Add summary file
                summary_content = self._generate_summary()
                zipf.writestr("session_summary.txt", summary_content)

            self.logger.info(f"Session archive created: {zip_path}")
            artifacts = self.session_data.setdefault("artifacts", {})
            artifacts["session_archive"] = str(zip_path)
            self._save_session_data()
            return str(zip_path)

        except Exception as e:
            self.logger.error(f"Failed to create session zip: {e}")
            return None

    def _generate_summary(self) -> str:
        """Generate a human-readable summary of the session."""
        lines = [
            "ARC Prize 2025 Session Summary",
            "=" * 50,
            f"Session Name: {self.session_name}",
            f"Start Time: {self.session_data['start_time']}",
            f"End Time: {self.session_data['end_time']}",
            f"Duration: {self.session_data.get('duration_seconds', 0):.2f} seconds",
            "",
            "Git Information:",
            f"  Commit: {self.session_data['git_info'].get('hash', 'unknown')}",
            f"  Branch: {self.session_data['git_info'].get('branch', 'unknown')}",
            f"  Message: {self.session_data['git_info'].get('message', 'unknown')}",
            "",
            "Execution Metrics:",
            f"  Total Runs: {self.session_data['run_count']}",
            f"  Top Score: {self.session_data['top_score']:.4f}",
            f"  Average Score: {sum(self.session_data['scores']) / max(1, len(self.session_data['scores'])):.4f}",
            "",
            "Models Used:",
        ]

        for i, model in enumerate(self.session_data["models_used"], 1):
            lines.append(f"  {i}. {model['name']}")

        if self.session_data["git_info"].get("modified_files") or self.session_data["git_info"].get("untracked_files"):
            lines.extend(
                [
                    "",
                    "Uncommitted Changes:",
                    f"  Modified Files: {len(self.session_data['git_info'].get('modified_files', []))}",
                    f"  Untracked Files: {len(self.session_data['git_info'].get('untracked_files', []))}",
                ]
            )

        lines.extend(
            [
                "",
                "Score History:",
            ]
        )

        for i, score in enumerate(self.session_data["scores"], 1):
            lines.append(f"  Run {i}: {score:.4f}")

        return "\n".join(lines)


# Global session logger instance
_session_logger: SessionLogger | None = None


def get_session_logger() -> SessionLogger | None:
    """Get the global session logger instance."""
    return _session_logger


def initialize_session_logger(session_name: str | None = None) -> SessionLogger:
    """Initialize the global session logger."""
    global _session_logger
    _session_logger = SessionLogger(session_name)
    return _session_logger


def finalize_session_logger() -> str | None:
    """Finalize the global session logger and return zip path."""
    global _session_logger
    if _session_logger:
        zip_path = _session_logger.finalize_session()
        _session_logger = None
        return zip_path
    return None
