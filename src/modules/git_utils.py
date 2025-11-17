"""
Git utilities for session tracking and change management
"""

import logging
import os
from pathlib import Path
import subprocess


logger = logging.getLogger(__name__)


def get_git_commit_info() -> dict[str, str]:
    """
    Get current git commit information.

    Returns:
        Dict with commit hash, branch, and message
    """
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()

        # Get branch name
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            branch = "unknown"

        # Get commit message
        try:
            commit_message = subprocess.check_output(
                ["git", "log", "-1", "--pretty=format:%s"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            commit_message = "unknown"

        return {"hash": commit_hash, "short_hash": commit_hash[:8], "branch": branch, "message": commit_message}

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to get git commit info - not in a git repository or git not available")
        return {"hash": "unknown", "short_hash": "unknown", "branch": "unknown", "message": "unknown"}


def get_uncommitted_changes() -> tuple[list[str], list[str]]:
    """
    Get lists of modified and untracked files.

    Returns:
        Tuple of (modified_files, untracked_files)
    """
    try:
        # Get modified files (staged and unstaged)
        modified_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True
        )
        modified_files = [f.strip() for f in modified_result.stdout.split("\n") if f.strip()]

        # Get untracked files
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, check=True
        )
        untracked_files = [f.strip() for f in untracked_result.stdout.split("\n") if f.strip()]

        return modified_files, untracked_files

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to get uncommitted changes - not in a git repository or git not available")
        return [], []


def create_changes_archive(output_path: Path, modified_files: list[str], untracked_files: list[str]) -> bool:
    """
    Create a tar archive of uncommitted changes.

    Args:
        output_path: Path where to save the archive
        modified_files: List of modified file paths
        untracked_files: List of untracked file paths

    Returns:
        True if archive created successfully, False otherwise
    """
    if not modified_files and not untracked_files:
        logger.info("No uncommitted changes to archive")
        return False

    try:
        import tarfile

        # Create archive
        with tarfile.open(output_path, "w:gz") as tar:
            all_files = modified_files + untracked_files

            for file_path in all_files:
                if os.path.exists(file_path):
                    try:
                        tar.add(file_path, arcname=file_path)
                        logger.debug(f"Added {file_path} to archive")
                    except Exception as e:
                        logger.warning(f"Failed to add {file_path} to archive: {e}")
                else:
                    logger.warning(f"File {file_path} not found, skipping")

        logger.info(f"Created changes archive: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create changes archive: {e}")
        return False


def is_git_repository() -> bool:
    """
    Check if the current directory is in a git repository.

    Returns:
        True if in a git repository, False otherwise
    """
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
