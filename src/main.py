#!/usr/bin/env python3
"""Main solution script for ARC Prize 2025."""

import sys
import time

from .modules.config import SUBMISSION_PATH
from .modules.pipeline_orchestrator import (
    finalize_pipeline,
    initialize_environment,
    load_tasks,
    prepare_initial_state,
    prepare_runtime,
    process_models,
)
from .modules.submission_handler import make_submission


current_run_logger = None


def main() -> None:
    global current_run_logger
    runtime = initialize_environment(time.time())
    current_run_logger = runtime.run_logger

    tasks = load_tasks(runtime)
    if not tasks.all_task_data:
        print("Critical Error: No data loaded.", file=sys.stderr)
        make_submission({}, [], SUBMISSION_PATH)
        return

    state = prepare_initial_state(tasks)
    prepare_runtime(runtime)
    process_models(runtime, tasks, state)
    finalize_pipeline(runtime, tasks, state)


if __name__ == "__main__":
    main()
