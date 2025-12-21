#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import signal
import logging
import logging.config
from pathlib import Path
from multiprocessing import Process

from vision_python.config.env_loader import EnvConfig
from vision_python.src.task_aruco_detection import task_aruco_detection

# Load environment configuration
EnvConfig()

# ---------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------

processes = []

# ---------------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------------


def signal_handler(signum, frame):
    """Signal handler for graceful shutdown (Ctrl+C)"""
    logger = logging.getLogger("main")
    logger.info("Stop signal received (Ctrl+C). Gracefully stopping processes...")

    # Terminate all processes
    for process in processes:
        if process and process.is_alive():
            logger.info(f"Stopping process {process.name}...")
            process.terminate()

    # Wait for all processes to terminate (with timeout)
    for process in processes:
        if process:
            process.join(timeout=3)
            if process.is_alive():
                logger.warning(f"Process {process.name} did not stop, forcing kill...")
                process.kill()
                process.join()

    logger.info("All processes have been stopped. Closing...")
    sys.exit(0)


def run_task(core_id, func, *args):
    """Executes a task on a specific core"""
    # Assign the process to a specific core
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass  # if not allowed, it continues without forced affinity

    # Execute the task
    func(*args)


if __name__ == "__main__":
    # Configure logging with absolute path
    repo_root = Path(__file__).resolve().parents[2]
    log_file_path = repo_root / "logs" / "aruco_detection_flow.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging_conf_path = repo_root / "vision_python" / "config" / "logging.conf"
    logging.config.fileConfig(
        str(logging_conf_path), defaults={"log_file": str(log_file_path)}
    )
    logger = logging.getLogger("main")

    # Configure signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:

        # Create processes
        p1 = Process(
            target=run_task,
            args=(0, task_aruco_detection),
            name="TaskArucoDetection",
        )

        # Store global references
        processes = [p1]

        # Start the processes
        p1.start()

        logger.info("All processes have started. Press Ctrl+C to stop...")

        # Wait for processes to finish
        p1.join()

    except KeyboardInterrupt:
        # Handle KeyboardInterrupt exception directly
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        # Clean up
        for process in processes:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()
