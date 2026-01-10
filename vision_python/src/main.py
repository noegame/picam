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
from multiprocessing import Process, Queue
from vision_python.config import config
from vision_python.src.tasks import task_aruco_detection
from vision_python.src.tasks import task_ui
from vision_python.src.hello import hello


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
    hello()
    # Configure logging with absolute path
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging_conf_path = repo_root / "vision_python" / "config" / "logging.conf"
    logging.config.fileConfig(
        str(logging_conf_path), defaults={"log_dir": str(log_dir)}
    )
    logger = logging.getLogger("main")

    # Configure signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create processes for enabled tasks
        task_list = []
        core_id = 0

        # Create shared queue for image data (only if both tasks are enabled)
        image_queue = None
        if config.is_aruco_detection_enabled() and config.is_ui_enabled():
            image_queue = Queue(maxsize=10)  # Limit queue size to avoid memory issues
            logger.info("Created shared image queue for ArUco detection and UI")

        if config.is_aruco_detection_enabled():
            p = Process(
                target=run_task,
                args=(core_id, task_aruco_detection.run, image_queue),
                name="TaskArucoDetection",
            )
            task_list.append(p)
            core_id += 1

        if config.is_ui_enabled():
            p = Process(
                target=run_task,
                args=(core_id, task_ui.run, image_queue),
                name="TaskUI",
            )
            task_list.append(p)
            core_id += 1

        # Store global references
        processes = task_list

        if not processes:
            logger.warning("No tasks enabled in config. Exiting...")
            sys.exit(0)

        # Start all processes
        for process in processes:
            process.start()
            logger.info(f"Started {process.name}")

        logger.info("All processes have started. Press Ctrl+C to stop...")

        # Wait for all processes to finish
        for process in processes:
            process.join()

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
