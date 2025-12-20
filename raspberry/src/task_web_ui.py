#!/usr/bin/env python3
"""
Web UI task for displaying ArUco tags
Receives tag data from a queue and displays it on a web page with a playground image
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import logging.config
import threading
import time
from multiprocessing import Queue
from pathlib import Path
from flask import Flask, jsonify, render_template

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Configure logging for this process
repo_root = Path(__file__).resolve().parents[2]
logging_conf_path = repo_root / "raspberry" / "config" / "logging.conf"
log_file_path = repo_root / "logs" / "aruco_detection_flow.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)

if not logging.getLogger().handlers:  # Only configure if not already configured
    logging.config.fileConfig(
        str(logging_conf_path), defaults={"log_file": str(log_file_path)}
    )

logger = logging.getLogger("task_web_ui")

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------

# Get the absolute path to the web-ui folder
template_folder = repo_root / "raspberry" / "src" / "web-ui"
static_folder = repo_root / "raspberry" / "src" / "web-ui"

app = Flask(
    __name__,
    template_folder=str(template_folder),
    static_folder=str(static_folder),
    static_url_path="",
)
data_lock = threading.Lock()

# Dictionary to store ArUco tag data
aruco_tags_data = []

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def update_data_from_queue(queue: Queue):
    """Retrieves the latest data from the queue by draining all pending elements."""
    global aruco_tags_data

    # Drain the queue: retrieve all pending elements and process only the latest one
    latest_data = None
    while not queue.empty():
        try:
            latest_data = queue.get_nowait()
        except:
            break

    # If we received data, process it
    if latest_data is not None:
        with data_lock:
            # Update the ArUco tag data
            # latest_data is directly a list of Aruco tags
            if isinstance(latest_data, list):
                aruco_tags_data = latest_data
                logger.debug(f"Updated aruco_tags_data with {len(latest_data)} tags")
            elif isinstance(latest_data, dict) and "aruco_tags" in latest_data:
                aruco_tags_data = latest_data["aruco_tags"]
                logger.debug(
                    f"Updated aruco_tags_data with {len(latest_data['aruco_tags'])} tags"
                )


def data_reader_thread(queue: Queue):
    """Thread that continuously reads data from the queue and processes the latest element."""
    logger.info("Starting data reader thread")
    while True:
        try:
            update_data_from_queue(queue)
            time.sleep(
                0.01
            )  # Small delay to avoid running too fast when the queue is empty
        except Exception as e:
            logger.error(f"Error in data reader thread: {e}", exc_info=True)


@app.route("/aruco_data")
def get_aruco_data():
    """Endpoint to retrieve ArUco tag data as JSON."""
    try:
        with data_lock:
            # Convert Aruco objects to dictionaries for JSON serialization
            serializable_data = [tag.to_dict() for tag in aruco_tags_data]
            return jsonify(serializable_data)
    except Exception as e:
        logger.error(f"Error in get_aruco_data: {e}", exc_info=True)
        return jsonify([])  # Return empty list on error


@app.route("/playground.png")
def get_playground_image():
    """Serve the playground image."""
    from flask import send_file

    image_path = template_folder / "playground.png"
    return send_file(str(image_path), mimetype="image/png")


@app.route("/")
def index():
    """Home page with the playground image and ArUco data."""
    return render_template("index.html")


def task_web_ui(queue: Queue):
    """Web UI task: retrieves data from the queue and serves it via Flask."""
    logger.info("Starting web UI task")

    # Start the data reader thread
    reader_thread = threading.Thread(
        target=data_reader_thread, args=(queue,), daemon=True
    )
    reader_thread.start()
    logger.info("Data reader thread started")

    # Disable Werkzeug logs to avoid cluttering the console
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # Launch the Flask server
    logger.info("Starting Flask server on 0.0.0.0:5000")

    try:
        app.run(
            host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False
        )
    except Exception as e:
        logger.error(f"Flask server error: {e}", exc_info=True)
        raise
