#!/usr/bin/env python3

"""
task_ui.py
Flask-based UI task for displaying ArUco detection results
"""

import logging
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import time


# Global variable to store latest frame
latest_frame = None
latest_tags = []
frame_lock = None


def generate_frames():
    """Generator function to yield frames for video streaming"""
    global latest_frame

    logger = logging.getLogger(__name__)
    logger.info("Frame generator started")

    while True:
        if latest_frame is not None:
            try:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode(".jpg", latest_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            except Exception as e:
                logger.error(f"Error generating frame: {e}")
        else:
            # Send a placeholder or wait
            pass

        time.sleep(0.05)  # Adjust frame rate as needed (~20 FPS)


# HTML template for the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ArUco Detection Live View</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #333;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 90%;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #333;
            border-radius: 5px;
        }
        .info {
            margin-top: 20px;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <img src="{{ url_for('video_feed') }}" alt="Live Stream">
    </div>
</body>
</html>
"""


def create_flask_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "aruco-detection-ui-key"

    @app.route("/")
    def index():
        """Main page with video stream"""
        return render_template_string(HTML_TEMPLATE)

    @app.route("/video_feed")
    def video_feed():
        """Video streaming route"""
        return Response(
            generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def run(image_queue=None) -> None:
    """
    Flask UI task function to display images from the queue

    Args:
        image_queue: Optional multiprocessing.Queue to receive images from ArUco detection task
    """
    global latest_frame, latest_tags

    logger = logging.getLogger(__name__)
    logger.info("UI task started")

    if image_queue is None:
        logger.warning("No image queue provided - UI will run but display no images")

    # Create Flask app
    app = create_flask_app()

    # Start background thread to consume from queue
    if image_queue is not None:
        import threading

        def queue_consumer():
            """Background thread to consume images from queue"""
            global latest_frame, latest_tags

            logger.info("Queue consumer thread started, waiting for frames...")

            while True:
                try:
                    # Get data from queue (blocking)
                    data = image_queue.get()

                    if data is not None:
                        # Decode JPEG bytes back to image
                        image_bytes = data.get("image_bytes")
                        if image_bytes:
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            # Convert tag dictionaries to objects for display
                            tags_data = data.get("tags", [])
                            latest_tags = [
                                type("Tag", (), tag_dict) for tag_dict in tags_data
                            ]

                            logger.info(
                                f"Received frame with {len(latest_tags)} tags from queue"
                            )
                        else:
                            logger.warning("Received data without image_bytes")

                except Exception as e:
                    logger.error(f"Error consuming from queue: {e}", exc_info=True)
                    time.sleep(0.1)

        consumer_thread = threading.Thread(target=queue_consumer, daemon=True)
        consumer_thread.start()
        logger.info("Started queue consumer thread")

    # Run Flask server
    try:
        logger.info("Starting Flask server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Error running Flask server: {e}")
        raise
