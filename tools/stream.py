#!/usr/bin/env python3

# Stream Video from Raspberry Pi Camera to Local Computer

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import io
import time
import threading
import picamera2
import cv2
from flask import Flask, Response

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Singleton pour la caméra - créée une seule fois
camera = None
camera_lock = threading.Lock()
frame_lock = threading.Lock()
current_frame = None


def initialize_camera():
    global camera
    if camera is None:
        with camera_lock:
            if camera is None:  # Double-check locking
                camera = picamera2.Picamera2()
                # Use RGB888 format for direct RGB output without conversion
                camera.configure(
                    camera.create_preview_configuration(
                        main={"format": "RGB888", "size": (2000, 2000)}
                    )
                )
                camera.start()


def capture_frame():
    """Capture une frame et la met en cache"""
    global current_frame
    try:
        # Capture l'image en tant qu'array (RGB888 format - already RGB)
        array = camera.capture_array()
        
        # Encode directly to JPEG using OpenCV (faster than PIL)
        # cv2.imencode expects BGR, so convert RGB to BGR
        array_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        success, jpeg_buffer = cv2.imencode('.jpg', array_bgr)
        
        if success:
            with frame_lock:
                current_frame = jpeg_buffer.tobytes()
        else:
            print("Erreur lors de l'encodage JPEG")
    except Exception as e:
        print(f"Erreur lors de la capture: {e}")
        import traceback
        traceback.print_exc()


def generate_frames():
    """Génère les frames pour le stream"""
    initialize_camera()

    # Capture les frames continuellement en arrière-plan
    def capture_loop():
        while True:
            capture_frame()
            time.sleep(0.20)  # ~25 FPS

    # Démarre le thread de capture
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # Envoie les frames aux clients
    while True:
        if current_frame:
            with frame_lock:
                frame_data = current_frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(
                len(frame_data)
            ).encode() + b"\r\n\r\n" + frame_data + b"\r\n"
        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return """
    <html>
    <head><title>Pi Camera Stream</title></head>
    <body>
        <h1>Raspberry Pi Camera Stream</h1>
        <img src="/video_feed" style="max-width: 100%; border: 1px solid #ccc;">
    </body>
    </html>
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initialize_camera()
    app.run(host="0.0.0.0", port=5000, threaded=True)
