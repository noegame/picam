"""
test of camera module
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from vision_python.src.camera.camera_factory import get_camera
from vision_python.config import config

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

output_folder = config.get_debug_directory()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def test_picamera():
    camera = get_camera("picamera2")
    camera.set_parameters(
        {
            "width": 4056,
            "height": 3040,
            "config_mode": "still",
            "save_format": "png",
            "output_folder": output_folder / "camera_tests",
        }
    )
    camera.start()
    camera.take_picture()
    camera.stop()


def test_webcam():
    camera = get_camera("webcam")
    camera.set_parameters(
        {
            "width": 640,
            "height": 480,
        }
    )
    camera.init()
    camera.start()
    camera.take_picture()
    camera.stop()


def test_emulated_camera():
    camera = get_camera("emulated")
    camera.set_parameters(
        {
            "width": 640,
            "height": 480,
        }
    )
    camera.init()
    camera.start()
    camera.take_picture()
    camera.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # test_picamera()
    test_webcam()
    # test_emulated_camera()


if __name__ == "__main__":
    main()
