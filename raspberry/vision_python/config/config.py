from enum import Enum
from pathlib import Path


# Camera Modes
class CameraMode(Enum):
    PI = 1
    COMPUTER = 2
    EMULATED = 3


# Feature Toggles
ENABLE = True
DISABLE = False

# Camera Configuration
CAMERA_WIDTH = 2000
CAMERA_HEIGHT = 2000
CAMERA = CameraMode.PI

# Project Paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RASPBERRY_DIR = PROJECT_DIR / "raspberry"
OUTPUT_DIR = PROJECT_DIR / "output"
CAMERA_DIR = OUTPUT_DIR / "camera"
VISION_DIR = RASPBERRY_DIR / "vision_python"
CALIBRATION_FILE = VISION_DIR / "config" / "camera_calibration_2000x2000.npz"

# Logging Configuration
LOG_LEVEL = "INFO"

# Task
ARUCO_DETECTION = ENABLE
UI = ENABLE

# Flask Server Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000


def get_camera_resolution():
    """Returns the camera resolution as a tuple (width, height)."""
    return CAMERA_WIDTH, CAMERA_HEIGHT


def get_calibration_file_path():
    """Returns the path to the camera calibration file."""
    return CALIBRATION_FILE


def is_aruco_detection_enabled():
    """Returns whether ArUco detection is enabled."""
    return ARUCO_DETECTION


def is_ui_enabled():
    """Returns whether the UI is enabled."""
    return UI


def get_flask_server_config():
    """Returns the Flask server configuration as a tuple (host, port)."""
    return FLASK_HOST, FLASK_PORT


def get_camera_mode():
    """Returns the current camera mode."""
    return CAMERA


def get_logging_level():
    """Returns the logging level."""
    return LOG_LEVEL


def get_output_directory():
    """Returns the output directory path."""
    return OUTPUT_DIR


def get_camera_directory():
    """Returns the camera output directory path."""
    return CAMERA_DIR


def get_vision_directory():
    """Returns the vision_python directory path."""
    return VISION_DIR


def get_project_directory():
    """Returns the project root directory path."""
    return PROJECT_DIR
