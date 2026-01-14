#!/usr/bin/env python3

"""
config.py
Configuration module for tests and main application.
Defines constants and getter functions for camera settings,
logging, feature toggles, and directory paths.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from enum import Enum
from pathlib import Path
from vision_python.src.aruco import aruco

# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class CameraMode(Enum):
    PI = 1
    COMPUTER = 2
    EMULATED = 3


ENABLE = True
DISABLE = False

# ---------------------------------------------------------------------------
# Image Processing and ArUco Detection Parameters
# ---------------------------------------------------------------------------

use_unround = True  # Désarrondissement de l'image (correction de l'effet fish-eye)
use_clahe = False  # Égalisation d'histogramme adaptative, False sauf éclairage variable
use_binarization = False  # Binarisation de l'image pour améliorer le contraste

# Prétraitement optimal
sharpen_alpha = 1.5  # Contraste
sharpen_beta = -0.5  # Netteté
sharpen_gamma = 0  # Luminosité

# Détection ArUco optimale
adaptive_thresh_constant = 7  # Ajustement aux conditions d'éclairage
min_marker_perimeter_rate = 0.01  # Taille minimale du marqueur
max_marker_perimeter_rate = 4.0  # Taille maximale du marqueur
polygonal_approx_accuracy_rate = 0.1  # Précision de détection des coins


# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------

CAMERA_WIDTH = 4000
CAMERA_HEIGHT = 4000
CAMERA = CameraMode.PI

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# Feature Toggles
# ---------------------------------------------------------------------------

ARUCO_DETECTION = ENABLE
UI = ENABLE

# ---------------------------------------------------------------------------
# Flask Server Configuration
# ---------------------------------------------------------------------------

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
PICTURES_DIR = PROJECT_DIR / "pictures"
CAMERA_DIR = PICTURES_DIR / "camera"
LOG_DIR = PROJECT_DIR / "logs"
VISION_DIR = PROJECT_DIR / "vision_python"
CALIBRATION_DIR = VISION_DIR / "config" / "calibrations"
CALIBRATION_FILE = CALIBRATION_DIR / "80_lens" / "camera_calibration_4000x4000.npz"
EMULATED_CAM_DIR = CAMERA_DIR / "2026-01-09-playground-ready"


# ---------------------------------------------------------------------------
# Fixed Aruco marker positions
# ---------------------------------------------------------------------------

# Define destination points (real world coordinates in mm)
A1 = aruco.Aruco(600, 600, 1, 20)
B1 = aruco.Aruco(1400, 600, 1, 22)
C1 = aruco.Aruco(600, 2400, 1, 21)
D1 = aruco.Aruco(1400, 2400, 1, 23)

# ---------------------------------------------------------------------------
# Getter Functions
# ---------------------------------------------------------------------------


def is_aruco_detection_enabled():
    """Returns whether ArUco detection is enabled."""
    return ARUCO_DETECTION


def is_ui_enabled():
    """Returns whether the UI is enabled."""
    return UI


def get_camera_resolution():
    """Returns the camera resolution as a tuple (width, height)."""
    return CAMERA_WIDTH, CAMERA_HEIGHT


def get_camera_width():
    """Returns the camera width."""
    return CAMERA_WIDTH


def get_camera_height():
    """Returns the camera height."""
    return CAMERA_HEIGHT


def get_camera_mode():
    """Returns the current camera mode."""
    return CAMERA


def get_flask_server_config():
    """Returns the Flask server configuration as a tuple (host, port)."""
    return FLASK_HOST, FLASK_PORT


def get_logging_level():
    """Returns the logging level."""
    return LOG_LEVEL


def get_image_processing_params():
    """Returns a dictionary of image processing parameters."""
    return {
        "use_unround": use_unround,
        "use_clahe": use_clahe,
        "use_binarization": use_binarization,
        "sharpen_alpha": sharpen_alpha,
        "sharpen_beta": sharpen_beta,
        "sharpen_gamma": sharpen_gamma,
        "adaptive_thresh_constant": adaptive_thresh_constant,
        "min_marker_perimeter_rate": min_marker_perimeter_rate,
        "max_marker_perimeter_rate": max_marker_perimeter_rate,
        "polygonal_approx_accuracy_rate": polygonal_approx_accuracy_rate,
    }


def get_fixed_aruco_positions():
    """Returns a list of fixed ArUco marker positions."""
    return [A1, B1, C1, D1]


# ---------------------------------------------------------------------------
# Directory Getter Functions
# ---------------------------------------------------------------------------


def get_camera_directory():
    """Returns the camera output directory path."""
    return CAMERA_DIR


def get_debug_directory():
    """Returns the debug output directory path."""
    return PICTURES_DIR / "debug"


def get_vision_directory():
    """Returns the vision_python directory path."""
    return VISION_DIR


def get_project_directory():
    """Returns the project root directory path."""
    return PROJECT_DIR


def get_pictures_directory():
    """Returns the pictures directory path."""
    return PICTURES_DIR


def get_log_directory():
    """Returns the log directory path."""
    return LOG_DIR


def get_camera_calibration_directory():
    """Returns the camera calibration directory path."""
    return CALIBRATION_DIR


def get_camera_calibration_file():
    """Returns the camera calibration file path."""
    return CALIBRATION_FILE


def get_emulated_cam_directory():
    """Returns the emulated images directory path."""
    return EMULATED_CAM_DIR
