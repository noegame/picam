#!/usr/bin/env python3
"""
Factory to get the appropriate Camera instance based on input string.
"""
from .camera import Camera


def get_camera(camera: str) -> Camera:
    """
    Factory function to return the appropriate Camera instance.
    :param camera: Type of camera ("emulated", "webcam", "picamera")
    :return: Instance of Camera subclass
    """
    if camera == "emulated":
        from .emulated_camera import EmulatedCamera

        return EmulatedCamera()

    elif camera == "webcam":
        from .webcam import Webcam

        return Webcam()

    elif camera == "picamera":
        from .picamera import PiCamera

        return PiCamera()

    else:
        raise ValueError(
            f"Unknown camera type: {camera}. Expected 'emulated', 'webcam', or 'picamera'."
        )
