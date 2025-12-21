#!/usr/bin/env python3
"""
Environment Loader - Load configuration from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class EnvConfig:
    """Load and manage environment variables from .env file"""

    def __init__(self, env_path: Path = None):
        """
        Initialize environment configuration.

        :param env_path: Path to .env file. If None, searches for it in parent directories.
        """
        if env_path is None:
            # Search for .env file in vision_python folder
            vision_python_root = Path(__file__).resolve().parents[1]
            env_path = vision_python_root / ".env"

        if not env_path.exists():
            raise FileNotFoundError(
                f".env file not found at {env_path}. "
                f"Run: bash raspberry/setup/setup_env.sh"
            )

        # Load environment variables from .env file
        load_dotenv(env_path)

    @staticmethod
    def get_camera_width() -> int:
        """Get camera width from environment"""
        return int(os.getenv("CAMERA_WIDTH", 2000))

    @staticmethod
    def get_camera_height() -> int:
        """Get camera height from environment"""
        return int(os.getenv("CAMERA_HEIGHT", 2000))

    @staticmethod
    def get_UI() -> bool:
        """Get UI flag from environment"""
        return os.getenv("UI", "false").lower() == "true"

    @staticmethod
    def get_flask_host() -> str:
        """Get Flask host from environment"""
        return os.getenv("FLASK_HOST", "0.0.0.0")

    @staticmethod
    def get_flask_port() -> int:
        """Get Flask port from environment"""
        return int(os.getenv("FLASK_PORT", 5000))

    @staticmethod
    def get_log_level() -> str:
        """Get log level from environment"""
        return os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def get_use_fake_camera() -> bool:
        """Get fake camera mode from environment"""
        return os.getenv("USE_FAKE_CAMERA", "false").lower() == "true"

    @staticmethod
    def get_calibration_filename() -> str:
        """Get calibration filename from environment"""
        return os.getenv(
            "CALIBRATION_FILENAME",
            f"camera_calibration_{EnvConfig.get_camera_width()}x{EnvConfig.get_camera_height()}.npz",
        )

    @staticmethod
    def print_config():
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("PiCam Configuration")
        print("=" * 60)
        print(
            f"Camera Resolution: {EnvConfig.get_camera_width()}x{EnvConfig.get_camera_height()}"
        )
        print(f"Flask Host: {EnvConfig.get_flask_host()}")
        print(f"Flask Port: {EnvConfig.get_flask_port()}")
        print(f"Log Level: {EnvConfig.get_log_level()}")
        print(f"Use Fake Camera: {EnvConfig.get_use_fake_camera()}")
        print(f"Calibration File: {EnvConfig.get_calibration_filename()}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        config = EnvConfig()
        config.print_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
