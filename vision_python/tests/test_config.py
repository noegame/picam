#!/usr/bin/env python3

"""
test_config.py
test environment configuration
test importation from file config.py
"""

from vision_python.config import config
import cv2

print(f" PROJECT_DIR: {config.PROJECT_DIR}")
print(f" OUTPUT_DIR: {config.PICTURES_DIR}")
print(f" RASPBERRY_DIR: {config.RASPBERRY_DIR}")

for item in config.PICTURES_DIR.iterdir():
    item_type = "folder" if item.is_dir() else "file"
    print(f"{item_type}: {item.name}")

image_path = config.PICTURES_DIR / "aruco_gray.png"
image = cv2.imread(str(image_path))
