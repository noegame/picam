#!/usr/bin/env python3

"""
task_aruco_detection.py
Detection and localization of ArUco tags using PiCamera2
Sends detected tag data to a queue for
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import time
import cv2
import logging
import numpy as np

from pathlib import Path
from multiprocessing import Queue

from raspberry.src import unround_image
from raspberry.src import aruco
from raspberry.src import detect_aruco
from raspberry.src.camera import camera_factory
from raspberry.config.env_loader import EnvConfig

# Load environment configuration
EnvConfig()

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# A1 = Point(600, 600, 20)
# B1 = Point(1400, 600, 22)
# C1 = Point(600, 2400, 21)
# D1 = Point(1400, 2400, 23)

A1 = aruco.Aruco(53, 53, 1, 20)  # SO
B1 = aruco.Aruco(123, 53, 1, 22)  # SE
C1 = aruco.Aruco(53, 213, 1, 21)  # NO
D1 = aruco.Aruco(123, 213, 1, 23)  # NE

FIXED_IDS = {20, 21, 22, 23}

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def task_aruco_detection(queue_to_ui: Queue, queue_to_com: Queue):
    """
    Task of
    - take a picture with PiCamera2
    - correct the rounding of the photo
    - find the src points (fixed ArUco tags) in the picture to compute the perspective transform matrix
    - find real world coordinates of all detected ArUco tags in the picture with the perspective transform matrix
    - send the list of detected tags with their real world coordinates to the UI task via queue
    """

    logger = logging.getLogger("task_aruco_detection")
    camera = None

    try:
        # Load environment configuration
        try:
            EnvConfig()
            image_width = EnvConfig.get_camera_width()
            image_height = EnvConfig.get_camera_height()
            calibration_filename = EnvConfig.get_calibration_filename()
            image_size = (image_width, image_height)

        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise

        # Prepare directories and files
        try:
            repo_root = Path(__file__).resolve().parents[2]
            camera_pictures_dir = repo_root / "output" / "camera"
            straightened_pictures_dir = repo_root / "output" / "straightened"
            unround_dir = repo_root / "output" / "unround"
            config_dir = repo_root / "raspberry" / "config"
            calibration_file = config_dir / calibration_filename

        except Exception as e:
            logger.error(f"Error while preparing output directories: {e}")
            raise

        # Initialize camera
        try:
            camera = camera_factory.get_camera(
                w=image_width, h=image_height, allow_fallback=False, config_mode="still"
            )
        except Exception as e:
            logger.error(f"Error while initializing the camera: {e}")
            raise

        # Importation unround parameters
        camera_matrix, dist_coeffs = unround_image.import_camera_calibration(
            str(calibration_file)
        )
        logger.info("Calibration parameters imported successfully")

        # Process
        newcameramtx = unround_image.process_new_camera_matrix(
            camera_matrix, dist_coeffs, image_size
        )
        logger.info("New optimized camera matrix calculated successfully")

        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        logger.info("ArUco detector initialized successfully")

        # Destination points for perspective transform (fixed positions in real world)
        dst_points = np.array(
            [[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32
        )

        end = False
        consecutive_errors = 0
        max_consecutive_errors = 5

        while end is False:
            # Take a picture
            try:
                original_img, original_filepath = camera.capture_image(
                    pictures_dir=camera_pictures_dir
                )
                consecutive_errors = 0  # Reset error counter on successful capture

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error while capturing image: {e}")

                # If too many consecutive errors, restart the camera
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(
                        f"Too many consecutive capture errors ({consecutive_errors}). Attempting to restart camera..."
                    )
                    try:
                        if camera:
                            camera.close()
                    except Exception as close_err:
                        logger.warning(f"Error closing camera: {close_err}")

                    try:
                        camera = camera_factory.get_camera(
                            w=image_width,
                            h=image_height,
                            allow_fallback=False,
                            config_mode="still",
                        )
                        consecutive_errors = 0
                        logger.info("Camera restarted successfully")
                    except Exception as restart_err:
                        logger.error(f"Failed to restart camera: {restart_err}")
                        raise

                time.sleep(1)  # Avoid overload in case of capture error in loop
                continue

            # TODO : check if conversion is redundant
            # Convert RGB to BGR for OpenCV processing
            img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

            # Correct image roundness
            img_unrounded = unround_image.unround(
                img=img_bgr,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                newcameramtx=newcameramtx,
            )
            logger.debug("Image roundness corrected successfully")

            # Detect ArUco markers sources points
            tags_from_img, img_annotated = detect_aruco.detect_in_image(
                img_unrounded, aruco_dict, None, aruco_params, draw=False
            )

            # Temporarily convert detected tags to Aruco objects (should be done inside detect_aruco)
            tags_from_img = detect_aruco.convert_detected_tags(tags_from_img)

            # Find source points by their ArUco IDs
            a2 = aruco.find_aruco_by_id(tags_from_img, 20)
            b2 = aruco.find_aruco_by_id(tags_from_img, 22)
            c2 = aruco.find_aruco_by_id(tags_from_img, 21)
            d2 = aruco.find_aruco_by_id(tags_from_img, 23)

            # Verify all reference markers were found
            if a2 is None or b2 is None or c2 is None or d2 is None:
                logger.info(f"Missing reference aruco markers in the image")

                # Use previous valid matrix if available
                # If no valid matrix yet, skip this frame
                if "matrix" not in locals():
                    logger.warning(
                        "Perspective transformation matrix not yet available, skipping frame"
                    )
                    time.sleep(1)  # Avoid overload in case of repeated missing markers
                    continue
                continue

            else:
                logger.info("All reference aruco markers found")
                # Define source points (corners of the area to be straightened in image coordinates)
                src_points = np.array(
                    [[a2.x, a2.y], [b2.x, b2.y], [d2.x, d2.y], [c2.x, c2.y]],
                    dtype=np.float32,
                )

                # Define destination points (where the corners should map to in real world coordinates)
                dst_points = np.array(
                    [[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]],
                    dtype=np.float32,
                )

            # Calculate the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Convert detected tags image coordinates to real world coordinates
            tags_from_real_world = []
            for tag in tags_from_img:
                # Create homogeneous coordinate [x, y, z]
                img_point = np.array([tag.x, tag.y, tag.z], dtype=np.float32).reshape(
                    3, 1
                )

                # Apply perspective transformation matrix to convert to real world coordinates
                real_world_point = matrix @ img_point

                # Normalize homogeneous coordinates
                real_x = real_world_point[0, 0] / real_world_point[2, 0]
                real_y = real_world_point[1, 0] / real_world_point[2, 0]

                # Create new Aruco object with transformed coordinates
                transformed_tag = aruco.Aruco(
                    real_x, real_y, tag.z, tag.aruco_id, tag.angle
                )
                tags_from_real_world.append(transformed_tag)

                logger.debug(
                    f"Tag ID {tag.aruco_id}: Image coords ({tag.x:.2f}, {tag.y:.2f}) -> Real world coords ({real_x:.2f}, {real_y:.2f})"
                )

            # Send detected tags to UI task
            try:
                queue_to_ui.put(tags_from_real_world)
                logger.info(
                    f"Sent {len(tags_from_real_world)} detected tags to UI task via queue"
                )
            except Exception as e:
                logger.error(f"Error while sending detected tags to UI task: {e}")

            # Send detected tags to Communication task
            # TODO

            logger.info(
                "Aruco detection iteration complete, waiting before next capture..."
            )
            time.sleep(1)  # Adjust delay as needed to control capture rate

    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise
    finally:
        # Ensure camera is properly closed on exit
        if camera:
            try:
                camera.close()
                logger.info("Camera closed during shutdown")
            except Exception as e:
                logger.warning(f"Error closing camera during shutdown: {e}")
