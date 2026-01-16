#!/usr/bin/env python3

"""
task_aruco_detection.py
Detection and localization of ArUco tags using PiCamera2
Sends detected tag data to a queue for UI display
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import time
import cv2
import logging
import numpy as np
import os
from datetime import datetime
from pathlib import Path

from vision_python.src.aruco import aruco
from vision_python.src.img_processing import detect_aruco
from vision_python.src.img_processing import unround_img
from vision_python.src.img_processing import processing_pipeline as pipeline
from vision_python.src.camera import camera_factory
from vision_python.config import config

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Get fixed ArUco markers from config
FIXED_MARKERS = config.get_fixed_aruco_markers()
FIXED_IDS = {marker.aruco_id for marker in FIXED_MARKERS}

# For backward compatibility
A1, B1, C1, D1 = FIXED_MARKERS[0], FIXED_MARKERS[1], FIXED_MARKERS[2], FIXED_MARKERS[3]

# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------


def run(image_queue=None) -> None:
    """
    Task of
    - take a picture with PiCamera2
    - correct the rounding of the photo
    - find the src points (fixed ArUco tags) in the picture to compute the perspective transform matrix
    - find real world coordinates of all detected ArUco tags in the picture with the perspective transform matrix
    - send the list of detected tags with their real world coordinates to the UI task via queue

    Args:
        image_queue: Optional multiprocessing.Queue to send images to the UI task
    """

    logger = logging.getLogger("task_aruco_detection")
    camera = None  # Initialize camera variable to avoid UnboundLocalError

    try:
        # Load configuration
        try:
            img_width, img_height = config.get_camera_resolution()
            image_size = config.get_camera_resolution()
            camera_mode = config.get_camera_mode()
            calibration_file = config.get_camera_calibration_file()

            # Get image processing parameters from config
            img_params = config.get_image_processing_params()
            use_unround = img_params["use_unround"]
            use_clahe = img_params["use_clahe"]
            use_thresholding = img_params["use_thresholding"]
            sharpen_alpha = img_params["sharpen_alpha"]
            sharpen_beta = img_params["sharpen_beta"]
            sharpen_gamma = img_params["sharpen_gamma"]

            if camera_mode == config.CameraMode.EMULATED:
                # Use existing folder from config, don't create today's folder
                daily_pictures_dir = config.get_emulated_cam_directory()
                logger.info(f"Using emulated camera directory: {daily_pictures_dir}")
            elif camera_mode == config.CameraMode.PI:
                # Create subdirectory for today's date for real camera
                pictures_dir = config.get_camera_directory()
                today_date = datetime.now().strftime("%Y-%m-%d")
                daily_pictures_dir = Path(pictures_dir) / today_date
                os.makedirs(daily_pictures_dir, exist_ok=True)
                logger.info(f"Using daily directory: {daily_pictures_dir}")

        except Exception as e:
            logger.error(
                f"Error while loading configuration or preparing directories: {e}"
            )
            raise

        # Initialize camera
        try:
            camera = camera_factory.get_camera(
                camera=camera_mode,
                w=img_width,
                h=img_height,
                camera_param=daily_pictures_dir,
            )
        except Exception as e:
            logger.error(f"Error while initializing the camera: {e}")
            raise

        # Load camera calibration matrices
        camera_matrix, dist_coeffs, newcameramtx, roi = (
            config.get_camera_calibration_matrices()
        )
        logger.info("Calibration parameters imported successfully")

        # Initialize ArUco detector
        aruco_detector = detect_aruco.init_aruco_detector()
        logger.info("ArUco detector initialized successfully")

        end = False
        consecutive_errors = 0
        max_consecutive_errors = 5

        while end is False:
            # Take a picture
            try:
                original_img, original_filepath = camera.capture_image(
                    pictures_dir=daily_pictures_dir
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
                            camera=camera_mode,
                            w=img_width,
                            h=img_height,
                            camera_param=pictures_dir,
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

            # Use centralized processing pipeline for ArUco detection
            detected_markers, final_img, perspective_matrix, metadata = (
                pipeline.process_image_for_aruco_detection(
                    img=img_bgr,
                    aruco_detector=aruco_detector,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    newcameramtx=newcameramtx,
                    fixed_markers=FIXED_MARKERS,
                    playground_corners=None,  # No playground masking for real-time detection
                    use_unround=use_unround,
                    use_clahe=use_clahe,
                    use_thresholding=use_thresholding,
                    sharpen_alpha=sharpen_alpha,
                    sharpen_beta=sharpen_beta,
                    sharpen_gamma=sharpen_gamma,
                    use_mask_playground=False,  # Disabled for real-time performance
                    use_straighten_image=False,  # Disabled for real-time performance
                    save_debug_images=False,  # No debug images in production
                    apply_contrast_boost=False,
                )
            )

            # Extract detected tags from markers
            tags_from_img = [marker for marker, _ in detected_markers]

            # Log detection results
            if not metadata["perspective_transform_computed"]:
                logger.info("Missing reference aruco markers in the image")
                logger.warning("Real world coordinates not calculated for this frame")
            else:
                logger.info("All reference aruco markers found")
                logger.info(f"Detected {len(tags_from_img)} ArUco markers")

            # sort aruco by id for easier reading
            tags_from_img.sort(key=lambda tag: tag.aruco_id)
            for tag in tags_from_img:
                tag.print()

            # Send image and detected tags to UI queue if enabled
            if image_queue is not None:
                try:
                    # Encode image to JPEG bytes for efficient transfer through multiprocessing queue
                    ret, jpeg_buffer = cv2.imencode(
                        ".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if not ret:
                        logger.warning("Failed to encode image to JPEG")
                    else:
                        # Convert tags to serializable format
                        serializable_tags = []
                        for tag in tags_from_img:
                            tag_data = {
                                "id": tag.aruco_id,
                                "x": float(tag.x),
                                "y": float(tag.y),
                                "z": float(tag.z),
                                "real_x": (
                                    float(tag.real_x)
                                    if hasattr(tag, "real_x")
                                    else None
                                ),
                                "real_y": (
                                    float(tag.real_y)
                                    if hasattr(tag, "real_y")
                                    else None
                                ),
                            }
                            serializable_tags.append(tag_data)

                        # Prepare data to send
                        queue_data = {
                            "image_bytes": jpeg_buffer.tobytes(),
                            "tags": serializable_tags,
                            "timestamp": time.time(),
                        }

                        # Non-blocking put - if queue is full, remove old item and add new
                        if image_queue.full():
                            try:
                                image_queue.get_nowait()  # Remove oldest item
                                logger.debug("Removed old frame from full queue")
                            except:
                                pass

                        image_queue.put_nowait(queue_data)
                        logger.info(
                            f"Image and {len(serializable_tags)} tags sent to UI queue"
                        )
                except Exception as e:
                    logger.error(f"Failed to send data to UI queue: {e}", exc_info=True)

            logger.info(
                "Aruco detection iteration complete, waiting before next capture..."
            )
            logger.info(
                "========================================================================"
            )

            time.sleep(0.1)  # Adjust delay as needed to control capture rate

    except Exception as e:
        logger.error(f"Erreur fatale dans la tâche ArUco: {e}")
        raise
    finally:
        # Ensure camera is properly closed on exit
        if camera is not None:
            try:
                camera.close()
                logger.info("Camera closed during shutdown")
            except Exception as e:
                logger.warning(f"Error closing camera during shutdown: {e}")
