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
from datetime import datetime
from pathlib import Path

from vision_python.src.aruco import aruco
from vision_python.src.camera import camera_factory
from vision_python.config import config


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

            if camera_mode == "emulated":
                # Use existing folder from config, don't create today's folder
                daily_pictures_dir = config.get_emulated_cam_directory()
                logger.info(f"Using emulated camera directory: {daily_pictures_dir}")
            else:
                # No need to create directories for real camera anymore
                logger.info(
                    "Using PiCamera in direct capture mode (no automatic saving)"
                )

        except Exception as e:
            logger.error(
                f"Error while loading configuration or preparing directories: {e}"
            )
            raise

        # Initialize camera
        try:
            camera = camera_factory.get_camera(camera=camera_mode)
            params = {
                "width": img_width,
                "height": img_height,
            }

            # Only emulated camera needs image_folder parameter
            if camera_mode == "emulated":
                params["image_folder"] = daily_pictures_dir

            camera.set_parameters(params)
            camera.init()
            camera.start()
        except Exception as e:
            logger.error(f"Error while initializing the camera: {e}")
            raise

        # Load camera calibration matrices
        camera_matrix = aruco.get_camera_matrix()
        dist_matrix = aruco.get_distortion_matrix()
        logger.info("Calibration parameters imported successfully")

        # Initialize ArUco detector
        aruco_detector = aruco.get_aruco_detector()
        logger.info("ArUco detector initialized successfully")

        # Initialize mask and inverse homography (will be computed on first frame)
        mask = None
        H_inv = None
        ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]

        end = False
        consecutive_errors = 0
        max_consecutive_errors = 5

        while end is False:
            # Take a picture
            try:
                original_img = camera.take_picture()
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
                        camera = camera_factory.get_camera(camera=camera_mode)
                        params = {
                            "width": img_width,
                            "height": img_height,
                        }

                        # Only emulated camera needs image_folder parameter
                        if camera_mode == "emulated":
                            params["image_folder"] = daily_pictures_dir

                        camera.set_parameters(params)
                        camera.init()
                        camera.start()
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

            # ========== STEP 1: COMPUTE MASK AND INVERSE HOMOGRAPHY (first frame only) ==========
            if mask is None or H_inv is None:
                try:
                    mask, H_inv = aruco.find_mask(aruco_detector, img_bgr, scale_y=1.1)
                    logger.info("Mask and inverse homography computed successfully")
                except Exception as e:
                    logger.error(f"Failed to compute mask and homography: {e}")
                    time.sleep(0.1)
                    continue

            # ========== STEP 2: DETECT MARKERS ==========
            corners, ids, rejected, detection_timings = aruco.detect_markers(
                aruco_detector, img_bgr, mask=mask
            )

            # ========== STEP 3: FILTER ALLOWED IDS ==========
            valid_ids = []
            valid_corners = []

            if ids is not None:
                for i, marker_id in enumerate(ids):
                    mid = (
                        marker_id[0] if isinstance(marker_id, np.ndarray) else marker_id
                    )
                    if mid in ALLOWED_IDS:
                        valid_ids.append(mid)
                        valid_corners.append(corners[i])

            # ========== STEP 4: CALCULATE CENTERS ==========
            centers = []
            if valid_corners:
                centers = aruco.find_center_coord(valid_corners)

            # ========== STEP 5: POSE ESTIMATION ==========
            real_coords = None
            if H_inv is not None and centers:
                z_values = [30] * len(centers)  # 30mm height for all markers
                real_coords = aruco.pose_estimation_homography(
                    points=centers,
                    homography_inv=H_inv,
                    K=camera_matrix,
                    D=dist_matrix,
                    z_values=z_values,
                )

                # Apply pose corrections
                for i in range(len(real_coords)):
                    x, y, z = real_coords[i]
                    x_corrected, y_corrected = aruco.apply_pose_correction(x, y)
                    real_coords[i] = (x_corrected, y_corrected, z)

            # ========== STEP 6: BUILD DETECTED MARKERS LIST ==========
            tags_from_img = []

            for i, mid in enumerate(valid_ids):
                # Create a simple tag object (dict) with detection data
                tag = {
                    "aruco_id": mid,
                    "center_x": int(centers[i][0]) if centers else None,
                    "center_y": int(centers[i][1]) if centers else None,
                    "x": real_coords[i][0] if real_coords else None,
                    "y": real_coords[i][1] if real_coords else None,
                    "z": real_coords[i][2] if real_coords else None,
                }
                tags_from_img.append(tag)

            # ========== STEP 7: ANNOTATE IMAGE FOR VISUALIZATION ==========
            final_img = img_bgr.copy()

            if len(tags_from_img) > 0:
                # Annotate with counter
                final_img = aruco.annotate_img_with_counter(
                    final_img, len(tags_from_img)
                )

                # Annotate with IDs
                if centers:
                    final_img = aruco.annotate_img_with_ids(
                        final_img, centers, [tag["aruco_id"] for tag in tags_from_img]
                    )

                # Annotate with real coordinates if available
                if real_coords:
                    final_img = aruco.annotate_img_with_real_coords(
                        final_img, centers, real_coords
                    )

            # Log detection results
            if H_inv is None:
                logger.info("Missing reference aruco markers in the image")
                logger.warning("Real world coordinates not calculated for this frame")
            else:
                logger.info("All reference aruco markers found")
                logger.info(f"Detected {len(tags_from_img)} ArUco markers")

            # sort aruco by id for easier reading
            tags_from_img.sort(key=lambda tag: tag["aruco_id"])
            for tag in tags_from_img:
                logger.info(
                    f"ID {tag['aruco_id']}: ({tag['x']:.1f}, {tag['y']:.1f}, {tag['z']:.1f}) mm"
                )

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
                                "id": tag["aruco_id"],
                                "x": float(tag["x"]) if tag["x"] is not None else None,
                                "y": float(tag["y"]) if tag["y"] is not None else None,
                                "z": float(tag["z"]) if tag["z"] is not None else None,
                                "real_x": None,
                                "real_y": None,
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
