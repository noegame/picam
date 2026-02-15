/**
 * @file rod_cv.h
 * @brief Computer vision helper functions for ROD
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_cv.c
 * @copyright Cecill-C (Cf. LICENCE.txt)
 */

#pragma once

/* ******************************************************* Includes ****************************************************** */

#include "opencv_wrapper.h"

/* ***************************************************** Public macros *************************************************** */

/* ************************************************** Public types definition ******************************************** */

/**
 * @brief Structure to hold 3D position and orientation
 */
typedef struct {
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;
} Pose3D;

/**
 * @brief Structure to hold 2D position and orientation
 */
typedef struct {
    float x;
    float y;
    float angle;
} Pose2D;

/* *********************************************** Public functions declarations ***************************************** */

/**
 * @brief Calculate the center point of a marker from its corners
 * @param corners Array of 4 corner points [x,y]
 * @return Center point
 */
Point2f calculate_marker_center(float corners[4][2]);

/**
 * @brief Calculate the angle of a marker from its corners
 * @param corners Array of 4 corner points [x,y]
 * @return Angle in radians (-PI to PI)
 */
float calculate_marker_angle(float corners[4][2]);

/**
 * @brief Calculate the perimeter of a marker
 * @param corners Array of 4 corner points [x,y]
 * @return Perimeter in pixels
 */
float calculate_marker_perimeter(float corners[4][2]);

/**
 * @brief Calculate the area of a marker
 * @param corners Array of 4 corner points [x,y]
 * @return Area in square pixels
 */
float calculate_marker_area(float corners[4][2]);

/**
 * @brief Convert angle from radians to degrees
 * @param radians Angle in radians
 * @return Angle in degrees
 */
float rad_to_deg(float radians);

/**
 * @brief Convert angle from degrees to radians
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
float deg_to_rad(float degrees);

/**
 * @brief Normalize angle to [-PI, PI] range
 * @param angle Angle in radians
 * @return Normalized angle
 */
float normalize_angle(float angle);

/**
 * @brief Pose estimation of aruco marker in a picture
 * @param image Image handle
 * @param marker_handle Marker handle containing marker ID and corners
 * @note This is a placeholder for future 3D pose estimation implementation
 */
// void estimate_marker_pose(ImageHandle* image, MarkerHandle* marker_handle);


/* ******************************************* Public callback functions declarations ************************************ */
