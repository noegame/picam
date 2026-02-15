/**
 * @file rod_cv.c
 * @brief Computer vision helper functions for ROD
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_cv.h
 * @copyright Cecill-C (Cf. LICENCE.txt)
 * 
 * This module provides computer vision utility functions for the ROD project:
 * - Pose estimation of ArUco markers
 * - Coordinate transformations
 * - Advanced marker detection utilities
 */

/* ******************************************************* Includes ****************************************************** */

#include "rod_cv.h"
#include "opencv_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ***************************************************** Public macros *************************************************** */

// Mathematical constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ************************************************** Public types definition ******************************************** */

/* *********************************************** Public functions declarations ***************************************** */

/* ******************************************* Public callback functions declarations ************************************ */

/* ********************************************* Function implementations *********************************************** */

Point2f calculate_marker_center(float corners[4][2]) {
    Point2f center;
    center.x = (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0f;
    center.y = (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0f;
    return center;
}

float calculate_marker_angle(float corners[4][2]) {
    // Calculate angle from corner 0 to corner 1 (top edge of marker)
    // This represents the orientation of the marker
    float dx = corners[1][0] - corners[0][0];
    float dy = corners[1][1] - corners[0][1];
    return atan2f(dy, dx);
}

float calculate_marker_perimeter(float corners[4][2]) {
    float perimeter = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        int next = (i + 1) % 4;
        float dx = corners[next][0] - corners[i][0];
        float dy = corners[next][1] - corners[i][1];
        perimeter += sqrtf(dx * dx + dy * dy);
    }
    
    return perimeter;
}

float calculate_marker_area(float corners[4][2]) {
    // Use the Shoelace formula for polygon area
    float area = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        int next = (i + 1) % 4;
        area += corners[i][0] * corners[next][1];
        area -= corners[next][0] * corners[i][1];
    }
    
    return fabsf(area) / 2.0f;
}

float rad_to_deg(float radians) {
    return radians * 180.0f / M_PI;
}

float deg_to_rad(float degrees) {
    return degrees * M_PI / 180.0f;
}

float normalize_angle(float angle) {
    // Normalize angle to [-PI, PI] range
    while (angle > M_PI) {
        angle -= 2.0f * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0f * M_PI;
    }
    return angle;
}
