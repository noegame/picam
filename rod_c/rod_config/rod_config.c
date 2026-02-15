/**
 * @file rod_config.c
 * @brief Centralized configuration for ROD system
 * @author Noé Game
 * @date 15/02/2026
 * @see rod_config.h
 * @copyright Cecill-C (Cf. LICENCE.txt)
 */

/* ******************************************************* Includes ****************************************************** */

#include "rod_config.h"

/* ******************************************* Public callback functions declarations ************************************ */

/* ********************************************* Function implementations *********************************************** */

int rod_config_is_valid_marker_id(int id) {
    return (id >= 1 && id <= 10) ||    // Robots (blue 1-5, yellow 6-10)
           (id >= 20 && id <= 23) ||   // Fixed markers
           (id == 36) ||                // Blue box
           (id == 41) ||                // Empty box (black)
           (id == 47);                  // Yellow box
}

MarkerCategory rod_config_get_marker_category(int id) {
    if (id >= 1 && id <= 5) {
        return MARKER_CATEGORY_ROBOT_BLUE;
    } else if (id >= 6 && id <= 10) {
        return MARKER_CATEGORY_ROBOT_YELLOW;
    } else if (id >= 20 && id <= 23) {
        return MARKER_CATEGORY_FIXED;
    } else if (id == 36) {
        return MARKER_CATEGORY_BOX_BLUE;
    } else if (id == 41) {
        return MARKER_CATEGORY_BOX_EMPTY;
    } else if (id == 47) {
        return MARKER_CATEGORY_BOX_YELLOW;
    } else {
        return MARKER_CATEGORY_INVALID;
    }
}

void rod_config_configure_detector_parameters(DetectorParametersHandle* params) {
    // Adaptive thresholding parameters
    // These values are CRITICAL - validated through extensive testing
    setAdaptiveThreshWinSizeMin(params, 3);
    setAdaptiveThreshWinSizeMax(params, 53);
    setAdaptiveThreshWinSizeStep(params, 4);
    
    // Marker size constraints
    setMinMarkerPerimeterRate(params, 0.01);
    setMaxMarkerPerimeterRate(params, 4.0);
    
    // Polygon approximation accuracy
    setPolygonalApproxAccuracyRate(params, 0.05);
    
    // Corner refinement for sub-pixel accuracy
    setCornerRefinementMethod(params, CORNER_REFINE_SUBPIX);
    setCornerRefinementWinSize(params, 5);
    setCornerRefinementMaxIterations(params, 50);
    
    // Detection constraints
    setMinDistanceToBorder(params, 0);
    setMinOtsuStdDev(params, 2.0);
    
    // Perspective removal
    setPerspectiveRemoveIgnoredMarginPerCell(params, 0.15);
}

int rod_config_get_aruco_dictionary_type(void) {
    return DICT_4X4_50;
}
