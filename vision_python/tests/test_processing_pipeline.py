#!/usr/bin/env python3

"""
test_processing_pipeline.py

Test simple pour vérifier que le module processing_pipeline fonctionne correctement.
"""

import sys
from pathlib import Path

# Test des imports
print("Testing imports...")
try:
    from vision_python.src.img_processing import processing_pipeline as pipeline

    print("✅ processing_pipeline imported successfully")
except ImportError as e:
    print(f"❌ Failed to import processing_pipeline: {e}")
    sys.exit(1)

# Test que toutes les fonctions sont disponibles
print("\nChecking available functions...")
expected_functions = [
    "load_and_convert_to_grayscale",
    "apply_sharpening",
    "apply_unrounding",
    "apply_clahe",
    "apply_thresholding",
    "detect_aruco_markers",
    "create_marker_objects",
    "annotate_image_with_markers",
    "annotate_image_with_rejected_markers",
    "compute_perspective_transform",
    "transform_markers_to_real_world",
    "mask_playground_area",
    "print_detection_summary",
    "print_detailed_markers",
    "save_debug_image",
    "build_filename",
    "save",
]

missing_functions = []
for func_name in expected_functions:
    if hasattr(pipeline, func_name):
        print(f"  ✅ {func_name}")
    else:
        print(f"  ❌ {func_name} - NOT FOUND")
        missing_functions.append(func_name)

if missing_functions:
    print(f"\n❌ Missing functions: {', '.join(missing_functions)}")
    sys.exit(1)
else:
    print(f"\n✅ All {len(expected_functions)} functions are available!")

print("\n" + "=" * 60)
print("✅ processing_pipeline module test PASSED!")
print("=" * 60)
