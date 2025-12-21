# Some code explanation
[readme](../README.md)

## Summary
- [Image unrounding](#image-unrounding)
- [Image straightening](#image-straightening)
- [ArUco tag detection](#aruco-tag-detection)
- [Camera configurations in Picamera2](#camera-configurations-in-picamera2)

## Image unrounding
The unround_image.py file provides functions to correct lens distortion in images using camera calibration data.
```python
# Import camera calibration data.
camera_matrix, dist_coeffs = unround_image.import_camera_calibration(
    str(calibration_file)
)
# Calculate a new optimal camera matrix for distortion correction.
newcameramtx = unround_image.process_new_camera_matrix(
    camera_matrix, dist_coeffs, image_size
)
# Unround the image using the calibration data.
img_unrounded = unround_image.unround(
    img=img,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    newcameramtx=newcameramtx,
)
```

## Image straightening
To correct perspective distortion in images, detected ArUco markers are used as reference points.
A transformation matrix is calculated to map the corners of the detected markers to predefined destination points, effectively straightening the image.
```python
# Define destination points
A1 = aruco.Aruco(53, 53, 1, 20)  # SO
B1 = aruco.Aruco(123, 53, 1, 22)  # SE
C1 = aruco.Aruco(53, 213, 1, 21)  # NO
D1 = aruco.Aruco(123, 213, 1, 23)  # NE

# Define source points (corners of the area to be straightened)
src_points = np.array(
    [[A2.x, A2.y], [B2.x, B2.y], [D2.x, D2.y], [C2.x, C2.y]],
    dtype=np.float32,
)

# Define destination points (where the corners should map to)
dst_points = np.array(
    [[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32
)

# Calculate the perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation to straighten the image
w, h = img.shape[:2]
print(f"  w : {w} h : {h} ")
straightened_img = cv2.warpPerspective(img, matrix, (w, h))
```

## ArUco tag detection

The detect_aruco.py file implements ArUco marker detection using OpenCV. Here's how it works:

1. Detector Initialization (init_aruco_detector()).
   Creates an ArUco detector with:
    - Dictionary: DICT_4X4_50 - supports 50 different 4x4 bit markers
    - Parameters: Default detection parameters for marker identification

2. Detection Pipeline (detect_aruco_in_img()).
The detection process:
   1. Convert to grayscale - Makes detection more reliable

   2. Detect markers - Uses detectMarkers() to find all ArUco tags in the image

   3. Extract information for each detected marker:
      - Center coordinates (cX, cY) - calculated as the mean of corner points
      - Angle correction - determines orientation:
        - Compares width vs height of the marker's bounding rectangle
        - Corrects the angle to a -180° to 180° range
      - Marker ID - unique identifier from the ArUco dictionary
Returns - List of Aruco objects containing position, ID, and angle data

## Camera configurations in Picamera2 

Here are the main differences between create_still_configuration() and create_preview_configuration() in Picamera2:

```python
create_still_configuration():
```

Optimized for single, high-quality image captures
Uses high resolution (often the full sensor resolution)
Slower frame rate (captures one frame at a time)
More processing/encoding overhead per frame
Better for photos where quality matters more than speed
Not suitable for continuous streaming

```python
create_preview_configuration():
```

Optimized for continuous video streaming/real-time preview
Lower resolution (typically 640x480 or similar)
Fast frame rate (25-30+ FPS)
Minimal latency between captures
Less processing overhead - frames update continuously
Ideal for live video feeds and continuous frame access