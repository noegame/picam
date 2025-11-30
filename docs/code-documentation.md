# Quelques bout de code expliqué
[readme](../README.md)

## Distorsion de l'image
```python
# Importation des coefficients de distorsion (calibration)
calibration_file = os.path.join(script_dir, "camera_calibration.npz")
data = np.load(calibration_file)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# Correction de la distorsion
img_undistorted = undistort(img, camera_matrix, dist_coeffs)
```

calibration_file : chemin vers le fichier de calibration de la caméra, isssu de la calibration réalisé avec [calibrate_camera.py](calibrate_camera.py).

## Redressement de l'image
```python
src_points = np.array([[A2.x, A2.y], [B2.x, B2.y], [D2.x, D2.y], [C2.x, C2.y]], dtype=np.float32)
dst_points = np.array([[A1.x, A1.y], [B1.x, B1.y], [D1.x, D1.y], [C1.x, C1.y]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
h, w = img_distorted.shape[:2]
transformed_img = cv2.warpPerspective(img_distorted, matrix, (h, w))
```
src_points : coordonnées des points dans l'image distordue (points détectés sur le tag ArUco).
dst_points : coordonnées des points dans l'image redressée (points attendus sur le tag ArUco).

## Détection des tags ArUco
