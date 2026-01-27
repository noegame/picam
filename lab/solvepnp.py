import cv2
import numpy as np

# -----------------------------
# Paramètres caméra (EXEMPLE)
# À remplacer par ta vraie calibration
# -----------------------------
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # supposé sans distorsion pour l'exemple

# -----------------------------
# Paramètres du tag ArUco
# -----------------------------
TAG_SIZE = 0.04  # taille du tag en mètres

# Coordonnées 3D des coins du tag (repère du tag)
object_points = np.array(
    [
        [-TAG_SIZE / 2, TAG_SIZE / 2, 0],
        [TAG_SIZE / 2, TAG_SIZE / 2, 0],
        [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
        [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
    ],
    dtype=np.float32,
)

# -----------------------------
# Chargement image
# -----------------------------
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Détection ArUco
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if ids is None:
    print("Aucun tag détecté")
    exit()

# -----------------------------
# Traitement du premier tag détecté
# -----------------------------
image_points = corners[0].reshape(4, 2).astype(np.float32)

# -----------------------------
# SolvePnP
# -----------------------------
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs
)

if not success:
    print("solvePnP a échoué")
    exit()

# -----------------------------
# Résultats
# -----------------------------
print("Vecteur rotation (rvec) :\n", rvec)
print("Vecteur translation (tvec) :\n", tvec)
print("Position du tag par rapport à la caméra (mètres) :")
print(f"X = {tvec[0][0]:.3f} m")
print(f"Y = {tvec[1][0]:.3f} m")
print(f"Z = {tvec[2][0]:.3f} m")

# -----------------------------
# Visualisation (axes)
# -----------------------------
cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
cv2.imshow("Pose ArUco", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
