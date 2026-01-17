import cv2
import numpy as np

# =============================
# CHARGEMENT IMAGE
# =============================

image_path = "image_phone.jpg"
img = cv2.imread(image_path)

if img is None:
    raise IOError("Impossible de charger l'image")

# =============================
# PRÉTRAITEMENT
# =============================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Amélioration du contraste (utile avec smartphone)
gray = cv2.equalizeHist(gray)

# =============================
# DÉTECTION ARUCO
# =============================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

params = cv2.aruco.DetectorParameters()

# Paramètres assouplis pour 4x4
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 55
params.adaptiveThreshWinSizeStep = 4

params.minMarkerPerimeterRate = 0.02
params.maxMarkerPerimeterRate = 4.0

params.polygonalApproxAccuracyRate = 0.08
params.minCornerDistanceRate = 0.01

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

corners, ids, rejected = detector.detectMarkers(gray)

# =============================
# AFFICHAGE
# =============================

output = img.copy()

if ids is not None:
    # Draw markers with thicker borders
    for i, corner in enumerate(corners):
        # Draw thick green border around marker
        pts = corner[0].astype(int)
        cv2.polylines(output, [pts], True, (0, 0, 255), 8)

        # Get the center of the marker
        center = corner[0].mean(axis=0).astype(int)
        # Draw ID with larger font
        cv2.putText(
            output,
            str(ids[i][0]),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 0, 0),
            6,
        )

    print("IDs détectés :", ids.flatten())
else:
    print("Aucun tag détecté")

# Affichage des candidats rejetés (diagnostic)
# for c in rejected:
#    cv2.polylines(output, [c.astype(int)], True, (0, 0, 255), 1)

# get img dimensions
height, width = img.shape[:2]
scale = 0.25
output = cv2.resize(output, (int(width * scale), int(height * scale)))
cv2.imshow("Aruco detection - smartphone", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("detection_smartphone_output.jpg", output)
