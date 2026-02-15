import cv2
import numpy as np
import time as t
import os
import glob


def detect_aruco_tags_advanced(image_path, output_path="output.png", verbose=True):
    """
    Détection optimisée avec configuration unique: Échelle 1.25 × Sharpened × Agressif petits tags
    Filtre uniquement les IDs autorisés: 20, 21, 22, 23, 41, 36, 47
    """

    t_start = t.time()

    # IDs autorisés
    ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]

    # Charger l'image (PNG ou JPG)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return

    # Convertir en BGR si l'image a un canal alpha
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Dictionnaire ArUco 4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Liste pour stocker tous les markers détectés
    all_detections = []

    if verbose:
        print("\n" + "=" * 60)
        print("Détection optimisée en cours...")
        print("Configuration: Échelle 1.25 × Sharpened × Agressif petits tags")
        print("=" * 60)

    # Prétraitement: Augmentation netteté
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)

    # Redimensionnement à l'échelle 1.25
    scale = 1.5
    width = int(sharpened.shape[1] * scale)
    height = int(sharpened.shape[0] * scale)
    img_scaled = cv2.resize(sharpened, (width, height))

    # Configuration: Très agressif pour petits tags
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.minDistanceToBorder = 0
    params.minOtsuStdDev = 2.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.15

    # Détection
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(img_scaled)

    rejected_count = 0

    if ids is not None:
        for i, marker_id in enumerate(ids):
            mid = marker_id[0]

            # FILTRAGE: Ignorer les IDs non autorisés
            if mid not in ALLOWED_IDS:
                rejected_count += 1
                continue

            corner = corners[i][0].copy()

            # Rescale corners
            corner = corner / scale

            # Calculer le centre
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])

            all_detections.append(
                {
                    "id": mid,
                    "corners": corner,
                    "center": (center_x, center_y),
                }
            )

    t_end = t.time()
    detection_time = t_end - t_start

    if verbose:
        print(f"\nIDs non autorisés rejetés: \t{rejected_count}")
        print(f"Tags valides détectés: \t\t{len(all_detections)}")
        print(f"Durée de la détection: \t\t{detection_time:.2f} secondes")

    # Afficher les résultats
    if verbose and len(all_detections) > 0:
        print(f"\n{'#':<6}{'ID':<10}{'Position Centre':<25}")
        print("-" * 41)
        # Trier par ID puis par position
        all_detections.sort(key=lambda x: (x["id"], x["center"][0], x["center"][1]))
        for idx, detection in enumerate(all_detections, 1):
            marker_id = detection["id"]
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            print(f"{idx:<6}{marker_id:<10}({center_x}, {center_y})")
    elif verbose:
        print("Aucun tag valide détecté")

    # Créer l'image de sortie
    output_image = image.copy()

    if len(all_detections) > 0:
        # Dessiner tous les markers
        for detection in all_detections:
            corner = detection["corners"].reshape(1, 4, 2).astype(np.float32)
            mid = np.array([[detection["id"]]], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(output_image, [corner], mid)

            # Ajouter texte
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])

            text = f"ID:{detection['id']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3

            cv2.putText(
                output_image,
                text,
                (center_x - 40, center_y + 10),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2,
            )
            cv2.putText(
                output_image,
                text,
                (center_x - 40, center_y + 10),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
            )

    # Ajouter compteur en haut de l'image
    cv2.putText(
        output_image,
        f"Tags detectes: {len(all_detections)}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 0),
        8,
    )
    cv2.putText(
        output_image,
        f"Tags detectes: {len(all_detections)}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 255, 0),
        5,
    )

    # Sauvegarder en PNG
    cv2.imwrite(output_path, output_image)
    if verbose:
        print(f"\nImage annotée sauvegardée: {output_path}")

    # Retourner les statistiques
    return {
        "detections": all_detections,
        "rejected_count": rejected_count,
        "valid_count": len(all_detections),
        "detection_time": detection_time,
    }


def process_folder(input_folder, output_folder="output"):
    """
    Traite toutes les images d'un dossier et affiche les statistiques
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Récupérer toutes les images (PNG et JPG)
    image_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"Aucune image trouvée dans le dossier: {input_folder}")
        return

    image_files.sort()
    total_images = len(image_files)

    print("=" * 80)
    print(f"TRAITEMENT DE {total_images} IMAGES")
    print("=" * 80)

    # Statistiques globales
    total_rejected = 0
    total_valid = 0
    total_time = 0

    # Traiter chaque image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"output_{image_name}")

        print(f"\n[{idx}/{total_images}] Traitement de: {image_name}")

        # Détecter les tags
        result = detect_aruco_tags_advanced(image_path, output_path, verbose=False)

        # Afficher le résumé pour cette image
        print(f"  → IDs non autorisés rejetés: \t{result['rejected_count']}")
        print(f"  → Tags valides détectés: \t{result['valid_count']}")
        print(f"  → Durée de la détection: \t{result['detection_time']:.2f} secondes")

        # Accumuler les statistiques
        total_rejected += result["rejected_count"]
        total_valid += result["valid_count"]
        total_time += result["detection_time"]

    # Afficher les moyennes
    print("\n" + "=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    print(f"Nombre total d'images traitées: \t{total_images}")
    print(f"\nMoyenne IDs non autorisés rejetés: \t{total_rejected / total_images:.2f}")
    print(f"Moyenne Tags valides détectés: \t\t{total_valid / total_images:.2f}")
    print(f"Moyenne Durée de détection: \t\t{total_time / total_images:.2f} secondes")
    print(f"\nDurée totale de traitement: \t\t{total_time:.2f} secondes")
    print("=" * 80)


# Utilisation
if __name__ == "__main__":
    input_folder = "2026-01-16-playground-ready"
    output_folder = "output"

    process_folder(input_folder, output_folder)
