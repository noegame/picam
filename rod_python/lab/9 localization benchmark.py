"""
On enlève la partie de l'image en dehors du terrain pour voir si cela améliore le temps de détection.
"""

import cv2
import numpy as np
import time as t
import os
import glob
import aruco_initial_positions

# Créer un dictionnaire des positions attendues
# Clé: (id, x, y) où x,y sont les positions réelles attendues
EXPECTED_POSITIONS = {}
for entry in aruco_initial_positions.initial_position:
    possible_ids = entry[0]
    x, y = entry[2], entry[3]
    for tag_id in possible_ids:
        if tag_id not in EXPECTED_POSITIONS:
            EXPECTED_POSITIONS[tag_id] = []
        EXPECTED_POSITIONS[tag_id].append((x, y))

# Matrice intrinsèque (valeurs de calibration)
K = np.array(
    [
        [2.49362477e03, 0.00000000e00, 1.97718701e03],
        [0.00000000e00, 2.49311358e03, 2.03491176e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

# Coefficients de distorsion fisheye (k1, k2, k3, k4)
D = np.array([[-0.1203345, 0.06802544, -0.13779641, 0.08243704]])


def find_closest_expected_position(tag_id, detected_pos):
    """
    Trouve la position attendue la plus proche d'une position détectée.
    Retourne (expected_x, expected_y, distance) ou None si aucune position attendue.
    """
    if tag_id not in EXPECTED_POSITIONS:
        return None

    expected_positions = EXPECTED_POSITIONS[tag_id]
    min_distance = float("inf")
    closest_pos = None

    for exp_x, exp_y in expected_positions:
        distance = np.sqrt(
            (detected_pos[0] - exp_x) ** 2 + (detected_pos[1] - exp_y) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            closest_pos = (exp_x, exp_y)

    return (*closest_pos, min_distance) if closest_pos else None


def detect_aruco_tags_advanced(
    image_path,
    detector,
    output_path="output.png",
    verbose=True,
    mask=None,
    homography_inv=None,
):
    """
    Détection optimisée avec configuration unique: Échelle 1.25 × Sharpened × Agressif petits tags
    Filtre uniquement les IDs autorisés: 20, 21, 22, 23, 41, 36, 47
    Retourne la liste tags détectés avec leurs coordonnées réelles si homography_inv est fourni
    """

    t_start = t.time()
    timings = {}  # Dictionnaire pour stocker les temps de chaque étape

    # IDs autorisés
    ALLOWED_IDS = [20, 21, 22, 23, 41, 36, 47]

    # Charger l'image (PNG ou JPG)
    t0 = t.time()
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    timings["chargement"] = t.time() - t0

    if image is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return

    # Convertir en BGR si l'image a un canal alpha
    t0 = t.time()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    timings["conversion_couleur"] = t.time() - t0

    # mask
    if mask is not None:
        t0 = t.time()
        # Redimensionner le masque à la taille de l'image redimensionnée
        image = cv2.bitwise_and(image, image, mask=mask)
        timings["application_masque"] = t.time() - t0

    # Liste pour stocker tous les markers détectés
    all_detections = []

    if verbose:
        print("\n" + "=" * 60)
        print("Détection optimisée en cours...")
        print("Configuration: Échelle 1.25 × Sharpened × Agressif petits tags")
        print("=" * 60)

    # Prétraitement: Augmentation netteté
    t0 = t.time()
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    timings["sharpening"] = t.time() - t0

    # Redimensionnement à l'échelle 1.25
    t0 = t.time()
    scale = 1.5
    width = int(sharpened.shape[1] * scale)
    height = int(sharpened.shape[0] * scale)
    img_scaled = cv2.resize(sharpened, (width, height))
    timings["redimensionnement"] = t.time() - t0

    # Détection
    t0 = t.time()
    corners, ids, rejected = detector.detectMarkers(img_scaled)
    timings["detection"] = t.time() - t0

    rejected_count = 0

    t0 = t.time()
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

            # Calculer les coordonnées réelles si homographie fournie
            real_coords = None
            expected_pos = None
            error_x = None
            error_y = None
            error_distance = None

            if homography_inv is not None:
                # Undistort le point avant de le transformer
                point_distorted = np.array([[[center_x, center_y]]], dtype=np.float32)
                point_undistorted = cv2.fisheye.undistortPoints(
                    point_distorted, K, D, None, K
                )
                point_real = cv2.perspectiveTransform(point_undistorted, homography_inv)
                real_coords = (point_real[0][0][0], point_real[0][0][1])

                # Trouver la position attendue la plus proche
                closest = find_closest_expected_position(mid, real_coords)
                if closest:
                    expected_pos = (closest[0], closest[1])
                    error_x = real_coords[0] - expected_pos[0]
                    error_y = real_coords[1] - expected_pos[1]
                    error_distance = closest[2]

            all_detections.append(
                {
                    "id": mid,
                    "corners": corner,
                    "center": (center_x, center_y),
                    "real_coords": real_coords,
                    "expected_pos": expected_pos,
                    "error_x": error_x,
                    "error_y": error_y,
                    "error_distance": error_distance,
                }
            )
    timings["filtrage"] = t.time() - t0

    # Calculer le temps total avant l'affichage
    t_end = t.time()
    detection_time = t_end - t_start

    if verbose:
        print(f"\nIDs non autorisés rejetés: \t{rejected_count}")
        print(f"Tags valides détectés: \t\t{len(all_detections)}")
        print(f"Durée de la détection: \t\t{detection_time:.2f} secondes")

    # Afficher les résultats
    if verbose and len(all_detections) > 0:
        if homography_inv is not None:
            print(
                f"\n{'#':<6}{'ID':<6}{'Position Image':<20}{'Détectée (mm)':<18}{'Attendue (mm)':<18}{'Écart (mm)':<20}"
            )
            print("-" * 88)
        else:
            print(f"\n{'#':<6}{'ID':<10}{'Position Centre':<25}")
            print("-" * 41)
        # Trier par ID puis par position
        all_detections.sort(key=lambda x: (x["id"], x["center"][0], x["center"][1]))
        for idx, detection in enumerate(all_detections, 1):
            marker_id = detection["id"]
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            if homography_inv is not None and detection["real_coords"] is not None:
                real_x = int(detection["real_coords"][0])
                real_y = int(detection["real_coords"][1])

                if detection["expected_pos"] is not None:
                    exp_x = int(detection["expected_pos"][0])
                    exp_y = int(detection["expected_pos"][1])
                    err_x = int(detection["error_x"])
                    err_y = int(detection["error_y"])
                    print(
                        f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y}){'':>7}({exp_x},{exp_y}){'':>7}(Δx:{err_x:+4}, Δy:{err_y:+4})"
                    )
                else:
                    print(
                        f"{idx:<6}{marker_id:<6}({center_x},{center_y}){'':>6}({real_x},{real_y}){'':>7}(N/A){'':>12}(N/A)"
                    )
            else:
                print(f"{idx:<6}{marker_id:<10}({center_x}, {center_y})")
    elif verbose:
        print("Aucun tag valide détecté")

    # Créer l'image de sortie
    t0 = t.time()
    output_image = image.copy()

    if len(all_detections) > 0:
        # Dessiner tous les markers en une seule fois (plus efficace)
        all_corners = [
            detection["corners"].reshape(1, 4, 2).astype(np.float32)
            for detection in all_detections
        ]
        all_ids = np.array(
            [[detection["id"]] for detection in all_detections], dtype=np.int32
        )
        # cv2.aruco.drawDetectedMarkers(output_image, all_corners, all_ids)

        # Ajouter les textes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for detection in all_detections:
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            text = f"ID:{detection['id']}"
            pos = (center_x, center_y)

            # Contour noir puis texte vert
            cv2.putText(
                output_image, text, pos, font, font_scale, (0, 0, 0), thickness + 2
            )
            cv2.putText(
                output_image, text, pos, font, font_scale, (0, 255, 0), thickness
            )

            # Ajouter les coordonnées réelles si disponibles
            if detection["real_coords"] is not None:
                real_x = int(detection["real_coords"][0])
                real_y = int(detection["real_coords"][1])
                real_text = f"({real_x}, {real_y})mm"
                real_pos = (center_x + 50, center_y)

                # Contour noir puis texte cyan
                cv2.putText(
                    output_image,
                    real_text,
                    real_pos,
                    font,
                    font_scale * 0.8,
                    (0, 0, 0),
                    thickness + 2,
                )
                cv2.putText(
                    output_image,
                    real_text,
                    real_pos,
                    font,
                    font_scale * 0.8,
                    (255, 255, 0),
                    thickness,
                )

            # Ajouter les coordonnées attendues et l'écart
            if detection["expected_pos"] is not None:
                exp_x = int(detection["expected_pos"][0])
                exp_y = int(detection["expected_pos"][1])
                err_x = int(detection["error_x"])
                err_y = int(detection["error_y"])

                # Texte position attendue
                exp_text = f"Att: ({exp_x}, {exp_y})mm"
                exp_pos = (center_x + 50, center_y + 20)
                cv2.putText(
                    output_image,
                    exp_text,
                    exp_pos,
                    font,
                    font_scale * 0.8,
                    (0, 0, 0),
                    thickness + 2,
                )
                cv2.putText(
                    output_image,
                    exp_text,
                    exp_pos,
                    font,
                    font_scale * 0.8,
                    (255, 0, 255),  # Magenta pour position attendue
                    thickness,
                )

    # Ajouter compteur en haut de l'image
    counter_text = f"Tags detectes: {len(all_detections)}"
    counter_pos = (50, 100)
    cv2.putText(
        output_image,
        counter_text,
        counter_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 0),
        8,
    )
    cv2.putText(
        output_image,
        counter_text,
        counter_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 255, 0),
        5,
    )
    timings["annotation"] = t.time() - t0

    # Sauvegarder avec compression optimisée
    t0 = t.time()
    # Compression PNG rapide (niveau 1 au lieu de 3 par défaut)
    cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    timings["sauvegarde"] = t.time() - t0

    if verbose:
        print(f"\nImage annotée sauvegardée: {output_path}")

    # Retourner les statistiques
    return {
        "detections": all_detections,
        "rejected_count": rejected_count,
        "valid_count": len(all_detections),
        "detection_time": detection_time,
        "timings": timings,
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

    # OPTIMISATION: Créer le detector UNE SEULE FOIS
    print("\nInitialisation du détecteur ArUco...")
    t_init = t.time()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

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

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    print(f"Détecteur initialisé en {t.time() - t_init:.4f} secondes\n")

    # Statistiques globales
    total_rejected = 0
    total_valid = 0
    total_time = 0

    # Statistiques de temps par étape
    cumulative_timings = {}

    # Statistiques d'écart de localisation
    all_errors_x = []
    all_errors_y = []
    all_errors_distance = []

    # find mask et calculer l'homographie inverse
    mask, homography_inv = find_mask(detector, image_files[0], scale_y=1.1)

    # Traiter chaque image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"output_{image_name}")

        print(f"\n[{idx}/{total_images}] Traitement de: {image_name}")

        # Mesurer le temps total incluant tous les overheads
        t_image_start = t.time()

        # Détecter les tags (passer le detector en paramètre)
        result = detect_aruco_tags_advanced(
            image_path,
            detector,
            output_path,
            verbose=True,
            mask=mask,
            homography_inv=homography_inv,
        )

        t_image_end = t.time()
        real_image_time = t_image_end - t_image_start

        # Afficher le résumé pour cette image
        print(f"  → IDs non autorisés rejetés: \t{result['rejected_count']}")
        print(f"  → Tags valides détectés: \t{result['valid_count']}")
        print(f"  → Durée rapportée: \t\t{result['detection_time']:.3f}s")
        print(f"  → Durée réelle (mesurée): \t{real_image_time:.3f}s")
        print(
            f"  → Overhead non tracé: \t{real_image_time - result['detection_time']:.3f}s"
        )

        # Afficher les temps détaillés
        timings = result["timings"]
        print(f"     • Chargement: \t\t{timings['chargement']:.3f}s")
        print(f"     • Conversion couleur: \t{timings['conversion_couleur']:.3f}s")
        print(f"     • masque: \t\t{timings.get('application_masque', 0):.3f}s")
        print(f"     • Sharpening: \t\t{timings['sharpening']:.3f}s")
        print(f"     • Redimensionnement: \t{timings['redimensionnement']:.3f}s")
        print(f"     • Détection ArUco: \t{timings['detection']:.3f}s")
        print(f"     • Filtrage: \t\t{timings['filtrage']:.3f}s")
        print(f"     • Annotation: \t\t{timings['annotation']:.3f}s")
        print(f"     • Sauvegarde: \t\t{timings['sauvegarde']:.3f}s")

        # Accumuler les statistiques (utiliser le temps RÉEL mesuré)
        total_rejected += result["rejected_count"]
        total_valid += result["valid_count"]
        total_time += real_image_time  # CORRECTION: utiliser le temps réel mesuré

        # Accumuler les temps par étape
        for key, value in timings.items():
            cumulative_timings[key] = cumulative_timings.get(key, 0) + value

        # Collecter les erreurs de localisation
        for detection in result["detections"]:
            if detection["error_x"] is not None:
                all_errors_x.append(detection["error_x"])
                all_errors_y.append(detection["error_y"])
                all_errors_distance.append(detection["error_distance"])

    # Afficher les moyennes
    print("\n" + "=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    print(f"Nombre total d'images traitées: \t{total_images}")
    print(f"\nMoyenne IDs non autorisés rejetés: \t{total_rejected / total_images:.2f}")
    print(f"Moyenne Tags valides détectés: \t\t{total_valid / total_images:.2f}")
    print(f"\nDurée totale de traitement: \t\t{total_time:.2f} secondes")
    print(f"Durée totale d'execution du script: \t{t.time() - t_init:.2f} secondes")
    print(f"Moyenne Durée par image: \t\t{total_time / total_images:.3f} secondes")

    print("\n" + "-" * 80)
    print("ANALYSE DES TEMPS PAR ÉTAPE (moyennes)")
    print("-" * 80)
    for key in [
        "chargement",
        "conversion_couleur",
        "sharpening",
        "redimensionnement",
        "detection",
        "filtrage",
        "annotation",
        "sauvegarde",
    ]:
        avg_time = cumulative_timings.get(key, 0) / total_images
        percentage = (cumulative_timings.get(key, 0) / total_time) * 100
        print(f"{key.capitalize():<25} {avg_time:.3f}s\t({percentage:.1f}%)")

    # Afficher les statistiques d'écart de localisation
    if len(all_errors_x) > 0:
        print("\n" + "=" * 80)
        print("STATISTIQUES D'ÉCART DE LOCALISATION")
        print("=" * 80)
        print(f"Nombre de tags localisés avec position attendue: {len(all_errors_x)}")
        print(f"\nÉcart moyen en X: \t\t{np.mean(all_errors_x):.2f} mm")
        print(f"Écart moyen en Y: \t\t{np.mean(all_errors_y):.2f} mm")
        print(f"Écart absolu moyen en X: \t{np.mean(np.abs(all_errors_x)):.2f} mm")
        print(f"Écart absolu moyen en Y: \t{np.mean(np.abs(all_errors_y)):.2f} mm")
        print(f"Distance euclidienne moyenne: \t{np.mean(all_errors_distance):.2f} mm")
        print(f"\nÉcart max en X: \t\t{np.max(np.abs(all_errors_x)):.2f} mm")
        print(f"Écart max en Y: \t\t{np.max(np.abs(all_errors_y)):.2f} mm")
        print(f"Distance max: \t\t\t{np.max(all_errors_distance):.2f} mm")
        print(f"\nÉcart min en X: \t\t{np.min(np.abs(all_errors_x)):.2f} mm")
        print(f"Écart min en Y: \t\t{np.min(np.abs(all_errors_y)):.2f} mm")
        print(f"Distance min: \t\t\t{np.min(all_errors_distance):.2f} mm")
        print(f"\nÉcart-type en X: \t\t{np.std(all_errors_x):.2f} mm")
        print(f"Écart-type en Y: \t\t{np.std(all_errors_y):.2f} mm")
    else:
        print("\n" + "=" * 80)
        print("Aucune position attendue trouvée pour les tags détectés")

    print("=" * 80)


def find_mask(
    detector,
    img_path,
    scale_y=1.0,
):
    """
    Génère un masque binaire correspondant au terrain.
    Permet d'étendre ou réduire la hauteur du masque via scale_y.

    - scale_y > 1.0 : masque plus haut
    - scale_y < 1.0 : masque plus bas
    """

    # Charger image de référence
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de charger {img_path}")

    h, w = img.shape[:2]

    # Détection des tags fixes
    result = detect_aruco_tags_advanced(
        img_path,
        detector,
        verbose=False,
        mask=None,
    )

    if result is None:
        raise RuntimeError("Aucune détection ArUco")

    # Positions IRL connues des tags
    tag_irl = {
        20: (600, 600),
        21: (600, 2400),
        22: (1400, 600),
        23: (1400, 2400),
    }

    src_pts = []
    dst_pts = []

    for det in result["detections"]:
        tid = det["id"]
        if tid in tag_irl:
            src_pts.append(tag_irl[tid])
            dst_pts.append(det["center"])

    if len(src_pts) != 4:
        raise RuntimeError(f"Tags fixes détectés: {len(src_pts)}/4")

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Undistort les points image avant de calculer l'homographie
    dst_pts_reshaped = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    dst_pts_undistorted = cv2.fisheye.undistortPoints(dst_pts_reshaped, K, D, None, K)
    dst_pts = dst_pts_undistorted.reshape(-1, 2)

    # Homographie IRL → image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise RuntimeError("Homographie non calculable")

    # Calculer l'homographie inverse (image → IRL)
    H_inv = np.linalg.inv(H)

    # Coins IRL du terrain
    terrain_irl = np.array(
        [
            [0, 0],
            [2000, 0],
            [2000, 3000],
            [0, 3000],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # Projection image
    terrain_img = cv2.perspectiveTransform(terrain_irl, H)
    terrain_img = terrain_img.reshape(-1, 2)

    # ----- EXTENSION VERTICALE DU MASQUE -----
    if scale_y != 1.0:
        center_y = np.mean(terrain_img[:, 1])
        terrain_img[:, 1] = center_y + (terrain_img[:, 1] - center_y) * scale_y

    # Clip dans l'image
    terrain_img[:, 0] = np.clip(terrain_img[:, 0], 0, w - 1)
    terrain_img[:, 1] = np.clip(terrain_img[:, 1], 0, h - 1)

    terrain_img = terrain_img.astype(np.int32)

    # Création du masque (TAILLE IDENTIQUE À L'IMAGE)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [terrain_img], 255)

    return mask, H_inv


# Utilisation
if __name__ == "__main__":
    input_folder = "2026-01-16-playground-ready"
    output_folder = "output9"

    process_folder(input_folder, output_folder)
