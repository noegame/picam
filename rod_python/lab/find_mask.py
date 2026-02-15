import cv2
import numpy as np
import os
import glob


def detect_aruco_tags_advanced(
    image_path, detector, output_path="output.png", verbose=True, mask=None
):
    """
    Détection optimisée avec configuration unique: Échelle 1.25 × Sharpened × Agressif petits tags
    Filtre uniquement les IDs autorisés: 20, 21, 22, 23, 41, 36, 47
    Retourne la liste tags détectés
    """

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

    # Sauvegarder l'image originale pour debug visuel
    original_image = image.copy()

    # mask
    if mask is not None:
        # Redimensionner le masque à la taille de l'image redimensionnée
        image = cv2.bitwise_and(image, image, mask=mask)

        # Debug visuel: sauvegarder l'image avec le masque appliqué
        mask_debug_path = output_path.replace(".png", "_masked.png").replace(
            ".jpg", "_masked.jpg"
        )
        cv2.imwrite(mask_debug_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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

    # Détection
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

    if verbose:
        print(f"\nIDs non autorisés rejetés: \t{rejected_count}")
        print(f"Tags valides détectés: \t\t{len(all_detections)}")

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
        # Dessiner tous les markers en une seule fois (plus efficace)
        all_corners = [
            detection["corners"].reshape(1, 4, 2).astype(np.float32)
            for detection in all_detections
        ]
        all_ids = np.array(
            [[detection["id"]] for detection in all_detections], dtype=np.int32
        )
        cv2.aruco.drawDetectedMarkers(output_image, all_corners, all_ids)

        # Ajouter les textes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        for detection in all_detections:
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            text = f"ID:{detection['id']}"
            pos = (center_x - 40, center_y + 10)

            # Contour noir puis texte vert
            cv2.putText(
                output_image, text, pos, font, font_scale, (0, 0, 0), thickness + 2
            )
            cv2.putText(
                output_image, text, pos, font, font_scale, (0, 255, 0), thickness
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

    # Sauvegarder avec compression optimisée
    # Compression PNG rapide (niveau 1 au lieu de 3 par défaut)
    cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    if verbose:
        print(f"\nImage annotée sauvegardée: {output_path}")

    # Retourner les statistiques et les détections
    stats = {
        "detections": all_detections,
        "rejected_count": rejected_count,
        "valid_count": len(all_detections),
    }
    return stats


def find_mask2(
    detector,
    img_path,
    scale=1.0,
    tag_ids=(20, 21, 22, 23),
):
    """
    Détermine un masque binaire pour conserver uniquement la zone du terrain.
    Fonctionne entièrement en espace image fisheye (pas de correction optique).

    - Détecte les tags fixes (coins du terrain)
    - Construit un polygone image
    - Génère un masque binaire

    Retourne: masque uint8 (0 / 255)
    """

    # Charger l'image de référence
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image {img_path}")

    h, w = img.shape[:2]

    # Détecter les tags (sans masque)
    stats = detect_aruco_tags_advanced(
        img_path,
        detector,
        verbose=False,
        mask=None,
    )

    if stats is None:
        raise RuntimeError("Aucune détection ArUco")

    # Récupérer les centres des tags de référence
    centers = {}
    for det in stats["detections"]:
        if det["id"] in tag_ids:
            centers[det["id"]] = det["center"]

    if len(centers) != 4:
        raise RuntimeError(
            f"Tags de référence détectés: {len(centers)}/4 ({list(centers.keys())})"
        )

    # Ordre des coins (important pour fillPoly)
    # Adapter si ton terrain est orienté différemment
    # ici: haut-gauche, bas-gauche, bas-droite, haut-droite
    ordered_ids = [20, 21, 23, 22]

    polygon = np.array(
        [centers[tag_id] for tag_id in ordered_ids],
        dtype=np.int32,
    )

    # Créer le masque
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # Redimensionnement optionnel
    if scale != 1.0:
        mask = cv2.resize(
            mask,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


def find_mask3(
    detector,
    img_path,
    scale=1.0,
    scale_y=None,
):
    """
    Génère un masque binaire correspondant à la totalité du terrain,
    même si les tags fixes sont à l'intérieur du terrain.

    Tout est fait en espace image fisheye (pas de correction optique).
    """

    # Charger image de référence
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Impossible de charger {img_path}")

    h, w = img.shape[:2]

    # Détecter les tags fixes
    stats = detect_aruco_tags_advanced(
        img_path,
        detector,
        verbose=False,
        mask=None,
    )

    if stats is None:
        raise RuntimeError("Aucune détection ArUco")

    # Positions IRL connues des tags (repère terrain)
    tag_irl = {
        20: (600, 600),
        21: (600, 2400),
        22: (1400, 600),
        23: (1400, 2400),
    }

    # Récupération des correspondances IRL ↔ image
    src_pts = []
    dst_pts = []

    for det in stats["detections"]:
        tid = det["id"]
        if tid in tag_irl:
            src_pts.append(tag_irl[tid])  # IRL
            dst_pts.append(det["center"])  # IMAGE

    if len(src_pts) != 4:
        raise RuntimeError(f"Tags fixes détectés: {len(src_pts)}/4")

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Homographie IRL → image
    H, _ = cv2.findHomography(src_pts, dst_pts)

    if H is None:
        raise RuntimeError("Homographie non calculable")

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

    # Projection des coins dans l'image
    terrain_img = cv2.perspectiveTransform(terrain_irl, H)
    terrain_img = terrain_img.reshape(-1, 2).astype(np.int32)

    # Création du masque
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [terrain_img], 255)

    # Redimensionnement optionnel
    if scale != 1.0:
        mask = cv2.resize(
            mask,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    if scale_y is not None:
        mask = cv2.resize(
            mask,
            (mask.shape[1], int(h * scale_y)),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


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
    stats = detect_aruco_tags_advanced(
        img_path,
        detector,
        verbose=False,
        mask=None,
    )

    if stats is None:
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

    for det in stats["detections"]:
        tid = det["id"]
        if tid in tag_irl:
            src_pts.append(tag_irl[tid])
            dst_pts.append(det["center"])

    if len(src_pts) != 4:
        raise RuntimeError(f"Tags fixes détectés: {len(src_pts)}/4")

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Homographie IRL → image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise RuntimeError("Homographie non calculable")

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

    return mask


def get_aruco_detector():
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
    return detector


# Exemple d'utilisation
if __name__ == "__main__":
    detector = get_aruco_detector()

    # Dossier d'images à traiter
    input_folder = "2026-01-16-playground-ready"
    output_folder = "output_optimized"
    os.makedirs(output_folder, exist_ok=True)

    image_files = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(
        os.path.join(input_folder, "*.jpg")
    )

    total_images = len(image_files)
    total_rejected = 0
    total_valid = 0

    # Créer le masque à partir de la première image (ou d'une image de référence)
    mask = None
    if len(image_files) > 0:
        try:
            print("Création du masque à partir de la première image...")
            mask = find_mask(detector, image_files[0], scale_y=1.1)
            print(f"Masque créé avec succès: {mask.shape}")
        except Exception as e:
            print(f"Impossible de créer le masque: {e}")
            print("Poursuite sans masque...")

    for img_file in image_files:
        base_name = os.path.basename(img_file)
        output_path = os.path.join(output_folder, f"output_{base_name}")

        stats = detect_aruco_tags_advanced(
            img_file,
            detector,
            output_path=output_path,
            verbose=False,
            mask=mask,
        )

        if stats is not None:
            valid_count = stats["valid_count"]
            rejected_count = stats["rejected_count"]
        else:
            valid_count = 0
            rejected_count = 0

        total_rejected += rejected_count
        total_valid += valid_count

    print("\n" + "=" * 80)
    print("Statistiques globales:")
    print("-" * 80)
    print(f"Total images traitées: \t\t{total_images}")
    print(f"Total IDs non autorisés rejetés: \t{total_rejected}")
    print(f"Total Tags valides détectés: \t\t{total_valid}")
    print("-" * 80)
