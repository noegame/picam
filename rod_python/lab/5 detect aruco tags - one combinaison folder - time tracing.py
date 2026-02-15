import cv2
import numpy as np
import time as t
import os
import glob


def detect_aruco_tags_advanced(
    image_path, detector, output_path="output.png", verbose=True
):
    """
    Détection optimisée avec configuration unique: Échelle 1.25 × Sharpened × Agressif petits tags
    Filtre uniquement les IDs autorisés: 20, 21, 22, 23, 41, 36, 47
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

            all_detections.append(
                {
                    "id": mid,
                    "corners": corner,
                    "center": (center_x, center_y),
                }
            )
    timings["filtrage"] = t.time() - t0

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
    timings["annotation"] = t.time() - t0

    # Sauvegarder avec compression optimisée
    t0 = t.time()
    # Compression PNG rapide (niveau 1 au lieu de 3 par défaut)
    cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    timings["sauvegarde"] = t.time() - t0

    # Calculer le temps total APRÈS toutes les opérations
    t_end = t.time()
    detection_time = t_end - t_start

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

    # Traiter chaque image
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"output_{image_name}")

        print(f"\n[{idx}/{total_images}] Traitement de: {image_name}")

        # Mesurer le temps total incluant tous les overheads
        t_image_start = t.time()

        # Détecter les tags (passer le detector en paramètre)
        result = detect_aruco_tags_advanced(
            image_path, detector, output_path, verbose=False
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
    print("=" * 80)


# Utilisation
if __name__ == "__main__":
    input_folder = "2026-01-16-playground-ready"
    output_folder = "output"

    process_folder(input_folder, output_folder)
