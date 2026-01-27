import cv2
import numpy as np
import time as t


def detect_aruco_tags_advanced(image_path, output_path="output.png"):
    """
    Détection avancée avec prétraitement et multi-résolution
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

    # Liste pour stocker tous les markers détectés (sans fusion par ID)
    all_detections = []

    # STRATÉGIE 1: Détection avec plusieurs configurations de paramètres
    param_configs = []

    # Config 1: Très agressif pour petits tags
    params1 = cv2.aruco.DetectorParameters()
    params1.adaptiveThreshWinSizeMin = 3
    params1.adaptiveThreshWinSizeMax = 53
    params1.adaptiveThreshWinSizeStep = 4
    params1.minMarkerPerimeterRate = 0.01
    params1.maxMarkerPerimeterRate = 4.0
    params1.polygonalApproxAccuracyRate = 0.05
    params1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params1.cornerRefinementWinSize = 5
    params1.cornerRefinementMaxIterations = 50
    params1.minDistanceToBorder = 0
    params1.minOtsuStdDev = 2.0
    params1.perspectiveRemoveIgnoredMarginPerCell = 0.15
    param_configs.append(("Agressif petits tags", params1))

    # Config 2: Pour tags moyens avec perspective déformée
    params2 = cv2.aruco.DetectorParameters()
    params2.adaptiveThreshWinSizeMin = 5
    params2.adaptiveThreshWinSizeMax = 25
    params2.adaptiveThreshWinSizeStep = 5
    params2.minMarkerPerimeterRate = 0.03
    params2.maxMarkerPerimeterRate = 4.0
    params2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params2.perspectiveRemoveIgnoredMarginPerCell = 0.2
    params2.minOtsuStdDev = 3.0
    param_configs.append(("Tags moyens", params2))

    # Config 3: Pour tags avec flou
    params3 = cv2.aruco.DetectorParameters()
    params3.adaptiveThreshWinSizeMin = 7
    params3.adaptiveThreshWinSizeMax = 35
    params3.minMarkerPerimeterRate = 0.02
    params3.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params3.minOtsuStdDev = 1.5
    param_configs.append(("Tags flous", params3))

    # STRATÉGIE 2: Prétraitement de l'image avec différentes méthodes
    preprocessed_images = []

    # Image originale
    preprocessed_images.append(("Original", image))

    # Augmentation du contraste (CLAHE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    preprocessed_images.append(("CLAHE", equalized_bgr))

    # Débruitage
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    preprocessed_images.append(("Denoised", denoised))

    # Augmentation netteté
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    preprocessed_images.append(("Sharpened", sharpened))

    # STRATÉGIE 3: Détection multi-échelle
    scales = [1.0, 0.75, 1.25]

    print("\n" + "=" * 60)
    print("Détection multi-stratégies en cours...")
    print("=" * 60)

    detection_count = 0
    rejected_count = 0

    # Statistiques par combinaison
    combination_stats = []

    # Combiner toutes les stratégies
    for scale in scales:
        for img_name, img_preprocessed in preprocessed_images:
            # Redimensionner si nécessaire
            if scale != 1.0:
                width = int(img_preprocessed.shape[1] * scale)
                height = int(img_preprocessed.shape[0] * scale)
                img_scaled = cv2.resize(img_preprocessed, (width, height))
            else:
                img_scaled = img_preprocessed

            for config_name, params in param_configs:
                combo_start = t.time()
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, rejected = detector.detectMarkers(img_scaled)
                combo_time = t.time() - combo_start

                combo_name = f"Échelle {scale:.2f} × {img_name} × {config_name}"
                tags_found = 0
                tags_rejected = 0
                new_tags_added = 0

                if ids is not None:
                    detection_count += 1
                    tags_found = len(ids)

                    for i, marker_id in enumerate(ids):
                        mid = marker_id[0]

                        # FILTRAGE: Ignorer les IDs non autorisés
                        if mid not in ALLOWED_IDS:
                            rejected_count += 1
                            tags_rejected += 1
                            continue

                        corner = corners[i][0].copy()

                        # Rescale corners si nécessaire
                        if scale != 1.0:
                            corner = corner / scale

                        # Calculer le centre
                        center_x = np.mean(corner[:, 0])
                        center_y = np.mean(corner[:, 1])

                        # Vérifier si ce tag n'est pas déjà détecté (même position)
                        is_duplicate = False
                        for existing in all_detections:
                            # Si même ID ET position proche (distance < 50 pixels), c'est un doublon
                            dist = np.sqrt(
                                (existing["center"][0] - center_x) ** 2
                                + (existing["center"][1] - center_y) ** 2
                            )
                            if existing["id"] == mid and dist < 50:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            all_detections.append(
                                {
                                    "id": mid,
                                    "corners": corner,
                                    "center": (center_x, center_y),
                                }
                            )
                            new_tags_added += 1

                # Enregistrer les stats de cette combinaison
                combination_stats.append(
                    {
                        "name": combo_name,
                        "tags_found": tags_found,
                        "tags_rejected": tags_rejected,
                        "new_tags": new_tags_added,
                        "time": combo_time,
                    }
                )

    t_end = t.time()
    detection_time = t_end - t_start

    print(f"\nDétections effectuées: \t\t{detection_count}")
    print(f"IDs non autorisés rejetés: \t{rejected_count}")
    print(f"Tags valides détectés: \t\t{len(all_detections)}")
    print(f"durée de la détection: \t\t{detection_time:.2f} secondes")

    # Afficher les statistiques par combinaison
    print("\n" + "=" * 100)
    print("ANALYSE DES PERFORMANCES PAR COMBINAISON")
    print("=" * 100)

    # Trier par nombre de nouveaux tags (décroissant)
    combination_stats.sort(key=lambda x: x["new_tags"], reverse=True)

    print(
        f"\n{'Combinaison':<60}{'Trouvés':<10}{'Rejetés':<10}{'Nouveaux':<10}{'Temps (s)':<10}"
    )
    print("-" * 100)

    for stat in combination_stats:
        print(
            f"{stat['name']:<60}{stat['tags_found']:<10}{stat['tags_rejected']:<10}{stat['new_tags']:<10}{stat['time']:<10.3f}"
        )

    # Enregistrer les statistiques dans un fichier csv
    with open("output/aruco_detection_stats.csv", "w") as f:
        f.write("Combinaison,Trouvés,Rejetés,Nouveaux,Temps (s)\n")
        for stat in combination_stats:
            f.write(
                f"{stat['name']},{stat['tags_found']},{stat['tags_rejected']},{stat['new_tags']},{stat['time']:.3f}\n"
            )

    # Résumé
    useful_combos = [s for s in combination_stats if s["new_tags"] > 0]
    useless_combos = [s for s in combination_stats if s["new_tags"] == 0]

    print(f"\n{'='*100}")
    print(f"RÉSUMÉ:")
    print(
        f"  • Combinaisons utiles (apportent de nouveaux tags): {len(useful_combos)}/{len(combination_stats)}"
    )
    print(
        f"  • Combinaisons inutiles (doublons uniquement): {len(useless_combos)}/{len(combination_stats)}"
    )
    print(f"  • Temps total: {detection_time:.2f}s")
    print(
        f"  • Temps moyen par combinaison: {detection_time/len(combination_stats):.3f}s"
    )

    if len(useful_combos) > 0:
        total_time_useful = sum(s["time"] for s in useful_combos)
        print(f"  • Temps cumulé des combinaisons utiles: {total_time_useful:.2f}s")

    if len(useless_combos) > 0:
        total_time_useless = sum(s["time"] for s in useless_combos)
        print(
            f"  • Temps gaspillé sur combinaisons inutiles: {total_time_useless:.2f}s ({total_time_useless/detection_time*100:.1f}%)"
        )

    print(f"{'='*100}\n")

    # Afficher les résultats
    if len(all_detections) > 0:
        print(f"\n{'#':<6}{'ID':<10}{'Position Centre':<25}")
        print("-" * 41)
        # Trier par ID puis par position
        all_detections.sort(key=lambda x: (x["id"], x["center"][0], x["center"][1]))
        for idx, detection in enumerate(all_detections, 1):
            marker_id = detection["id"]
            center_x = int(detection["center"][0])
            center_y = int(detection["center"][1])
            print(f"{idx:<6}{marker_id:<10}({center_x}, {center_y})")
    else:
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
    print(f"\nImage annotée sauvegardée: {output_path}")

    return all_detections


# Utilisation
if __name__ == "__main__":
    input_image = "data/1.jpg"  # Accepte PNG et JPG
    output_image = "output/output.png"

    detections = detect_aruco_tags_advanced(input_image, output_image)
