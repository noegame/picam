#!/usr/bin/env python3

"""
example_usage_aruco_detection_pipeline.py

Exemple d'utilisation de la fonction centralisée process_image_for_aruco_detection()
pour détecter des marqueurs ArUco dans une image.
"""

import cv2
from pathlib import Path
from vision_python.src.img_processing import processing_pipeline as pipeline
from vision_python.src.img_processing import unround_img
from vision_python.src.img_processing import detect_aruco
from vision_python.src.aruco import aruco
from vision_python.config import config


def main():
    """
    Exemple simple d'utilisation du pipeline de détection ArUco.
    """

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------

    # Charger les paramètres depuis la configuration
    img_params = config.get_image_processing_params()

    # Récupérer les marqueurs fixes et coins du terrain depuis config
    FIXED_MARKERS = config.get_fixed_aruco_markers()
    PLAYGROUND_CORNERS = config.get_playground_corners()

    # ---------------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------------

    # Charger les paramètres de calibration de la caméra
    camera_matrix, dist_coeffs, newcameramtx, roi = (
        config.get_camera_calibration_matrices()
    )
    
    # Créer le détecteur ArUco
    aruco_detector = detect_aruco.create_aruco_detector()
    # ---------------------------------------------------------------------------

    # Exemple : charger une image de test
    img_path = (
        config.get_camera_directory() / "2026-01-09-playground-ready" / "image_001.jpg"
    )

    if not img_path.exists():
        print(f"❌ Image non trouvée : {img_path}")
        print("Veuillez spécifier un chemin d'image valide")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Impossible de charger l'image depuis {img_path}")

    print(f"✓ Image chargée : {img_path}")

    # ---------------------------------------------------------------------------
    # Traitement avec le pipeline centralisé
    # ---------------------------------------------------------------------------

    print("\n🔍 Traitement de l'image...")

    detected_markers, final_img, perspective_matrix, metadata = (
        pipeline.process_image_for_aruco_detection(
            # Image et détecteur
            img=img,
            aruco_detector=aruco_detector,
            # Calibration de la caméra
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            newcameramtx=newcameramtx,
            # Marqueurs de référence et terrain
            fixed_markers=FIXED_MARKERS,
            playground_corners=PLAYGROUND_CORNERS,
            # Étapes de prétraitement (depuis config)
            use_unround=img_params["use_unround"],
            use_clahe=img_params["use_clahe"],
            use_thresholding=img_params["use_binarization"],
            sharpen_alpha=img_params["sharpen_alpha"],
            sharpen_beta=img_params["sharpen_beta"],
            sharpen_gamma=img_params["sharpen_gamma"],
            # Options avancées
            use_mask_playground=True,  # Masquer la zone hors terrain
            use_straighten_image=True,  # Redresser l'image
            apply_contrast_boost=True,  # Boost de contraste avant détection
            contrast_alpha=1.3,
            # Débogage
            save_debug_images=True,
            debug_dir=config.get_debug_directory() / "example",
            base_name=img_path.stem,
        )
    )

    # ---------------------------------------------------------------------------
    # Affichage des résultats
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RÉSULTATS DE DÉTECTION")
    print("=" * 80)

    # Métadonnées du traitement
    print("\n📊 Métadonnées du traitement :")
    if metadata["masking_applied"]:
        print("  ✓ Masquage du terrain appliqué")
    if metadata["straightening_applied"]:
        print("  ✓ Redressement de l'image appliqué")
    if metadata["all_fixed_markers_found"]:
        print("  ✓ Tous les marqueurs fixes trouvés")
    if metadata["perspective_transform_computed"]:
        print("  ✓ Transformation perspective calculée")
        print("  ✓ Coordonnées réelles calculées")
    else:
        print("  ⚠️  Transformation perspective non calculée")
        print("     (Tous les marqueurs fixes n'ont pas été détectés)")

    # Statistiques de détection
    print(f"\n🎯 Détection :")
    print(f"  - Nombre total de marqueurs détectés : {len(detected_markers)}")

    tags_from_img = [marker for marker, _ in detected_markers]

    # Compter par type (exemple basé sur les IDs)
    yellow_tags = [tag for tag in tags_from_img if tag.aruco_id == 47]
    blue_tags = [tag for tag in tags_from_img if tag.aruco_id == 36]
    black_tags = [tag for tag in tags_from_img if tag.aruco_id == 41]
    fixed_tags = [tag for tag in tags_from_img if tag.aruco_id in [20, 21, 22, 23]]

    print(f"  - Marqueurs fixes : {len(fixed_tags)}")
    print(f"  - Marqueurs jaunes : {len(yellow_tags)}")
    print(f"  - Marqueurs bleus : {len(blue_tags)}")
    print(f"  - Marqueurs noirs : {len(black_tags)}")

    # Détails des marqueurs détectés
    print("\n📋 Détails des marqueurs :")
    tags_from_img.sort(key=lambda tag: tag.aruco_id)

    for tag in tags_from_img:
        if hasattr(tag, "real_x") and hasattr(tag, "real_y"):
            print(
                f"  ID {tag.aruco_id:2d}: "
                f"img=({tag.x:7.1f}, {tag.y:7.1f}) → "
                f"real=({tag.real_x:7.1f}, {tag.real_y:7.1f}) mm, "
                f"angle={tag.angle:6.1f}°"
            )
        else:
            print(
                f"  ID {tag.aruco_id:2d}: "
                f"img=({tag.x:7.1f}, {tag.y:7.1f}), "
                f"angle={tag.angle:6.1f}° (coordonnées réelles non calculées)"
            )

    print("\n" + "=" * 80)
    print("✅ Traitement terminé avec succès !")
    print("=" * 80)


if __name__ == "__main__":
    main()
