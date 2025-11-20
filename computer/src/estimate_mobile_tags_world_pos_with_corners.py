#!/usr/bin/env python3
"""
estimate_mobile_tags_world_pos_with_corners.py

Usage:
    python estimate_mobile_tags_world_pos_with_corners.py --calib camera_calibration2.npz --video 0
    python estimate_mobile_tags_world_pos_with_corners.py --calib camera_calibration2.npz --tags-json aruco_tags.json --video 0

Pré-requis:
    pip install opencv-contrib-python numpy

Ce script :
- charge la calibration (.npz),
- charge les coordonnées des coins des tags depuis un fichier JSON,
- détecte ArUco,
- construit correspondances 3D (coins des tags fixes) <-> 2D (coins détectés),
- estime la pose caméra via solvePnPRansac,
- estime la pose des tags mobiles via estimatePoseSingleMarkers,
- transforme les positions des tags mobiles dans le repère monde (mètres).
"""

import cv2
import numpy as np
import argparse
import time
import json

# ------------- CONFIG -------------
# Dictionnaire ArUco utilisé (compatibilité selon versions OpenCV)
if hasattr(cv2.aruco, 'getPredefinedDictionary'):
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
elif hasattr(cv2.aruco, 'Dictionary_get'):
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
else:
    ARUCO_DICT = None

# Paramètres du détecteur (fallback compat)
if hasattr(cv2.aruco, 'DetectorParameters_create'):
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
elif hasattr(cv2.aruco, 'DetectorParameters'):
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
else:
    ARUCO_PARAMS = None

# utilitaire pour dessiner les axes avec compatibilité OpenCV
def safe_draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length):
    try:
        # préférence pour aruco.drawAxis
        if hasattr(cv2.aruco, 'drawAxis'):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
            return
    except Exception:
        pass
    try:
        # fallback to cv2.drawFrameAxes (OpenCV >=4.5)
        if hasattr(cv2, 'drawFrameAxes'):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, length)
            return
    except Exception:
        pass
    # no-op if neither is available
    return

# ------------- Fonctions utilitaires -------------
def load_calibration(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    # essais sur clés usuelles
    K = None
    dist = None
    for k in ['camera_matrix', 'cameraMatrix', 'mtx', 'K', 'camera_matrix']:
        if k in d:
            K = d[k]
            break
    for k in ['dist', 'distCoeffs', 'distortion_coefficients', 'distortion']:
        if k in d:
            dist = d[k]
            break
    if K is None:
        # essayer toutes les clés
        for k in d.files:
            if d[k].shape == (3,3):
                K = d[k]; break
    if dist is None:
        # fallback zeros
        dist = np.zeros((5,1))
    return K, dist

def load_tag_corners_from_json(json_path):
    """
    Charge les coordonnées world des coins des tags depuis un fichier JSON.
    Format attendu:
    {
        "tags": {
            "0": {
                "corners": [
                    [x1, y1, z1],
                    [x2, y2, z2],
                    [x3, y3, z3],
                    [x4, y4, z4]
                ]
            },
            ...
        }
    }
    Retourne: dict id -> (4,3) array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fixed_tag_corners_world = {}
    tags = data.get('tags', {})
    for tag_id_str, tag_info in tags.items():
        tag_id = int(tag_id_str)
        corners_list = tag_info.get('corners', [])
        if len(corners_list) == 4:
            fixed_tag_corners_world[tag_id] = np.array(corners_list, dtype=np.float64)
        else:
            print(f"Warning: Tag {tag_id} n'a pas exactement 4 coins dans le JSON")
    
    return fixed_tag_corners_world

def marker_corners_to_center(corners):
    # corners shape (1,4,2) from detectMarkers
    return np.mean(corners.reshape((4,2)), axis=0)

# ------------- Pipeline principal -------------
def main(args):
    K, dist = load_calibration(args.calib)
    if K is None:
        raise SystemExit("Impossible de charger la matrice caméra depuis le fichier de calibration.")
    print("Loaded calibration. K=\n", K)
    print("distCoeffs=\n", dist.flatten())

    # Charger coins world des tags fixes depuis le fichier JSON
    fixed_tag_corners_world = load_tag_corners_from_json(args.tags_json)
    if not fixed_tag_corners_world:
        raise SystemExit("Aucun tag chargé depuis le fichier JSON.")
    print(f"Loaded {len(fixed_tag_corners_world)} tags from JSON")

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la vidéo / caméra: " + args.video)

    aruco_dict = ARUCO_DICT
    aruco_params = ARUCO_PARAMS

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if aruco_dict is None:
            raise SystemExit("cv2.aruco: impossible de créer le dictionnaire ArUco sur cette build OpenCV")
        if aruco_params is not None:
            corners_list, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        else:
            corners_list, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None:
            cv2.putText(frame, "No markers", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)

        # Construire correspondances 3D (world corners) <-> 2D (image corners)
        image_pts = []
        world_pts = []
        for i, marker_id in enumerate(ids):
            mid = int(marker_id)
            if mid in fixed_tag_corners_world:
                # corners_list[i] shape (1,4,2)
                img_corners = corners_list[i].reshape((4,2))  # order matches compute_tag_corners_world
                world_corners = fixed_tag_corners_world[mid]  # shape (4,3)
                # ajouter les 4 coins dans le même ordre
                for k in range(4):
                    image_pts.append([float(img_corners[k,0]), float(img_corners[k,1])])
                    world_pts.append([float(world_corners[k,0]), float(world_corners[k,1]), float(world_corners[k,2])])

        image_pts = np.array(image_pts, dtype=np.float64)
        world_pts = np.array(world_pts, dtype=np.float64)

        cam_pose_available = False
        if len(image_pts) >= 4:
            # solvePnPRansac expects objectPoints Nx3 and imagePoints Nx2
            success, rvec_cam, tvec_cam, inliers = cv2.solvePnPRansac(
                objectPoints=world_pts,
                imagePoints=image_pts,
                cameraMatrix=K,
                distCoeffs=dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=200
            )
            if success:
                cam_pose_available = True
                R_cam, _ = cv2.Rodrigues(rvec_cam)
                t_cam = tvec_cam.reshape((3,))
                R_wc = R_cam.T
                t_wc = -R_wc @ t_cam
                cv2.putText(frame, f"Camera pose estimated (inliers:{0 if inliers is None else len(inliers)})", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame, "solvePnP failed", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.putText(frame, f"Need >=4 fixed-corners (found markers providing {len(image_pts)} corners)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Estimer pose des marqueurs détectés (relative à la caméra)
        tag_size = 0.10  # taille moyenne pour afficher les axes
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners_list, tag_size, K, dist)

        for i, marker_id in enumerate(ids):
            safe_draw_axis(frame, K, dist, rvecs[i], tvecs[i], tag_size * 0.5)
            mid = int(marker_id)
            center = marker_corners_to_center(corners_list[i]).astype(int)

            if cam_pose_available:
                pos_cam = tvecs[i].reshape((3,))
                pos_world = R_wc @ pos_cam + t_wc
                xw, yw, zw = pos_world
                color = (0,255,0) if mid in fixed_tag_corners_world else (255,0,0)
                label = f"id:{mid} -> ({xw:.3f}, {yw:.3f}, {zw:.3f}) m"
                cv2.putText(frame, label, (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"[{time.strftime('%H:%M:%S')}] Marker {mid} world pos (m): x={xw:.4f}, y={yw:.4f}, z={zw:.4f}")
            else:
                pos_cam = tvecs[i].reshape((3,))
                cv2.putText(frame, f"id:{mid} cam ({pos_cam[0]:.3f},{pos_cam[1]:.3f},{pos_cam[2]:.3f})m",
                            (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"frame_{ts}.png", frame)
            print("Saved frame_", ts, ".png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estime positions de tags ArUco (utilise coins des tags fixes).")
    parser.add_argument("--calib", required=True, help="fichier npz de calibration (ex: camera_calibration2.npz)")
    parser.add_argument("--tags-json", required=True, help="fichier JSON contenant les coordonnées des coins des tags (ex: aruco.json)")
    parser.add_argument("--video", default="0", help="index caméra (0) ou chemin vidéo")
    args = parser.parse_args()
    main(args)
