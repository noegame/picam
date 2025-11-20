"""
match.py
Détection et annotation de tags ArUco .
1. Détecte les 4 tags ArUco "balise" dans une image donnée 
2. Trouve les coordonnées du centre des 4 tags
Comme on connait la position des 4 tags dans le repère de la table, on peut trouver la transformation
qui permet de passer des coordonnées image (pixels) aux coordonnées réelles sur la table (mm).



 Usage exemple :
  .\.venv\Scripts\Activate.ps1
  python .\match.py "C:\HOME\WORK\raspberrypi\ZOD\picam\data\data_qr\aruco tel balise 1.jpg" --outdir "C:\HOME\WORK\raspberrypi\ZOD\picam\output" --dict DICT_4X4_50 --show

"""

import cv2
import numpy as np
import math
import argparse
import sys
import json
from pathlib import Path

def get_aruco_dict(dict_name='DICT_4X4_50'):
    """
    Retourne un objet dictionnaire ArUco utilisable par OpenCV à partir d'un nom lisible.

    Args:
        dict_name (str): Nom du dictionnaire ArUco (ex: 'DICT_4X4_50').

    Returns:
        cv2.aruco.Dictionary: objet dictionnaire ArUco prêt à l'emploi.

    Raises:
        ValueError: si le nom du dictionnaire n'est pas supporté.
        RuntimeError: si l'API cv2.aruco ne fournit pas de méthode pour obtenir le dictionnaire.
    """
    # Dans le cadre de la ZOD nous utilisons toujours des tags 4x4.
    # On ne propose que les dictionnaires 4x4 pour éviter toute confusion.
    ARUCO_DICT = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    }
    if dict_name not in ARUCO_DICT:
        raise ValueError(f"Dictionnaire ArUco inconnu: {dict_name} (seuls 'DICT_4X4_50' et 'DICT_4X4_100' sont pris en charge)")

    # Compatibilité selon version OpenCV
    if hasattr(cv2.aruco, "Dictionary_get"):
        return cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_name])
    raise RuntimeError("Impossible d'obtenir le dictionnaire ArUco depuis cv2.aruco")

def create_detector_and_params(aruco_dict):
    """
    Crée et retourne un détecteur ArUco et les paramètres de détection selon la version d'OpenCV.

    La fonction gère les différences d'API entre anciennes et nouvelles versions
    d'OpenCV (ex: existence de ArucoDetector, DetectorParameters_create, ...).

    Args:
        aruco_dict: objet dictionnaire ArUco retourné par `get_aruco_dict`.

    Returns:
        tuple: (detector, aruco_params)
            - detector: instance de cv2.aruco.ArucoDetector si disponible, sinon None
            - aruco_params: instance de paramètres (DetectorParameters) si utilisée par l'ancienne API,
              sinon None. Lorsque `detector` n'est pas None, `aruco_params` renvoyé est None
              car le paramètre est passé au constructeur du détecteur.
    """
    # crée params si disponible
    aruco_params = None
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    elif hasattr(cv2.aruco, "DetectorParameters"):
        try:
            aruco_params = cv2.aruco.DetectorParameters()
        except Exception:
            aruco_params = None

    # nouvelle API
    if hasattr(cv2.aruco, "ArucoDetector"):
        if aruco_params is not None:
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        else:
            detector = cv2.aruco.ArucoDetector(aruco_dict)
        return detector, None

    # ancienne API
    return None, aruco_params

def load_aruco_descriptions(json_path='aruco.json'):
    """
    Charge les descriptions des tags ArUco depuis un fichier JSON.
    Returns:
        dict: Dictionnaire des descriptions des tags {id (str): description (str)}
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Impossible de charger {json_path}: {e}")
        return {}

def detect_in_image(img, aruco_dict, detector, aruco_params, draw=True, aruco_descriptions=None):
    """
    Détecte les tags ArUco dans une image et ajoute leurs descriptions si disponibles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection
    if detector is not None:
        corners_list, ids, rejected = detector.detectMarkers(gray)
    else:
        corners_list, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    results = []
    if ids is None or len(ids) == 0:
        return results, img

    ids = ids.flatten()
    for corners, id_val in zip(corners_list, ids):
        corners = corners.reshape((4,2))
        cX = float(np.mean(corners[:,0]))
        cY = float(np.mean(corners[:,1]))

        rect = cv2.minAreaRect(corners.astype(np.float32))
        angle = rect[2]
        w, h = rect[1]
        if w < h:
            corrected_angle = (angle + 90) % 360
        else:
            corrected_angle = angle % 360
        if corrected_angle > 180:
            corrected_angle -= 360

        results.append({
            'id': int(id_val),
            'center_x': cX,
            'center_y': cY,
            'angle_deg': float(corrected_angle),
            'corners': corners
        })

        if draw:
            int_corners = corners.astype(int)
            cv2.polylines(img, [int_corners], isClosed=True, color=(0,255,0), thickness=2)
            cv2.circle(img, (int(round(cX)), int(round(cY))), 4, (255,0,0), -1)
            
            # Affiche l'ID et la description si disponible
            label = f"ID:{id_val}"
            if aruco_descriptions and str(id_val) in aruco_descriptions:
                label += f" - {aruco_descriptions[str(id_val)]}"
            
            cv2.putText(img, label, (int(cX)+10, int(cY)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            L = max(40, int(min(img.shape[0], img.shape[1]) * 0.05))
            theta = math.radians(corrected_angle)
            dx = L * math.cos(theta)
            dy = -L * math.sin(theta)
            pt1 = (int(round(cX)), int(round(cY)))
            pt2 = (int(round(cX + dx)), int(round(cY + dy)))
            cv2.arrowedLine(img, pt1, pt2, (0,0,255), 2, tipLength=0.2)

    return results, img

def process_path(input_path_str, out_dir, dict_name='DICT_4X4_50', show=False):
    """
    Traite un fichier image ou un dossier d'images: détecte les tags ArUco, annote
    les images et sauvegarde les images annotées.

    Args:
        input_path_str (str): chemin vers un fichier image ou un dossier d'images.
        out_dir (str): dossier de sortie pour les images annotées (sera créé si besoin).
        dict_name (str): nom du dictionnaire ArUco (ex: 'DICT_4X4_50').
        show (bool): si True, affiche chaque image annotée à l'écran (fermer la fenêtre pour continuer).

    Comportement:
        - charge le dictionnaire ArUco approprié
        - crée le détecteur suivant l'API disponible
        - charge les descriptions depuis `aruco.json` si présent
        - pour chaque image trouvée, détecte les tags, imprime les résultats et sauvegarde
          une copie annotée dans `out_dir`.
    """
    input_path = Path(input_path_str)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Forcer / valider l'utilisation de tags 4x4 (convention ZOD)
    if not dict_name.startswith('DICT_4X4'):
        print(f"Warning: Seuls les dictionnaires 4x4 sont supportés pour la ZOD. Forçage vers 'DICT_4X4_50' (vous aviez '{dict_name}').")
        dict_name = 'DICT_4X4_50'

    aruco_dict = get_aruco_dict(dict_name)
    detector, aruco_params = create_detector_and_params(aruco_dict)
    
    # Charge les descriptions des tags ArUco
    descriptions = load_aruco_descriptions()

    files = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        # extensions courantes images
        files = sorted([p for p in input_path.glob("*") if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']])
    else:
        raise FileNotFoundError(f"Chemin introuvable: {input_path}")

    if not files:
        print("Aucune image trouvée dans", input_path)
        return

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[SKIP] Impossible de lire {f}")
            continue

        results, annotated = detect_in_image(img, aruco_dict, detector, aruco_params, draw=True, aruco_descriptions=descriptions)

        if results:
            for d in results:
                print(f"{f.name} -> Tag ID={d['id']} | x={d['center_x']:.1f} px y={d['center_y']:.1f} px angle={d['angle_deg']:.1f}°")
        else:
            print(f"{f.name} -> Aucun tag détecté.")

        out_path = out_dir / f"{f.stem}_annotated{f.suffix}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Annotated saved: {out_path}")

        if show:
            # Affiche la fenêtre (fermer avec une touche)
            winname = f"Annotated - {f.name}"
            cv2.imshow(winname, annotated)
            print("  Appuie sur une touche dans la fenêtre d'image pour continuer...")
            cv2.waitKey(0)
            cv2.destroyWindow(winname)

def main():
    """
    Point d'entrée CLI du script. Analyse les arguments et lance le traitement.
    Options prises en charge:
        - input: fichier image ou dossier d'images
        - --outdir: dossier de sortie
        - --dict: dictionnaire ArUco à utiliser
        - --show: afficher les images annotées
    """
    parser = argparse.ArgumentParser(description="Detect ArUco tags and save annotated images (Windows compatible).")
    parser.add_argument("input", help="fichier image ou dossier d'images")
    parser.add_argument("--outdir", help="dossier de sortie pour images annotées", default="out_annotated")
    parser.add_argument("--dict", help="nom dictionnaire ArUco (ex: DICT_4X4_50)", default="DICT_4X4_50")
    parser.add_argument("--show", help="afficher chaque image annotée (fermer fenêtre pour passer à la suivante)", action="store_true")
    args = parser.parse_args()

    try:
        process_path(args.input, args.outdir, dict_name=args.dict, show=args.show)
    except Exception as e:
        print("Erreur:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
