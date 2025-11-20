#!/usr/bin/env python3
"""
qr_readability_checker.py

Ouvre une boîte de dialogue Tkinter pour choisir une image, puis
teste plusieurs stratégies pour lire des QR codes dans l'image.
Génère un rapport dans la console et sauvegarde une image annotée
<originalname>_annotated.png à côté de l'image choisie.

Dépendances:
 - opencv-python
 - pyzbar
 - pillow
 - numpy
"""

import os
import sys
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from PIL import Image

# -----------------------
# Prétraitements utiles
# -----------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_equalize(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def hist_equalize(gray):
    return cv2.equalizeHist(gray)

def denoise(gray):
    # bilateral filter to keep edges
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

def sharpen(gray):
    # unsharp mask-ish
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

def threshold_adaptive(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def morphological_close(bin_img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

# -----------------------
# Décodage avec pyzbar
# -----------------------
def decode_with_pyzbar(image):
    # pyzbar works on PIL or OpenCV images; convert to PIL
    try:
        # If grayscale, convert to 3-channel before PIL conversion to keep compatibility
        if len(image.shape) == 2:
            pil = Image.fromarray(image)
        else:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception:
        pil = Image.fromarray(image)
    decs = pyzbar_decode(pil)
    results = []
    for d in decs:
        data = d.data.decode('utf-8', errors='replace')
        bbox = d.rect  # left, top, width, height
        polygon = [(p.x, p.y) for p in d.polygon] if hasattr(d, 'polygon') else None
        results.append({'data': data, 'bbox': bbox, 'polygon': polygon})
    return results

# -----------------------
# Décodage avec OpenCV QRCodeDetector
# -----------------------
def decode_with_cv2_qr(image):
    detector = cv2.QRCodeDetector()
    # cv2 wants color or gray; pass color
    try:
        data, points, straight_qrcode = detector.detectAndDecodeMulti(image)
    except Exception:
        # some cv2 builds might not have detectAndDecodeMulti stable; fallback to single
        data_single, points_single, sq = detector.detectAndDecode(image)
        if data_single:
            return [{'data': data_single, 'points': points_single}]
        return []
    results = []
    # when using detectAndDecodeMulti, `data` is a tuple/list of decoded strings or empty strings
    if isinstance(data, (list, tuple, np.ndarray)):
        for idx, s in enumerate(data):
            if s:
                pts = None
                try:
                    if points is not None:
                        pts = points[idx]
                except Exception:
                    pts = None
                results.append({'data': s, 'points': pts})
    else:
        # sometimes returns single string
        if data:
            results.append({'data': data, 'points': points})
    return results

# -----------------------
# Utilitaires annotation
# -----------------------
def draw_annotations(orig_bgr, decoded_entries):
    img = orig_bgr.copy()
    for ent in decoded_entries:
        if 'polygon' in ent and ent['polygon']:
            pts = np.array(ent['polygon'], dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            # put text at first point
            x,y = ent['polygon'][0]
            text = ent['data'][:40]
            cv2.putText(img, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        elif 'bbox' in ent and ent['bbox']:
            l,t,w,h = ent['bbox'].left, ent['bbox'].top, ent['bbox'].width, ent['bbox'].height
            cv2.rectangle(img, (l,t), (l+w, t+h), (255,0,0), 2)
            cv2.putText(img, ent['data'][:40], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        elif 'points' in ent and ent['points'] is not None:
            try:
                pts = np.array(ent['points'], dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
                x,y = int(ent['points'][0][0]), int(ent['points'][0][1])
                cv2.putText(img, ent['data'][:40], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            except Exception:
                pass
    return img

# -----------------------
# Main evaluation routine
# -----------------------
def evaluate_image(path, verbose=True):
    if verbose: print("Chargement de l'image:", path)
    orig = cv2.imread(path)
    if orig is None:
        raise FileNotFoundError("Impossible de lire l'image. Vérifie le chemin et le format.")
    # keep original size
    h0, w0 = orig.shape[:2]

    found = []  # list of dicts {data, method, details...}
    unique_data = set()

    # 1) Tentative directe pyzbar sur l'image originale
    if verbose: print("Tentative 1: pyzbar sur image originale")
    r1 = decode_with_pyzbar(orig)
    for r in r1:
        if r['data'] not in unique_data:
            unique_data.add(r['data'])
            found.append({'data': r['data'], 'method': 'pyzbar_original', 'bbox': r.get('bbox'), 'polygon': r.get('polygon')})

    # 2) OpenCV detector
    if verbose: print("Tentative 2: OpenCV QRCodeDetector")
    r2 = decode_with_cv2_qr(orig)
    for r in r2:
        if r['data'] not in unique_data:
            unique_data.add(r['data'])
            found.append({'data': r['data'], 'method': 'cv2_detector', 'points': r.get('points')})

    # 3) Série de prétraitements + échelles + rotations
    if verbose: print("Tentative 3+: Prétraitements, redimensionnements et rotations")
    gray = to_gray(orig)
    preprocessings = [
        ('gray', lambda g: g),
        ('clahe', clahe_equalize),
        ('histeq', hist_equalize),
        ('denoise', denoise),
        ('sharpen', sharpen),
        ('adaptive_thresh', threshold_adaptive),
    ]

    scales = [1.0, 1.5, 2.0, 3.0]  # agrandir peut aider pour petits QR flous
    rotations = [0, 90, 180, 270]

    attempt_count = 0
    for pname, pfunc in preprocessings:
        try:
            img_p = pfunc(gray)
        except Exception as e:
            if verbose: print(f"Erreur preprocess {pname}: {e}")
            continue

        for scale in scales:
            if scale != 1.0:
                new_size = (int(w0*scale), int(h0*scale))
                img_scaled = cv2.resize(img_p, new_size, interpolation=cv2.INTER_CUBIC)
            else:
                img_scaled = img_p

            # If we have a binary image after adaptive threshold, optionally morph close
            if img_scaled.ndim == 2 and img_scaled.max() <= 255 and (img_scaled.dtype == np.uint8):
                # try morphological close to fill holes for thresholded images
                try:
                    img_scaled = morphological_close(img_scaled, ksize=3)
                except Exception:
                    pass

            for rot in rotations:
                attempt_count += 1
                if rot != 0:
                    M = cv2.getRotationMatrix2D((img_scaled.shape[1]/2, img_scaled.shape[0]/2), rot, 1.0)
                    img_rot = cv2.warpAffine(img_scaled, M, (img_scaled.shape[1], img_scaled.shape[0]), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE)
                else:
                    img_rot = img_scaled

                # pyzbar
                try:
                    decs = decode_with_pyzbar(img_rot)
                    for d in decs:
                        if d['data'] and d['data'] not in unique_data:
                            unique_data.add(d['data'])
                            found.append({'data': d['data'],
                                          'method': f'pyzbar_preproc({pname})_s{scale}_r{rot}',
                                          'bbox': d.get('bbox'),
                                          'polygon': d.get('polygon')})
                except Exception:
                    pass

                # try OpenCV detector on color version if needed
                try:
                    if img_rot.ndim == 2:
                        color_for_cv = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2BGR)
                    else:
                        color_for_cv = img_rot
                    od = decode_with_cv2_qr(color_for_cv)
                    for d in od:
                        if d['data'] and d['data'] not in unique_data:
                            unique_data.add(d['data'])
                            found.append({'data': d['data'],
                                          'method': f'cv2_preproc({pname})_s{scale}_r{rot}',
                                          'points': d.get('points')})
                except Exception:
                    pass

    if verbose:
        print(f"Nombre total de tentatives de prétraitement effectuées: ~{attempt_count}")
        print("Résultats trouvés :")
        for i, f in enumerate(found):
            print(f" {i+1}. '{f['data']}'  (méthode: {f['method']})")

    # Annotate image with all found bboxes/polygons
    annotated = draw_annotations(orig, found)
    base, ext = os.path.splitext(path)
    out_path = base + "_annotated.png"
    cv2.imwrite(out_path, annotated)
    if verbose:
        print("Image annotée sauvegardée :", out_path)

    # Build a simple readability assessment:
    # - if >=1 unique decoded strings found -> likely readable
    # - else -> probably not (but may still be readable with better photos)
    assessment = "NOT_READABLE"
    if len(unique_data) >= 1:
        assessment = "READABLE"
    # More nuanced: report how many distinct codes found
    report = {
        'image_path': path,
        'annotated_image': out_path,
        'num_codes_found': len(unique_data),
        'codes': [{'data': d, 'method': next((f['method'] for f in found if f['data']==d), None)} for d in unique_data],
        'assessment': assessment,
        'details': found
    }
    return report

# -----------------------
# Tkinter UI
# -----------------------
def run_gui():
    root = tk.Tk()
    root.withdraw()  # hide main window

    messagebox.showinfo("QR readability checker",
                        "Choisis une image contenant les QR codes à tester (vue de dessus OK). Le programme va essayer plusieurs méthodes pour lire les QR.")

    filetypes = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
    path = filedialog.askopenfilename(title="Sélectionne une image", filetypes=filetypes)
    if not path:
        messagebox.showwarning("Aucune image", "Aucune image sélectionnée. Fin.")
        return

    try:
        report = evaluate_image(path, verbose=True)
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue pendant l'analyse :\n{e}")
        return

    # Build human-friendly message
    if report['assessment'] == "READABLE":
        title = "QR lisibles détectés"
        msg = (f"{report['num_codes_found']} code(s) décodé(s) trouvé(s).\n"
               f"Exemples :\n")
        for c in report['codes'][:5]:
            msg += f" - {c['data']}  (méthode: {c['method']})\n"
        msg += f"\nImage annotée sauvegardée :\n{report['annotated_image']}\n\n"
        msg += "Interprétation : il serait raisonnable d'implémenter la lecture des QR sur ton système.\n"
    else:
        title = "Aucun QR décodable trouvé"
        msg = ("Aucun QR code n'a été décodé automatiquement après plusieurs tentatives.\n"
               "Cela signifie que, sur cette image, les QR semblent illisibles —\n"
               "possible causes : flou optique, faible contraste, réflexion, QR trop petits, ou QR partiellement masqués.\n\n"
               f"Image annotée sauvegardée :\n{report['annotated_image']}\n\n"
               "Interprétation : je te conseille de vérifier la qualité d'image (augmenter résolution/zoom, améliorer contraste, réduire flou) "
               "avant d'implémenter la lecture, ou de prévoir une étape de recadrage/zoom automatique.\n")
    # show a messagebox with summary
    messagebox.showinfo(title, msg)

    # Print detailed report in console as well
    print("\n=== Rapport détaillé ===")
    print(f"Image: {report['image_path']}")
    print(f"Annotated image: {report['annotated_image']}")
    print(f"Assessment: {report['assessment']}")
    print(f"Num codes found: {report['num_codes_found']}")
    for c in report['codes']:
        print(f" - Data: {c['data']}  (method: {c['method']})")

if __name__ == "__main__":
    run_gui()
