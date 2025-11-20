# qr_pyzbar.py
"""
Lire et afficher le contenu d'un QR code dans une image avec pyzbar + Pillow/OpenCV.
Usage: python qr_pyzbar.py chemin/vers/image.jpg
"""

import sys
from PIL import Image
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import numpy as np

def read_qr_with_pyzbar(image_path):
    img = Image.open(image_path).convert("RGB")
    decoded = decode(img, symbols=[ZBarSymbol.QRCODE])
    results = []
    for d in decoded:
        data = d.data.decode('utf-8')
        rect = d.rect  # left, top, width, height
        results.append({'data': data, 'rect': rect, 'type': d.type})
    return results

def draw_boxes_and_save(image_path, results, out_path="out_qr_pyzbar.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    for r in results:
        x, y, w, h = r['rect'].left, r['rect'].top, r['rect'].width, r['rect'].height
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, r['data'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(out_path, img)
    print(f"Image annotée enregistrée: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qr_pyzbar.py chemin/vers/image.jpg")
        sys.exit(1)
    path = sys.argv[1]
    res = read_qr_with_pyzbar(path)
    if not res:
        print("Aucun QR code détecté.")
    else:
        for i, r in enumerate(res, 1):
            print(f"[{i}] Type: {r['type']}, Data: {r['data']}, Rect: {r['rect']}")
        draw_boxes_and_save(path, res)
