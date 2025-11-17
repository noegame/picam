# qr_opencv.py
"""
Lire QR code avec OpenCV QRCodeDetector.
Usage: python qr_opencv.py chemin/vers/image.jpg
"""

import sys
import cv2

def read_qr_with_opencv(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {image_path}")
    detector = cv2.QRCodeDetector()
    data, points, straight_qrcode = detector.detectAndDecode(img)
    return data, points

def draw_boxes_and_save(image_path, points, data, out_path="out_qr_opencv.png"):
    img = cv2.imread(image_path)
    if points is not None:
        pts = points.astype(int).reshape((-1,2))
        # dessiner polygone
        cv2.polylines(img, [pts], isClosed=True, color=(255,0,0), thickness=2)
        # écrire texte
        x, y = int(pts[0][0]), int(pts[0][1]) - 10
        cv2.putText(img, data, (x, max(10,y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.imwrite(out_path, img)
        print(f"Image annotée enregistrée: {out_path}")
    else:
        print("Aucun point de QR code trouvé pour dessin.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qr_opencv.py chemin/vers/image.jpg")
        sys.exit(1)
    path = sys.argv[1]
    data, points = read_qr_with_opencv(path)
    if not data:
        print("Aucun QR code décodé.")
    else:
        print("Données décodées :", data)
        draw_boxes_and_save(path, points, data)
