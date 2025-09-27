# stream_pi_to_vlc.py
from picamera2 import Picamera2
import socket
import cv2
import numpy as np
import time

# ------------------------
# Configuration caméra
# ------------------------
print("🎥 Initialisation de la caméra...")
picam2 = Picamera2()

# Vérifier les caméras disponibles
cameras = Picamera2.global_camera_info()
print(f"📷 Caméras détectées: {len(cameras)}")
if not cameras:
    print("❌ Aucune caméra trouvée!")
    exit(1)

picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
print("✅ Caméra initialisée")

# Attendre stabilisation
time.sleep(2)

# ------------------------
# Création serveur TCP
# ------------------------
HOST = ''    # toutes interfaces
PORT = 8554  # port pour VLC

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print(f"En attente de connexion sur port {PORT}...")
conn, addr = s.accept()
print("Connecté à", addr)

try:
    frame_count = 0
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convertir RGB vers BGR pour OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode en JPEG pour transmission
        _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = jpeg.tobytes()
        
        # Envoi taille (4 octets) + données
        try:
            conn.send(len(data).to_bytes(4, 'big') + data)
            frame_count += 1
            if frame_count % 30 == 0:  # Afficher toutes les 30 frames
                print(f"📺 {frame_count} frames envoyées")
        except BrokenPipeError:
            print("❌ Connexion fermée par le client")
            break
            
        # Limiter le framerate à ~30 FPS
        time.sleep(1/30)

except KeyboardInterrupt:
    print("\nFermeture du serveur...")

finally:
    print("🔄 Nettoyage...")
    try:
        conn.close()
    except:
        pass
    try:
        picam2.stop()
    except:
        pass
    print("✅ Arrêt complet")
