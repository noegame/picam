# stream_pi_to_vlc.py
from picamera2 import Picamera2
import socket
import cv2
import numpy as np

# ------------------------
# Configuration caméra
# ------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

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
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Encode en JPEG pour transmission
        _, jpeg = cv2.imencode('.jpg', frame)
        data = jpeg.tobytes()
        
        # Envoi taille (4 octets) + données
        conn.send(len(data).to_bytes(4, 'big') + data)

except KeyboardInterrupt:
    print("\nFermeture du serveur...")

finally:
    conn.close()
    picam2.close()
