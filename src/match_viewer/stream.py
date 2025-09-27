from picamera2 import Picamera2
import socket
import cv2
import numpy as np

picam2 = Picamera2()
picam2.start()

# TCP server
HOST = ''
PORT = 8554
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("En attente de connexion...")
conn, addr = s.accept()
print("Connecté à", addr)

while True:
    frame = picam2.capture_array()
    # Encode en JPEG pour transmission
    _, jpeg = cv2.imencode('.jpg', frame)
    data = jpeg.tobytes()
    # Envoi taille + données
    conn.send(len(data).to_bytes(4, 'big') + data)
