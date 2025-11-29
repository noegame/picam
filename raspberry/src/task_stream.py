#!/usr/bin/env python3
"""
Task de streaming vidéo via Flask
Reçoit les données de la détection (images et tags) d'une queue et les affiche sur une page web
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import threading
from multiprocessing import Queue
from logging import Logger
from flask import Flask, Response, jsonify, render_template

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder='templates')
data_lock = threading.Lock()

# Dictionnaire pour stocker les données binaires des images
image_data = {
    "original_img": None,
    "undistorted_img": None,
    "warped_img": None
}
# Liste pour stocker les informations des tags ArUco
aruco_tags_data = []

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

def update_data_from_queue(queue: Queue, logger: Logger):
    """Récupère les données de la queue, lit les images et met à jour les variables globales."""
    global image_data, aruco_tags_data
    
    data = queue.get()
    
    with data_lock:
        # Met à jour les données des tags ArUco
        aruco_tags_data = data["aruco_tags"]

        # Lit et met à jour chaque image
        for key in image_data.keys():
            try:
                with open(data[key], 'rb') as f:
                    image_data[key] = f.read()
            except FileNotFoundError:
                logger.warning(f"Fichier image non trouvé: {data[key]}")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier image {data[key]}: {e}")

def data_reader_thread(queue: Queue, logger: Logger):
    """Thread qui lit continuellement les données de la queue."""
    logger.info("Démarrage du thread de lecture des données")
    while True:
        try:
            update_data_from_queue(queue, logger)
        except Exception as e:
            logger.error(f"Erreur dans le thread de lecture des données: {e}", exc_info=True)

def generate_frames(image_key: str):
    """Génère les frames pour un stream MJPEG à partir de la clé d'image spécifiée."""
    while True:
        with data_lock:
            frame_bytes = image_data.get(image_key)
        
        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                   str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Attendre un court instant pour ne pas surcharger le CPU
        threading.Event().wait(0.04) # ~25 fps

@app.route('/stream/<image_key>')
def stream(image_key: str):
    """Endpoint pour le streaming vidéo (ex: /stream/original_img)."""
    if image_key not in image_data:
        return "Invalid image key", 404
    return Response(generate_frames(image_key), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/aruco_data')
def get_aruco_data():
    """Endpoint pour récupérer les données des tags ArUco en JSON."""
    with data_lock:
        return jsonify(aruco_tags_data)

@app.route('/')
def index():
    """Page d'accueil avec les lecteurs vidéo et les données ArUco."""
    return render_template("index.html")

def task_stream(queue: Queue, logger: Logger):
    """Tâche de streaming: récupère les données de la queue et les sert via Flask."""
    logger.info("Démarrage de la tâche de streaming")
    
    # Démarre le thread de lecture des données
    reader_thread = threading.Thread(target=data_reader_thread, args=(queue, logger), daemon=True)
    reader_thread.start()
    
    # Lance le serveur Flask
    logger.info("Démarrage du serveur Flask sur 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
