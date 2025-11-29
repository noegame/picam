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
from flask import Flask, Response, jsonify

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

app = Flask(__name__)
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
    return '''
    <html>
    <head>
        <title>Pi Camera Stream - Processed</title>
        <style>
            body { font-family: sans-serif; margin: 0; padding: 1em; background-color: #f4f4f4; }
            h1 { text-align: center; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 1em; }
            .stream-box { border: 1px solid #ccc; padding: 0.5em; background-color: white; }
            .stream-box h2 { margin: 0 0 0.5em 0; text-align: center; font-size: 1em; }
            .stream-box img { max-width: 100%; display: block; }
            #aruco-info { margin-top: 1em; }
            #aruco-info pre { background-color: #eee; border: 1px solid #ddd; padding: 1em; white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi - Multi-stream</h1>
        <div class="container">
            <div class="stream-box">
                <h2>Caméra Originale</h2>
                <img src="/stream/original_img">
            </div>
            <div class="stream-box">
                <h2>Image Détordue</h2>
                <img src="/stream/undistorted_img">
            </div>
            <div class="stream-box">
                <h2>Image Redressée</h2>
                <img src="/stream/warped_img">
            </div>
        </div>
        <div id="aruco-info">
            <h2>Tags ArUco Détectés</h2>
            <pre id="aruco-data-display">En attente de données...</pre>
        </div>
        <script>
            function fetchArucoData() {
                fetch('/aruco_data')
                    .then(response => response.json())
                    .then(data => {
                        const display = document.getElementById('aruco-data-display');
                        if (data && data.length > 0) {
                            display.textContent = JSON.stringify(data, null, 2);
                        } else {
                            display.textContent = 'Aucun tag ArUco détecté.';
                        }
                    })
                    .catch(error => {
                        console.error('Erreur lors de la récupération des données ArUco:', error);
                        document.getElementById('aruco-data-display').textContent = 'Erreur de chargement.';
                    });
            }
            // Récupérer les données toutes les secondes
            setInterval(fetchArucoData, 1000);
            // Premier appel au chargement de la page
            fetchArucoData();
        </script>
    </body>
    </html>
    '''

def task_stream(queue: Queue, logger: Logger):
    """Tâche de streaming: récupère les données de la queue et les sert via Flask."""
    logger.info("Démarrage de la tâche de streaming")
    
    # Démarre le thread de lecture des données
    reader_thread = threading.Thread(target=data_reader_thread, args=(queue, logger), daemon=True)
    reader_thread.start()
    
    # Lance le serveur Flask
    logger.info("Démarrage du serveur Flask sur 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
