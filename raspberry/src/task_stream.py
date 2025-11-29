#!/usr/bin/env python3

"""
Task de streaming vidéo via Flask
Reçoit les frames d'une queue et les envoie aux clients
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import io
import threading
from multiprocessing import Queue
from logging import Logger
from flask import Flask, Response

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

app = Flask(__name__)
frame_lock = threading.Lock()
current_frame = None

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

def get_frame(queue: Queue):
    """Récupère une frame du chemin et la met en cache"""
    global current_frame
    frame_path = queue.get()
    try:
        with open(frame_path, 'rb') as f:
            with frame_lock:
                current_frame = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du frame: {e}")


def frame_reader_thread(queue: Queue, logger: Logger):
    """Thread qui lit continuellement les frames de la queue"""
    logger.info("Démarrage du thread de lecture des frames")
    while True:
        try:
            get_frame(queue)
        except Exception as e:
            logger.error(f"Erreur dans le thread de lecture: {e}")


def generate_frames():
    """Génère les frames pour le stream MJPEG"""
    while True:
        if current_frame:
            with frame_lock:
                frame_data = current_frame
            yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                   str(len(frame_data)).encode() + b'\r\n\r\n' + frame_data + b'\r\n')
        threading.Event().wait(0.01)


@app.route('/video_feed')
def video_feed():
    """Endpoint pour le streaming vidéo"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Page d'accueil avec le lecteur vidéo"""
    return '''
    <html>
    <head><title>Pi Camera Stream</title></head>
    <body>
        <h1>Raspberry Pi Camera Stream</h1>
        <img src="/video_feed" style="max-width: 100%; border: 1px solid #ccc;">
    </body>
    </html>
    '''


def task_stream(queue: Queue, logger: Logger):
    """Tâche de streaming: récupère les frames de la queue et les servie via Flask"""
    logger.info("Démarrage de la tâche de streaming")
    
    # Démarre le thread de lecture des frames
    reader_thread = threading.Thread(target=frame_reader_thread, args=(queue, logger), daemon=True)
    reader_thread.start()
    
    # Lance le serveur Flask
    logger.info("Démarrage du serveur Flask sur 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)



