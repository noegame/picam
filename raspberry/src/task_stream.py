#!/usr/bin/env python3
"""
Task de streaming vidéo via Flask
Reçoit les données de la détection (images et tags) d'une queue et les affiche sur une page web
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import threading
import time
from multiprocessing import Queue
from pathlib import Path
from flask import Flask, Response, jsonify, render_template

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

logger = logging.getLogger("task_stream")
app = Flask(__name__, template_folder="templates")
data_lock = threading.Lock()

# Dictionnaire pour stocker les données binaires des images
image_data = {"undistorted_img": None, "img_width": 2000, "img_height": 2000}
# Liste pour stocker les informations des tags ArUco
aruco_tags_data = []

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def update_data_from_queue(queue: Queue):
    """Récupère les dernières données de la queue en drainant tous les éléments en attente."""
    global image_data, aruco_tags_data

    # Drainer la queue : récupérer tous les éléments en attente et ne traiter que le dernier
    latest_data = None
    while not queue.empty():
        try:
            latest_data = queue.get_nowait()
        except:
            break

    # Si on a reçu des données, les traiter
    if latest_data:
        with data_lock:
            # Met à jour les données des tags ArUco
            if "aruco_tags" in latest_data:
                aruco_tags_data = latest_data["aruco_tags"]

            # Met à jour l'image et les dimensions
            if "undistorted_img" in latest_data:
                image_data["undistorted_img"] = latest_data["undistorted_img"]
            if "img_width" in latest_data:
                image_data["img_width"] = latest_data["img_width"]
            if "img_height" in latest_data:
                image_data["img_height"] = latest_data["img_height"]


def data_reader_thread(queue: Queue):
    """Thread qui lit continuellement les données de la queue et traite le dernier élément."""
    logger.info("Démarrage du thread de lecture des données")
    while True:
        try:
            update_data_from_queue(queue)
            time.sleep(
                0.01
            )  # Petit délai pour éviter de tourner trop vite quand la queue est vide
        except Exception as e:
            logger.error(
                f"Erreur dans le thread de lecture des données: {e}", exc_info=True
            )


def generate_frames(image_key: str):
    """Génère les frames pour un stream MJPEG à partir de la clé d'image spécifiée."""
    while True:
        with data_lock:
            frame_bytes = image_data.get(image_key)

        if frame_bytes is not None:
            # frame_bytes is already JPEG encoded from the detection task
            # No need for conversion, just stream it directly
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame_bytes)).encode()
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        # Attendre un court instant pour ne pas surcharger le CPU
        time.sleep(0.04)  # ~25 fps


@app.route("/stream/<image_key>")
def stream(image_key: str):
    """Endpoint pour le streaming vidéo (ex: /stream/undistorted_img)."""
    if image_key not in image_data:
        return "Invalid image key", 404
    return Response(
        generate_frames(image_key), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/aruco_data")
def get_aruco_data():
    """Endpoint pour récupérer les données des tags ArUco en JSON."""
    with data_lock:
        # Convert Point objects to dictionaries for JSON serialization
        serializable_data = [point.to_dict() for point in aruco_tags_data]
        return jsonify(serializable_data)


@app.route("/")
def index():
    """Page d'accueil avec les lecteurs vidéo et les données ArUco."""
    return render_template("index.html")


def task_stream(queue: Queue):
    """Tâche de streaming: récupère les données de la queue et les sert via Flask."""
    logger.info("Démarrage de la tâche de streaming")

    # Démarre le thread de lecture des données
    reader_thread = threading.Thread(
        target=data_reader_thread, args=(queue,), daemon=True
    )
    reader_thread.start()

    # Lance le serveur Flask
    logger.info("Démarrage du serveur Flask sur 0.0.0.0:5000")

    # Désactiver les logs de Werkzeug pour ne pas polluer la console
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)
