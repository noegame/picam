#!/usr/bin/env python3
"""
Script pour capturer des images pour la calibration de la caméra avec un aperçu en direct.

Ce script démarre un serveur web qui diffuse le flux vidéo de la caméra.
Vous pouvez visualiser ce flux depuis un navigateur à l'adresse http://<ip_raspberry>:5000.

Pour capturer une image, appuyez sur la touche 'Entrée' dans la console où le script est lancé.
Les images sont sauvegardées dans 'output/calibration'.

Appuyez sur Ctrl+C pour arrêter le script.
"""

import logging
import time
import cv2
from pathlib import Path
from flask import Flask, Response
import threading

from raspberry.src.camera.camera_factory import get_camera
from raspberry.config.env_loader import EnvConfig

# Load environment configuration
EnvConfig()


def setup_logging():
    """Configure le logging pour l'application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


app = Flask(__name__)
cam = None
output_path = None
frame_lock = threading.Lock()
current_frame_bytes = None
import cv2


def capture_thread_func():
    """Thread qui capture continuellement les images de la caméra."""
    global current_frame_bytes
    logger = logging.getLogger("capture_thread")
    logger.info("Démarrage du thread de capture.")
    while True:
        try:
            # Utiliser la méthode capture_array() du wrapper
            frame = cam.capture_array()
            if frame is not None:
                # Convertir de RGB à BGR pour l'affichage correct (Picamera2 retourne RGB)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode(".jpg", frame_bgr)
                if ret:
                    with frame_lock:
                        current_frame_bytes = buffer.tobytes()
        except Exception as e:
            logger.error(f"Erreur dans le thread de capture : {e}")
        # Attendre un court instant pour ne pas surcharger le CPU
        time.sleep(0.04)  # Vise environ 25 images/seconde


def generate_frames():
    """Générateur pour le flux vidéo en multipart/x-mixed-replace."""
    global current_frame_bytes
    while True:
        with frame_lock:
            if current_frame_bytes:
                frame_to_send = current_frame_bytes
            else:
                frame_to_send = None

        if frame_to_send:
            frame_len = len(frame_to_send)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(frame_len).encode()
                + b"\r\n\r\n"
                + frame_to_send
                + b"\r\n"
            )

        # Attendre un peu pour que le client puisse suivre
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    """Route pour le streaming vidéo."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    """Page d'accueil qui affiche le flux vidéo."""
    return """
    <html>
        <head>
            <title>Aperçu pour Calibration</title>
            <style>
                body { font-family: sans-serif; text-align: center; margin-top: 2rem; background-color: #f0f0f0; }
                h1 { color: #333; }
                .container { padding: 20px; }
                img { border: 2px solid #ccc; max-width: 90%; height: auto; background-color: #fff; }
                .instructions { color: #666; font-size: 16px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Aperçu pour Calibration</h1>
                <p class="instructions">Positionnez le damier et appuyez sur 'Entrée' dans la console pour capturer une image.</p>
                <img src="/video_feed" alt="Flux vidéo de la caméra">
            </div>
        </body>
    </html>
    """


def main():
    """Fonction principale du script de capture."""
    setup_logging()
    logger = logging.getLogger("capture_for_calibration")

    global cam, output_path

    try:
        # Chemin vers le fichier de configuration
        config_path = repo_root / "raspberry" / "config" / "config.ini"
        if not config_path.exists():
            logger.error(
                f"Le fichier de configuration '{config_path}' est introuvable."
            )
            return

        # Lire la configuration
        config = configparser.ConfigParser()
        config.read(config_path)
        image_width = config.getint("camera", "width")
        image_height = config.getint("camera", "height")

        # Créer le dossier de sortie
        output_path = (
            repo_root / "output" / "calibration"
        )  # Défini pour la capture d'image
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Les images seront sauvegardées dans : {output_path}")

        # Initialiser la caméra
        logger.info(
            f"Initialisation de la caméra avec une résolution de {image_width}x{image_height}"
        )
        cam = get_camera(w=image_width, h=image_height, use_fake_camera=False)

        # Démarrer le thread de capture d'images en arrière-plan
        cap_thread = threading.Thread(target=capture_thread_func, daemon=True)
        cap_thread.start()

        # Attendre un peu que la première frame soit capturée
        time.sleep(0.5)

        # Démarrer le serveur Flask dans un thread séparé
        flask_thread = threading.Thread(
            target=lambda: app.run(
                host="0.0.0.0", port=5000, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        flask_thread.start()

        # Attendre que le serveur Flask soit prêt
        time.sleep(1)

        logger.info("=" * 50)
        logger.info("Serveur de streaming démarré.")
        logger.info(
            "Ouvrez votre navigateur et allez sur http://<ip_du_raspberry>:5000"
        )
        logger.info("=" * 50)

        logger.info(
            "\nAppuyez sur [Entrée] pour prendre une photo, ou Ctrl+C pour quitter.\n"
        )

        # Boucle pour attendre l'input de l'utilisateur
        while True:
            input()  # Attend que l'utilisateur appuie sur Entrée
            logger.info("Capture d'une image...")
            _, filepath = cam.capture_image(pictures_dir=output_path)
            logger.info(f"Image sauvegardée : {filepath.name}")

    except KeyboardInterrupt:
        logger.info("\nArrêt du script par l'utilisateur.")
    except Exception as e:
        logger.error(f"Une erreur fatale est survenue : {e}")
    finally:
        if cam and hasattr(cam, "stop"):
            cam.stop()
        logger.info("Caméra arrêtée.")


if __name__ == "__main__":
    main()
