#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys
import signal
import logging
import logging.config
from pathlib import Path
from multiprocessing import Process, Queue

from raspberry.config.env_loader import EnvConfig
from raspberry.src.task_aruco_detection import task_aruco_detection
from raspberry.src.task_communication import task_communication
from raspberry.src.task_stream import task_stream

# Load environment configuration
EnvConfig()

# ---------------------------------------------------------------------------
# Variables globales
# ---------------------------------------------------------------------------

processes = []

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def signal_handler(signum, frame):
    """Gestionnaire de signaux pour arrêt propre (Ctrl+C)"""
    logger = logging.getLogger("main")
    logger.info("Signal d'arrêt reçu (Ctrl+C). Arrêt gracieux des processus...")

    # Terminer tous les processus
    for process in processes:
        if process and process.is_alive():
            logger.info(f"Arrêt du processus {process.name}...")
            process.terminate()

    # Attendre que tous les processus se terminent (avec timeout)
    for process in processes:
        if process:
            process.join(timeout=3)
            if process.is_alive():
                logger.warning(
                    f"Processus {process.name} ne s'est pas arrêté, forçage du kill..."
                )
                process.kill()
                process.join()

    logger.info("Tous les processus ont été arrêtés. Fermeture en cours...")
    sys.exit(0)


def run_task(core_id, func, *args):
    """Exécute une tâche sur un cœur spécifique"""
    # Assignation du processus à un cœur spécifique
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass  # si pas permis, ça continue sans affinité forcée

    # Exécuter la tâche
    func(*args)


if __name__ == "__main__":
    # Configuration du logging avec chemin absolu
    repo_root = Path(__file__).resolve().parents[2]
    log_file_path = repo_root / "logs" / "aruco_detection_flow.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging_conf_path = repo_root / "raspberry" / "config" / "logging.conf"
    logging.config.fileConfig(
        str(logging_conf_path), defaults={"log_file": str(log_file_path)}
    )
    logger = logging.getLogger("main")

    # Configurer le gestionnaire de signaux pour Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:

        queue_aruco_detection_to_stream = Queue()
        queue_aruco_detection_to_com = Queue()

        # Créer les processus
        p1 = Process(
            target=run_task,
            args=(
                0,
                task_aruco_detection,
                queue_aruco_detection_to_stream,
                queue_aruco_detection_to_com,
            ),
            name="TaskArucoDetection",
        )
        p2 = Process(
            target=run_task,
            args=(1, task_communication, queue_aruco_detection_to_com),
            name="TaskCommunication",
        )
        p3 = Process(
            target=run_task,
            args=(2, task_stream, queue_aruco_detection_to_stream),
            name="TaskStream",
        )

        # Stocker les références globales
        processes = [p1, p2, p3]

        # Démarrer les processus
        p1.start()
        p2.start()
        p3.start()

        logger.info(
            "Tous les processus ont démarré. Appuyez sur Ctrl+C pour arrêter..."
        )

        # Attendre la fin des processus
        p1.join()
        p2.join()
        p3.join()

    except KeyboardInterrupt:
        # Gérer l'exception KeyboardInterrupt directement
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Erreur dans le main: {e}")
        # Clean up
        for process in processes:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()
