#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
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
# Fonctions principales
# ---------------------------------------------------------------------------


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

    p1, p2, p3 = None, None, None

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
        )
        p2 = Process(
            target=run_task, args=(1, task_communication, queue_aruco_detection_to_com)
        )
        p3 = Process(
            target=run_task, args=(2, task_stream, queue_aruco_detection_to_stream)
        )

        # Démarrer les processus
        p1.start()
        p2.start()
        p3.start()

        # Attendre la fin des processus
        p1.join()
        p2.join()
        p3.join()

    except Exception as e:
        logger.error(f"Erreur dans le main: {e}")
        # Clean up
        if p1:
            p1.terminate()
        if p2:
            p2.terminate()
        if p3:
            p3.terminate()
