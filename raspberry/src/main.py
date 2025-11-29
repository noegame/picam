#!/usr/bin/env python3

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import logging
import logging.config
from pathlib import Path
from multiprocessing import Process, Queue
from task_aruco_detection import task_aruco_detection
from task_communication import task_communication
from task_stream import task_stream

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def run_task(core_id, func, queue):
    """Exécute une tâche sur un cœur spécifique"""
    # Assignation du processus à un cœur spécifique
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass  # si pas permis, ça continue sans affinité forcée

    # Exécuter la tâche
    func(queue)

def main():
    # Configuration du logging avec chemin absolu
    logging_conf_path = Path(__file__).parent / 'logging.conf'
    logging.config.fileConfig(str(logging_conf_path))
    logger = logging.getLogger('main')
    queue = Queue()

    # Créer les processus
    p1 = Process(target=run_task, args=(0, task_aruco_detection, queue))
    p2 = Process(target=run_task, args=(1, task_communication, queue))
    p3 = Process(target=run_task, args=(2, task_stream, queue))

    # Démarrer les processus
    p1.start()
    p2.start()
    p3.start()
    
    # Attendre la fin des processus
    p1.join()
    p2.join()
    p3.join()

if __name__ == "__main__":
    main()
