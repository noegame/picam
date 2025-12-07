#!/usr/bin/env python3

"""
Tâche de communication avec le robot
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from multiprocessing import Queue
import logging

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------


def task_communication(queue: Queue):
    logger = logging.getLogger("task_communication")

    while True:
        try:
            message = queue.get()
            logger.debug(f"Message reçu: {message}")
            # Traiter le message ici (envoyer au robot, etc.)
        except Exception as e:
            logger.error(f"Erreur dans la tâche de communication: {e}")
            raise Exception(f"Erreur dans la tâche de communication: {e}")
