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

def task_communication(queue:Queue):
    logger = logging.getLogger('task_communication')
    pass