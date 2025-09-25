#!/bin/bash
# Script de démarrage pour le système PiCam

echo "Démarrage du système PiCam..."
cd "$(dirname "$0")"

# Vérifier si Python3 est disponible
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python3 n'est pas installé"
    exit 1
fi

# Démarrer le programme principal
python3 src/main.py