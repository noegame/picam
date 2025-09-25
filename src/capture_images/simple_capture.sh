#!/bin/bash
# Script Shell Simple - Capture d'images PiCam
# Version simplifiée qui prend une photo toutes les 5 secondes

# Configuration
INTERVAL=5
DATA_DIR="data"

# Créer le répertoire s'il n'existe pas
mkdir -p "$DATA_DIR"

echo "======================================"
echo "🎥 PiCam Capture Simple - Shell"
echo "======================================"
echo "Intervalle: ${INTERVAL} secondes"
echo "Dossier: ${DATA_DIR}/"
echo "Appuyez sur Ctrl+C pour arrêter"
echo "--------------------------------------"

# Fonction de nettoyage
cleanup() {
    echo
    echo "Arrêt de la capture..."
    exit 0
}

# Capturer le signal Ctrl+C
trap cleanup SIGINT

# Compteur d'images
count=0

# Boucle principale
while true; do
    # Générer le nom de fichier avec timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    filename="picam_${timestamp}.jpg"
    filepath="${DATA_DIR}/${filename}"
    
    # Essayer libcamera-still (Raspberry Pi OS Bookworm)
    if command -v libcamera-still &> /dev/null; then
        if libcamera-still --width 1920 --height 1080 --output "$filepath" --timeout 1000 --nopreview &>/dev/null; then
            ((count++))
            echo "[$(date '+%H:%M:%S')] ✅ Image $count: $filename"
        else
            echo "[$(date '+%H:%M:%S')] ❌ Échec de capture"
        fi
    
    # Essayer raspistill (Raspberry Pi OS Bullseye)
    elif command -v raspistill &> /dev/null; then
        if raspistill -w 1920 -h 1080 -o "$filepath" -t 1000 -n &>/dev/null; then
            ((count++))
            echo "[$(date '+%H:%M:%S')] ✅ Image $count: $filename"
        else
            echo "[$(date '+%H:%M:%S')] ❌ Échec de capture"
        fi
    
    # Fallback avec fswebcam
    elif command -v fswebcam &> /dev/null; then
        if fswebcam -r 1920x1080 --no-banner --save "$filepath" &>/dev/null; then
            ((count++))
            echo "[$(date '+%H:%M:%S')] ✅ Image $count: $filename (webcam)"
        else
            echo "[$(date '+%H:%M:%S')] ❌ Échec de capture"
        fi
    
    else
        echo "❌ Aucun outil de capture trouvé!"
        echo "Installez: sudo apt install libcamera-apps"
        exit 1
    fi
    
    # Attendre 5 secondes
    sleep $INTERVAL
done