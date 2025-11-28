#!/bin/bash
# Script d'installation et configuration PiCam

# === Vérification des privilèges sudo ===
if [ "$EUID" -ne 0 ]; then
  echo "Ce script doit être exécuté avec sudo."
  exit 1
fi

# === Mise à jour du système et installation des paquets nécessaires ===
# Installation et mises à jour des paquets
sudo apt update
sudo apt upgrade -y

# Installation de libcamera
sudo apt install -y libcamera-apps

# Installation de python et des bibliothèques nécessaires
sudo apt install -y python3 
sudo apt install -y python3-pip 
sudo apt install -y python3-picamera2
sudo apt install -y python3-opencv

# === Configuration du Wi-Fi ===
sudo bash ./setup_wifi.sh

# Redémarrage de la Raspberry Pi
echo "Un redémarrage est recommandé : sudo reboot"
