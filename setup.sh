#!/bin/bash
# Script d'installation et configuration PiCam

# Installation et mises à jour des paquets
sudo apt update
sudo apt upgrade -y

# Installation de libcamera
sudo apt install -y libcamera-apps

# Installation de python et des bibliothèques nécessaires
sudo apt install -y python3 
sudo apt install -y python3-pip 
sudo apt install -y python3-picamera2
sudo apt install -y python3-socket
sudo apt install -y python3-cv2
