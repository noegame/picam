#!/bin/bash
# Script d'installation et configuration PiCam

# Installation et mises à jour des paquets
sudo apt update
sudo apt upgrade -y

# Installation de libcamera
sudo apt install -y libcamera-apps

# Installation de python et des bibliothèques nécessaires
sudo apt install -y python3 python3-pip python3-picamera2