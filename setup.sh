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
sudo apt install -y python3-opencv

# Configuration d'une IP statique pour la Raspberry Pi
echo "Configuration de l'IP statique..."
sudo nano /etc/dhcpcd.conf
interface eth0
static ip_address=192.168.1.50/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8

# Connexion au wifi du club

set -e

SSID="RobotESEO-IoT"
PASSWORD="RobotE5E0*"
WPA_CONF="/etc/wpa_supplicant/wpa_supplicant.conf"

echo "Configuration du Wi-Fi : $SSID"

# Vérifie si le réseau est déjà présent
if grep -q "$SSID" "$WPA_CONF"; then
    echo "Le réseau $SSID est déjà configuré."
else
    echo "Ajout du réseau $SSID à $WPA_CONF..."
    sudo bash -c "cat >> $WPA_CONF <<EOF

network={
    ssid=\"$SSID\"
    psk=\"$PASSWORD\"
    key_mgmt=WPA-PSK
}
EOF"
fi

# Redémarre le service pour appliquer les changements
echo "Redémarrage du service wpa_supplicant..."
sudo systemctl restart wpa_supplicant

# Optionnel : tenter de se connecter immédiatement
sudo wpa_cli -i wlan0 reconfigure || true

echo "Configuration Wi-Fi terminée."
echo "La raspberry Pi va redémarrer."
wait 5 

# Redémarrage de la Raspberry Pi
sudo reboot