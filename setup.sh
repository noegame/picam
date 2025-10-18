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
