#!/bin/bash
# Script d'installation et configuration PiCam

# Vérification des privilèges sudo
if [ "$EUID" -ne 0 ]; then
  echo "Ce script doit être exécuté avec sudo."
  exit 1
fi

# Installation et mises à jour des paquets
sudo apt update
sudo apt upgrade -y

# Installation de libcamera
sudo apt install -y libcamera-apps

# Installation de python et des bibliothèques nécessaires
sudo apt install -y python3 
sudo apt install -y python3-pip 
sudo apt install -y python3-picamera2
#sudo apt install -y python3-opencv

# === Configuration d'une IP statique ===
echo "Configuration de l'IP statique..."

STATIC_IP="192.168.1.50"            # IP fixe souhaitée
ROUTER_IP="192.168.1.1"             # IP de la passerelle / routeur
DNS_SERVERS="192.168.1.1 8.8.8.8"   # DNS local + Google
INTERFACE="wlan0"                   # Interface Wi-Fi
SUBNET="/24"                        # Masque de sous-réseau

# Sauvegarde du fichier /etc/dhcpcd.conf.
cp /etc/dhcpcd.conf /etc/dhcpcd.conf.backup_$(date +%Y%m%d_%H%M%S)
# Ajout de la configuration statique

cat <<EOF >> /etc/dhcpcd.conf

# --- Configuration statique ajoutée automatiquement ---
interface $INTERFACE
static ip_address=${STATIC_IP}${SUBNET}
static routers=${ROUTER_IP}
static domain_name_servers=${DNS_SERVERS}
# --------------------------------------------------------
EOF

sudo service dhcpcd restart

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

sudo systemctl restart wpa_supplicant
sudo wpa_cli -i wlan0 reconfigure || true
echo "Configuration Wi-Fi terminée."

# Redémarrage de la Raspberry Pi
echo "Notez bien mon adresse IP : $STATIC_IP"
hostname -I
echo "La raspberry Pi va redémarrer."
sleep 5
sudo reboot
