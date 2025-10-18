#!/bin/bash
# Script pour configurer une IP statique sur Raspberry Pi OS Bookworm
# Fonctionne avec systemd-networkd (par défaut depuis Bookworm)

set -e

echo "=== Configuration d'une IP statique sur Raspberry Pi OS Bookworm ==="

# Vérifie que l'utilisateur est root
if [ "$EUID" -ne 0 ]; then
  echo "Veuillez exécuter ce script avec sudo."
  exit 1
fi

# Liste les interfaces réseau
echo "Interfaces détectées :"
ip -o link show | awk -F': ' '{print $2}' | grep -v lo
echo
read -p "Entrez le nom de l'interface (ex: eth0 ou wlan0) : " IFACE

# Demande les informations réseau
read -p "Adresse IP (ex: 192.168.1.50) : " IP_ADDR
read -p "Masque CIDR (ex: 24 pour 255.255.255.0) : " CIDR
read -p "Passerelle (Gateway) (ex: 192.168.1.1) : " GATEWAY
read -p "Serveur DNS (ex: 8.8.8.8) : " DNS

# Chemin du fichier de config
CONF_FILE="/etc/systemd/network/10-${IFACE}.network"

echo
echo "Création du fichier de configuration : ${CONF_FILE}"
cat <<EOF > "$CONF_FILE"
[Match]
Name=${IFACE}

[Network]
Address=${IP_ADDR}/${CIDR}
Gateway=${GATEWAY}
DNS=${DNS}
EOF

echo "Configuration écrite :"
cat "$CONF_FILE"
echo

# Active systemd-networkd et systemd-resolved
echo "Activation des services réseau..."
systemctl enable systemd-networkd --now
systemctl enable systemd-resolved --now

# Liens symboliques (utile si resolv.conf n’est pas bien configuré)
ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

echo
echo "Redémarrage du service réseau..."
systemctl restart systemd-networkd

echo
echo "Configuration terminée !"
echo "L'adresse IP ${IP_ADDR} est désormais statique sur ${IFACE}."
echo "Un redémarrage est recommandé : sudo reboot"
