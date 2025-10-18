#!/bin/bash
# =====================================================
# Script : set_static_ip.sh (non-interactif)
# Objectif : Définir une adresse IP statique sur Raspberry Pi OS Bookworm
# Auteur : ChatGPT (GPT-5)
# =====================================================

set -e

# -----------------------------
# CONFIGURATION À MODIFIER ICI
# -----------------------------
IFACE="eth0"            # Nom de l'interface à configurer (eth0 ou wlan0)
IP_ADDR="192.168.1.50"  # Adresse IP statique
CIDR="24"               # Masque réseau en CIDR (ex: 24 → 255.255.255.0)
GATEWAY="192.168.1.1"   # Passerelle par défaut
DNS="8.8.8.8"           # Serveur DNS

# -----------------------------
# SCRIPT
# -----------------------------
echo "Configuration automatique d'une IP statique sur $IFACE"

# Vérifie que le script est exécuté en root
if [ "$EUID" -ne 0 ]; then
  echo "Veuillez exécuter ce script avec sudo."
  exit 1
fi

# Vérifie que l’interface existe
if ! ip link show "$IFACE" &> /dev/null; then
  echo "Interface '$IFACE' introuvable."
  exit 1
fi

# Chemin du fichier de configuration systemd-networkd
NETWORK_FILE="/etc/systemd/network/10-${IFACE}.network"

echo "Création du fichier de configuration réseau : $NETWORK_FILE"

cat <<EOF > "$NETWORK_FILE"
[Match]
Name=${IFACE}

[Network]
Address=${IP_ADDR}/${CIDR}
Gateway=${GATEWAY}
DNS=${DNS}
EOF

echo "Fichier réseau créé :"
cat "$NETWORK_FILE"

# Active systemd-networkd et systemd-resolved
echo "Activation des services réseau..."
systemctl enable systemd-networkd --now
systemctl enable systemd-resolved --now

# Assure que le résolveur DNS pointe vers systemd
ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

# Redémarre les services pour appliquer la configuration
echo "Redémarrage du service réseau..."
systemctl restart systemd-networkd

echo "Configuration terminée."
echo "L'adresse IP ${IP_ADDR} est maintenant statique sur ${IFACE}."
echo "Redémarrage recommandé : sudo reboot"
