#!/bin/bash
# =====================================================
# Script : setup_wifi.sh
# Objectif : Configurer automatiquement plusieurs réseaux Wi-Fi
#             sur Raspberry Pi OS Bookworm à partir d’un fichier JSON.
# Auteur : ChatGPT (GPT-5)
# =====================================================

set -e

CONFIG_FILE="./wifi_config.json"
WPA_SUPPLICANT_FILE="/etc/wpa_supplicant/wpa_supplicant.conf"

echo "=== Configuration automatique des réseaux Wi-Fi ==="

# Vérifie que le script est exécuté en root
if [ "$EUID" -ne 0 ]; then
  echo "Veuillez exécuter ce script avec sudo."
  exit 1
fi

# Vérifie que le fichier de configuration existe
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Fichier de configuration introuvable : $CONFIG_FILE"
  exit 1
fi

# Vérifie que jq est installé
if ! command -v jq &> /dev/null; then
  echo "Installation de jq (parseur JSON)..."
  apt update -qq
  apt install -y jq
fi

# Sauvegarde de l’ancien fichier wpa_supplicant.conf
if [ -f "$WPA_SUPPLICANT_FILE" ]; then
  cp "$WPA_SUPPLICANT_FILE" "${WPA_SUPPLICANT_FILE}.bak"
  echo "Sauvegarde de l’ancien fichier : ${WPA_SUPPLICANT_FILE}.bak"
fi

# Création du nouveau fichier
cat <<EOF > "$WPA_SUPPLICANT_FILE"
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=FR
EOF

# Lecture des réseaux Wi-Fi depuis le JSON
echo "Lecture du fichier $CONFIG_FILE..."
WIFI_COUNT=$(jq '.wifis | length' "$CONFIG_FILE")

for ((i=0; i<$WIFI_COUNT; i++)); do
  SSID=$(jq -r ".wifis[$i].ssid" "$CONFIG_FILE")
  PASSWORD=$(jq -r ".wifis[$i].password" "$CONFIG_FILE")

  # Vérifie que les valeurs ne sont pas vides
  if [[ -z "$SSID" || -z "$PASSWORD" || "$SSID" == "null" ]]; then
    echo "Réseau $i invalide, ignoré."
    continue
  fi

  cat <<EOF >> "$WPA_SUPPLICANT_FILE"

network={
    ssid="$SSID"
    psk="$PASSWORD"
    key_mgmt=WPA-PSK
    scan_ssid=1
}
EOF

  echo "Réseau ajouté : $SSID"
done

# Protection du fichier
chmod 600 "$WPA_SUPPLICANT_FILE"

# Redémarrage du service Wi-Fi
echo
echo "Redémarrage du service wpa_supplicant..."
systemctl restart wpa_supplicant
systemctl enable wpa_supplicant

echo
echo "Configuration Wi-Fi terminée avec succès !"
echo "Les réseaux définis dans $CONFIG_FILE seront utilisés automatiquement au démarrage."
