#!/bin/bash

# Script to add WiFi networks from wifi.json to Raspberry Pi known networks
# This allows the Pi to automatically connect to these networks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIFI_CONFIG_FILE="$SCRIPT_DIR/wifi.json"
WPA_CONFIG_FILE="/etc/wpa_supplicant/wpa_supplicant.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Check if wifi.json exists
if [[ ! -f "$WIFI_CONFIG_FILE" ]]; then
    echo -e "${RED}Error: $WIFI_CONFIG_FILE not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting WiFi setup...${NC}"

# Check if jq is installed for JSON parsing
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}jq not found. Installing jq...${NC}"
    apt-get update
    apt-get install -y jq
fi

# Backup the original wpa_supplicant.conf
if [[ -f "$WPA_CONFIG_FILE" ]]; then
    cp "$WPA_CONFIG_FILE" "$WPA_CONFIG_FILE.backup"
    echo -e "${GREEN}Backed up original config to $WPA_CONFIG_FILE.backup${NC}"
fi

# Parse JSON and add networks
networks=$(jq -r '.wifis[] | "\(.ssid)|\(.password)"' "$WIFI_CONFIG_FILE")

while IFS='|' read -r ssid password; do
    # Skip empty lines
    [[ -z "$ssid" ]] && continue
    
    echo -e "${YELLOW}Adding WiFi network: $ssid${NC}"
    
    # Check if network already exists in nmcli
    if nmcli connection show | grep -q "^$ssid\s"; then
        echo -e "${YELLOW}  Network '$ssid' already exists. Updating...${NC}"
        nmcli connection delete "$ssid" 2>/dev/null || true
    fi
    
    # Add the network using nmcli (NetworkManager)
    # This works with modern Raspbian/Raspberry Pi OS
    nmcli device wifi connect "$ssid" password "$password" --ask 2>/dev/null || \
    nmcli connection add type wifi ifname wlan0 con-name "$ssid" ssid "$ssid" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$password" 2>/dev/null || {
        echo -e "${RED}  Failed to add network '$ssid' with nmcli${NC}"
        continue
    }
    
    echo -e "${GREEN}  Successfully added network: $ssid${NC}"
done <<< "$networks"

# Restart networking
echo -e "${YELLOW}Restarting networking...${NC}"
systemctl restart networking || systemctl restart wpa_supplicant || true

echo -e "${GREEN}WiFi setup complete!${NC}"
echo -e "${YELLOW}The Pi will now automatically connect to these networks when available.${NC}"

# Show configured networks
echo -e "${YELLOW}Configured networks:${NC}"
nmcli connection show --active 2>/dev/null | grep -E "^NAME|^TYPE" || true
