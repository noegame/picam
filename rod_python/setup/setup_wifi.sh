#!/bin/bash

# Script to interactively add WiFi networks to Raspberry Pi known networks
# This allows the Pi to automatically connect to these networks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

echo -e "Starting WiFi setup..."
echo -e "This script will add WiFi networks to your Raspberry Pi"
echo ""

# Loop to add multiple WiFi networks
while true; do
    echo -e "Enter WiFi network SSID (or press Enter to finish):"
    read -r ssid
    
    # Exit if SSID is empty
    if [[ -z "$ssid" ]]; then
        echo -e "${GREEN}No more networks to add.${NC}"
        break
    fi
    
    echo -e "Enter password for '$ssid':"
    read -rs password
    echo ""
    
    # Check if password is empty
    if [[ -z "$password" ]]; then
        echo -e "${RED}Error: Password cannot be empty. Skipping network.${NC}"
        echo ""
        continue
    fi
    
    echo -e "Adding WiFi network: $ssid"
    
    # Check if network already exists in nmcli
    if nmcli connection show | grep -q "^$ssid\s"; then
        echo -e "  Network '$ssid' already exists. Updating..."
        nmcli connection delete "$ssid" 2>/dev/null || true
    fi
    
    # Add the network using nmcli (NetworkManager)
    # This works with modern Raspbian/Raspberry Pi OS
    nmcli device wifi connect "$ssid" password "$password" --ask 2>/dev/null || \
    nmcli connection add type wifi ifname wlan0 con-name "$ssid" ssid "$ssid" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$password" 2>/dev/null || {
        echo -e "${RED}  Failed to add network '$ssid' with nmcli${NC}"
        echo ""
        continue
    }
    
    echo -e "${GREEN}  Successfully added network: $ssid${NC}"
    echo ""
done

# Restart networking
echo -e "Restarting networking..."
systemctl restart networking || systemctl restart wpa_supplicant || true

echo -e "${GREEN}WiFi setup complete!${NC}"
echo -e "The Pi will now automatically connect to these networks when available."

# Show configured networks
echo -e "Configured networks:"
nmcli connection show --active 2>/dev/null | grep -E "^NAME|^TYPE" || true
