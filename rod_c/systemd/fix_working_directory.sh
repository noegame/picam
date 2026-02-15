#!/bin/bash
# Script to fix WorkingDirectory in systemd service files
# This ensures debug images are saved to the correct location

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Fixing systemd WorkingDirectory ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if services are installed
if [ ! -f "/etc/systemd/system/rod-detection.service" ]; then
    echo "Error: rod-detection.service not installed"
    echo "Run 'sudo ./install.sh' first"
    exit 1
fi

# Backup existing service files
echo "Creating backups..."
sudo cp /etc/systemd/system/rod-detection.service \
        /etc/systemd/system/rod-detection.service.backup
sudo cp /etc/systemd/system/rod-communication.service \
        /etc/systemd/system/rod-communication.service.backup

echo "Backups created in /etc/systemd/system/*.backup"
echo ""

# Fix rod-detection.service
echo "Updating rod-detection.service..."
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_ROOT|g" \
    /etc/systemd/system/rod-detection.service

# Fix rod-communication.service
echo "Updating rod-communication.service..."
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_ROOT|g" \
    /etc/systemd/system/rod-communication.service

echo ""
echo "=== Changes applied ==="
echo ""

# Show the modified lines
echo "rod-detection.service WorkingDirectory:"
grep "^WorkingDirectory=" /etc/systemd/system/rod-detection.service
echo ""

echo "rod-communication.service WorkingDirectory:"
grep "^WorkingDirectory=" /etc/systemd/system/rod-communication.service
echo ""

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Restart services: sudo systemctl restart rod.target"
echo "  2. Check logs: sudo journalctl -u rod-detection.service -f"
echo "  3. Verify debug images: ls -lh $PROJECT_ROOT/pictures/debug/"
echo ""
echo "To restore backups if needed:"
echo "  sudo cp /etc/systemd/system/rod-detection.service.backup \\"
echo "          /etc/systemd/system/rod-detection.service"
echo "  sudo systemctl daemon-reload"