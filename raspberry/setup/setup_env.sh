#!/bin/bash

################################################################################
# Setup Environment Configuration Script
# This script automatically creates and configures a .env file for the PiCam
# project on a Raspberry Pi.
################################################################################

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
DEFAULT_CAMERA_WIDTH=2000
DEFAULT_CAMERA_HEIGHT=2000
DEFAULT_FLASK_PORT=5000
DEFAULT_FLASK_HOST="0.0.0.0"
DEFAULT_LOG_LEVEL="INFO"

# Get the script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RASPBERRY_DIR="$PROJECT_ROOT/raspberry"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   PiCam Environment Configuration Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if .env already exists
ENV_FILE="$RASPBERRY_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  .env file already exists at: $ENV_FILE${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✓ Keeping existing .env file${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Creating backup: ${ENV_FILE}.backup${NC}"
    cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

echo -e "${BLUE}Step 1: Camera Configuration${NC}"
echo "═══════════════════════════════════════════════════════"

read -p "Camera width (default: $DEFAULT_CAMERA_WIDTH): " CAMERA_WIDTH
CAMERA_WIDTH=${CAMERA_WIDTH:-$DEFAULT_CAMERA_WIDTH}

read -p "Camera height (default: $DEFAULT_CAMERA_HEIGHT): " CAMERA_HEIGHT
CAMERA_HEIGHT=${CAMERA_HEIGHT:-$DEFAULT_CAMERA_HEIGHT}

echo ""
echo -e "${BLUE}Step 2: Flask Server Configuration${NC}"
echo "═══════════════════════════════════════════════════════"

read -p "Flask host (default: $DEFAULT_FLASK_HOST): " FLASK_HOST
FLASK_HOST=${FLASK_HOST:-$DEFAULT_FLASK_HOST}

read -p "Flask port (default: $DEFAULT_FLASK_PORT): " FLASK_PORT
FLASK_PORT=${FLASK_PORT:-$DEFAULT_FLASK_PORT}

echo ""
echo -e "${BLUE}Step 3: Logging Configuration${NC}"
echo "═══════════════════════════════════════════════════════"

echo "Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
read -p "Log level (default: $DEFAULT_LOG_LEVEL): " LOG_LEVEL
LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}

echo ""
echo -e "${BLUE}Step 4: Optional Settings${NC}"
echo "═══════════════════════════════════════════════════════"

read -p "Use fake camera for testing? (y/N): " -n 1 -r
echo
USE_FAKE_CAMERA=$([ "$REPLY" = "y" ] && echo "true" || echo "false")

# Get hostname for reference
HOSTNAME=$(hostname)
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Create .env file
echo -e "${BLUE}Creating .env file...${NC}"

cat > "$ENV_FILE" << EOF
################################################################################
# PiCam Environment Configuration
# Generated: $(date)
# Hostname: $HOSTNAME
# Local IP: $LOCAL_IP
################################################################################

# Camera Configuration
CAMERA_WIDTH=$CAMERA_WIDTH
CAMERA_HEIGHT=$CAMERA_HEIGHT

# Flask Server Configuration
FLASK_HOST=$FLASK_HOST
FLASK_PORT=$FLASK_PORT

# Logging Configuration
LOG_LEVEL=$LOG_LEVEL

# Camera Mode (true for fake camera, false for real camera)
USE_FAKE_CAMERA=$USE_FAKE_CAMERA

# Project Paths (automatically set)
PROJECT_ROOT=$PROJECT_ROOT
RASPBERRY_DIR=$RASPBERRY_DIR

# Calibration
CALIBRATION_FILENAME=camera_calibration_${CAMERA_WIDTH}x${CAMERA_HEIGHT}.npz
EOF

echo -e "${GREEN}✓ .env file created successfully${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Environment Configuration Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "📍 .env file location:"
echo "   $ENV_FILE"
echo ""
echo "📷 Camera Settings:"
echo "   Resolution: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo ""
echo "🌐 Flask Server:"
echo "   Host: $FLASK_HOST"
echo "   Port: $FLASK_PORT"
echo "   URL: http://$LOCAL_IP:$FLASK_PORT"
echo ""
echo "📝 Logging:"
echo "   Level: $LOG_LEVEL"
echo ""
echo "🔧 Mode:"
echo "   Fake Camera: $USE_FAKE_CAMERA"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Verify the .env file: cat $ENV_FILE"
echo "2. Activate virtual environment: source /path/to/.venv/bin/activate"
echo "3. Install requirements: pip install -r requirements.txt"
echo "4. Run the capture script: python tools/capture_for_calibration.py"
echo ""
