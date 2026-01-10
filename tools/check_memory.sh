#!/bin/bash
# Check the Raspberry Pi's memory usage

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠️]${NC} $1"
}

echo "$(printf '=%.0s' {1..60})"
echo "🧠 Vérification Mémoire Raspberry Pi"
echo "$(printf '=%.0s' {1..60})"

# Check total and free memory
print_info "Vérification de la mémoire RAM..."
MEM_INFO=$(free -h | awk 'NR==2{print "Total: "$2", Utilisé: "$3", Libre: "$4}')
if [ -n "$MEM_INFO" ]; then
    print_success "Mémoire RAM: $MEM_INFO"
else
    print_error "Impossible de récupérer les informations RAM."
fi

# Check disk space
print_info "Vérification de l'espace disque..."
DISK_INFO=$(df -h / | awk 'NR==2{print "Total: "$2", Utilisé: "$3", Disponible: "$4", Utilisation: "$5}')
if [ -n "$DISK_INFO" ]; then
    print_success "Espace disque (/): $DISK_INFO"
else
    print_error "Impossible de récupérer les informations d'espace disque."
fi

echo "$(printf '=%.0s' {1..60})"
