#!/bin/bash
# quick_camera_test.sh - Test rapide caméra sans Python
# Usage: ./quick_camera_test.sh [IP_PC] [USERNAME_PC]

# Configuration
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
PHOTO_NAME="camera_test_${TIMESTAMP}.jpg"
TIMEOUT=2000  # 2 secondes

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
echo "🎯 Test Rapide Caméra Raspberry Pi (Shell)"
echo "$(printf '=%.0s' {1..60})"

# Fonction pour prendre la photo
take_photo() {
    print_info "Test de la caméra..."
    
    # Vérifier si rpicam-still est disponible (Bookworm)
    if command -v rpicam-still &> /dev/null; then
        print_info "Utilisation de rpicam-still (Raspberry Pi OS Bookworm)"
        
        if rpicam-still --output "$PHOTO_NAME" --timeout $TIMEOUT --width 1920 --height 1080 --nopreview 2>/dev/null; then
            print_success "Photo prise avec rpicam-still: $PHOTO_NAME"
            return 0
        else
            print_error "Échec avec rpicam-still"
            return 1
        fi
        
    # Vérifier si libcamera-still est disponible
    elif command -v libcamera-still &> /dev/null; then
        print_info "Utilisation de libcamera-still"
        
        if libcamera-still --output "$PHOTO_NAME" --timeout $TIMEOUT --width 1920 --height 1080 --nopreview 2>/dev/null; then
            print_success "Photo prise avec libcamera-still: $PHOTO_NAME"
            return 0
        else
            print_error "Échec avec libcamera-still"
            return 1
        fi
        
    # Vérifier si raspistill est disponible (legacy)
    elif command -v raspistill &> /dev/null; then
        print_info "Utilisation de raspistill (legacy)"
        
        if raspistill -o "$PHOTO_NAME" -t $TIMEOUT -w 1920 -h 1080 -n 2>/dev/null; then
            print_success "Photo prise avec raspistill: $PHOTO_NAME"
            return 0
        else
            print_error "Échec avec raspistill"
            return 1
        fi
        
    else
        print_error "Aucun outil de capture trouvé!"
        print_error "Installez: sudo apt install libcamera-apps"
        return 1
    fi
}

# Fonction pour envoyer la photo
send_photo() {
    local pc_ip=$1
    local username=$2
    
    if [ ! -f "$PHOTO_NAME" ]; then
        print_error "Photo non trouvée: $PHOTO_NAME"
        return 1
    fi
    
    local file_size=$(stat -c%s "$PHOTO_NAME" 2>/dev/null || echo "0")
    print_info "Taille du fichier: $file_size bytes"
    
    if [ "$file_size" -eq 0 ]; then
        print_error "Fichier vide ou corrompu"
        return 1
    fi
    
    print_info "Envoi vers ${username}@${pc_ip}..."
    
    # Créer le dossier Downloads s'il n'existe pas sur le PC distant
    ssh "${username}@${pc_ip}" "mkdir -p ~/Downloads" 2>/dev/null
    
    # Envoyer avec SCP
    if scp "$PHOTO_NAME" "${username}@${pc_ip}:~/Downloads/" 2>/dev/null; then
        print_success "Photo envoyée dans ~/Downloads/ sur $pc_ip"
        print_success "Fichier: $PHOTO_NAME"
        return 0
    else
        print_error "Échec de l'envoi SCP"
        return 1
    fi
}

# Fonction de nettoyage
cleanup() {
    if [ -f "$PHOTO_NAME" ] && [ "$1" = "success" ]; then
        rm -f "$PHOTO_NAME"
        print_info "Fichier local supprimé: $PHOTO_NAME"
    elif [ -f "$PHOTO_NAME" ]; then
        print_warning "Photo conservée localement: $PHOTO_NAME"
        print_info "Pour la récupérer manuellement:"
        print_info "  scp $(whoami)@$(hostname -I | awk '{print $1}'):$(pwd)/$PHOTO_NAME ~/Downloads/"
    fi
}

# Script principal
main() {
    # Prendre la photo
    if ! take_photo; then
        print_error "Test échoué - Problème avec la caméra"
        exit 1
    fi
    
    print_success "Test caméra réussi!"
    print_info "Fichier local: $(pwd)/$PHOTO_NAME"
    
    # Vérifier les arguments pour l'envoi
    if [ $# -ge 2 ]; then
        PC_IP=$1
        USERNAME=$2
        
        echo
        print_info "Tentative d'envoi vers ${USERNAME}@${PC_IP}..."
        
        if send_photo "$PC_IP" "$USERNAME"; then
            echo
            print_success "🎉 Test complet réussi!"
            cleanup "success"
        else
            echo
            print_warning "Envoi échoué, mais photo disponible localement"
            cleanup "failed"
        fi
        
    else
        echo
        print_info "💡 Pour envoyer automatiquement sur votre PC:"
        print_info "   $0 IP_DE_VOTRE_PC VOTRE_USERNAME"
        print_info "   Exemple: $0 192.168.1.10 votrenom"
        echo
        cleanup "manual"
    fi
}

# Vérification des permissions
if [ ! -w "." ]; then
    print_error "Pas de permission d'écriture dans le répertoire courant"
    exit 1
fi

# Lancer le script principal
main "$@"