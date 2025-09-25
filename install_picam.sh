#!/bin/bash
# Script d'installation et configuration PiCam

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "================================================"
    echo "🎥 Installation PiCam - Scripts Shell"
    echo "================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[ÉTAPE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[ATTENTION]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERREUR]${NC} $1"
}

# Vérifier si on est sur Raspberry Pi
check_raspberry_pi() {
    print_step "Vérification du système..."
    
    if [ -f /proc/device-tree/model ]; then
        model=$(cat /proc/device-tree/model)
        if [[ $model == *"Raspberry Pi"* ]]; then
            print_success "Système détecté: $model"
            return 0
        fi
    fi
    
    print_warning "Système non-Raspberry Pi détecté"
    return 1
}

# Vérifier et activer la caméra
setup_camera() {
    print_step "Configuration de la caméra..."
    
    # Vérifier si la caméra est activée
    if ! grep -q "^camera_auto_detect=1" /boot/config.txt 2>/dev/null; then
        print_warning "La caméra pourrait ne pas être activée"
        echo "Pour l'activer manuellement:"
        echo "  sudo raspi-config"
        echo "  -> Interface Options -> Camera -> Enable"
    else
        print_success "Caméra activée dans /boot/config.txt"
    fi
}

# Installer les dépendances
install_dependencies() {
    print_step "Installation des dépendances..."
    
    # Mettre à jour les paquets
    sudo apt update
    
    # Identifier la version de Raspberry Pi OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_success "OS détecté: $PRETTY_NAME"
        
        # Raspberry Pi OS Bookworm (recommandé)
        if [[ $VERSION_CODENAME == "bookworm" ]]; then
            print_step "Installation de libcamera-apps (Bookworm)..."
            sudo apt install -y libcamera-apps
            
        # Raspberry Pi OS Bullseye (legacy)
        elif [[ $VERSION_CODENAME == "bullseye" ]]; then
            print_step "Installation des outils legacy (Bullseye)..."
            sudo apt install -y raspberrypi-kernel-headers
            
        else
            print_warning "Version OS non reconnue, installation des deux..."
            sudo apt install -y libcamera-apps raspberrypi-kernel-headers
        fi
    fi
    
    # Installer fswebcam comme fallback
    print_step "Installation de fswebcam (fallback)..."
    sudo apt install -y fswebcam
    
    print_success "Dépendances installées"
}

# Configurer les permissions
setup_permissions() {
    print_step "Configuration des permissions..."
    
    # Ajouter l'utilisateur au groupe video
    sudo usermod -a -G video $USER
    
    # Rendre les scripts exécutables
    chmod +x capture_images.sh
    chmod +x simple_capture.sh
    chmod +x start_picam.sh
    
    print_success "Permissions configurées"
    print_warning "Redémarrez votre session pour que les permissions prennent effet"
}

# Tester la caméra
test_camera() {
    print_step "Test de la caméra..."
    
    mkdir -p test_data
    
    # Tester libcamera-still
    if command -v libcamera-still &> /dev/null; then
        print_step "Test avec libcamera-still..."
        if libcamera-still --width 640 --height 480 --output test_data/test_libcamera.jpg --timeout 1000 --nopreview; then
            print_success "Test libcamera-still réussi!"
            rm -f test_data/test_libcamera.jpg
        else
            print_error "Échec du test libcamera-still"
        fi
    fi
    
    # Tester raspistill
    if command -v raspistill &> /dev/null; then
        print_step "Test avec raspistill..."
        if raspistill -w 640 -h 480 -o test_data/test_raspistill.jpg -t 1000 -n; then
            print_success "Test raspistill réussi!"
            rm -f test_data/test_raspistill.jpg
        else
            print_error "Échec du test raspistill"
        fi
    fi
    
    rmdir test_data 2>/dev/null
}

# Afficher les instructions d'utilisation
show_usage() {
    echo
    print_step "Instructions d'utilisation:"
    echo
    echo "📋 Scripts disponibles:"
    echo "  • ./capture_images.sh    - Script complet avec options"
    echo "  • ./simple_capture.sh    - Script simple et rapide"
    echo "  • ./start_picam.sh       - Lance le programme Python"
    echo
    echo "🚀 Exemples d'utilisation:"
    echo "  ./simple_capture.sh                    # Capture simple toutes les 5s"
    echo "  ./capture_images.sh                    # Capture avec logs"
    echo "  ./capture_images.sh -i 10              # Toutes les 10 secondes"
    echo "  ./capture_images.sh -d photos -i 15    # Dans dossier 'photos/'"
    echo "  ./capture_images.sh -s                 # Voir les statistiques"
    echo
    echo "⚠️  Pour arrêter: Ctrl+C"
    echo
}

# Menu principal
main_menu() {
    print_header
    
    echo "Choisissez une option:"
    echo "1) Installation complète (recommandé)"
    echo "2) Vérifier le système seulement"
    echo "3) Installer les dépendances seulement"
    echo "4) Tester la caméra"
    echo "5) Afficher les instructions"
    echo "6) Quitter"
    echo
    read -p "Votre choix [1-6]: " choice
    
    case $choice in
        1)
            check_raspberry_pi
            setup_camera
            install_dependencies
            setup_permissions
            test_camera
            show_usage
            ;;
        2)
            check_raspberry_pi
            setup_camera
            ;;
        3)
            install_dependencies
            ;;
        4)
            test_camera
            ;;
        5)
            show_usage
            ;;
        6)
            echo "Au revoir!"
            exit 0
            ;;
        *)
            print_error "Option invalide"
            main_menu
            ;;
    esac
}

# Lancer le menu si aucun argument
if [ $# -eq 0 ]; then
    main_menu
else
    # Installation automatique si argument --auto
    if [ "$1" = "--auto" ]; then
        print_header
        check_raspberry_pi
        setup_camera
        install_dependencies
        setup_permissions
        show_usage
    else
        echo "Usage: $0 [--auto]"
        echo "  --auto    Installation automatique"
        echo "  (sans arg) Menu interactif"
    fi
fi