#!/bin/bash
# Script Shell - Capture automatique d'images PiCam
# Prend une photo toutes les 5 secondes et les sauvegarde dans le dossier data/

# Configuration
INTERVAL= 5                  # Intervalle en secondes
DATA_DIR="data"              # Répertoire de sauvegarde
RESOLUTION="1920x1080"       # Résolution des images
LOG_FILE="picam.log"         # Fichier de log

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} - ${level} - ${message}" | tee -a "${LOG_FILE}"
}

# Fonction d'affichage coloré
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log_message "INFO" "$1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log_message "SUCCESS" "$1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log_message "WARNING" "$1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log_message "ERROR" "$1"
}

# Fonction pour vérifier les dépendances
check_dependencies() {
    print_info "Vérification des dépendances..."
    
    # Vérifier si libcamera-still est disponible (Raspberry Pi OS Bookworm)
    if command -v libcamera-still &> /dev/null; then
        CAMERA_CMD="libcamera-still"
        CAMERA_TYPE="libcamera"
        print_success "libcamera-still trouvé (Raspberry Pi OS Bookworm)"
        return 0
    fi
    
    # Vérifier si raspistill est disponible (Raspberry Pi OS Bullseye et antérieur)
    if command -v raspistill &> /dev/null; then
        CAMERA_CMD="raspistill"
        CAMERA_TYPE="raspistill"
        print_success "raspistill trouvé (Raspberry Pi OS Bullseye)"
        return 0
    fi
    
    # Vérifier si fswebcam est disponible (fallback pour webcam USB)
    if command -v fswebcam &> /dev/null; then
        CAMERA_CMD="fswebcam"
        CAMERA_TYPE="fswebcam"
        print_warning "Utilisation de fswebcam (webcam USB)"
        return 0
    fi
    
    print_error "Aucun outil de capture d'image trouvé!"
    print_error "Installez: sudo apt install libcamera-apps"
    print_error "Ou pour legacy: sudo apt install raspberrypi-kernel-headers"
    return 1
}

# Fonction pour créer le répertoire de données
setup_data_directory() {
    if [ ! -d "${DATA_DIR}" ]; then
        mkdir -p "${DATA_DIR}"
        print_success "Répertoire ${DATA_DIR}/ créé"
    else
        print_info "Répertoire ${DATA_DIR}/ existe déjà"
    fi
}

# Fonction pour capturer une image
capture_image() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local filename="picam_${timestamp}.jpg"
    local filepath="${DATA_DIR}/${filename}"
    
    case $CAMERA_TYPE in
        "libcamera")
            # Commande pour libcamera-still (Raspberry Pi OS Bookworm)
            if libcamera-still --width 1920 --height 1080 --output "${filepath}" --timeout 1000 --nopreview &>/dev/null; then
                print_success "Image capturée: ${filename}"
                return 0
            else
                print_error "Échec de la capture avec libcamera-still"
                return 1
            fi
            ;;
            
        "raspistill")
            # Commande pour raspistill (Raspberry Pi OS Bullseye)
            if raspistill -w 1920 -h 1080 -o "${filepath}" -t 1000 -n &>/dev/null; then
                print_success "Image capturée: ${filename}"
                return 0
            else
                print_error "Échec de la capture avec raspistill"
                return 1
            fi
            ;;
            
        "fswebcam")
            # Commande pour fswebcam (webcam USB)
            if fswebcam -r 1920x1080 --no-banner --save "${filepath}" &>/dev/null; then
                print_success "Image capturée: ${filename}"
                return 0
            else
                print_error "Échec de la capture avec fswebcam"
                return 1
            fi
            ;;
            
        *)
            print_error "Type de caméra non reconnu: ${CAMERA_TYPE}"
            return 1
            ;;
    esac
}

# Fonction pour afficher les statistiques
show_stats() {
    if [ -d "${DATA_DIR}" ]; then
        local image_count=$(find "${DATA_DIR}" -name "picam_*.jpg" | wc -l)
        local disk_usage=$(du -sh "${DATA_DIR}" | cut -f1)
        local data_path=$(realpath "${DATA_DIR}")
        
        echo
        print_info "Statistiques actuelles:"
        echo "  📁 Répertoire: ${data_path}"
        echo "  📸 Images: ${image_count}"
        echo "  💾 Espace utilisé: ${disk_usage}"
        echo
    fi
}

# Fonction de nettoyage à l'arrêt
cleanup() {
    echo
    print_info "Arrêt du système de capture..."
    print_info "Capture terminée proprement"
    show_stats
    exit 0
}

# Fonction principale de capture en boucle
start_capture_loop() {
    print_info "Démarrage de la capture automatique (intervalle: ${INTERVAL}s)"
    print_info "Appuyez sur Ctrl+C pour arrêter"
    echo "$(printf '=%.0s' {1..50})"
    
    local capture_count=0
    
    while true; do
        # Capturer une image
        if capture_image; then
            ((capture_count++))
            print_info "Prochaine capture dans ${INTERVAL} secondes... (Total: ${capture_count})"
        else
            print_warning "Échec de la capture, nouvelle tentative dans ${INTERVAL}s..."
        fi
        
        # Attendre l'intervalle spécifié
        sleep ${INTERVAL}
    done
}

# Fonction d'aide
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -i, --interval SECONDS    Intervalle entre captures (défaut: 5)"
    echo "  -d, --data-dir DIR        Répertoire de sauvegarde (défaut: data)"
    echo "  -r, --resolution WxH      Résolution (défaut: 1920x1080)"
    echo "  -h, --help               Afficher cette aide"
    echo "  -s, --stats              Afficher les statistiques seulement"
    echo
    echo "Exemples:"
    echo "  $0                       # Capture toutes les 5 secondes"
    echo "  $0 -i 10                 # Capture toutes les 10 secondes"
    echo "  $0 -d photos -i 15       # Capture dans 'photos/' toutes les 15s"
    echo "  $0 -s                    # Afficher les statistiques"
}

# Traitement des arguments de ligne de commande
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -r|--resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        -s|--stats)
            show_stats
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
done

# Script principal
main() {
    echo "$(printf '=%.0s' {1..50})"
    echo "🎥 Système PiCam - Capture Shell"
    echo "$(printf '=%.0s' {1..50})"
    
    # Vérifier les dépendances
    if ! check_dependencies; then
        exit 1
    fi
    
    # Configurer le répertoire de données
    setup_data_directory
    
    # Afficher les statistiques initiales
    show_stats
    
    # Configurer le gestionnaire de signal pour Ctrl+C
    trap cleanup SIGINT SIGTERM
    
    # Démarrer la capture
    start_capture_loop
}

# Lancer le script principal
main "$@"