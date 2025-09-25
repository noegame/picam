# PiCam

## Description
- captures_images : Programme Python qui capture automatiquement des photos toutes les 15 secondes avec la caméra du Raspberry Pi et les sauvegarde dans le dossier `data/`.

- match viewer : Surveille le terrain et communique les états des caisses au robot.

## Tester la caméra rapidement

libcamera-vid -t 0 \
  --width 640 --height 480 \
  --framerate 25 \
  --codec h264 \
  --inline \
  --listen on \
  -o tcp://0.0.0.0:8554

tcp/h264://10.153.210.115:8554

## Prérequis
- Raspberry Pi 3 ou supérieur
- Caméra PiCam connectée et activée
- Python 3.7+

## Installation

### 1. Activer la caméra
```bash
sudo raspi-config
# Interface Options > Camera > Enable
```

### 2. Installer les dépendances
```bash
# Pour Raspberry Pi OS Bookworm (recommandé)
sudo apt update
sudo apt install python3-picamera2

# Ou installer via pip
pip3 install -r requirements.txt
```

## Utilisation

### Installation rapide
```bash
# Rendre les scripts exécutables
chmod +x *.sh

# Installation automatique (recommandé)
./install_picam.sh --auto

# Ou installation interactive
./install_picam.sh
```

### Capture avec Shell (NOUVEAU)
```bash
# Script simple - capture toutes les 5 secondes
./simple_capture.sh

# Script complet avec options
./capture_images.sh

# Avec options personnalisées
./capture_images.sh -i 10 -d photos    # Toutes les 10s dans dossier 'photos/'
```

### Capture avec Python
```bash
# Démarrage rapide
./start_picam.sh

# Démarrage manuel
cd src/
python3 main.py
```

### Arrêter la capture
Appuyez sur `Ctrl+C` pour arrêter proprement le programme.

## Configuration

### Modifier l'intervalle de capture
Éditez le fichier `src/main.py` ligne 168 :
```python
picam = PiCamCapture(data_dir="../data", interval=15)  # 15 secondes
```

### Modifier la résolution
Éditez les lignes 56-62 dans `src/main.py` :
```python
# Pour PiCamera2
main={"size": (1920, 1080)}  # Résolution Full HD

# Pour PiCamera legacy
self.camera.resolution = (1920, 1080)
```

## Structure des fichiers
```
picam/
├── data/                    # Photos capturées
│   ├── picam_20240925_143022.jpg
│   └── picam_20240925_143037.jpg
├── docs/
│   └── requirements.md      # Spécifications du projet
├── src/
│   └── main.py             # Programme principal Python
├── capture_images.sh       # Script Shell complet
├── simple_capture.sh       # Script Shell simplifié
├── install_picam.sh        # Script d'installation
├── start_picam.sh         # Démarrage Python
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## Format des noms de fichiers
Les photos sont nommées selon le format :
`picam_YYYYMMDD_HHMMSS.jpg`

Exemple : `picam_20240925_143022.jpg`
- 2024-09-25 à 14h30m22s

## Logs
Le programme affiche des logs détaillés :
```
2024-09-25 14:30:22,123 - INFO - PiCamera2 initialisée
2024-09-25 14:30:22,124 - INFO - Démarrage de la capture automatique (intervalle: 15s)
2024-09-25 14:30:22,456 - INFO - Image capturée: picam_20240925_143022.jpg
2024-09-25 14:30:22,457 - INFO - Prochaine capture dans 15 secondes...
```

## Dépannage

### Erreur "Camera not found"
```bash
# Vérifier que la caméra est détectée
libcamera-hello --list-cameras

# Vérifier les connexions physiques
# Redémarrer le Raspberry Pi
```

### Erreur d'importation PiCamera
```bash
# Installer les bonnes dépendances selon votre OS
sudo apt install python3-picamera2  # Bookworm
# ou
sudo apt install python3-picamera   # Bullseye
```

### Permissions insuffisantes
```bash
# Ajouter l'utilisateur au groupe video
sudo usermod -a -G video $USER
# Redémarrer la session
```

## Développement

### Scripts Shell disponibles

#### 1. Script Simple (`simple_capture.sh`)
- Script basique et rapide
- Capture toutes les 5 secondes
- Support libcamera-still et raspistill
- Fallback fswebcam

#### 2. Script Complet (`capture_images.sh`)
- Interface colorée avec logs
- Options personnalisables (intervalle, dossier, résolution)
- Statistiques en temps réel
- Gestion d'erreurs avancée
- Support multi-plateforme

#### 3. Script d'Installation (`install_picam.sh`)
- Installation automatique des dépendances
- Configuration des permissions
- Test de la caméra
- Menu interactif

### Tests sans Raspberry Pi
Le programme Python utilise OpenCV comme fallback pour les tests sur PC :
```bash
pip3 install opencv-python
python3 src/main.py
```

Les scripts Shell nécessitent fswebcam sur PC :
```bash
sudo apt install fswebcam  # Linux
brew install fswebcam      # macOS
```

## Licence
Ce projet est sous licence MIT.