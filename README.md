# PiCam

## Sommaire
- [Description](#description)
- [Comment Tester la caméra rapidement](#comment-tester-la-caméra-rapidement)
- [Installation de l'environnement de développement](#installation-de-lenvironnement-de-développement)



## Description
- captures_images : Programme Python qui capture automatiquement des photos toutes les 15 secondes avec la caméra du Raspberry Pi et les sauvegarde dans le dossier `data/`.
- match viewer : Surveille le terrain et communique les états des caisses au robot.

## Comment Tester la caméra rapidement

D'abord vérifier que la caméra est bien détectée :
```
# Lister les caméras
rpicam-hello --list-cameras

# Test rapide d'affichage (5 secondes)
rpicam-hello --timeout 5000

# Test de capture d'image
rpicam-still --output test.jpg --timeout 2000
ls -la test.jpg

# Si l'image est créée, l'afficher pour vérifier
file test.jpg
```

Sur la raspberry : commande linux pour lancer le streaming vidéo avec libcamera-vid :
```
libcamera-vid -t 0 \
  --width 640 --height 480 \
  --framerate 25 \
  --codec h264 \
  --inline \
  --listen on \
  -o tcp://0.0.0.0:8554

libcamera-vid -t 0 --width 640 --height 480 --framerate 25 --codec h264 --inline --listen on -o tcp://0.0.0.0:8554


```
sur votre ordinateur : ouvrir VLC et se connecter au flux réseau :
```tcp/h264://[IP de la PI]:8554```

### Prérequis

- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/)


## Installation de l'environnement de développement

1. Télécharger Raspberry PI Imager et flasher une carte SD avec Raspberry Pi OS (32-bit) Lite.
2. Insérer la carte SD dans le Raspberry Pi, connecter un clavier, une souris et un écran, puis démarrer le Raspberry Pi.
3. Connecter le Raspberry Pi à Internet et mettre à jour le système :
```
sudo apt update && sudo apt upgrade -y && sudo reboot
```
3. Configurer la connexion SSH. Nous allons autoriser temporairement l'authentification par mot de passe afin de copier la clé publique SSH.
```
sudo nano /etc/ssh/sshd_config
```
Modifier les lignes suivantes :
```
PasswordAuthentication yes
PubkeyAuthentication yes
```
Puis redémarrer le service SSH :
```
sudo systemctl restart ssh
```
4. Créer une paire de clés SSH sur votre ordinateur depuis powershell :
```
ssh-keygen 
```
5. Copier la clé publique sur le Raspberry Pi depuis powershell :
```
type $env:USERPROFILE\.ssh\raspberrypi_robot.pub | ssh pi@192.168.68.52 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
6. Repasser en mode sécurisé
```
sudo nano /etc/ssh/sshd_config
```
Modifier les lignes suivantes :
```
PasswordAuthentication no
PubkeyAuthentication yes
```
Puis redémarrer le service SSH :
```
sudo systemctl restart ssh
```

**Pour se connecter à la raspberry pi:**
```
ssh -i $env:USERPROFILE\.ssh\raspberrypi_robot roboteseo@192.168.68.52
```

7. Installer les dépendances nécessaires pour le projet
```
sudo apt install git
git clone https://github.com/noegame/picam
cd picam
chmod u+x setup.sh 
./setup.sh
```
La Raspberry Pi est maintenant fonctionnelle.