# How to setup the project

## Sommaire
- [Prérequis](#prérequis)
- [Installation de l'environnement raspberry pi](#installation-de-lenvironnement-raspberry-pi)
- [Comment tester la caméra rapidement](#comment-tester-la-caméra-rapidement)

## Prérequis

- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [MobaXterm](https://mobaxterm.mobatek.net/download-home-edition.html) (optionnel, mais recommandé, utile pour accéder au fichiers du Raspberry Pi)

## Installation de l'environnement raspberry pi

1. Télécharger Raspberry PI Imager et flasher une carte SD avec Raspberry Pi OS (32-bit) Lite.
2. Insérer la carte SD dans le Raspberry Pi, connecter un clavier, une souris et un écran, puis démarrer le Raspberry Pi.
3. Connecter le Raspberry Pi à Internet et mettre à jour le système :
``` shell
sudo apt update && sudo apt upgrade -y && sudo reboot
```
3. Configurer la connexion SSH. Nous allons autoriser temporairement l'authentification par mot de passe afin de copier la clé publique SSH.
``` shell 
sudo nano /etc/ssh/sshd_config
```
Modifier les lignes suivantes :
``` shell
PasswordAuthentication yes
PubkeyAuthentication yes
```
Puis redémarrer le service SSH :
``` shell 
sudo systemctl restart ssh
```
4. Aller dans C:\Users\<VotreNomUtilisateur>\.ssh
Créer une paire de clés SSH (si vous n'en avez pas déjà une) :
``` powershell
ssh-keygen 
```
5. Copier la clé publique sur le Raspberry Pi :
``` shell
sudo nano .ssh/authorized_keys
```
6. Repasser en mode sécurisé
``` shell
sudo nano /etc/ssh/sshd_config
```
Modifier les lignes suivantes :
```shell
PasswordAuthentication no
PubkeyAuthentication yes
```
Puis redémarrer le service SSH :
``` shell
sudo systemctl restart ssh
```

**Pour se connecter à la raspberry pi:**
``` powershell
ssh roboteseo@raspberrypi-robot.local
```
Vous pouvez remplacer au besoin "raspberrypi-robot.local" par l'adresse IP de votre Raspberry Pi ou le nom d'hôte que vous avez configuré ainsi que "roboteseo" par le nom d'utilisateur que vous avez choisi. 

7. Installer les dépendances nécessaires pour le projet
``` shell
sudo mkdir dev
cd dev
sudo apt install git
git clone https://github.com/noegame/picam
cd picam/setup/
chmod u+x setup.sh 
./setup.sh
```
La Raspberry Pi est maintenant fonctionnelle.


## Comment tester la caméra rapidement

Vérifier que la caméra est bien détectée :
``` shell
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
Prendre une photo.
``` shell
cd ~/picam/src/test_camera/
./camera_check.sh
```
Récupérer la photo sur votre PC :
Si vous utilisez MobaXterm, vous pouvez aussi simplement glisser-déposer le fichier depuis la fenêtre de terminal.
Sinon, utilisez la commande suivante pour le récupérer :
``` powershell
# Puis sur votre PC : récupérer le fichier
scp roboteseo@192.168.68.100:~/picam/src/test_camera/camera_test_*.jpg $env:USERPROFILE\Downloads\
```