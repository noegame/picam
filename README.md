# PiCam

## Sommaire
- [Description](#description)
- [Comment Tester la caméra rapidement](#comment-tester-la-caméra-rapidement)

## Description
- captures_images : Programme Python qui capture automatiquement des photos toutes les 15 secondes avec la caméra du Raspberry Pi et les sauvegarde dans le dossier `data/`.
- match viewer : Surveille le terrain et communique les états des caisses au robot.

## Comment Tester la caméra rapidement

Sur la raspberry : commande linux pour lancer le streaming vidéo avec libcamera-vid :

libcamera-vid -t 0 \
  --width 640 --height 480 \
  --framerate 25 \
  --codec h264 \
  --inline \
  --listen on \
  -o tcp://0.0.0.0:8554

sur votre ordinateur : ouvrir VLC et se connecter au flux réseau :
tcp/h264://10.153.210.115:8554
