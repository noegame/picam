## How to test the camera quickly
[readme](../README.md)

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