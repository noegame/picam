# Guide rapide - Services systemd ROD

## Installation

```bash
cd systemd
sudo ./install.sh
```

## Commandes essentielles

### Démarrer ROD
```bash
sudo systemctl start rod.target
```

### Arrêter ROD
```bash
sudo systemctl stop rod.target
```

### Redémarrer ROD
```bash
sudo systemctl restart rod.target
```

### Vérifier le statut
```bash
systemctl status rod.target
systemctl status rod-detection.service
systemctl status rod-communication.service
```

### Voir les logs en temps réel
```bash
# Les deux services
sudo journalctl -u rod-detection.service -u rod-communication.service -f

# Seulement detection
sudo journalctl -u rod-detection.service -f

# Seulement communication
sudo journalctl -u rod-communication.service -f
```

### Voir les derniers logs
```bash
sudo journalctl -u rod-detection.service -n 50
```

### Activer au démarrage
```bash
sudo systemctl enable rod.target
```

### Désactiver au démarrage
```bash
sudo systemctl disable rod.target
```

## Désinstallation

```bash
cd systemd
sudo ./uninstall.sh
```

## Dépannage

### Le service ne démarre pas
```bash
# Vérifier les erreurs
sudo systemctl status rod-detection.service
sudo journalctl -xe

# Vérifier les permissions
ls -l /home/noegame/ROD/rod_c/build/rod_*
```

### Le socket n'est pas créé
```bash
# Vérifier si le socket existe
ls -l /tmp/rod_detection.sock

# Si non, regarder les logs de detection
sudo journalctl -u rod-detection.service -n 100
```

### Modifier le dossier d'images
```bash
# Éditer le service
sudo nano /etc/systemd/system/rod-detection.service

# Modifier la ligne ExecStart:
# ExecStart=/home/noegame/ROD/rod_c/build/rod_detection /nouveau/chemin

# Recharger et redémarrer
sudo systemctl daemon-reload
sudo systemctl restart rod-detection.service
```
