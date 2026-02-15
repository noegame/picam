# Systemd Services for ROD

This directory contains systemd service files to run ROD as system services.

## Architecture

- **rod-detection.service**: Computer vision thread (server, creates socket)
- **rod-communication.service**: Communication thread (client, connects to socket)
- **rod.target**: Groups both services together for easy management

## Installation rapide

```bash
cd systemd
sudo ./install.sh
```

Le script d'installation va :
- Vérifier que les exécutables sont compilés
- Copier les fichiers de service vers `/etc/systemd/system/`
- Recharger systemd

## Désinstallation

```bash
cd systemd
sudo ./uninstall.sh
```

## Installation manuelle

Si vous préférez installer manuellement :

### 1. Copier les fichiers de service

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo cp systemd/*.target /etc/systemd/system/
```

### 2. Recharger systemd

```bash
sudo systemctl daemon-reload
```

### 3. Activer les services au démarrage (optionnel)

```bash
sudo systemctl enable rod.target
```

## Usage

### Start/Stop services

```bash
# Start both services via target
sudo systemctl start rod.target

# Or start individually
sudo systemctl start rod-detection.service
sudo systemctl start rod-communication.service

# Stop services
sudo systemctl stop rod.target
```

### Check status

```bash
# Check status of both services
systemctl status rod-detection.service
systemctl status rod-communication.service

# Or check the target
systemctl status rod.target
```

### View logs

```bash
# Follow logs in real-time
sudo journalctl -fu rod-detection.service
sudo journalctl -fu rod-communication.service

# View recent logs
sudo journalctl -u rod-detection.service -n 50
sudo journalctl -u rod-communication.service -n 50

# View logs from both services
sudo journalctl -u rod-detection.service -u rod-communication.service -f
```

### Restart services

```bash
# Restart a service
sudo systemctl restart rod-detection.service

# Restart both
sudo systemctl restart rod.target
```

## Service Dependencies

The communication service has these dependencies:
- `After=rod-detection.service`: Starts after detection
- `Requires=rod-detection.service`: Won't start if detection fails
- `BindsTo=rod-detection.service`: Stops if detection stops

This ensures the socket is created before communication tries to connect.

## Configuration

### Change image folder

Edit `/etc/systemd/system/rod-detection.service` and modify the `ExecStart` line:

```ini
ExecStart=/home/noegame/ROD/rod_c/build/rod_detection /path/to/your/images
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart rod-detection.service
```

### Adjust resource limits

Edit the service files to modify:
- `MemoryMax`: Maximum memory usage
- `CPUQuota`: Maximum CPU usage (percentage)
- `RestartSec`: Delay before restart after crash

## Troubleshooting

### Services won't start

```bash
# Check for errors
sudo systemctl status rod-detection.service
sudo journalctl -xe
```

### Permission issues

Make sure:
1. User `noegame` exists and has correct permissions
2. Executable files are accessible: `chmod +x /home/noegame/ROD/rod_c/build/rod_*`
3. Working directory exists and is readable

### Socket connection issues

Check if socket is created:
```bash
ls -l /tmp/rod_detection.sock
```

If not, check detection service logs:
```bash
sudo journalctl -u rod-detection.service -n 50
```

## Uninstall

```bash
# Stop services
sudo systemctl stop rod.target

# Disable services
sudo systemctl disable rod-detection.service
sudo systemctl disable rod-communication.service
sudo systemctl disable rod.target

# Remove service files
sudo rm /etc/systemd/system/rod-detection.service
sudo rm /etc/systemd/system/rod-communication.service
sudo rm /etc/systemd/system/rod.target

# Reload systemd
sudo systemctl daemon-reload
```
