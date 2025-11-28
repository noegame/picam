# Comment se connecter à la rapsberry pi
[readme](../README.md)
## Sommaire
- [Se connecter en SSH](#se-connecter-en-ssh)
- [Utiliser SSH Tunnel dans VSCode](#utiliser-ssh-tunnel-dans-vscode)

## Se connecter en SSH

1. Allumer la Raspberry Pi et la connecter au réseau local (via Ethernet ou Wi-Fi).
2. Depuis un autre ordinateur sur le même réseau, ouvrir un terminal ou une invite de commande.
``` powershell
ssh roboteseo@raspberrypi-robot.local
```

## Utiliser SSH Tunnel dans VSCode

Pour éditer des fichiers directement sur la Raspberry Pi depuis Visual Studio Code, vous pouvez utiliser l'extension "Remote - SSH".

1. Installer l'extension "Remote - SSH" dans VSCode.
2. Ouvrir la palette de commandes (Ctrl+Shift+P) et sélectionner "Remote-SSH: Connect to Host...".
3. Ajouter une nouvelle configuration SSH avec l'adresse de la Raspberry Pi :
```
ssh roboteseo@raspberrypi-robot.local
```
4. Se connecter à la Raspberry Pi en sélectionnant l'hôte ajouté.
5. Une fois connecté, ouvrir le dossier de travail sur la Raspberry Pi via "File > Open Folder...".

Remarque : Vous pouvez remplacer au besoin "raspberrypi-robot.local" par l'adresse IP de votre Raspberry Pi ou le nom d'hôte que vous avez configuré ainsi que "roboteseo" par le nom d'utilisateur que vous avez choisi.