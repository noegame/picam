# Comment se connecter à la rapsberry pi

## Sommaire
- [Se connecter en SSH](#se-connecter-en-ssh)

## Se connecter en SSH

### Comment trouver l'adresse IP de la Raspberry Pi
to do

### Comment se connecter en SSH

La connexion SSH permet de se connecter à distance à la Raspberry Pi depuis un autre ordinateur. La commande pour se connecter est la suivante :
``` powershell
ssh -i $env:USERPROFILE\.ssh\rapsberry_pi_name user_name@ip_address
```
Remplacer :
- `rapsberry_pi_name` par le nom de la clé privée SSH que vous avez créée lors de la configuration initiale (par exemple : `raspberrypi_robot`).
- `user_name` par le nom d'utilisateur sur la Raspberry Pi (par exemple : `roboteseo`).
- `ip_address` par l'adresse IP de la Raspberry Pi (par exemple : `192.168.68.100`).

Exemple complet :
``` powershell
ssh -i $env:USERPROFILE\.ssh\raspberrypi_robot roboteseo@192.168.68.100
```
