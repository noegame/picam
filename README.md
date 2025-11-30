# ZOD

## Sommaire
- [Description](#description)
- [Documentation](#documentation)

## Description
Dans le cadre du concours [Eurobot](https://www.eurobot.org/) et de la [Coupe de France de Robotique](https://www.coupederobotique.fr/), des robots s'affrontent lors de match sur une table de jeu. Les règles de la compétition autorisent l'utilisation d'une Zone d'Observation Déportée (ZOD) : un système équipé d'une caméra qui permet de capturer des images de la table de jeu et de transmettre des informations au robot.

Ce projet de ZOD a pour objectif :
- Reconnaissance de TAG ArUco des éléments de jeu au travers d'une caméra montée au-dessus de la table
- Être capable de localiser précisement les éléments du jeu dans le repère du terrain 
- Transmission des positions des éléments de jeu à un robot

## Documentation
- [How to set up the raspberry pi environment](docs/how-to-set-up.md)
- [How to connect to the raspberry pi](docs/how-to-connect.md)
- [How to test the camera quickly](docs/how-to-test-the-camera.md)
- [How to calibrate the camera](docs/how-to-calibrate-the-camera.md)
- [Some code documentation](docs/code-documentation.md)
