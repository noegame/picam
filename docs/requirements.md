# Requirements - Système PiCam

## Exigences Fonctionnelles

**RF-001** - Le système doit être capable de reconnaitre en temps réel les caisses sur le terrain.

**RF-002** - Le système doit être capable de reconnaitre la couleur des caisses (jaunes ou bleues).

**RF-003** - Le système doit être capable de d'identifier dans quelle zone se trouve chaque caisse

**RF-004** - Le système doit être capable de communiquer avec le robot via une connexion sans fil.

**RF-005** - Le système doit être capable dans un mode dégradé d'au moins d'identifier la couleur des caisses les plus présentes dans une zone.

## Exigences Non-Fonctionnelles

**RNF-001** - Le système doit traiter les images avec une latence maximale de 100ms.

**RNF-002** - La précision de reconnaissance des caisses doit être supérieure à 95%.

**RNF-003** - Le système doit fonctionner de manière continue pendant au moins 10 minutes.

## Exigences Techniques

**RT-001** - Le système doit fonctionner sur Raspberry Pi 4 ou supérieur.

**RT-002** - Le système doit utiliser une caméra compatible avec le module PiCam.

**RT-003** - La communication sans fil doit utiliser le protocole WiFi ou Bluetooth.

## Exigences d'Interface

**RI-001** - Le système doit fournir une interface de visualisation en temps réel.

**RI-002** - Le système doit exposer une API REST pour la communication avec le robot.

## Exigences de Performance

**RP-001** - Le système doit traiter au minimum 10 images par seconde.

**RP-002** - La portée de communication sans fil doit être d'au moins 5 mètres.
