# Architecture du Système ROD (Remote Observation Device)

## 📋 Vue d'Ensemble

ROD est un système de vision par ordinateur pour détecter des marqueurs ArUco sur des éléments de jeu Eurobot et transmettre leurs positions aux robots via socket Unix.

## 🏗️ Architecture Modulaire

```
rod_c/
├── rod_detection.c          # Thread principal (CV + orchestration)
├── rod_communication.c      # Thread IPC (réception données)
│
├── rod_config/              # ⚙️ Configuration centralisée
│   ├── IDs valides Eurobot 2026
│   ├── Paramètres détecteur ArUco optimisés
│   └── Constantes système (chemins, intervalles)
│
├── rod_cv/                  # 🔍 Vision par ordinateur
│   ├── Calculs géométriques (centre, angle, périmètre)
│   ├── Filtrage des marqueurs valides
│   ├── Comptage par catégorie
│   └── Types standards (MarkerData, MarkerCounts)
│
├── rod_visualization/       # 🎨 Visualisation & Debug
│   ├── Annotation avec IDs
│   ├── Annotation avec centres
│   ├── Annotation avec compteurs
│   └── Sauvegarde images debug
│
├── rod_socket/              # 🔌 Communication inter-processus
│   ├── Serveur socket Unix domain
│   ├── Gestion connexions clients
│   └── Envoi données de détection (JSON-like)
│
├── rod_camera/              # 📷 Abstraction caméra
│   ├── emulated_camera (test)
│   └── libcamera (production)
│
└── opencv_wrapper/          # 🔧 Interface C vers OpenCV
    └── Wrapper C++ → C pour détection ArUco
```

## 🔄 Flux de Données

```
┌─────────────────────────────────────────────────────────────┐
│                    rod_detection (Thread CV)                 │
│                                                               │
│  1. Caméra → Capture image                                   │
│  2. ArUco  → Détection (opencv_wrapper)                      │
│  3. rod_cv → Filtrage IDs valides                            │
│  4. rod_socket → Envoi détections                            │
│  5. rod_visualization → Sauvegarde debug                     │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       │ Unix Socket: /tmp/rod_detection.sock
                       │ Format: [[id,x,y,angle], ...]
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              rod_communication (Thread IPC)                  │
│                                                               │
│  1. Socket client → Connexion                                │
│  2. Réception → Données détection                            │
│  3. Affichage → Console (debug)                              │
│  4. TODO: Transmission → Robot principal                     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Modules et Responsabilités

### rod_config - Configuration
**Rôle** : Point unique de configuration  
**Exports** :
- `rod_config_is_valid_marker_id()` - Validation IDs Eurobot
- `rod_config_configure_detector_parameters()` - Paramètres ArUco
- Macros : `ROD_SOCKET_PATH`, `ROD_DEBUG_OUTPUT_FOLDER`, etc.

**Avantage** : Modifier IDs/paramètres = éditer 1 seul fichier

### rod_cv - Vision par Ordinateur
**Rôle** : Opérations géométriques et traitement détections  
**Exports** :
- `calculate_marker_center()`, `calculate_marker_angle()`
- `filter_valid_markers()` - Filtrage + conversion DetectionResult → MarkerData[]
- `count_markers_by_category()` - Comptage par type
- Types : `MarkerData`, `MarkerCounts`, `Point2f`, `Pose2D/3D`

**Avantage** : Réutilisable dans tous les programmes (tests, prod)

### rod_visualization - Visualisation
**Rôle** : Annotations et debug visuel  
**Exports** :
- `rod_viz_annotate_with_ids()` - Affiche IDs
- `rod_viz_annotate_with_centers()` - Affiche coordonnées
- `rod_viz_annotate_with_counter()` - Affiche compteurs
- `rod_viz_save_debug_image()` - Sauvegarde complète annotée

**Avantage** : Séparation nette détection ↔ visualisation

### rod_socket - Communication
**Rôle** : Encapsulation socket Unix domain  
**Exports** :
- `rod_socket_server_create()` - Création serveur
- `rod_socket_server_accept()` - Acceptation client (non-bloquant)
- `rod_socket_server_send_detections()` - Envoi JSON-like
- `rod_socket_server_destroy()` - Nettoyage

**Avantage** : Logique socket isolée, facilement testable

### rod_camera - Abstraction Caméra
**Rôle** : Interface unifiée caméras  
**Implémentations** :
- `emulated_camera` - Lecture images dossier (test)
- `libcamera` - Caméra Raspberry Pi (production)

**Avantage** : Switch caméra sans modifier code détection

### opencv_wrapper - Bridge C/C++
**Rôle** : Interface C vers OpenCV C++  
**Pattern** : Handles opaques + wrappers fonctions  
**Exports** : `detectMarkersWithConfidence()`, `sharpen_image()`, etc.

**Avantage** : Code C pur dans rod_detection.c

## 🎯 Pipeline de Détection

```
Image RGB → Sharpen → Resize 1.5x → Detect ArUco → Filter IDs → MarkerData[]
                                           ↓
                                    opencv_wrapper
                                           ↓
                                       rod_cv
                                           ↓
                                  [MarkerData array]
                                     ↙         ↘
                            rod_socket      rod_visualization
                                ↓                  ↓
                           Envoi JSON        Debug Images
```

## 🔑 Avantages Architecture Actuelle

✅ **Zéro duplication** - Chaque fonction existe en un seul endroit  
✅ **Modules indépendants** - Testables séparément  
✅ **Configuration centralisée** - Modification simplifiée  
✅ **Responsabilités claires** - 1 module = 1 rôle  
✅ **Évolutivité** - Facile d'ajouter nouveaux modules  
✅ **Maintenabilité** - Code organisé et documenté

## 📊 Métriques

| Métrique | Valeur |
|----------|--------|
| Modules réutilisables | 6 |
| Lignes dupliquées | 0 |
| Taille rod_detection.c | 295 lignes (-62%) |
| Points de configuration | 1 (rod_config) |
| Dépendances circulaires | 0 |

## 🚀 Évolutions Futures

- **Phase 3** : Extraction threads caméra/détection séparés
- **Phase 4** : Pipeline de détection configurable via JSON
- **Phase 5** : Support multi-caméras
- **Phase 6** : Intégration protocole robot principal
