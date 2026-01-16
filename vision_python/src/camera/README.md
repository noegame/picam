# Module Camera - Architecture POO

## Vue d'ensemble

Ce module fournit une architecture orientée objet pour la gestion de différents types de caméras. Il utilise les principes de la programmation orientée objet (POO) avec une classe abstraite de base et plusieurs implémentations concrètes.

## Architecture

### Hiérarchie des classes

```
Camera (Abstraite)
├── PiCamera (Raspberry Pi)
├── Webcam (USB)
└── EmulatedCamera (Simulation)
```

### Classe abstraite : `Camera`

La classe `Camera` définit l'interface commune pour tous les types de caméras. Elle contient les méthodes abstraites suivantes :

#### Méthodes abstraites

- **`init()`** : Initialise le matériel de la caméra
- **`start()`** : Démarre le flux vidéo
- **`stop()`** : Arrête le flux et libère les ressources
- **`set_parameters(parameters: Dict[str, Any])`** : Configure les paramètres de la caméra
- **`capture_photo()`** : Capture une photo et la retourne sous forme de `np.ndarray`

## Implémentations

### 1. PiCamera

Implémentation pour le module caméra du Raspberry Pi utilisant la bibliothèque `picamera2`.

**Utilisation :**

```python
from vision_python.src.camera import PiCamera

# Création de l'instance
camera = PiCamera()

# Configuration des paramètres de base
camera.set_parameters({
    "width": 1920,
    "height": 1080,
    "config_mode": "still"
})

# Initialisation du matériel
camera.init()

# Démarrage du flux
camera.start()

# Configuration des contrôles caméra (optionnel)
camera.set_parameters({"ExposureTime": 10000, "AnalogueGain": 1.0})

# Capture d'une photo
image = camera.capture_photo()

# Fermeture
camera.stop()
```

**Paramètres de configuration (via set_parameters) :**
- `width` : Largeur de l'image (requis avant init)
- `height` : Hauteur de l'image (requis avant init)
- `config_mode` : Mode de configuration ("preview" ou "still", par défaut "preview")

**Contrôles caméra (après init) :**
- `ExposureTime` : Temps d'exposition
- `AnalogueGain` : Gain analogique
- etc.

### 2. Webcam

Implémentation pour les webcams USB standard utilisant OpenCV.

**Utilisation :**

```python
from vision_python.src.camera import Webcam

# Création de l'instance
camera = Webcam()

# Configuration des paramètres de base
camera.set_parameters({
    "width": 1280,
    "height": 720,
    "device_id": 0
})

# Initialisation du matériel
camera.init()

# Démarrage du flux
camera.start()

# Configuration des propriétés OpenCV (optionnel)
camera.set_parameters({
    "brightness": 50,
    "contrast": 40,
    "exposure": -5
})

# Capture d'une photo
image = camera.capture_photo()

# Fermeture
camera.stop()
```

**Paramètres de configuration (via set_parameters) :**
- `width` : Largeur de l'image (requis avant init)
- `height` : Hauteur de l'image (requis avant init)
- `device_id` : ID du périphérique (0 pour la première webcam, par défaut 0)

**Propriétés OpenCV configurables (après init) :**
- `brightness` : Luminosité
- `contrast` : Contraste
- `saturation` : Saturation
- `exposure` : Exposition
- `fps` : Images par seconde
- `gain` : Gain
- `auto_exposure` : Exposition automatique

### 3. EmulatedCamera

Implémentation virtuelle pour le test et le débogage sans matériel réel. Lit des images depuis un dossier.

**Utilisation :**

```python
from pathlib import Path
from vision_python.src.camera import EmulatedCamera

# Création de l'instance
camera = EmulatedCamera()

# Configuration des paramètres
image_folder = Path("/home/user/test_images")
camera.set_parameters({
    "width": 1920,
    "height": 1080,
    "image_folder": image_folder
})

# Initialisation (charge la liste des images)
camera.init()

# Démarrage du flux
camera.start()

# Capture d'une "photo" (lit une image du dossier)
image = camera.capture_photo()

# Fermeture
camera.stop()
```

**Paramètres de configuration (via set_parameters) :**
- `width` : Largeur de l'image (pour information)
- `height` : Hauteur de l'image (pour information)
- `image_folder` : Chemin vers le dossier contenant les images (requis avant init)

## Factory Pattern

Le module fournit une fonction factory pour créer facilement des instances de caméras.

### Fonction `get_camera()`

```python
from pathlib import Path
from vision_python.config import config
from vision_python.src.camera import get_camera

# Création d'une PiCamera
camera = get_camera(camera=config.CameraMode.PI)
camera.set_parameters({
    "width": 1920,
    "height": 1080,
    "config_mode": "still"
})
camera.init()

# Création d'une Webcam
camera = get_camera(camera=config.CameraMode.COMPUTER)
camera.set_parameters({
    "width": 1280,
    "height": 720,
    "device_id": 0
})
camera.init()

# Création d'une EmulatedCamera
camera = get_camera(camera=config.CameraMode.EMULATED)
camera.set_parameters({
    "width": 1920,
    "height": 1080,
    "image_folder": Path("/home/user/test_images")
})
camera.init()
```

**Paramètres de la factory :**
- `camera` : Type de caméra (`CameraMode.PI`, `CameraMode.COMPUTER`, ou `CameraMode.EMULATED`)

**Note :** Après avoir créé la caméra, vous devez appeler `set_parameters()` avec les paramètres appropriés, puis `init()` pour initialiser le matériel.

## Exemple d'utilisation complet

```python
from pathlib import Path
from vision_python.config import config
from vision_python.src.camera import get_camera, PiCamera, Webcam

# Sélection du type de caméra selon la configuration
camera = get_camera(camera=config.CAMERA)

# Configuration des paramètres de base
if isinstance(camera, PiCamera):
    camera.set_parameters({
        "width": 1920,
        "height": 1080,
        "config_mode": "still"
    })
elif isinstance(camera, Webcam):
    camera.set_parameters({
        "width": 1280,
        "height": 720,
        "device_id": 0
    })
else:  # EmulatedCamera
    camera.set_parameters({
        "width": 1920,
        "height": 1080,
        "image_folder": Path("pictures/camera/test")
    })

# Initialisation du matériel
camera.init()

# Démarrage de la caméra
camera.start()

# Configuration optionnelle des contrôles
if isinstance(camera, PiCamera):
    camera.set_parameters({"ExposureTime": 10000})
elif isinstance(camera, Webcam):
    camera.set_parameters({"brightness": 50, "contrast": 40})

# Capture de photos
for i in range(10):
    image = camera.capture_photo()
    print(f"Photo {i+1} capturée : shape={image.shape}")

# Fermeture propre
camera.stop()
```

## Principes POO appliqués

1. **Abstraction** : La classe `Camera` définit l'interface commune
2. **Encapsulation** : Chaque classe gère ses propres ressources
3. **Héritage** : Les classes concrètes héritent de `Camera`
4. **Polymorphisme** : Toutes les caméras peuvent être utilisées via l'interface `Camera`
5. **Factory Pattern** : Création simplifiée d'instances via `get_camera()`

## Structure des fichiers

```
vision_python/src/camera/
├── __init__.py           # Exports du module
├── camera.py             # Classe abstraite Camera + PiCamera
├── webcam.py             # Implémentation Webcam
├── emulated_camera.py    # Implémentation EmulatedCamera
├── camera_factory.py     # Factory pour créer des instances
└── README.md             # Cette documentation
```

## Notes importantes

- Toutes les méthodes `capture_photo()` retournent des images au format **RGB** (pas BGR)
- Les paramètres de configuration doivent être définis via `set_parameters()` **avant** d'appeler `init()`
- La méthode `init()` doit être appelée explicitement après la configuration des paramètres
- La méthode `close()` est un alias pour `stop()` pour la compatibilité
- Les exceptions sont loguées et propagées pour un débogage facile
- L'initialisation en deux étapes (instantiation puis configuration) permet une plus grande flexibilité
