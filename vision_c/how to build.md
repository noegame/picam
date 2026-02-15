## Comment Build


### Build rapide

```bash
cd vision_c
./build.sh
```

### Build manuel

```bash
cd vision_c
mkdir -p build
cd build
cmake ..
make test_aruco_detection_pipeline
```

### Exécution

```bash
./build/apps/test_aruco_detection_pipeline <image_path> [output_path]
```

## Programmes disponibles

Tous les programmes peuvent être compilés avec CMake :

```bash
cd build
make test_aruco_detection_pipeline  # Test de détection ArUco complet
make detect_aruco                   # Détecteur ArUco simple
make grayscale_converter            # Convertisseur de niveaux de gris
make                                # Compiler tout
```

## Structure du build

```
vision_c/
├── build.sh                        # Script CMake (nouveau)
├── build_test_aruco.sh            # Script bash (obsolète)
├── CMakeLists.txt                 # Configuration CMake (existant)
├── build/                         # Dossier de build
│   ├── apps/                      # Exécutables
│   │   ├── test_aruco_detection_pipeline
│   │   ├── detect_aruco
│   │   └── grayscale_converter
│   └── libopencv_wrapper.a        # Bibliothèque wrapper
```


### Nettoyer le build
```bash
cd vision_c
rm -rf build
./build.sh
```

