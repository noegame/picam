# Pipeline de Détection ArUco - Guide Complet

Ce document détaille chaque étape du processus de détection des marqueurs ArUco, du chargement de l'image brute jusqu'à l'obtention des coordonnées réelles des marqueurs sur le terrain.

---

## Vue d'ensemble du Pipeline

```
Image brute (BGR)
    ↓
1. Conversion en niveaux de gris
    ↓
2. Correction de la distorsion optique (unround)
    ↓
3. Égalisation de l'histogramme (CLAHE) [optionnel]
    ↓
4. Accentuation de la netteté (sharpening)
    ↓
5. Détection des marqueurs ArUco
    ↓
6. Raffinement des coins
    ↓
7. Calcul des centres et angles
    ↓
8. Transformation de perspective
    ↓
Coordonnées réelles (mm)
```

---

## Étapes Détaillées

### 1. Conversion en Niveaux de Gris

**Opération :** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

**Explication :**
La détection ArUco fonctionne uniquement avec des images en niveaux de gris car elle se base sur les contrastes de luminosité, pas sur les couleurs. Cette conversion réduit les données de 3 canaux (BGR) à 1 canal (intensité lumineuse), ce qui :
- Accélère tous les traitements suivants (×3 plus rapide)
- Élimine les variations de teinte qui pourraient perturber la détection
- Conserve toute l'information nécessaire (les marqueurs ArUco sont monochromes)

**Impact sur la détection :** Essentiel. Sans cette étape, les algorithmes suivants ne fonctionneront pas correctement.

---

### 2. Correction de la Distorsion Optique (Unround)

**Opération :** `cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)`

**Paramètres :**
- `camera_matrix` : Matrice intrinsèque de la caméra (focale, centre optique)
- `dist_coeffs` : Coefficients de distorsion radiale et tangentielle
- `newcameramtx` : Nouvelle matrice optimale calculée avec `getOptimalNewCameraMatrix()`

**Explication :**
Les objectifs de caméra, surtout grand-angle, déforment l'image selon une distorsion barrel (effet "fisheye") ou pincushion. Cette distorsion :
- Courbe les lignes droites, surtout en périphérie
- Déforme les carrés des marqueurs ArUco
- Fausse les angles et les proportions

La correction utilise un modèle mathématique obtenu par calibration préalable de la caméra pour "redresser" l'image. Elle déplace chaque pixel selon les coefficients de distorsion pour retrouver la géométrie réelle.

**Impact sur la détection :**
- **Améliore la détection en bordure d'image** : Les marqueurs périphériques conservent leur forme carrée
- **Augmente la précision angulaire** : Les angles des marqueurs sont préservés
- **Réduit l'erreur de position** : Particulièrement important pour la transformation de perspective

**Ordre d'application :** APRÈS la conversion en grayscale pour optimiser les performances (1 canal au lieu de 3).

---

### 3. Égalisation Adaptative de l'Histogramme (CLAHE)

**Opération :** `cv2.createCLAHE(clipLimit, tileGridSize).apply(img)`

**Paramètres :**
- `clipLimit` (2.0-3.0) : Limite de contraste pour éviter l'amplification excessive du bruit
- `tileGridSize` (8×8 ou 16×16) : Taille des régions traitées indépendamment

**Explication :**
CLAHE (Contrast Limited Adaptive Histogram Equalization) améliore le contraste local de l'image en divisant l'image en tuiles et en égalisant l'histogramme de chaque tuile séparément. Contrairement à l'égalisation globale, elle :
- Préserve les détails dans les zones sombres ET claires simultanément
- S'adapte aux variations d'éclairage local (ombres, reflets)
- Limite l'amplification du bruit grâce au `clipLimit`

**Cas d'usage :**
- Éclairage non uniforme sur le terrain
- Présence d'ombres ou de zones très éclairées
- Faible contraste entre les marqueurs et le fond

**Impact sur la détection :**
- **Peut augmenter les détections** dans les zones mal éclairées
- **Risque d'amplifier le bruit** si clipLimit trop élevé
- **Coût computationnel** : Ralentit le traitement

**Recommandation :** Tester avec et sans CLAHE. Souvent utile en conditions d'éclairage variable, mais peut être contre-productif avec un éclairage contrôlé.

---

### 4. Accentuation de la Netteté (Sharpening)

**Opération :** 
```python
blur = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.addWeighted(img, sharpen_alpha, blur, sharpen_beta, 0)
```

**Paramètres :**
- `sharpen_alpha` (1.0-2.0) : Poids de l'image originale (plus élevé = plus de contraste)
- `sharpen_beta` (-0.5 à -1.0) : Poids de l'image floue (négatif pour soustraire)
- Kernel de flou : (5, 5) pour le filtre gaussien

**Explication :**
La technique d'unsharp masking accentue les contours en soustrayant une version floue de l'image. Le principe :
1. Créer une version floue (GaussianBlur élimine les hautes fréquences)
2. Soustraire cette version floue de l'originale → amplifie les différences (contours)
3. Ajouter ce résultat à l'image originale

Formule : `sharpened = original × alpha - blurred × |beta|`

**Impact sur la détection :**
- **Améliore la détection des contours** : Les bords des marqueurs deviennent plus nets
- **Aide avec les images légèrement floues** : Compense un défaut de mise au point
- **Risque d'amplifier le bruit** : Si alpha trop élevé ou beta trop négatif
- **Peut créer des artefacts** : Halos autour des contours si paramètres extrêmes

**Valeurs typiques :**
- Légère accentuation : alpha=1.5, beta=-0.5
- Accentuation forte : alpha=2.0, beta=-1.0

---

### 5. Détection des Marqueurs ArUco

**Opération :** `aruco_detector.detectMarkers(img)`

#### 5.1 Seuillage Adaptatif

**Paramètre :** `adaptiveThreshConstant` (5-10)

**Explication :**
Le seuillage adaptatif convertit l'image en noir et blanc en calculant un seuil local pour chaque pixel basé sur son voisinage. La formule :
```
T(x,y) = mean(voisinage) - adaptiveThreshConstant
pixel_binaire = 255 si pixel > T(x,y), sinon 0
```

Cette méthode s'adapte aux variations d'éclairage local, contrairement au seuillage global. La constante ajuste la "sensibilité" :
- **Valeur basse (5)** : Seuil plus permissif, plus de pixels blancs, détecte dans zones sombres
- **Valeur haute (10)** : Seuil plus strict, moins de pixels blancs, réduit le bruit

**Impact :** Critique pour la première étape de détection des contours. Influence directement le nombre de candidats détectés.

---

#### 5.2 Filtrage par Périmètre

**Paramètres :**
- `minMarkerPerimeterRate` (0.01-0.05) : Taille minimale relative
- `maxMarkerPerimeterRate` (3.0-5.0) : Taille maximale relative

**Explication :**
Ces paramètres définissent la plage de tailles acceptables pour un marqueur, en fonction du périmètre de l'image :
```
perimetre_image = 2 × (largeur + hauteur)
perimetre_min = perimetre_image × minMarkerPerimeterRate
perimetre_max = perimetre_image × maxMarkerPerimeterRate
```

Les contours dont le périmètre est hors de cette plage sont rejetés. Cela élimine :
- Les très petits contours (bruit, pixels isolés)
- Les très grands contours (bords de l'image, objets non-marqueurs)

**Impact :**
- **minMarkerPerimeterRate trop élevé** : Rate les petits marqueurs éloignés
- **minMarkerPerimeterRate trop bas** : Trop de faux positifs (bruit)
- **maxMarkerPerimeterRate trop bas** : Rate les grands marqueurs proches

**Valeurs typiques :**
- Petite image ou marqueurs éloignés : min=0.01
- Grande image ou marqueurs proches : min=0.03-0.05

---

#### 5.3 Approximation Polygonale

**Paramètre :** `polygonalApproxAccuracyRate` (0.03-0.1)

**Explication :**
Cette étape utilise l'algorithme de Douglas-Peucker pour approximer les contours détectés par des polygones. Le paramètre définit la tolérance d'approximation :
```
epsilon = perimetre_contour × polygonalApproxAccuracyRate
```

L'algorithme simplifie le contour en conservant uniquement les points "importants" selon epsilon :
- **Valeur basse (0.03)** : Approximation précise, polygone proche du contour réel
- **Valeur haute (0.1)** : Approximation grossière, moins de points

Les marqueurs ArUco étant carrés, on recherche des polygones à 4 côtés. Une approximation trop précise peut créer des polygones à 5+ côtés (bruit sur les contours).

**Impact :**
- **Trop bas** : Contours bruités ne forment pas des quadrilatères propres
- **Trop haut** : Perte de précision angulaire, peut fusionner des marqueurs proches

---

#### 5.4 Contraintes Géométriques

**Paramètres :**
- `minCornerDistanceRate` (0.05-0.1) : Distance minimale entre coins
- `minDistanceToBorder` (3-5 pixels) : Marge minimale aux bords

**Explication :**

**minCornerDistanceRate :**
Définit la distance minimale entre deux coins d'un même marqueur, relative à sa taille :
```
distance_min = perimetre_marqueur × minCornerDistanceRate
```
Rejette les quadrilatères dont les coins sont trop rapprochés (forme dégénérée, marqueur partiellement visible).

**minDistanceToBorder :**
Marge de sécurité en pixels par rapport aux bords de l'image. Les marqueurs trop proches des bords sont rejetés car :
- Probablement partiellement coupés
- Zone où la distorsion optique est maximale
- Risque de coins manquants

**Impact :**
- Élimine les faux positifs (formes dégénérées)
- Peut rejeter de vrais marqueurs partiellement visibles
- Réduit les détections en bordure d'image

---

### 6. Raffinement des Coins (Subpixel)

**Paramètre :** `cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX`

**Explication :**
Après la détection initiale, cette étape affine la position des coins avec une précision subpixellique. Le processus :

1. **Détection initiale** : Précision au pixel près (erreur ~0.5 pixel)
2. **Raffinement subpixel** : Analyse le gradient d'intensité autour de chaque coin
3. **Optimisation** : Trouve la position exacte du coin avec précision ~0.1 pixel

L'algorithme cherche le point où les gradients de deux bords se croisent, en utilisant une fenêtre de recherche locale et une méthode itérative.

**Alternatives :**
- `CORNER_REFINE_NONE` : Pas de raffinement (plus rapide, moins précis)
- `CORNER_REFINE_CONTOUR` : Raffinement basé sur le contour
- `CORNER_REFINE_APRILTAG` : Méthode AprilTag (plus robuste, plus lent)

**Impact :**
- **Améliore la précision de position** : Erreur réduite de ~5× (0.5→0.1 pixel)
- **Améliore la précision angulaire** : Angles plus précis
- **Essentiel pour transformation perspective** : Erreurs de coins = erreurs amplifiées en coordonnées réelles
- **Coût** : +20-30% de temps de traitement (mais vaut généralement le coût)

---

### 7. Calcul des Caractéristiques des Marqueurs

#### 7.1 Centre du Marqueur

**Calcul :**
```python
cX = mean(corners[:, 0])  # Moyenne des X des 4 coins
cY = mean(corners[:, 1])  # Moyenne des Y des 4 coins
```

**Explication :**
Le centre est calculé comme le barycentre (centre de gravité) des 4 coins. Pour un quadrilatère quelconque, c'est l'intersection des diagonales. Cette position sert de point de référence pour :
- La transformation de perspective
- La localisation du marqueur
- La distance entre marqueurs

**Précision :** Hérite de la précision du raffinement subpixel (~0.1 pixel).

---

#### 7.2 Angle d'Orientation

**Calcul :**
```python
dx = corners[1][0] - corners[0][0]  # Top-right X - Top-left X
dy = corners[1][1] - corners[0][1]  # Top-right Y - Top-left Y
angle = degrees(arctan2(dy, dx))
```

**Explication :**
L'angle est calculé à partir du bord supérieur du marqueur (du coin top-left au coin top-right). La fonction `arctan2(dy, dx)` :
- Calcule l'angle du vecteur (dx, dy) par rapport à l'horizontale
- Gère tous les quadrants correctement (±180°)
- Représente la rotation du marqueur dans l'image

**Convention :**
- 0° : Marqueur horizontal, bord haut parallèle à l'axe X
- +90° : Marqueur tourné de 90° sens horaire
- -90° : Marqueur tourné de 90° sens anti-horaire

**Usage :**
- Orientation des objets de jeu (boîtes, robots)
- Validation de la cohérence (marqueurs fixes doivent avoir angle constant)
- Calculs de trajectoire

---

### 8. Transformation de Perspective

**Opération :** `cv2.getPerspectiveTransform(src_points, dst_points)`

#### 8.1 Points Source et Destination

**Explication :**
La transformation de perspective établit une correspondance entre :
- **Points source** : Positions des marqueurs fixes dans l'image (pixels)
- **Points destination** : Positions réelles connues sur le terrain (mm)

Les 4 marqueurs fixes (IDs 20, 21, 22, 23) aux coins du terrain servent de référence. Leurs positions réelles sont connues avec précision :
```
A1 (ID 20) : (600, 600) mm
B1 (ID 22) : (1400, 600) mm
C1 (ID 21) : (600, 2400) mm
D1 (ID 23) : (1400, 2400) mm
```

**Ordre critique :** Les points source et destination doivent être dans le MÊME ordre, sinon la matrice sera incorrecte.

---

#### 8.2 Matrice de Transformation

**Calcul :**
```python
H = getPerspectiveTransform(src_points, dst_points)
```

**Explication :**
Cette fonction calcule une matrice de transformation homographique 3×3 qui projette le plan image sur le plan terrain. La matrice H satisfait :
```
[x_terrain]       [x_image]
[y_terrain] = H × [y_image]
[   1     ]       [   1   ]
```

Le calcul résout un système d'équations linéaires avec les 4 paires de points (minimum requis pour une homographie). Plus de 4 points peuvent être utilisés avec méthode RANSAC pour plus de robustesse.

**Propriétés :**
- Conserve les lignes droites (mais pas les angles ni les distances)
- Transforme les parallèles en lignes convergentes (effet de perspective)
- Inverse calculable : peut convertir terrain → image

---

#### 8.3 Application de la Transformation

**Opération :**
```python
real_point = cv2.perspectiveTransform(img_point, H)
```

**Explication :**
Pour chaque marqueur détecté, on transforme son centre (x_img, y_img) en coordonnées réelles (x_real, y_real) :

1. Le point image est converti en coordonnées homogènes
2. Multiplié par la matrice H
3. Normalisé pour obtenir les coordonnées réelles

**Sources d'erreur :**
- **Erreur de détection des coins** des marqueurs fixes (~0.1-1 pixel)
- **Erreur sur positions réelles** des marqueurs fixes (installation physique)
- **Déformation du terrain** (planéité imparfaite)
- **Distorsion résiduelle** après correction optique

**Impact de l'erreur :**
Une erreur de 1 pixel sur les marqueurs fixes peut se traduire par 5-20mm d'erreur sur les coordonnées réelles, selon :
- La distance à la caméra
- La position par rapport aux marqueurs de référence
- La qualité de la correction de distorsion

---


## Cas Particuliers et Problèmes Courants

### Marqueurs Non Détectés

**Causes possibles :**
1. **Éclairage insuffisant** → Test CLAHE
2. **Flou de bougé/mise au point** → Augmenter sharpening
3. **Marqueur trop petit** → Réduire minMarkerPerimeterRate
4. **Marqueur trop proche du bord** → Réduire minDistanceToBorder
5. **Distorsion excessive** → Vérifier calibration caméra

### Faux Positifs

**Causes possibles :**
1. **Bruit amplifié** → Réduire sharpening ou CLAHE
2. **Seuillage trop permissif** → Augmenter adaptiveThreshConstant
3. **Objets similaires** → Augmenter minMarkerPerimeterRate
4. **Patterns réguliers** → Améliorer validation des IDs ArUco

### Erreurs de Position Élevées

**Causes possibles :**
1. **Marqueurs fixes mal positionnés** → Vérifier installation physique
2. **Coins imprécis** → Activer CORNER_REFINE_SUBPIX
3. **Distorsion non corrigée** → Recalibrer caméra
4. **Terrain non plan** → Modèle de transformation inadéquat

---

## Recommandations Finales

### Configuration Standard

```python
# Prétraitement optimal
sharpen_alpha = 1.5
sharpen_beta = -0.5
use_clahe = False  # Sauf éclairage variable

# Détection ArUco optimale
adaptive_thresh_constant = 7
min_marker_perimeter_rate = 0.03
max_marker_perimeter_rate = 4.0
polygonal_approx_accuracy_rate = 0.03
corner_refinement_method = CORNER_REFINE_SUBPIX
```


