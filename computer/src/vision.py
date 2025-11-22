# vision.py
# 1. Prends une photo avec la caméra ou importe une image depuis un fichier.
# 2. Corrige la distorsion de l'image en utilisant les paramètres de distorsion de la caméra.
# 3. Détecte les tags ArUco dans l'image corrigée.
# 4. Grace au 4 tags aruco fixes dont on connait la position dans le monde réel, calcule la transformation 
#    entre le repère de la caméra et le repère du monde réel.
# 5. Utilise cette transformation pour estimer la position et l'orientation de tout autre tag ArUco détecté dans l'image.

