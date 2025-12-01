# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import math

# ---------------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------------

class Point:
    def __init__(self, x, y, aruco_id, angle=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.aruco_id = aruco_id

    def __str__(self):
        return f"Point(x={self.x}, y={self.y}, aruco_id={self.aruco_id}, angle={self.angle})"
    
    def __print__(self):
        return self.__str__()

def distance(p1: Point, p2: Point) -> float:
    """Calcule la distance euclidienne entre deux points"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def find_point_by_id(points: list[Point], target_id: int):
    """
    Trouve un point par son aruco_id dans une liste de points.

    Args:
        points: La liste des points dans laquelle chercher.
        target_id: L'aruco_id du point à trouver.

    Returns:
        Le Point correspondant à l'aruco_id, ou None si aucun point n'est trouvé.
    """
    for p in points:
        if p.aruco_id == target_id:
            return p
    return None

def print_points(points : list) -> None:
    """Affiche une liste de points"""
    for i in range(len(points)):
        print(f"{i}: {points[i].aruco_id} -> ({points[i].x}, {points[i].y})")