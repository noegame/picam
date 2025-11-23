"""

"""

import math

class Point:
    def __init__(self, x, y, ID, angle=None):
        self.x = x
        self.y = y
        self.ID = ID
        self.angle = angle 

    def __str__(self):
        return f"Point(x={self.x}, y={self.y}, ID={self.ID}, angle={self.angle})"
    
    def __print__(self):
        return self.__str__()

def distance(p1: Point, p2: Point):
    """Calcule la distance euclidienne entre deux points"""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def find_point_by_id(points, target_id):
    """Trouve un point par son ID dans une liste de points"""
    for p in points:
        if p.ID == target_id:
            return p
    return None

def print_points(points : list):
    """Affiche une liste de points"""
    for i in range(len(points)):
        print(f"{i}: {points[i].ID} -> ({points[i].x}, {points[i].y})")