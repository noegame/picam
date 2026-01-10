# usr/bin/env python3

"""
aruco.py
Module for handling ArUco markers and related functionalities.
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BLACK = "\033[90m"

aruco_smiley = {
    20: "⚪",
    21: "⚪",
    22: "⚪",
    23: "⚪",
    36: "🔵",
    47: "🟡",
    41: "⚫",
}

aruco_color = {
    20: WHITE,
    21: WHITE,
    22: WHITE,
    23: WHITE,
    36: BLUE,
    47: YELLOW,
    41: BLACK,
}


# ---------------------------------------------------------------------------
# Classes and Methods
# ---------------------------------------------------------------------------


class Aruco:
    def __init__(self, x, y, z, aruco_id, angle=None):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle
        self.aruco_id = aruco_id
        self.real_x = None
        self.real_y = None

    def __str__(self):
        return f"Aruco(id={self.aruco_id}, x={self.x}, y={self.y}, z={self.z}, angle={self.angle} real_x={self.real_x}, real_y={self.real_y})"

    def get_smiley(self, tag_id: int) -> str:
        return aruco_smiley.get(tag_id, "unknown")

    def get_color(self, tag_id: int) -> str:
        return aruco_color.get(tag_id, RESET)

    def set_real_world_coords(self, real_x: float, real_y: float):
        self.real_x = real_x
        self.real_y = real_y

    def print(self):
        smiley = self.get_smiley(self.aruco_id)
        color = self.get_color(self.aruco_id)
        if self.real_x is not None and self.real_y is not None:
            print(
                f"{color}{smiley} Aruco ID: {self.aruco_id} \t Image Coords: ({self.x:.2f}, {self.y:.2f}) \t Real World Coords: ({self.real_x:.2f}, {self.real_y:.2f}){color}{RESET}"
            )
        else:
            print(
                f"{color}{smiley} Aruco ID: {self.aruco_id} \t Image Coords: ({self.x:.2f}, {self.y:.2f}) \t Real World Coords: N/A{color}{RESET}"
            )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def find_aruco_by_id(points: list[Aruco], target_id: int):
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
