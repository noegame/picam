aruco_color = {
    20: "white",
    21: "white",
    22: "white",
    23: "white",
    36: "blue",
    47: "yellow",
}


aruco_smiley = {
    20: "⚪",
    21: "⚪",
    22: "⚪",
    23: "⚪",
    36: "🔵",
    47: "🟡",
}


def get_aruco_color(tag_id: int) -> str:
    return aruco_color.get(tag_id, "unknown")


def get_aruco_smiley(tag_id: int) -> str:
    return aruco_smiley.get(tag_id, "")


def get_aruco_smiley_dict() -> dict:
    return aruco_smiley
