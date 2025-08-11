import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

import math

def calculate_vertical_angle(a, b):
    """
    Calculates the vertical angle (in degrees) of a segment AB relative to the vertical axis.
    a, b: tuples or lists of (x, y) coordinates.
    """
    ax, ay = a
    bx, by = b
    
    # Difference in coordinates
    dx = bx - ax
    dy = by - ay

    # Angle from vertical: we compare with vertical axis, so atan2(dx, dy)
    angle = math.degrees(math.atan2(dx, dy))

    return abs(angle)  # always positive
