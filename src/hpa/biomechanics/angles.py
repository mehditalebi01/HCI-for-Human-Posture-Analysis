"""Reusable geometry helpers for 2D biomechanical features."""

import math


def angle_between_three_points(point_a, point_b, point_c):
    """Return the angle ABC in degrees.

    The middle point, B, is the joint center. For example, hip-knee-ankle gives
    a 2D knee angle measured in the image plane.
    """
    ax, ay = point_a
    bx, by = point_b
    cx, cy = point_c

    vector_ba = (ax - bx, ay - by)
    vector_bc = (cx - bx, cy - by)

    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    length_ba = math.hypot(vector_ba[0], vector_ba[1])
    length_bc = math.hypot(vector_bc[0], vector_bc[1])

    if length_ba == 0 or length_bc == 0:
        return None

    cosine_angle = dot_product / (length_ba * length_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    return math.degrees(math.acos(cosine_angle))


def line_angle_from_vertical(point_top, point_bottom):
    """Return the absolute 2D lean angle of a body segment from vertical."""
    top_x, top_y = point_top
    bottom_x, bottom_y = point_bottom

    dx = top_x - bottom_x
    dy = top_y - bottom_y

    if dx == 0 and dy == 0:
        return None

    # Image y points downward. Using abs(dx) gives left/right lean magnitude.
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def absolute_difference(value_a, value_b):
    """Return the absolute difference when both values are available."""
    if value_a is None or value_b is None:
        return None

    return abs(value_a - value_b)
