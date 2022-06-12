"""Utilisation functions used by the field boundary detection model."""
from typing import Any, List, Tuple

import cv2
import numpy as np


def mask_to_polygons(
    mask: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """Transform a mask back to a polygon boundary."""
    polygons = []
    for v in sorted(set(np.unique(mask)) - {0}):
        # Extract polygon
        contours, _ = cv2.findContours(
            np.asarray(mask == v, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        polygon = []
        for point in sorted(contours, key=lambda x: -len(x))[
            0
        ]:  # If multiple (shouldn't be); use largest
            polygon.append((int(point[0][0]), int(point[0][1])))

        # Trim redundant points
        i = 1
        while i < len(polygon) - 1:
            if _is_line(polygon[i - 1 : i + 2]):
                del polygon[i]
            else:
                i += 1
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        polygons.append(polygon)
    return polygons


def _is_line(data: Any) -> bool:
    """Check if straight line."""
    x, y = zip(*data)
    assert len(x) == 3
    for i in range(-2, 2 + 1):
        if x[0] - i == x[1] == x[2] + i:
            return True
        if y[0] - i == y[1] == y[2] + i:
            return True
    return False
