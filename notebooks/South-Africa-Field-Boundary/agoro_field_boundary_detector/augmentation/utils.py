"""Utilisation functions for data manipulation."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_annotations(path: Path) -> Dict[str, List[List[Tuple[int, int]]]]:
    """Load in the annotations made by VGG."""
    with open(path, "r") as f:
        annotations = json.load(f)

    def _transform(values: List[Any]) -> List[List[Tuple[int, int]]]:
        """Transform the Label Studio syntax to image-coordinates."""
        boundaries = []
        for value in values:
            x = value["shape_attributes"]["all_points_x"]
            x += [x[-1]]
            y = value["shape_attributes"]["all_points_y"]
            y += [y[-1]]
            boundaries.append(list(zip(x, y)))
        return boundaries

    # Load in the boundaries by field
    field_annotations = {}
    for key, annotation in annotations.items():
        field_annotations[key] = _transform(annotation["regions"])
    return field_annotations


def polygons_to_mask(
    polygons: List[List[Tuple[int, int]]],
    width: int = 1024,
    height: int = 1024,
) -> np.ndarray:
    """Create a masking image using the provided polygon."""
    mask = np.zeros((width, height))
    for idx, polygon in enumerate(reversed(polygons)):  # Latest annot likely worse annot
        img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += (idx + 1) * np.array(img)  # Add binary mask (0=background, (idx+1)=field)
        mask = np.clip(mask, a_min=0, a_max=(idx + 1))  # type: ignore
    print(mask)
    return mask
