"""Interface around the field boundary detector model used for inference."""
import os
from pathlib import Path
from random import getrandbits
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.agoro_field_boundary_detector.field_detection import FieldBoundaryDetector


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class FieldBoundaryDetectorInterface:
    """Interface around the field boundary detector model used for inference."""

    def __init__(
        self,
        model_path: Path,
        new_session: bool = True,
    ) -> None:
        """
        Initialise the interface/wrapper by loading in the model and starting a GEE session.

        :param model_path: Path towards a pre-trained model file
        :param new_session: Start a new GEE session
        """
        self.model = FieldBoundaryDetector(model_path=model_path)
        if new_session:
            start_session()

    def __call__(
        self, lat: float, lng: float, thr: float = 0.5
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Get the field-polygon of the field overlapping with the provided coordinate.

        :param lat: Latitude coordinate
        :param lng: Longitude coordinate
        :param thr: Certainty threshold
        :return: Polygon (list of (lat,lng) coordinates) or None if no field recognised
        """
        # Create temporal file to write extracted image to
        f = Path.cwd() / f"{getrandbits(128)}.png"

        # Fetch image using GEE
        box = to_polygon(create_bounding_box(lat=lat, lng=lng))
        coll = NaipCollection(region=box)
        coll.export_as_png(file_name=f)
        field = np.array(Image.open(f))[:, :, :3]  # Assure RGB

        # Make prediction
        self.model.thr = thr
        polygon = self.model(field)

        # Convert polygon's pixel-coordinates to (lat,lng) coordinates
        polygon_adj = adjust_polygon(
            coordinate=(lat, lng),
            center=(512, 512),
            polygon=polygon,
        )

        # Remove the temporal file and return the result
        f.unlink(missing_ok=True)
        return polygon_adj if polygon_adj else None

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "FieldBoundaryDetectorInterface"

    def __repr__(self) -> str:
        """Representation of the data collection."""
        return str(self)


if __name__ == "__main__":
    # Demo: Load in the model
    model = FieldBoundaryDetectorInterface(Path.cwd() / "../../models/mask_rcnn")

    # Make the prediction
    pred = model(lat=39.6679328199836, lng=-95.4287818841267, thr=0.9)
    print(pred)
