"""Wrapper around GEE's NAIP dataset collection."""
from pathlib import Path
from typing import Any, Dict, Tuple

import ee
from requests import get


class NaipCollection:
    """National Agriculture Imagery Program data collection."""

    def __init__(
        self,
        region: ee.Geometry.Polygon,
        startdate: str = "2017-01-01",
        enddate: str = "2020-12-31",
    ) -> None:
        """
        Initialise the National Agriculture Imagery Program (NAIP) collection.

        For more information, visit:
        https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ

        :param region: Region to load data for
        :param startdate: Minimal date of the image to capture
        :param enddate: Maximal date of the image to capture
        """
        self.region = region
        self.collection = (
            ee.ImageCollection("USDA/NAIP/DOQQ").filterDate(startdate, enddate).filterBounds(region)
        )

    def __str__(self) -> str:
        """Representation of the data collection."""
        return "NAIP Collection"

    def __repr__(self) -> str:
        """Representation of the data collection."""
        return str(self)

    def get_image(self) -> ee.Image:
        """Get the representative image from the collection."""
        return self.collection.mosaic()

    def get_vis_params(self) -> Dict[str, Any]:
        """Get the visualisation parameters of this collection."""
        return {
            "min": 0.0,
            "max": 255.0,
            "bands": ["R", "G", "B"],
            "region": self.region,
        }

    def get_size(self) -> int:
        """Get the number of images captured in the collection."""
        return self.collection.size().getInfo()  # type: ignore

    def export_as_png(
        self,
        file_name: Path,
        dimensions: Tuple[int, int] = (1024, 1024),
    ) -> None:
        """
        Export the data collection as PNG images.

        :param file_name: Complete path with filename to where to store the retrieved thumbnail image
        :param dimensions: Dimensions of the output image expressed in pixels (width, height)
        """
        # Update the visualisation parameters for the export
        params = self.get_vis_params()
        params["dimensions"] = f"{dimensions[0]}x{dimensions[1]}"

        # Create a thumbnail image for the collection and download
        url = self.get_image().getThumbURL(params)
        img_data = get(url).content
        with open(file_name, "wb") as handler:
            handler.write(img_data)
