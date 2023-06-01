"""Collect and export field data using GEE's NAIP dataset."""
import json
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from agoro_field_boundary_detector.google_earth_engine import (
    NaipCollection,
    create_bounding_box,
    start_session,
    to_polygon,
)


def main(coordinates: List[Tuple[float, float]], write_folder: Path) -> None:
    """
    Export a list of coordinates' thumbnail images.

    :param coordinates: Coordinates for which to export a PNG thumbnail
    :param write_folder: Folder to write the data to
    """
    for c in tqdm(coordinates, "Exporting"):
        export(coordinate=c, write_path=write_folder / f"{c[0]}-{c[1]}.png")


def export(coordinate: Tuple[float, float], write_path: Path) -> None:
    """
    Collect the data (thumbnail) of a bounding box around the specified coordinate and write as PNG.

    :param coordinate: Coordinate to export
    :param write_path: Path to write the generated thumbnail to
    """
    # Create bounding box
    region = to_polygon(
        create_bounding_box(
            lat=coordinate[0],
            lng=coordinate[1],
        )
    )

    # Load in the collection
    collection = NaipCollection(
        region=region,
    )

    # Export the image
    collection.export_as_png(
        file_name=write_path,
    )


if __name__ == "__main__":
    # Start a new GEE session
    start_session()

    # Load in the coordinates to sample
    DATA_PATH = Path(__file__).parent / "../../data"
    (DATA_PATH / "raw").mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH / "coordinates.json", "r") as f:
        coordinate_f = json.load(f)

    # Export the given coordinates
    main(
        coordinates=coordinate_f["train"] + coordinate_f["test"],
        write_folder=DATA_PATH / "raw",
    )
