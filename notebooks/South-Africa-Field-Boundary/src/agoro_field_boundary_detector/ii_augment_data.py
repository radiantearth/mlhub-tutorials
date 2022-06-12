"""Augment the training data."""
import json
from pathlib import Path
from shutil import rmtree
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from agoro_field_boundary_detector.augmentation import (
    get_random_noise,
    load_annotations,
    polygons_to_mask,
    t_linear,
    t_quartile,
    transform,
)


def generate(
    field: np.ndarray,
    mask: np.ndarray,
    write_folder: Path,
    dupl: int = 2,
    prefix: str = "",
) -> None:
    """
    Generate data augmentations of the provided field and corresponding mask.

    Every field is augmented dupl*5 times, where every 5 times includes:
     - Linear (no) transformation
     - Four quartile cuts of the input image
    Aside from these translation augmentations, a random noise is added, chosen from:
     - Linear (no) transformation
     - Rotation
     - Horizontal or vertical flip
     - Gaussian filter (blur)
     - Gamma filter (brightness)

    :param field: Input array of the field to augment
    :param mask: Input array of the corresponding mask to augment
    :param write_folder: Folder (path) to write the results (augmentations) to
    :param dupl: Number of times to add random noise to the 5 translation augmentations
    :param prefix: Field-specific prefix used when writing the augmentation results
    """
    # Generate transformations
    idx = 0
    for _ in range(dupl):
        for t, t_idx in (
            (t_linear, 0),
            (t_quartile, 0),
            (t_quartile, 1),
            (t_quartile, 2),
            (t_quartile, 3),
        ):
            n, n_idx = get_random_noise()
            field_t, mask_t = transform(
                field=field,
                mask=mask,
                translation=t,  # type: ignore
                t_idx=t_idx,
                noise=n,
                n_idx=n_idx,
            )
            # Save as PNG; slower but more memory efficient than pure numpy
            Image.fromarray(np.uint8(field_t)).save(write_folder / f"fields/{prefix}_{idx}.png")
            Image.fromarray(np.uint8(mask_t)).save(write_folder / f"masks/{prefix}_{idx}.png")
            idx += 1


def main(
    fields: List[np.ndarray],
    masks: List[np.ndarray],
    prefixes: List[str],
    write_folder: Path,
    dupl: int = 2,
) -> None:
    """
    Generate and save data augmentations for all the fields and corresponding masks.

    Every field is augmented dupl*5 times, where every 5 times includes:
     - Linear (no) transformation
     - Four quartile cuts of the input image
    Aside from these translation augmentations, a random noise is added, chosen from:
     - Linear (no) transformation
     - Rotation
     - Horizontal or vertical flip
     - Gaussian filter (blur)
     - Gamma filter (brightness)

    :param fields: Fields to augment
    :param masks: Corresponding masks to augment
    :param prefixes: Field-specific prefixes corresponding each field
    :param write_folder: Path to write the results (augmentations) to
    :param dupl: Number of times to add random noise to the 5 translation augmentations
    """
    for field, mask, prefix in tqdm(
        zip(fields, masks, prefixes), total=len(prefixes), desc="Generating"
    ):
        generate(
            field=field,
            mask=mask,
            prefix=prefix,
            dupl=dupl,
            write_folder=write_folder,
        )


if __name__ == "__main__":
    # Load in the annotations
    DATA_PATH = Path(__file__).parent / "../../data"
    annotations = load_annotations(DATA_PATH / "annotations.json")

    # Load in the requested coordinates, keep only training
    with open(DATA_PATH / "coordinates.json", "r") as f:
        coordinates = json.load(f)
    coordinate_names = [f"{c[0]}-{c[1]}" for c in coordinates["train"]]
    names = [k for k in annotations.keys() if k in coordinate_names]

    # Load in fields and corresponding masks
    annotated_fields, annotated_masks = [], []
    for name in names:
        annotated_fields.append(np.asarray(Image.open(DATA_PATH / f"raw/{name}.png")))
        annotated_masks.append(polygons_to_mask(annotations[name]))

    # Ensure folders exist
    if (DATA_PATH / "augmented").is_dir():
        rmtree(DATA_PATH / "augmented")
    (DATA_PATH / "augmented/fields").mkdir(exist_ok=True, parents=True)
    (DATA_PATH / "augmented/masks").mkdir(exist_ok=True, parents=True)

    # Export the given coordinates
    main(
        fields=annotated_fields,
        masks=annotated_masks,
        prefixes=names,
        write_folder=DATA_PATH / "augmented",
    )

    # Write test-data as-is
    if (DATA_PATH / "test").is_dir():
        rmtree(DATA_PATH / "test")
    (DATA_PATH / "test/fields").mkdir(exist_ok=True, parents=True)
    (DATA_PATH / "test/masks").mkdir(exist_ok=True, parents=True)
    test_coordinate_names = [f"{c[0]}-{c[1]}" for c in coordinates["test"]]
    test_names = [k for k in annotations.keys() if k in test_coordinate_names]
    for name in tqdm(test_names, "Generating test"):
        f = np.asarray(Image.open(DATA_PATH / f"raw/{name}.png"))  # type: ignore
        Image.fromarray(np.uint8(f)).save(DATA_PATH / f"test/fields/{name}.png")
        m = polygons_to_mask(annotations[name])
        Image.fromarray(np.uint8(m)).save(DATA_PATH / f"test/masks/{name}.png")
