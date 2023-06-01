"""Dataset class."""
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F_vis


class Dataset(torch.utils.data.Dataset):  # type: ignore
    """Dataset used to train the Mask RCNN model."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        """Initialise the dataset."""
        self.path = path
        self.fields_path = self.path / "fields"
        self.masks_path = self.path / "masks"
        self.tags = [Path(x).with_suffix("").name for x in glob(str(self.fields_path / "*"))]

    def __getitem__(self, idx: int) -> Any:
        """Get the item at the given index from the dataset."""
        # load image and mask under specified index
        field = np.array(Image.open(self.fields_path / f"{self.tags[idx]}.png"))
        mask = np.array(Image.open(self.masks_path / f"{self.tags[idx]}.png"))

        # Mask are identified by their unique ID (zero for background)
        obj_ids = np.unique(mask)  # Sorted
        obj_ids = obj_ids[1:]

        # Split the index-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bounding box (target) coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # type: ignore

        # Create target-response
        target = {
            "boxes": boxes,
            "labels": torch.ones((num_objs,), dtype=torch.int64),  # type: ignore
            "masks": torch.as_tensor(masks, dtype=torch.uint8),  # type: ignore
            "image_id": torch.tensor([idx]),  # type: ignore
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),  # type: ignore #(boxes[:, 3,] - boxes[:, 1,]) * (boxes[:, 2,] - boxes[:, 0,]),  # type: ignore
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),  # type: ignore
        }
        return F_vis.to_tensor(field), target

    def __len__(self) -> int:
        """Get the size of the dataset."""
        return len(self.tags)
