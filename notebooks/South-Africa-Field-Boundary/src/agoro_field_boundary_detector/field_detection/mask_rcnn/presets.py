"""Configure presets, code from https://github.com/pytorch/vision."""
from typing import Any

import src.agoro_field_boundary_detector.field_detection.mask_rcnn.transforms as T


class DetectionPresetTrain:
    """Detection presets for training."""

    def __init__(self, hflip_prob: float = 0.5) -> None:
        """Initialise the training preset detection."""
        trans = [T.ToTensor()]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        self.transforms = T.Compose(trans)

    def __call__(self, img: Any, target: Any) -> Any:
        """Transform the image and target."""
        return self.transforms(img, target)


class DetectionPresetEval:
    """Detection preset for evaluation."""

    def __init__(self) -> None:
        """Initialise the evaluation preset detection."""
        self.transforms = T.ToTensor()

    def __call__(self, img: Any, target: Any) -> Any:
        """Transform the image and target."""
        return self.transforms(img, target)
