"""Transformation methods and classes, code from https://github.com/pytorch/vision."""
from typing import Any

from torchvision.transforms import functional as F


class Compose(object):
    """Compose a target object."""

    def __init__(self, transforms: Any) -> None:
        """Initialise the composer."""
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> Any:
        """Compose a (image,target) object couple."""
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """Transform image to PyTorch tensor."""

    def __call__(self, image: Any, target: Any) -> Any:
        """Transform image to PyTorch tensor."""
        image = F.to_tensor(image)
        return image, target
