"""
Group images by aspect ratio to improve training speed by clustering same-ratio images in the same batch.

This code is from https://github.com/pytorch/vision.
"""
import bisect
import copy
import math
from collections import defaultdict
from itertools import chain, repeat
from typing import Any

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import BatchSampler, Sampler
from tqdm import tqdm


def _repeat_to_at_least(iterable: Any, n: int) -> Any:
    """Repeat the iterable."""
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.

    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    :param sampler: Base sampler.
    :param group_ids: If the sampler produces indices in range [0, N),
        `group_ids` must be a list of `N` ints which contains the group id of each sample.
        The group ids must be a continuous set of integers starting from
        0, i.e. they must be in the range [0, num_groups).
    :param batch_size: Size of mini-batch.
    """

    def __init__(self, sampler: Any, group_ids: Any, batch_size: Any) -> None:
        """Initialise the batch sampler."""
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                f"torch.utils.data.Sampler, but got sampler={sampler}"
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self) -> Any:
        """Iterate the next sample."""
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(
                buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True
            ):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self) -> int:
        """Get the size of the batch sampler."""
        return len(self.sampler) // self.batch_size  # type: ignore


def _compute_aspect_ratios_slow(dataset: Any, indices: Any = None) -> Any:
    """Compute the aspect ratios."""
    print(
        "Your dataset doesn't support the fast path for "
        "computing the aspect ratios, so will iterate over "
        "the full dataset and load every image instead. "
        "This might take some time..."
    )
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):  # type: ignore
        """Subset sampler."""

        def __init__(self, indices: Any) -> None:
            self.indices = indices

        def __iter__(self) -> Any:
            return iter(self.indices)

        def __len__(self) -> int:
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset: Any, indices: Any = None) -> Any:
    """Compute the aspect ratios of the custom dataset."""
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset: Any, indices: Any = None) -> Any:
    """Compute the aspect ratios of the COCO dataset."""
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset: Any, indices: Any = None) -> Any:
    """Compute the aspect ratios of the VOC dataset."""
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset: Any, indices: Any = None) -> Any:
    """Compute the aspect ratios for the subset of the dataset."""
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset: Any, indices: Any = None) -> Any:
    """Compute all the aspect ratios."""
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x: Any, bins: Any) -> Any:
    """Quantize."""
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset: Any, k: int = 0) -> Any:
    """Create aspect ratios for the groups."""
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print(f"Using {fbins} as bins for aspect ratio quantization")
    print(f"Count of instances per bin: {counts}")
    return groups
