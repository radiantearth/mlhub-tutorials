"""Main engine of the Mask RCNN, code from https://github.com/pytorch/vision."""
import math
import sys
import time
from typing import Any

import torch
import torchvision.models.detection.mask_rcnn

from src.agoro_field_boundary_detector.field_detection.mask_rcnn.coco_eval import CocoEvaluator
from src.agoro_field_boundary_detector.field_detection.mask_rcnn.coco_utils import (
    get_coco_api_from_dataset,
)
from src.agoro_field_boundary_detector.field_detection.mask_rcnn.utils import  (
    MetricLogger,
    SmoothedValue,
    reduce_dict,
    warmup_lr_scheduler,
)
# from agoro_field_boundary_detector.field_detection.mask_rcnn.utils import (
#     MetricLogger,
#     SmoothedValue,
#     reduce_dict,
#     warmup_lr_scheduler,
# )


def train_one_epoch(
    model: Any, optimizer: Any, data_loader: Any, device: Any, epoch: Any, print_freq: Any
) -> Any:
    """Train for a single epoch."""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()  # type: ignore

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()  # type: ignore
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model: Any) -> Any:
    """Get the IoU types."""
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model: Any, data_loader: Any, device: Any) -> Any:
    """Evaluate the model."""
    n_threads = torch.get_num_threads()  # type: ignore
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)  # type: ignore
    cpu_device = torch.device("cpu")  # type: ignore
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)  # type: ignore

    # Return the F1 score of IoU=.5:.95
    stats = coco_evaluator.coco_eval["segm"].stats
    p, r = stats[0], stats[8]
    return 2 * (p * r) / max(p + r, 1e-5)