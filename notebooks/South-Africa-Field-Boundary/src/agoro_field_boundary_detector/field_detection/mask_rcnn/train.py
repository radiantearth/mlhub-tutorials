"""PyTorch Detection Training, code from https://github.com/pytorch/vision."""
import datetime
import os
import time
from typing import Any

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

import src.agoro_field_boundary_detector.field_detection.mask_rcnn.presets as presets
import src.agoro_field_boundary_detector.field_detection.mask_rcnn.utils as utils
from agoro_field_boundary_detector.field_detection.mask_rcnn.coco_utils import get_coco, get_coco_kp
from agoro_field_boundary_detector.field_detection.mask_rcnn.engine import evaluate, train_one_epoch


def get_dataset(name: Any, image_set: Any, transform: Any, data_path: Any) -> Any:
    """Get the COCO dataset."""
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)  # type: ignore
    return ds, num_classes


def get_transform(train: Any) -> Any:
    """Transform the presets."""
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def main(args: Any) -> None:  # noqa C901
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)  # type: ignore

    # Data loading code
    print("Loading data")
    dataset, num_classes = get_dataset(
        args.dataset, "train", get_transform(train=True), args.data_path
    )
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)  # type: ignore
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)  # type: ignore
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)  # type: ignore
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)  # type: ignore

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=num_classes, pretrained=args.pretrained, **kwargs
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")  # type: ignore
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"model_{epoch}.pth"),
            )

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", help="dataset")
    parser.add_argument("--dataset", default="coco", help="dataset")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", help="model")
    parser.add_argument("--device", default="cuda", help="device")  # TODO: Detected later on?
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs", default=26, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training "
        "on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn"
    )
    parser.add_argument(
        "--trainable-backbone-layers",
        default=None,
        type=int,
        help="number of trainable layers of backbone",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)
    main(args)
