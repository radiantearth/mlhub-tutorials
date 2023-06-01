"""COCO evaluation classes and methods, code from https://github.com/pytorch/vision."""
import copy
import json
from collections import defaultdict
from typing import Any

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch._six
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.agoro_field_boundary_detector.field_detection.mask_rcnn.utils import all_gather


class CocoEvaluator(object):
    """COCO evaluator class."""

    def __init__(self, coco_gt: Any, iou_types: Any) -> None:
        """Initialise the evaluator."""
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []  # type: ignore
        self.eval_imgs = {k: [] for k in iou_types}  # type: ignore

    def update(self, predictions: Any) -> None:
        """Update the evaluation stats regarding the new predictions."""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self) -> None:
        """Synchronise the IoU types between parallel processes."""
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self) -> None:
        """Accumulate all stats."""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> None:
        """Print out a summary of the stats."""
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions: Any, iou_type: Any) -> Any:
        """Initialise for evaluation."""
        if iou_type == "bbox":
            return prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")


def prepare_for_coco_detection(predictions: Any) -> Any:
    """Initialise for evaluation on boundary detection."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def prepare_for_coco_segmentation(predictions: Any) -> Any:
    """Initialise for evaluation on segmentation."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        masks = prediction["masks"] > 0.5

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def prepare_for_coco_keypoint(predictions: Any) -> Any:
    """Initialise for keypoint detection."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        keypoints = prediction["keypoints"]
        keypoints = keypoints.flatten(start_dim=1).tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "keypoints": keypoint,
                    "score": scores[k],
                }
                for k, keypoint in enumerate(keypoints)
            ]
        )
    return coco_results


def convert_to_xywh(boxes: Any) -> Any:
    """XYWH conversion."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids: Any, eval_imgs: Any) -> Any:
    """Merge the masking images."""
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)
    merged_eval_imgs =np.array(merged_eval_imgs)
    merged_img_ids = np.array(merged_img_ids)  # type: ignore
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]  # type: ignore

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval: Any, img_ids: Any, eval_imgs: Any) -> None:
    """Combine the COCO evaluations."""
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions


def createIndex(self: Any) -> None:
    """Create index."""
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if "annotations" in self.dataset:
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

    if "images" in self.dataset:
        for img in self.dataset["images"]:
            imgs[img["id"]] = img

    if "categories" in self.dataset:
        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

    if "annotations" in self.dataset and "categories" in self.dataset:
        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self: Any, resFile: str) -> Any:  # noqa C901
    """
    Load result file and return a result api object.

    :param self: coco object with ground truth annotations
    :param resFile: file name of result file
    :return: Result api object
    """
    res = COCO()
    res.dataset["images"] = list(self.dataset["images"])

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:  # type: ignore
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(self.getImgIds())
    ), "Results do not correspond to current coco set"
    if "caption" in anns[0]:
        imgIds = {img["id"] for img in res.dataset["images"]} & {ann["image_id"] for ann in anns}
        res.dataset["images"] = [img for img in res.dataset["images"] if img["id"] in imgIds]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann["area"] = maskUtils.area(ann["segmentation"])
            if "bbox" not in ann:
                ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann["area"] = (x2 - x1) * (y2 - y1)
            ann["id"] = id + 1
            ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset["annotations"] = anns
    createIndex(res)
    return res


def evaluate(self: Any) -> Any:
    """Run per image evaluation on given images and store results (a list of dict) in self.evalImgs."""
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(f"useSegm (deprecated) is not None. Running {p.iouType} evaluation")
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))  # type: ignore
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
