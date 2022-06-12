"""Methods related to the Mask-RCNN model."""
from datetime import datetime
from pathlib import Path
from random import getrandbits
from shutil import rmtree
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F_vis

from src.agoro_field_boundary_detector.field_detection.dataset import Dataset
from src.agoro_field_boundary_detector.field_detection.utils import mask_to_polygons
from src.agoro_field_boundary_detector.field_detection.mask_rcnn.engine import evaluate, train_one_epoch

class FieldBoundaryDetector:
    """Field Boundary Detector model implemented in PyTorch."""

    def __init__(
        self,
        model_path: Path,
        n_classes: int = 2,
        n_hidden: int = 512,
        thr: float = 0.5,
        pretrained_resnet: bool = True,
        reset: bool = False,
    ) -> None:
        """
        Initialise the Field Boundary Detector model, either by loading in previous version or creating a new one.
        :param model_path: Path under which the model is saved or will be saved
        :param n_classes: Number of the classes (outputs) the model should have
        :param n_hidden: Number of hidden layers used in the MaskRCNN predictor
        :param thr: Certainty threshold when defining masks (pixel-level)
        :param pretrained_resnet: Use a pretrained ResNet backbone when creating a new Mask RCNN model
        :param reset: Create a new model instead of loading a previously existing one
        """
        self.path = model_path
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.thr = thr
        self.model: Optional[torchvision.models.detection.maskrcnn_resnet50_fpn] = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore
        if reset or not self.load():
            self.create_instance_segmentation_model(pretrained_resnet)

    def __call__(
        self,
        im: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Predict the polygon of the mask covering the image's center pixel.
        :param im: Image for which the model will make its prediction
        :return: Field polygon as a list of pixel-coordinates, empty list if no field recognised
        """
        # Make masking predictions
        mask = self.get_mask(im=im)

        # Extract polygon of center pixel, if exists
        center = im.shape[0] // 2, im.shape[1] // 2
        m_value = mask[center[0], center[1]]
        if m_value == 0:
            return []
        else:
            mask[mask != m_value] = 0
            mask[mask == m_value] = 1
            return mask_to_polygons(mask)[0]

    def get_all_polygons(
        self,
        im: np.ndarray,
    ) -> List[List[Tuple[int, int]]]:
        """Extract all the detected field-polygons (in pixel coordinates) from the given image."""
        # Make masking predictions
        mask = self.get_mask(im=im)
        return mask_to_polygons(mask)

    def __str__(self) -> str:
        """Textual model representation."""
        return f"FieldBoundaryDetector(n_hid={self.n_hidden}, n_out={self.n_classes})"

    def __repr__(self) -> str:
        """Textual model representation."""
        return str(self)

    def create_instance_segmentation_model(self, pretrained: bool = True) -> None:
        """Create an instance segmentation model."""
        # Load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

        # Replace the pre-trained head (box and mask predictor)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.model.roi_heads.box_predictor.cls_score.in_features,
            self.n_classes,
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels,
            self.n_hidden,
            self.n_classes,
        )

        # Move model to correct device
        self.model.to(self.device)
        print("Created a new (untrained) FieldBoundaryDetector model")

    def train(
        self,
        dataset: Dataset,
        n_epoch: int = 20,
        batch_size: int = 1,
        n_workers: int = 1,
        val_frac: float =0.0005, #0.1
        early_stop: bool = True,
        patience: int = 3,
    ) -> None:
        """
        Train the model.
        :param dataset: Dataset to train ane evaluate the model on
        :param n_epoch: Number of epochs to train over
        :param batch_size: Training and validation batch size
        :param n_workers: Number of dataloader workers (0 for sequential training)
        :param val_frac: Fraction of dataset to use for validation
        :param early_stop: Whether or not to stop training when validation F1 score starts to decrease
        :param patience: Early stopping patience, which indicates when training halts for decreasing F1
        """
        # Split the dataset in train and test set
        frac_idx = round(val_frac * len(dataset))
        indices = torch.randperm(len(dataset)).tolist()  # type: ignore
        dataset_val = torch.utils.data.Subset(dataset, indices[:frac_idx])
        dataset_train = torch.utils.data.Subset(dataset, indices[frac_idx:])

        # Define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: tuple(zip(*x)),
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]  # type: ignore
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005,
        )

        # Learning rate scheduler which decreases the learning rate by 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1,
        )

        # Create temporal folder to store models in
        temp_folder = Path.cwd() / f"{getrandbits(128)}"
        temp_folder.mkdir(exist_ok=False, parents=True)

        # Train the model
        best_f1, best_epoch, last_improvement = 0.0, 0, 0
        for epoch in range(n_epoch):
            # Train for one epoch and update the learning rate afterwards
            train_one_epoch(
                self.model,
                optimizer,
                data_loader,
                self.device,
                epoch,
                print_freq=len(data_loader) // 0.1 #10,
            )
            lr_scheduler.step()

            # Evaluate on the validation dataset
            f1 = evaluate(
                self.model,
                data_loader_val,
                device=self.device,
            )
            print(f" => F1 epoch {epoch}: {f1}")

            # Check if improvement made, act accordingly
            if best_f1 > f1:
                last_improvement += 1
                print(f" ! No improvement for {last_improvement} epochs")
            else:
                last_improvement = 0
                best_f1 = f1
                best_epoch = epoch
                torch.save(
                    self.model,
                    temp_folder / f"{epoch}",
                )

            # Stop if validation F1 starts to decrease
            if early_stop and last_improvement >= patience:
                break

        # Revert back to best-performing model and delete temporal files
        self.model = torch.load(temp_folder / f"{best_epoch}")  # type: ignore
        rmtree(temp_folder)
        self.save()

    def test(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        n_workers: int = 0,
        n_show: int = 0,
        write_path: Optional[Path] = None,
    ) -> None:
        """
        Evaluate the model.
        :param dataset: Complete dataset (same split used as seen during training)
        :param batch_size: Testing batch size
        :param n_workers: Number of dataloader workers (0 for sequential training)
        :param n_show: Number of test-images for which to show the masked fields
        :param write_path: Path to write evaluation-images to (only if n_show > 0)
        """
        if n_show:
            assert write_path is not None
            write_path.mkdir(parents=True, exist_ok=True)

        # Ensure evaluation mode
        self.model.eval()  # type: ignore

        # Extract evaluation-portion of model
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        # Evaluate the model, print out the resulting F1-score
        f1 = evaluate(
            self.model,
            data_loader,
            device=self.device,
        )
        print(f" ==> F1: {f1}")

        # Eye-ball evaluation
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        for idx in range(min(len(dataset), n_show)):
            img, _ = dataset[idx]
            polygons = self.get_all_polygons(img)
            img_tc = img.mul(255).permute(1, 2, 0).byte().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(img_tc, interpolation="nearest")
            for polygon in polygons:
                x, y = zip(*polygon)
                plt.plot(x, y)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(write_path / f"{time}_eval_{idx}.png")  # type: ignore
            plt.close()

    def get_mask(
        self,
        im: Union[np.ndarray, torch.Tensor],
        size_thr: float = 0.001,
        overlap_thr: float = 0.8,
    ) -> np.ndarray:
        """
        Predict a layered mask for the given image.
        :param im: Image to make the prediction on
        :param size_thr: Proportion of the image a mask must occupy in order to be valid
        :param overlap_thr: Proportion of the image that does not overlap in order to be valid
        :return: Image masks
        """
        # Assure model in evaluation mode
        if self.model.training:  # type: ignore
            self.model.eval()  # type: ignore

        # Transform image to PyTorch tensor and put on right device
        im_t: torch.Tensor = im if type(im) == torch.Tensor else F_vis.to_tensor(im)  # type: ignore
        im_t = im_t.to(self.device)
        with torch.no_grad():
            # Predict all masks
            prediction = self.model([im_t])[0]  # type: ignore

            # Merge the masks together
            i = 1
            result = np.zeros(im_t.shape[1:], dtype=np.int8)
            for mask in prediction["masks"]:
                mask = mask[0].cpu().numpy()
                mask[mask >= self.thr] = 1  # Ensure binary mask
                mask[mask < self.thr] = 0
                mask = mask.astype(np.int8)

                # Ignore masks if they occupy less than 0.1% of the image
                if mask.sum() / len(mask.flatten()) < size_thr:
                    continue

                # Ignore masks that mainly (90% or more) overlap with previously detected masks
                previous = np.clip(result, a_min=0, a_max=1)
                if (mask * previous).sum() > (1 - overlap_thr) * mask.sum():
                    continue

                # Add mask to the result
                result += i * mask
                result = np.clip(result, a_min=0, a_max=i)  # type: ignore
                i += 1
        return result

    def load(self) -> bool:
        """Load a previously saved model."""
        if self.path.is_file():
            self.model = torch.load(self.path, map_location=self.device)  # type: ignore
            return True
        return False

    def save(self) -> None:
        """Save the model."""
        self.model.to(torch.device("cpu"))  # type: ignore
        torch.save(
            self.model,
            self.path,
        )