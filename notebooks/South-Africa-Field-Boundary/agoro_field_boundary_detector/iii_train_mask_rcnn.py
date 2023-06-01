"""Create, train and evaluate the Field Boundary Detector model."""
import os
from pathlib import Path
from typing import Optional

from agoro_field_boundary_detector.field_detection import Dataset, FieldBoundaryDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train(
    model: FieldBoundaryDetector,
    path: Path,
    n_epoch: int = 20,
    batch_size: int = 1,
    n_workers: int = 0,
    val_frac: float = 0.1,
    early_stop: bool = True,
    patience: int = 3,
) -> None:
    """
    Train the model.

    :param model: Mask R-CNN model to train
    :param path: Path where data is stored used for training
    :param n_epoch: Number of training epochs
    :param batch_size: Batch-size used during training
    :param n_workers: Number of workers used to load in the training data every batch
    :param val_frac: Fraction of the training data dedicated for validation
    :param early_stop: Whether or not to use early stopping
    :param patience: Early stopping patience
    """
    dataset = Dataset(path=path)
    model.train(
        dataset=dataset,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_workers=n_workers,
        val_frac=val_frac,
        early_stop=early_stop,
        patience=patience,
    )


def evaluate(
    model: FieldBoundaryDetector,
    path: Path,
    batch_size: int = 1,
    n_workers: int = 0,
    n_show: int = 0,
    write_path: Optional[Path] = None,
) -> None:
    """
    Evaluate the model.

    :param model: Mask R-CNN model to train
    :param path: Path where data is stored used for training
    :param batch_size: Batch-size used during training
    :param n_workers: Number of workers used to load in the training data every batch
    :param n_show: Number of test-images to show the results for (eye-ball evaluation)
    :param write_path: Path to which the eye-ball evaluated test results are written to
    """
    if write_path is None:
        write_path = Path.cwd()
    dataset = Dataset(path=path)
    model.test(
        dataset=dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        n_show=n_show,
        write_path=write_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", default=Path(__file__).parent / "../../models/mask_rcnn", type=str
    )
    parser.add_argument("--train", default=0, type=int)
    parser.add_argument(
        "--train-path", default=Path(__file__).parent / "../../data/augmented", type=str
    )
    parser.add_argument("--test", default=1, type=int)
    parser.add_argument("--test-path", default=Path(__file__).parent / "../../data/test", type=str)
    args = parser.parse_args()

    # Load in the model
    field_detector = FieldBoundaryDetector(model_path=args.model_path)

    # Train, if requested
    if args.train:
        train(
            model=field_detector,
            path=args.train_path,
        )
    # Test, if requested
    if args.test:
        evaluate(
            model=field_detector,
            path=args.test_path,
            n_show=10,
            write_path=Path(__file__).parent / "../../data/test_results",
        )
