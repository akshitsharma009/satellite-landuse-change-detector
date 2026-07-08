"""
split.py
--------
Utilities for creating train and validation splits.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import EUROSAT_DIR


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif"}


def create_train_val_split(
    dataset_dir=EUROSAT_DIR,
    test_size=0.2,
    random_state=42,
):
    """
    Create a stratified train-validation split.
    """

    dataset_dir = Path(dataset_dir)

    classes = sorted(
        folder.name
        for folder in dataset_dir.iterdir()
        if folder.is_dir()
    )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    image_paths = []
    labels = []

    for class_name in classes:

        class_path = dataset_dir / class_name

        for image_path in class_path.iterdir():

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image_paths.append(str(image_path))
            labels.append(class_to_idx[class_name])

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        class_to_idx,
    )