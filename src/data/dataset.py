"""
dataset.py
----------
Utilities for exploring and loading the EuroSAT dataset.
"""

from pathlib import Path
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset

from src.config import EUROSAT_DIR


class EuroSATExplorer:
    """
    Utility class for exploring the EuroSAT dataset.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif"}

    def __init__(self, dataset_path=EUROSAT_DIR):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        self.classes = self._get_classes()

    def _get_classes(self):
        """
        Return all class folders.
        """
        return sorted(
            folder.name
            for folder in self.dataset_path.iterdir()
            if folder.is_dir()
        )

    def count_images(self):
        """
        Count images in every class.
        """
        image_counts = Counter()

        for class_name in self.classes:

            class_path = self.dataset_path / class_name

            total = sum(
                1
                for file in class_path.iterdir()
                if file.suffix.lower() in self.VALID_EXTENSIONS
            )

            image_counts[class_name] = total

        return image_counts

    def total_images(self):
        """
        Return total number of images.
        """
        return sum(self.count_images().values())

    def summary(self):
        """
        Print dataset summary.
        """
        counts = self.count_images()

        print("=" * 60)
        print("EuroSAT Dataset Summary")
        print("=" * 60)

        print(f"Dataset Path  : {self.dataset_path}")
        print(f"Total Classes : {len(self.classes)}")
        print(f"Total Images  : {self.total_images()}")

        print("\nImages per Class\n")

        for class_name in self.classes:
            print(f"{class_name:<25} {counts[class_name]}")

        print("=" * 60)

    def verify_images(self):
        """
        Verify dataset integrity and image resolutions.
        """
        total_images = 0
        corrupted_images = 0
        image_sizes = Counter()

        for class_name in self.classes:

            class_path = self.dataset_path / class_name

            for image_path in class_path.iterdir():

                if image_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue

                total_images += 1

                try:
                    with Image.open(image_path) as img:
                        image_sizes[img.size] += 1

                except Exception:
                    corrupted_images += 1

        print("=" * 60)
        print("Dataset Integrity Report")
        print("=" * 60)
        print(f"Total Images      : {total_images}")
        print(f"Corrupted Images  : {corrupted_images}")
        print("\nImage Resolutions")

        for size, count in image_sizes.items():
            print(f"{size} : {count}")

        print("=" * 60)


class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for training and evaluation.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label