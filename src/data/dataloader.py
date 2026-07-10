"""
dataloader.py
-------------
Utilities for creating PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, NUM_WORKERS
from src.data.dataset import EuroSATDataset
from src.data.split import create_train_val_split
from src.data.transforms import get_train_transforms, get_val_transforms


def create_dataloaders():
    """
    Create train and validation DataLoaders.
    """

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        _
    ) = create_train_val_split()

    train_dataset = EuroSATDataset(
        train_paths,
        train_labels,
        transform=get_train_transforms(),
    )

    val_dataset = EuroSATDataset(
        val_paths,
        val_labels,
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader