"""
ESC-50 spectrogram dataset loader.

Provides a PyTorch Dataset for loading pre-generated spectrogram images
and a helper function to create train/test DataLoaders.
Includes support for Data Augmentations (SpecAugment, Gaussian Noise).
"""

import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from src import config

logger = logging.getLogger(__name__)


class ESC50SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading ESC-50 spectrogram images.

    Reads pre-generated spectrogram PNG images organized by category,
    filtered by target classes and fold numbers.

    Args:
        csv_path: Path to the ESC-50 metadata CSV file.
        img_dir: Root directory containing spectrogram images.
        target_classes: List of ESC-50 category names to include.
        folds: List of fold numbers to include (1-5).
        augment: If True, applies random TimeMasking, FrequencyMasking, and noise.
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        target_classes: list,
        folds: list,
        augment: bool = False,
    ):
        self.img_dir = img_dir
        self.augment = augment

        # Torchaudio transforms for SpecAugment
        if self.augment:
            import torchaudio.transforms as T
            self.time_masking = T.TimeMasking(time_mask_param=10)
            self.freq_masking = T.FrequencyMasking(freq_mask_param=10)

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find CSV at {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate classes
        available_classes = set(df["category"].unique())
        target_classes = list(target_classes)

        invalid = set(target_classes) - available_classes
        if invalid:
            raise ValueError(
                f"Invalid class(es): {invalid}\nAvailable: {sorted(available_classes)}"
            )

        # Filter by category and folds
        df = df[df["category"].isin(target_classes)]
        df = df[df["fold"].isin(folds)].reset_index(drop=True)

        self.df = df
        logger.info(
            "Loaded %d spectrograms for classes %s (Folds: %s, Augment: %s)",
            len(self.df), target_classes, folds, self.augment
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        filename = row["filename"]
        category = row["category"]

        # Construct path: img_dir / category / filename.png
        img_name = filename.replace(".wav", ".png")
        file_path = os.path.join(self.img_dir, category, img_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spectrogram not found: {file_path}")

        # Load image (C, H, W) - uint8
        img = read_image(file_path)

        # Convert to float [0, 1]
        img = img.float() / 255.0

        # Ensure single channel (if saved as RGB by mistake, take first)
        if img.shape[0] > 1:
            img = img[0:1, :, :]

        # Apply Augmentations (SpecAugment + Noise)
        if self.augment:
            noise = torch.randn_like(img) * 0.05
            img = img + noise
            img = self.freq_masking(img)
            img = self.time_masking(img)
            img = torch.clamp(img, 0.0, 1.0)

        return img


def get_dataloaders(
    class_names: list,
    batch_size: int = None,
    root_path: str = None,
) -> tuple:
    """
    Create train and test DataLoaders for the given ESC-50 classes.

    Args:
        class_names: List of ESC-50 category names.
        batch_size: Batch size for DataLoaders. Defaults to config.BATCH_SIZE.
        root_path: Root path for data. Defaults to config.PROJECT_ROOT.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if root_path is None:
        root_path = config.PROJECT_ROOT

    csv_path = os.path.join(root_path, "meta", "esc50.csv")
    spectrogram_path = os.path.join(root_path, "spectrograms")

    if not os.path.exists(spectrogram_path):
        raise FileNotFoundError(
            f"Spectrograms directory not found at {spectrogram_path}. "
            "Run 'python scripts/generate_spectrograms.py' first."
        )

    train_ds = ESC50SpectrogramDataset(
        csv_path, spectrogram_path, class_names, folds=config.TRAIN_FOLDS, augment=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = ESC50SpectrogramDataset(
        csv_path, spectrogram_path, class_names, folds=config.TEST_FOLDS, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
