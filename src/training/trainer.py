"""
Training and evaluation routines for the AutoEncoder.

Handles the training loop, model evaluation, and best-model checkpointing.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src import config

logger = logging.getLogger(__name__)


def evaluate(model: nn.Module, test_loader, device: torch.device = None) -> tuple:
    """
    Evaluate the model on a test set.

    Args:
        model: Trained AutoEncoder model.
        test_loader: DataLoader for the test set.
        device: Computation device. Defaults to config.DEVICE.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    if device is None:
        device = config.DEVICE

    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for spectrogram in test_loader:
            spectrogram = spectrogram.to(device)
            outputs = model(spectrogram)
            loss = criterion(outputs, spectrogram)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    accuracy = (1 - np.sqrt(avg_loss)) * 100

    logger.info("Test Loss: %.4f, Test Accuracy: %.2f%%", avg_loss, accuracy)
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader,
    test_loader,
    num_epochs: int = None,
    learning_rate: float = None,
    device: torch.device = None,
    checkpoint_path: str = None,
) -> nn.Module:
    """
    Train the AutoEncoder model.

    Args:
        model: AutoEncoder model to train.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        num_epochs: Number of training epochs. Defaults to config.NUM_EPOCHS.
        learning_rate: Learning rate. Defaults to config.LEARNING_RATE.
        device: Computation device. Defaults to config.DEVICE.
        checkpoint_path: Path to save the best model. Defaults to config.BEST_MODEL_PATH.

    Returns:
        The trained model.
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if device is None:
        device = config.DEVICE
    if checkpoint_path is None:
        checkpoint_path = config.BEST_MODEL_PATH

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = -float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, spectrogram in enumerate(train_loader):
            spectrogram = spectrogram.to(device)

            optimizer.zero_grad()
            outputs = model(spectrogram)
            loss = criterion(outputs, spectrogram)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (1 - np.sqrt(epoch_loss)) * 100

        logger.info(
            "Epoch [%d/%d], Train Loss: %.4f, Train Accuracy: %.2f%%",
            epoch + 1, num_epochs, epoch_loss, epoch_acc,
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, device=device)

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                "New best model saved with accuracy: %.2f%% → %s",
                best_accuracy, checkpoint_path,
            )

    return model
