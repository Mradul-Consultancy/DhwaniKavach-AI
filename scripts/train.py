#!/usr/bin/env python
"""
Entry point: Train the AutoEncoder model.

Usage:
    python scripts/train.py
    python scripts/train.py --classes engine chainsaw --epochs 20 --lr 0.001
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.models.autoencoder import AutoEncoder
from src.data.loader import get_dataloaders
from src.training.trainer import train, evaluate


def main():
    parser = argparse.ArgumentParser(description="Train the AutoEncoder model")
    parser.add_argument(
        "--classes", nargs="+", default=config.DEFAULT_NORMAL_CLASSES,
        help=f"Normal sound classes to train on (default: {config.DEFAULT_NORMAL_CLASSES})",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help=f"Number of training epochs (default: {config.NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Learning rate (default: {config.LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Training AutoEncoder on classes: %s", args.classes)
    logger.info("Device: %s", config.DEVICE)

    try:
        # Load data
        train_loader, test_loader = get_dataloaders(
            args.classes, batch_size=args.batch_size,
        )

        # Initialize model
        model = AutoEncoder().to(config.DEVICE)
        logger.info("Model initialized.")

        # Train
        train(
            model, train_loader, test_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )

        # Final evaluation
        logger.info("Training complete. Final evaluation:")
        evaluate(model, test_loader)

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
