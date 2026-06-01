#!/usr/bin/env python
"""
Entry point: Run anomaly detection inference.

Usage:
    # Single file prediction
    python scripts/predict.py --file path/to/audio.wav

    # Batch evaluation with visualization
    python scripts/predict.py --batch

    # Both
    python scripts/predict.py --file audio.wav --batch
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predict import (
    load_model,
    predict_anomaly,
    batch_evaluate,
    visualize_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Machine Sound Anomaly Detection")
    parser.add_argument(
        "--file", type=str,
        help="Path to a single .wav file to test",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run batch evaluation on normal vs. anomaly classes",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="MSE threshold for anomaly detection (default: from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (default: checkpoints/best_autoencoder.pth)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not args.file and not args.batch:
        parser.print_help()
        print("\nError: Specify --file and/or --batch.")
        sys.exit(1)

    try:
        # Load model
        model = load_model(args.model)

        # Single file prediction
        if args.file:
            result = predict_anomaly(model, args.file, threshold=args.threshold)
            if result:
                print(f"\n{'='*50}")
                print(f"  File:     {result['file']}")
                print(f"  Loss:     {result['loss']:.4f}")
                print(f"  Accuracy: {result['accuracy']:.2f}%")
                print(f"  Result:   {'🚨 ANOMALY' if result['is_anomaly'] else '✅ NORMAL'}")
                print(f"{'='*50}")

        # Batch evaluation
        if args.batch:
            results = batch_evaluate(model)

            print(f"\n{'='*50}")
            print(f"  Normal  — Loss: {results['normal']['avg_loss']:.4f}, "
                  f"Accuracy: {results['normal']['avg_accuracy']:.2f}%")
            print(f"  Anomaly — Loss: {results['anomaly']['avg_loss']:.4f}, "
                  f"Accuracy: {results['anomaly']['avg_accuracy']:.2f}%")
            diff = results['normal']['avg_accuracy'] - results['anomaly']['avg_accuracy']
            print(f"  Gap:      {diff:.2f}%")
            print(f"{'='*50}")

            # Generate visualization
            visualize_comparison(model)

    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
