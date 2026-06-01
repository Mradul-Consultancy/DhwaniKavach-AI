#!/usr/bin/env python
"""
Entry point: Generate mel-spectrogram images from audio files.

Usage:
    python scripts/generate_spectrograms.py
    python scripts/generate_spectrograms.py --csv meta/esc50.csv --audio audio/ --output spectrograms/
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generate_spectrograms import generate_spectrograms


def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from audio files")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to ESC-50 metadata CSV (default: meta/esc50.csv)",
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Directory containing WAV files (default: audio/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for spectrograms (default: spectrograms/)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting spectrogram generation...")

    generate_spectrograms(
        csv_path=args.csv,
        audio_dir=args.audio,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
