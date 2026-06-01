"""
Batch spectrogram generation from audio files.

Converts all WAV files referenced in the ESC-50 metadata CSV into
mel-spectrogram PNG images, organized by category.

Uses the shared preprocessing module to ensure consistency with
the inference pipeline.
"""

import logging
import os

import pandas as pd
from torchvision.utils import save_image
from tqdm import tqdm

from src import config
from src.data.preprocessing import load_audio, waveform_to_spectrogram

logger = logging.getLogger(__name__)


def generate_spectrograms(
    csv_path: str = None,
    audio_dir: str = None,
    output_dir: str = None,
) -> None:
    """
    Generate mel-spectrogram images for all audio files in the dataset.

    Args:
        csv_path: Path to ESC-50 metadata CSV. Defaults to config path.
        audio_dir: Directory containing WAV files. Defaults to config path.
        output_dir: Output directory for spectrograms. Defaults to config path.
    """
    csv_path = csv_path or config.META_CSV_PATH
    audio_dir = audio_dir or config.AUDIO_DIR
    output_dir = output_dir or config.SPECTROGRAM_DIR

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Using device: %s", config.DEVICE)

    # Load metadata
    df = pd.read_csv(csv_path)
    logger.info("Found %d audio files in metadata.", len(df))

    # Process each file
    success_count = 0
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Spectrograms"):
        filename = row["filename"]
        category = row["category"]

        # Create category folder
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Check source file
        file_path = os.path.join(audio_dir, filename)
        if not os.path.exists(file_path):
            logger.warning("File not found: %s", file_path)
            error_count += 1
            continue

        try:
            # Use shared preprocessing pipeline
            waveform = load_audio(file_path)
            spectrogram = waveform_to_spectrogram(waveform, device=config.DEVICE)

            # Save as image
            out_filename = filename.replace(".wav", ".png")
            out_path = os.path.join(cat_dir, out_filename)
            save_image(spectrogram, out_path)
            success_count += 1

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)
            error_count += 1

    logger.info(
        "Done! %d spectrograms generated, %d errors. Saved to %s",
        success_count, error_count, output_dir,
    )
