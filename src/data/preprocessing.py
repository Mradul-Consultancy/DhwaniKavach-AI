"""
Shared audio preprocessing utilities.

This module is the SINGLE source of truth for converting audio waveforms
to mel-spectrograms. Both batch spectrogram generation and single-file
inference use these functions, eliminating code duplication.
"""

import logging
import torch
import torchaudio

from src import config

logger = logging.getLogger(__name__)


def load_audio(file_path: str) -> torch.Tensor:
    """
    Load an audio file and preprocess the waveform.

    Resamples to the target sample rate and pads/crops to the fixed length
    defined in config.

    Args:
        file_path: Path to a .wav audio file.

    Returns:
        Preprocessed waveform tensor of shape (1, FIXED_LENGTH).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If torchaudio cannot load the file.
    """
    import soundfile as sf
    waveform_np, sr = sf.read(file_path)
    # soundfile returns (time, channels), so we transpose to (channels, time)
    if waveform_np.ndim == 1:
        waveform_np = waveform_np.reshape(1, -1)
    else:
        waveform_np = waveform_np.transpose()
    waveform = torch.from_numpy(waveform_np).float()

    # Resample to target sample rate
    if sr != config.SAMPLE_RATE:
        logger.debug("Resampling from %d Hz to %d Hz", sr, config.SAMPLE_RATE)
        waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

    # Pad or crop to fixed length
    if waveform.shape[1] < config.FIXED_LENGTH:
        pad_len = config.FIXED_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :config.FIXED_LENGTH]

    # Take first channel if stereo
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]

    return waveform


def waveform_to_spectrogram(
    waveform: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert a waveform to a normalized log-mel spectrogram.

    Args:
        waveform: Audio waveform tensor of shape (1, num_samples).
        device: Device to run the transform on. Defaults to config.DEVICE.

    Returns:
        Normalized spectrogram tensor of shape (1, N_MELS, time),
        with values in [0, 1].
    """
    if device is None:
        device = config.DEVICE

    waveform = waveform.to(device)

    # Mel spectrogram transform
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
    ).to(device)

    spectrogram = spec_transform(waveform)  # Shape: (1, N_MELS, time)

    # Log transform
    spectrogram = torch.log(spectrogram + config.LOG_OFFSET)

    # Normalize to [0, 1] per sample
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if spec_max - spec_min > config.LOG_OFFSET:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spectrogram = spectrogram - spec_min

    return spectrogram


def load_and_preprocess(file_path: str, device: torch.device = None) -> torch.Tensor:
    """
    Full pipeline: load audio file → preprocess → mel spectrogram.

    This is a convenience function combining load_audio() and
    waveform_to_spectrogram().

    Args:
        file_path: Path to a .wav audio file.
        device: Device for spectrogram computation.

    Returns:
        Spectrogram tensor of shape (1, 1, N_MELS, time) ready for model input.
        Returns None if processing fails.
    """
    try:
        waveform = load_audio(file_path)
        spectrogram = waveform_to_spectrogram(waveform, device=device)

        # Ensure single channel
        if spectrogram.shape[0] > 1:
            spectrogram = spectrogram[0:1, :, :]

        # Add batch dimension → (1, 1, N_MELS, time)
        spectrogram = spectrogram.unsqueeze(0)

        return spectrogram

    except Exception as e:
        logger.error("Error processing %s: %s", file_path, e)
        return None
