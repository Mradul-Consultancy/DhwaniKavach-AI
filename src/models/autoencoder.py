"""
Convolutional AutoEncoder for machine sound anomaly detection.

The encoder compresses a mel-spectrogram into a latent representation,
and the decoder reconstructs it. High reconstruction error indicates
an anomalous (unfamiliar) sound.

Architecture:
    Encoder: Conv2d(1→16) → ReLU → MaxPool
             Conv2d(16→32) → ReLU → MaxPool
             Conv2d(32→64) → ReLU → MaxPool  → Latent [64, 8, 39]

    Decoder: ConvTranspose2d(64→32) → ReLU
             ConvTranspose2d(32→16) → ReLU
             ConvTranspose2d(16→1)  → Sigmoid
"""

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """Convolutional AutoEncoder for spectrogram reconstruction."""

    def __init__(self, base_filters: int = 16, latent_dim: int = 64):
        """
        Initialize the AutoEncoder.
        
        Args:
            base_filters: Number of filters in the first convolutional layer.
            latent_dim: Number of filters in the bottleneck layer.
        """
        super(AutoEncoder, self).__init__()
        
        f1 = base_filters
        f2 = base_filters * 2
        f3 = latent_dim

        # --- ENCODER (Compressing) ---
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> f1
            nn.Conv2d(1, f1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 2: f1 -> f2
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 3: f2 -> f3 (Bottleneck)
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # LATENT REPRESENTATION
        )

        # --- DECODER (Reconstructing) ---
        self.decoder = nn.Sequential(
            # Layer 3 Reverse: f3 -> f2
            nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2),
            nn.ReLU(),

            # Layer 2 Reverse: f2 -> f1
            nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2),
            nn.ReLU(),

            # Layer 1 Reverse: f1 -> 1
            nn.ConvTranspose2d(f1, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Output must be 0-1 range to match input
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input spectrogram tensor of shape (B, 1, H, W).

        Returns:
            Reconstructed spectrogram of the same shape as input.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Handle potential size mismatch from pooling/unpooling
        if decoded.shape != x.shape:
            decoded = nn.functional.interpolate(decoded, size=x.shape[2:])

        return decoded
