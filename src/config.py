"""
Central configuration for Acoustic Machine Fault Detection and Monitoring System.

All hyperparameters, paths, and constants are defined here.
No other file should contain hardcoded values — import from this module.
"""

import os
import torch

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths (relative to PROJECT_ROOT)
META_CSV_PATH = os.path.join(PROJECT_ROOT, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio")
SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "spectrograms")

# Model checkpoint
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_autoencoder.pth")

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ==============================================================================
# Audio Processing
# ==============================================================================
SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 5
FIXED_LENGTH = SAMPLE_RATE * AUDIO_DURATION_SEC  # 80000 samples
N_MELS = 64
LOG_OFFSET = 1e-9  # Small constant to avoid log(0)

# ==============================================================================
# Training
# ==============================================================================
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
TRAIN_FOLDS = [1, 2, 3, 4]
TEST_FOLDS = [5]

# ==============================================================================
# Inference
# ==============================================================================
ANOMALY_THRESHOLD = 0.035  # MSE threshold: above = anomaly, below = normal

# ==============================================================================
# Classes (ESC-50 Dataset)
# ==============================================================================
DEFAULT_NORMAL_CLASSES = ["engine"]
EVAL_NORMAL_CLASSES = [
    "engine", "chainsaw", "keyboard_typing",
    "mouse_click", "vacuum_cleaner", "washing_machine",
]
EVAL_ANOMALY_CLASSES = ["footsteps"]

# ==============================================================================
# Device
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
