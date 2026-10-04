"""
Inference and anomaly detection module.

Provides functions to:
- Load a trained model
- Predict anomalies on single audio files
- Batch-evaluate normal vs. anomaly classes
- Visualize reconstruction results
"""

import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src import config
from src.models.autoencoder import AutoEncoder
from src.data.preprocessing import load_and_preprocess
from src.data.loader import get_dataloaders

logger = logging.getLogger(__name__)


def load_model(model_path: str = None) -> AutoEncoder:
    """
    Load a trained AutoEncoder from a checkpoint file.

    Args:
        model_path: Path to the .pth checkpoint file.
                    Defaults to config.BEST_MODEL_PATH.

    Returns:
        Loaded AutoEncoder model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Train the model first "
            "using 'python scripts/train.py'."
        )

    model = AutoEncoder().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    logger.info("Loaded model from %s", model_path)
    return model


def predict_anomaly(
    model: AutoEncoder,
    file_path: str,
    threshold: float = None,
) -> dict:
    """
    Predict whether an audio file contains an anomaly.

    Args:
        model: Trained AutoEncoder model.
        file_path: Path to a .wav audio file.
        threshold: MSE threshold for anomaly classification.
                   Defaults to config.ANOMALY_THRESHOLD.

    Returns:
        Dictionary with keys: 'file', 'loss', 'accuracy', 'is_anomaly'.
        Returns None if preprocessing fails.
    """
    if threshold is None:
        threshold = config.ANOMALY_THRESHOLD

    spectrogram = load_and_preprocess(file_path)
    if spectrogram is None:
        return None

    with torch.no_grad():
        reconstruction = model(spectrogram)
        loss = nn.functional.mse_loss(reconstruction, spectrogram)
        loss_val = float(loss.item())
        accuracy = (1 - (loss_val ** 0.5)) * 100

    from src.inference.diagnostics import IndustrialDiagnosticEngine
    engine = IndustrialDiagnosticEngine(sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS)
    report = engine.analyze(spectrogram, reconstruction, threshold=threshold)

    result = {
        "file": os.path.basename(file_path),
        "loss": loss_val,
        "accuracy": accuracy,
        "threshold": threshold,
        "is_anomaly": loss_val > threshold,
        "severity": report.severity,
        "status_summary": report.status_summary,
        "recommended_action": report.recommended_action,
        "dominant_band": report.dominant_fault_band,
        "band_analysis": report.band_analysis,
    }

    logger.info(
        "Prediction for %s — Loss: %.4f, Severity: %s, Result: %s",
        result["file"], loss_val, result["severity"],
        "ANOMALY" if result["is_anomaly"] else "NORMAL",
    )

    return result


def batch_evaluate(
    model: AutoEncoder,
    normal_classes: list = None,
    anomaly_classes: list = None,
) -> dict:
    """
    Evaluate the model on normal and anomaly class datasets.

    Args:
        model: Trained AutoEncoder model.
        normal_classes: List of normal class names.
                        Defaults to config.EVAL_NORMAL_CLASSES.
        anomaly_classes: List of anomaly class names.
                         Defaults to config.EVAL_ANOMALY_CLASSES.

    Returns:
        Dictionary with 'normal' and 'anomaly' stats.
    """
    if normal_classes is None:
        normal_classes = config.EVAL_NORMAL_CLASSES
    if anomaly_classes is None:
        anomaly_classes = config.EVAL_ANOMALY_CLASSES

    criterion = nn.MSELoss()
    results = {}

    for label, classes in [("normal", normal_classes), ("anomaly", anomaly_classes)]:
        logger.info("Evaluating on %s data %s...", label.upper(), classes)
        _, loader = get_dataloaders(classes, batch_size=8)

        losses = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(config.DEVICE)
                out = model(batch)
                loss = criterion(out, batch)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        avg_acc = (1 - (avg_loss ** 0.5)) * 100

        results[label] = {"avg_loss": avg_loss, "avg_accuracy": avg_acc}
        logger.info(
            "%s — Avg Loss: %.4f, Avg Accuracy: %.2f%%",
            label.upper(), avg_loss, avg_acc,
        )

    diff = results["normal"]["avg_accuracy"] - results["anomaly"]["avg_accuracy"]
    logger.info("Accuracy difference: %.2f%% (higher = better detection)", diff)

    return results


def visualize_comparison(
    model: AutoEncoder,
    normal_classes: list = None,
    anomaly_classes: list = None,
    save_path: str = None,
) -> None:
    """
    Generate a side-by-side visualization of normal vs. anomaly reconstructions.

    Args:
        model: Trained AutoEncoder model.
        normal_classes: Normal class names. Defaults to config value.
        anomaly_classes: Anomaly class names. Defaults to config value.
        save_path: Path to save the output image. Defaults to outputs dir.
    """
    if normal_classes is None:
        normal_classes = config.EVAL_NORMAL_CLASSES
    if anomaly_classes is None:
        anomaly_classes = config.EVAL_ANOMALY_CLASSES
    if save_path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.OUTPUT_DIR, "anomaly_detection_results.png")

    _, normal_loader = get_dataloaders(normal_classes, batch_size=8)
    _, anomaly_loader = get_dataloaders(anomaly_classes, batch_size=8)

    normal_batch = next(iter(normal_loader)).to(config.DEVICE)
    anomaly_batch = next(iter(anomaly_loader)).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        normal_recon = model(normal_batch)
        anomaly_recon = model(anomaly_batch)

    # Move to CPU for plotting
    nb = normal_batch.cpu()
    nr = normal_recon.cpu()
    ab = anomaly_batch.cpu()
    ar = anomaly_recon.cpu()

    num_samples = 3
    fig, axes = plt.subplots(4, num_samples, figsize=(10, 12))

    for i in range(num_samples):
        # Normal Original
        axes[0, i].imshow(nb[i][0], cmap="inferno", origin="lower")
        axes[0, i].set_title("Normal (Original)")
        axes[0, i].axis("off")

        # Normal Reconstructed
        loss_n = nn.functional.mse_loss(nr[i], nb[i]).item()
        acc_n = (1 - loss_n ** 0.5) * 100
        axes[1, i].imshow(nr[i][0], cmap="inferno", origin="lower")
        axes[1, i].set_title(f"Recon (Acc: {acc_n:.1f}%)")
        axes[1, i].axis("off")

        # Anomaly Original
        axes[2, i].imshow(ab[i][0], cmap="inferno", origin="lower")
        axes[2, i].set_title("Anomaly (Original)")
        axes[2, i].axis("off")

        # Anomaly Reconstructed
        loss_a = nn.functional.mse_loss(ar[i], ab[i]).item()
        acc_a = (1 - loss_a ** 0.5) * 100
        axes[3, i].imshow(ar[i][0], cmap="inferno", origin="lower")
        axes[3, i].set_title(f"Recon (Acc: {acc_a:.1f}%)")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info("Saved comparison visualization to %s", save_path)
