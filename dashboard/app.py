"""
Acoustic Machine Fault Detection and Monitoring System — Web Dashboard Backend (v2).

Professional-grade Flask application with:
- Single file and batch audio analysis
- Microphone recording support
- Model performance evaluation (ROC, Confusion Matrix, Precision/Recall)
- Interactive threshold analysis
- Training history visualization
"""

import base64
import io
import json
import logging
import os
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from flask import Flask, jsonify, render_template, request

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.models.autoencoder import AutoEncoder
from src.data.preprocessing import load_audio, waveform_to_spectrogram

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# ── Matplotlib styling for dark theme ────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#2a3040",
    "axes.labelcolor": "#8892a0",
    "text.color": "#e0e6ed",
    "xtick.color": "#8892a0",
    "ytick.color": "#8892a0",
    "grid.color": "#1a2030",
    "grid.alpha": 0.5,
}

# ── Global State ─────────────────────────────────────────────────────────────
model = None
model_loaded = False


def load_model():
    """Load the trained AutoEncoder model."""
    global model, model_loaded
    try:
        model_path = config.BEST_MODEL_PATH
        if not os.path.exists(model_path):
            logger.warning("Model not found at %s", model_path)
            return False
        model = AutoEncoder().to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        model_loaded = True
        logger.info("Model loaded from %s", model_path)
        return True
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        model_loaded = False
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#   VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="none", edgecolor="none", transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _apply_dark_style():
    """Apply dark theme to matplotlib."""
    for k, v in PLOT_STYLE.items():
        plt.rcParams[k] = v


def generate_waveform_image(waveform):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(10, 2.5))
    wave_np = waveform.squeeze().cpu().numpy()
    t = np.linspace(0, len(wave_np) / config.SAMPLE_RATE, len(wave_np))
    ax.fill_between(t, wave_np, alpha=0.25, color="#00d4aa")
    ax.plot(t, wave_np, color="#00d4aa", linewidth=0.4, alpha=0.8)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)
    return fig_to_base64(fig)


def generate_spectrogram_image(spec, title=""):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    spec_np = spec.squeeze().cpu().numpy()
    im = ax.imshow(spec_np, cmap="inferno", origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#8892a0", labelsize=7)
    cbar.outline.set_edgecolor("#2a3040")
    if title:
        ax.set_title(title, color="#e0e6ed", fontsize=11, fontweight="600", pad=10)
    ax.set_xlabel("Time Frame", fontsize=9)
    ax.set_ylabel("Mel Band", fontsize=9)
    ax.tick_params(labelsize=7)
    return fig_to_base64(fig)


def generate_error_heatmap(original, reconstructed):
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    error = (original - reconstructed).abs().squeeze().cpu().numpy()
    im = ax.imshow(error, cmap="hot", origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#8892a0", labelsize=7)
    cbar.outline.set_edgecolor("#2a3040")
    ax.set_title("Reconstruction Error Map", color="#e0e6ed", fontsize=11, fontweight="600", pad=10)
    ax.set_xlabel("Time Frame", fontsize=9)
    ax.set_ylabel("Mel Band", fontsize=9)
    ax.tick_params(labelsize=7)
    return fig_to_base64(fig)


def generate_roc_curve(normal_losses, anomaly_losses):
    """Generate ROC curve from normal and anomaly loss distributions."""
    _apply_dark_style()

    y_true = np.array([0] * len(normal_losses) + [1] * len(anomaly_losses))
    y_scores = np.array(normal_losses + anomaly_losses)

    thresholds = np.linspace(0, max(y_scores) * 1.1, 200)
    tprs, fprs = [], []

    for t in thresholds:
        predictions = (y_scores > t).astype(int)
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        tn = np.sum((predictions == 0) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC (trapezoidal)
    sorted_indices = np.argsort(fprs)
    sorted_fprs = np.array(fprs)[sorted_indices]
    sorted_tprs = np.array(tprs)[sorted_indices]
    auc = np.trapz(sorted_tprs, sorted_fprs)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fprs, tprs, color="#00d4aa", linewidth=2.5, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#555e6e", linewidth=1, alpha=0.6, label="Random Classifier")
    ax.fill_between(sorted_fprs, sorted_tprs, alpha=0.08, color="#00d4aa")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve", color="#e0e6ed", fontsize=13, fontweight="700", pad=12)
    ax.legend(loc="lower right", fontsize=9, facecolor="#0f1420", edgecolor="#2a3040",
              labelcolor="#e0e6ed")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.15)
    ax.set_aspect("equal")

    return fig_to_base64(fig), round(auc, 4)


def generate_confusion_matrix(normal_losses, anomaly_losses, threshold):
    """Generate confusion matrix visualization."""
    _apply_dark_style()

    y_true = np.array([0] * len(normal_losses) + [1] * len(anomaly_losses))
    y_scores = np.array(normal_losses + anomaly_losses)
    predictions = (y_scores > threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))

    matrix = np.array([[tn, fp], [fn, tp]])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors = np.array([
        ["#0a3d2e", "#3d1a1a"],
        ["#3d2e1a", "#0a2d3d"],
    ])

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colors[i][j],
                                       edgecolor="#2a3040", linewidth=2))
            ax.text(j + 0.5, i + 0.55, str(val), ha="center", va="center",
                    fontsize=26, fontweight="800", color="#e0e6ed")
            labels = [["TN", "FP"], ["FN", "TP"]]
            ax.text(j + 0.5, i + 0.25, labels[i][j], ha="center", va="center",
                    fontsize=10, color="#8892a0", fontweight="500")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Normal", "Anomaly"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual", fontsize=11, labelpad=10)
    ax.set_title("Confusion Matrix", color="#e0e6ed", fontsize=13, fontweight="700", pad=12)
    ax.invert_yaxis()

    return fig_to_base64(fig), {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def generate_loss_distribution(normal_losses, anomaly_losses, threshold):
    """Generate loss distribution histogram for normal vs anomaly."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0, max(max(normal_losses), max(anomaly_losses)) * 1.1, 40)
    ax.hist(normal_losses, bins=bins, alpha=0.6, color="#00d4aa", label="Normal", edgecolor="#00d4aa",
            linewidth=0.5)
    ax.hist(anomaly_losses, bins=bins, alpha=0.6, color="#ef4444", label="Anomaly", edgecolor="#ef4444",
            linewidth=0.5)
    ax.axvline(threshold, color="#f59e0b", linewidth=2.5, linestyle="--",
               label=f"Threshold ({threshold:.4f})")

    ax.set_xlabel("Reconstruction Loss (MSE)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Loss Distribution: Normal vs Anomaly", color="#e0e6ed",
                 fontsize=13, fontweight="700", pad=12)
    ax.legend(fontsize=9, facecolor="#0f1420", edgecolor="#2a3040", labelcolor="#e0e6ed")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig_to_base64(fig)


def generate_threshold_analysis(normal_losses, anomaly_losses):
    """Generate threshold vs metrics chart."""
    _apply_dark_style()

    y_true = np.array([0] * len(normal_losses) + [1] * len(anomaly_losses))
    y_scores = np.array(normal_losses + anomaly_losses)

    thresholds = np.linspace(0, max(y_scores) * 1.1, 100)
    precisions, recalls, f1s, accuracies = [], [], [], []

    for t in thresholds:
        preds = (y_scores > t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        a = (tp + tn) / len(y_true)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        accuracies.append(a)

    # Find optimal threshold (max F1)
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precisions, color="#3b82f6", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls, color="#f59e0b", linewidth=2, label="Recall")
    ax.plot(thresholds, f1s, color="#00d4aa", linewidth=2.5, label="F1-Score")
    ax.plot(thresholds, accuracies, color="#8b5cf6", linewidth=1.5, linestyle="--",
            label="Accuracy", alpha=0.7)

    ax.axvline(best_threshold, color="#ef4444", linewidth=1.5, linestyle=":",
               label=f"Optimal ({best_threshold:.4f})", alpha=0.8)
    ax.axvline(config.ANOMALY_THRESHOLD, color="#f59e0b", linewidth=1.5, linestyle="--",
               label=f"Current ({config.ANOMALY_THRESHOLD})", alpha=0.6)

    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Threshold vs Classification Metrics", color="#e0e6ed",
                 fontsize=13, fontweight="700", pad=12)
    ax.legend(fontsize=8, facecolor="#0f1420", edgecolor="#2a3040", labelcolor="#e0e6ed",
              loc="center right")
    ax.set_xlim(0, max(thresholds))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig_to_base64(fig), round(best_threshold, 6)


def generate_model_architecture_diagram():
    """Generate model architecture visualization."""
    _apply_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    layers = [
        {"x": 0.5, "label": "Input\n1×64×T", "color": "#1e3a5f", "border": "#3b82f6"},
        {"x": 2.0, "label": "Conv1\n16×32×T/2", "color": "#1a3d2e", "border": "#00d4aa"},
        {"x": 3.5, "label": "Conv2\n32×16×T/4", "color": "#1a3d2e", "border": "#00d4aa"},
        {"x": 5.0, "label": "Latent\n64×8×T/8", "color": "#3d1a3d", "border": "#7c5cfc"},
        {"x": 6.5, "label": "DeConv1\n32×16×T/4", "color": "#3d2e1a", "border": "#f59e0b"},
        {"x": 8.0, "label": "DeConv2\n16×32×T/2", "color": "#3d2e1a", "border": "#f59e0b"},
        {"x": 9.5, "label": "Output\n1×64×T", "color": "#1e3a5f", "border": "#3b82f6"},
    ]

    h, w = 1.8, 1.2
    y_center = 3.0

    for i, layer in enumerate(layers):
        rect = plt.Rectangle((layer["x"] - w / 2, y_center - h / 2), w, h,
                              facecolor=layer["color"], edgecolor=layer["border"],
                              linewidth=2, alpha=0.85, zorder=2)
        ax.add_patch(rect)
        ax.text(layer["x"], y_center, layer["label"], ha="center", va="center",
                fontsize=8, color="#e0e6ed", fontweight="500", zorder=3)

        if i < len(layers) - 1:
            next_x = layers[i + 1]["x"]
            ax.annotate("", xy=(next_x - w / 2 - 0.05, y_center),
                        xytext=(layer["x"] + w / 2 + 0.05, y_center),
                        arrowprops=dict(arrowstyle="->", color="#555e6e", lw=1.5),
                        zorder=1)

    # Labels
    ax.text(2.75, y_center + h / 2 + 0.5, "ENCODER", ha="center", fontsize=11,
            color="#00d4aa", fontweight="700")
    ax.text(7.25, y_center + h / 2 + 0.5, "DECODER", ha="center", fontsize=11,
            color="#f59e0b", fontweight="700")
    ax.text(5.0, y_center + h / 2 + 0.5, "BOTTLENECK", ha="center", fontsize=9,
            color="#7c5cfc", fontweight="600")

    ax.set_title("AutoEncoder Architecture", color="#e0e6ed", fontsize=14,
                 fontweight="800", pad=15)
    return fig_to_base64(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#   PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _run_inference(file_path):
    """Run model inference on an audio file. Returns dict with results."""
    waveform = load_audio(file_path)
    spec = waveform_to_spectrogram(waveform, device=config.DEVICE)
    if spec.shape[0] > 1:
        spec = spec[0:1, :, :]
    spec_input = spec.unsqueeze(0)

    with torch.no_grad():
        reconstruction = model(spec_input)
        loss = nn.functional.mse_loss(reconstruction, spec_input)
        loss_val = loss.item()
        accuracy = (1 - (loss_val ** 0.5)) * 100

    return {
        "waveform": waveform,
        "spec_input": spec_input,
        "reconstruction": reconstruction,
        "loss": loss_val,
        "accuracy": accuracy,
        "is_anomaly": loss_val > config.ANOMALY_THRESHOLD,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#   ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    param_count = sum(p.numel() for p in model.parameters()) if model else 0
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(config.DEVICE),
        "model_params": param_count,
        "threshold": config.ANOMALY_THRESHOLD,
        "sample_rate": config.SAMPLE_RATE,
        "n_mels": config.N_MELS,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Single file anomaly prediction with full visualizations."""
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Train first."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        audio_file.save(tmp.name)
        tmp.close()

        result = _run_inference(tmp.name)

        return jsonify({
            "filename": audio_file.filename,
            "loss": round(result["loss"], 6),
            "accuracy": round(result["accuracy"], 2),
            "threshold": config.ANOMALY_THRESHOLD,
            "is_anomaly": result["is_anomaly"],
            "verdict": "ANOMALY" if result["is_anomaly"] else "NORMAL",
            "device": str(config.DEVICE),
            "images": {
                "waveform": generate_waveform_image(result["waveform"]),
                "original_spectrogram": generate_spectrogram_image(
                    result["spec_input"], "Original Spectrogram"),
                "reconstructed_spectrogram": generate_spectrogram_image(
                    result["reconstruction"], "Reconstructed Spectrogram"),
                "error_map": generate_error_heatmap(
                    result["spec_input"], result["reconstruction"]),
            },
        })
    except Exception as e:
        logger.error("Prediction failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """Batch prediction on multiple uploaded files."""
    if not model_loaded:
        return jsonify({"error": "Model not loaded."}), 503

    files = request.files.getlist("audio")
    if not files:
        return jsonify({"error": "No files provided."}), 400

    results = []
    for audio_file in files:
        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            audio_file.save(tmp.name)
            tmp.close()
            r = _run_inference(tmp.name)
            results.append({
                "filename": audio_file.filename,
                "loss": round(r["loss"], 6),
                "accuracy": round(r["accuracy"], 2),
                "is_anomaly": r["is_anomaly"],
                "verdict": "ANOMALY" if r["is_anomaly"] else "NORMAL",
            })
        except Exception as e:
            results.append({
                "filename": audio_file.filename,
                "error": str(e),
            })
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    return jsonify({
        "results": results,
        "threshold": config.ANOMALY_THRESHOLD,
        "total": len(results),
        "anomalies": sum(1 for r in results if r.get("is_anomaly")),
        "normal": sum(1 for r in results if not r.get("is_anomaly") and "error" not in r),
        "errors": sum(1 for r in results if "error" in r),
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Run full model evaluation: ROC curve, confusion matrix, loss distribution,
    threshold analysis. Requires dataset to be present.
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        from src.data.loader import get_dataloaders
    except ImportError:
        return jsonify({"error": "Data loader not available."}), 500

    data = request.get_json() or {}
    normal_classes = data.get("normal_classes", config.EVAL_NORMAL_CLASSES)
    anomaly_classes = data.get("anomaly_classes", config.EVAL_ANOMALY_CLASSES)
    threshold = data.get("threshold", config.ANOMALY_THRESHOLD)

    try:
        # Collect losses for normal samples
        logger.info("Evaluating normal classes: %s", normal_classes)
        _, normal_loader = get_dataloaders(normal_classes, batch_size=8)
        normal_losses = []
        model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in normal_loader:
                batch = batch.to(config.DEVICE)
                out = model(batch)
                for i in range(batch.size(0)):
                    loss = criterion(out[i:i+1], batch[i:i+1]).item()
                    normal_losses.append(loss)

        # Collect losses for anomaly samples
        logger.info("Evaluating anomaly classes: %s", anomaly_classes)
        _, anomaly_loader = get_dataloaders(anomaly_classes, batch_size=8)
        anomaly_losses = []
        with torch.no_grad():
            for batch in anomaly_loader:
                batch = batch.to(config.DEVICE)
                out = model(batch)
                for i in range(batch.size(0)):
                    loss = criterion(out[i:i+1], batch[i:i+1]).item()
                    anomaly_losses.append(loss)

        logger.info("Collected %d normal, %d anomaly samples",
                     len(normal_losses), len(anomaly_losses))

        # Generate all charts
        roc_img, auc_score = generate_roc_curve(normal_losses, anomaly_losses)
        cm_img, cm_metrics = generate_confusion_matrix(normal_losses, anomaly_losses, threshold)
        dist_img = generate_loss_distribution(normal_losses, anomaly_losses, threshold)
        thresh_img, optimal_threshold = generate_threshold_analysis(
            normal_losses, anomaly_losses)

        return jsonify({
            "success": True,
            "normal_count": len(normal_losses),
            "anomaly_count": len(anomaly_losses),
            "normal_avg_loss": round(float(np.mean(normal_losses)), 6),
            "anomaly_avg_loss": round(float(np.mean(anomaly_losses)), 6),
            "auc": auc_score,
            "optimal_threshold": optimal_threshold,
            "current_threshold": threshold,
            "metrics": cm_metrics,
            "images": {
                "roc_curve": roc_img,
                "confusion_matrix": cm_img,
                "loss_distribution": dist_img,
                "threshold_analysis": thresh_img,
            },
        })

    except FileNotFoundError as e:
        return jsonify({
            "error": f"Dataset not found: {str(e)}. Generate spectrograms first.",
            "hint": "Run: python scripts/generate_spectrograms.py",
        }), 404
    except Exception as e:
        logger.error("Evaluation failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model architecture details and diagram."""
    if not model:
        return jsonify({"error": "Model not loaded."}), 503

    arch_img = generate_model_architecture_diagram()

    # Layer details
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d,
                               nn.ReLU, nn.Sigmoid)):
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "details": str(module),
            })

    return jsonify({
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "architecture_image": arch_img,
        "layers": layers,
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "n_mels": config.N_MELS,
            "audio_duration": config.AUDIO_DURATION_SEC,
            "threshold": config.ANOMALY_THRESHOLD,
            "device": str(config.DEVICE),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════

def create_app():
    load_model()
    return app


if __name__ == "__main__":
    create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
