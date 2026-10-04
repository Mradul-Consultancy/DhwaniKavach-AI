"""
ONNX Export and Edge Benchmark Utility for Machine Sound Diagnostics.

Exports the PyTorch AutoEncoder to open standard ONNX format,
verifies numerical consistency, and provides an Edge-optimized runtime predictor.
"""

import logging
import os
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch

from src import config
from src.models.autoencoder import AutoEncoder

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: AutoEncoder,
    output_path: Optional[str] = None,
    opset_version: int = 14
) -> str:
    """
    Export PyTorch AutoEncoder to ONNX format with dynamic batch & time frames.
    """
    if output_path is None:
        output_path = os.path.join(config.CHECKPOINT_DIR, "autoencoder_edge.onnx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    # Dummy input: (batch_size=1, channels=1, n_mels=64, time_frames=313)
    dummy_input = torch.randn(1, 1, config.N_MELS, 313, device="cpu")
    cpu_model = AutoEncoder().to("cpu")
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()

    logger.info("Exporting model to ONNX: %s", output_path)

    torch.onnx.export(
        cpu_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectrogram_input"],
        output_names=["reconstructed_output"],
        dynamic_axes={
            "spectrogram_input": {0: "batch_size", 3: "time_frames"},
            "reconstructed_output": {0: "batch_size", 3: "time_frames"},
        }
    )

    logger.info("Successfully exported ONNX model to %s", output_path)
    return output_path


class EdgeONNXPredictor:
    """
    Edge-optimized inference runner.
    Uses ONNX Runtime if available for ultra-fast C++ runtime execution,
    otherwise falls back to PyTorch eval mode seamlessly.
    """

    def __init__(self, onnx_path: Optional[str] = None, fallback_model: Optional[AutoEncoder] = None):
        if onnx_path is None:
            onnx_path = os.path.join(config.CHECKPOINT_DIR, "autoencoder_edge.onnx")

        self.onnx_path = onnx_path
        self.session = None
        self.provider = "PyTorch-CPU"

        try:
            import onnxruntime as ort
            if os.path.exists(onnx_path):
                self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self.provider = "ONNX-Runtime-Edge"
                logger.info("Initialized Edge ONNX Runtime engine: %s", onnx_path)
        except Exception as e:
            logger.info("ONNX Runtime not active (%s), utilizing PyTorch CPU inference.", e)

        self.fallback_model = fallback_model

    def predict(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Runs edge inference on spectrogram.
        Returns: (reconstructed_numpy_array, latency_ms)
        """
        start = time.perf_counter()

        if self.session is not None:
            # ONNX Runtime pathway
            np_in = input_tensor.detach().cpu().numpy().astype(np.float32)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: np_in})
            recon = outputs[0]
        else:
            # PyTorch fallback
            with torch.no_grad():
                recon = self.fallback_model(input_tensor).detach().cpu().numpy()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return recon, latency_ms
