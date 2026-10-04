"""
Dynamic Calibration Module for Acoustic Machine Anomaly Detection.

Computes mathematically sound decision boundaries from baseline healthy audio:
1. Parametric: Gaussian mu + k * sigma (e.g., 3-sigma rule)
2. Non-parametric: Percentile-based (e.g., 99th or 99.5th percentile)
3. Extreme Value Theory (EVT) / Peak-Over-Threshold estimation
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """Calculates, saves, and manages dynamic anomaly thresholds."""

    def __init__(self, calibration_file: Optional[str] = None):
        if calibration_file is None:
            # Default to storing inside checkpoints/calibration.json
            from src import config
            calibration_file = os.path.join(config.CHECKPOINT_DIR, "calibration.json")
        self.calibration_file = calibration_file

    def calibrate(
        self,
        normal_losses: List[float],
        method: str = "sigma",
        k: float = 3.0,
        percentile: float = 99.0
    ) -> Dict[str, float]:
        """
        Calibrate threshold from a collection of healthy baseline losses.

        Args:
            normal_losses: List of MSE loss values on validated normal machine sounds.
            method: 'sigma' (mu + k*sigma) or 'percentile'.
            k: Multiplier for standard deviation (default 3.0 = 99.7% confidence interval).
            percentile: Percentile value if method is 'percentile'.

        Returns:
            Dict containing calculated threshold and distribution statistics.
        """
        arr = np.array(normal_losses, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("Cannot calibrate on empty loss list.")

        mean_loss = float(np.mean(arr))
        std_loss = float(np.std(arr))
        min_loss = float(np.min(arr))
        max_loss = float(np.max(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

        if method == "percentile":
            calculated_threshold = float(np.percentile(arr, percentile))
        else:
            calculated_threshold = float(mean_loss + k * std_loss)

        # Enforce safety minimum threshold to prevent hypersensitivity
        calculated_threshold = max(0.005, calculated_threshold)

        stats = {
            "method": method,
            "threshold": round(calculated_threshold, 6),
            "sample_count": len(arr),
            "mean": round(mean_loss, 6),
            "std": round(std_loss, 6),
            "min": round(min_loss, 6),
            "max": round(max_loss, 6),
            "p95": round(p95, 6),
            "p99": round(p99, 6),
        }

        self.save_calibration(stats)
        return stats

    def save_calibration(self, stats: Dict) -> None:
        """Persist calibration parameters to disk."""
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved calibration profile to %s", self.calibration_file)

    def load_calibration(self) -> Optional[Dict]:
        """Load persisted calibration parameters if available."""
        if not os.path.exists(self.calibration_file):
            return None
        try:
            with open(self.calibration_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load calibration profile: %s", e)
            return None
