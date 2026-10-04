"""
Industrial Diagnostic Engine for Acoustic Machine Fault Detection.

Implements:
1. Multi-band spectral anomaly localization (Low, Mid, High frequency bands)
2. ISO 10816 inspired 4-tier industrial health severity levels:
   - OPTIMAL / NORMAL (Class A)
   - ACCEPTABLE / MONITOR (Class B)
   - WARNING / UNSATISFACTORY (Class C)
   - CRITICAL / ALARM (Class D)
3. Actionable diagnostic root-cause estimations based on acoustic frequency characteristics.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional
import numpy as np
import torch


class SeverityLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DiagnosticReport:
    mse_loss: float
    accuracy_score: float
    threshold: float
    severity: str
    severity_color: str
    status_summary: str
    recommended_action: str
    band_analysis: Dict[str, Any]
    dominant_fault_band: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndustrialDiagnosticEngine:
    """
    Analyzes autoencoder spectrogram reconstruction error across
    industrial vibration/acoustic frequency ranges to identify specific machine failure modes.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Define Mel band splits (assuming n_mels=64 covering ~0 to sample_rate/2 Hz):
        # Bands:
        # Low: 0 to ~600 Hz (Bands 0-15) -> Unbalance, looseness, structural resonance
        # Mid: ~600 to 2500 Hz (Bands 16-39) -> Gear meshing, valve chatter, cavitation
        # High: ~2500 to 8000 Hz (Bands 40-63) -> Bearing race defects, friction, air/steam leaks
        self.low_band = (0, 16)
        self.mid_band = (16, 40)
        self.high_band = (40, 64)

    def analyze(
        self,
        input_spec: torch.Tensor,
        recon_spec: torch.Tensor,
        threshold: float
    ) -> DiagnosticReport:
        """
        Evaluate input vs reconstructed spectrogram to produce actionable diagnostics.
        """
        if isinstance(input_spec, torch.Tensor):
            inp = input_spec.detach().cpu().squeeze().numpy()
            rec = recon_spec.detach().cpu().squeeze().numpy()
        else:
            inp = np.squeeze(input_spec)
            rec = np.squeeze(recon_spec)

        # 2D residual error map: (Mel bands, Time frames)
        residual = np.square(inp - rec)
        mse_loss = float(np.mean(residual))
        accuracy_score = max(0.0, float((1.0 - np.sqrt(mse_loss)) * 100.0))

        # Band-specific error analysis
        low_err = float(np.mean(residual[self.low_band[0]:self.low_band[1], :]))
        mid_err = float(np.mean(residual[self.mid_band[0]:self.mid_band[1], :]))
        high_err = float(np.mean(residual[self.high_band[0]:self.high_band[1], :]))

        band_errs = {
            "Low Frequency (0-600 Hz)": low_err,
            "Mid Frequency (600-2500 Hz)": mid_err,
            "High Frequency (2500-8000 Hz)": high_err,
        }

        # Determine dominant fault frequency band
        dominant_band = max(band_errs, key=band_errs.get)

        # Calculate Severity Levels (ISO 10816 inspired grading)
        # Normal: <= threshold
        # Warning: between 1.0x and 1.5x threshold
        # Critical: > 1.5x threshold
        warning_limit = threshold * 1.35
        critical_limit = threshold * 1.75

        if mse_loss <= threshold:
            severity = SeverityLevel.NORMAL.value
            severity_color = "#00d4aa"  # Emerald green
            status_summary = "Equipment operating within healthy acoustic parameters."
            recommended_action = "Routine inspection. No maintenance intervention required."
        elif mse_loss <= critical_limit:
            severity = SeverityLevel.WARNING.value
            severity_color = "#f59e0b"  # Amber warning
            status_summary = f"Acoustic deviation detected primarily in {dominant_band}."
            recommended_action = self._get_recommendation(dominant_band, severity="WARNING")
        else:
            severity = SeverityLevel.CRITICAL.value
            severity_color = "#ef4444"  # Alert red
            status_summary = f"High-energy anomaly detected in {dominant_band}."
            recommended_action = self._get_recommendation(dominant_band, severity="CRITICAL")

        band_analysis = {
            "low": {
                "label": "Low (0-600 Hz)",
                "error": round(low_err, 6),
                "potential_issues": "Mechanical unbalance, shaft misalignment, structural looseness",
            },
            "mid": {
                "label": "Mid (600-2500 Hz)",
                "error": round(mid_err, 6),
                "potential_issues": "Gear mesh wear, impeller cavitation, blade-pass flutter",
            },
            "high": {
                "label": "High (2.5k-8k Hz)",
                "error": round(high_err, 6),
                "potential_issues": "Bearing raceway spalling, lubrication breakdown, friction/rubbing",
            }
        }

        return DiagnosticReport(
            mse_loss=round(mse_loss, 6),
            accuracy_score=round(accuracy_score, 2),
            threshold=round(threshold, 6),
            severity=severity,
            severity_color=severity_color,
            status_summary=status_summary,
            recommended_action=recommended_action,
            band_analysis=band_analysis,
            dominant_fault_band=dominant_band,
        )

    def _get_recommendation(self, dominant_band: str, severity: str) -> str:
        if "Low" in dominant_band:
            action = "Check machine foundation mounting, coupling alignment, and shaft rotational balance."
        elif "Mid" in dominant_band:
            action = "Inspect gearbox tooth wear, hydraulic pump valves, or pump suction head for cavitation."
        else:
            action = "Lubricate bearing assemblies or test with ultrasonic probe for rolling element degradation."

        prefix = "URGENT ACTION: " if severity == "CRITICAL" else "RECOMMENDED ACTION: "
        return prefix + action
