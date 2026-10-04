"""
Industrial Alert and Telemetry Dispatcher for Machine Faults.

Sends real-time telemetry and alarms to:
1. Standard Webhooks (Slack, Microsoft Teams, Discord, custom SCADA backend)
2. MQTT Industrial Brokers (for PLCs, Ignition SCADA, Edge Gateways)
3. Local audit log files for ISO-55000 asset management compliance
"""

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IndustrialAlertDispatcher:
    """Dispatches industrial fault alerts and telemetry to external systems."""

    def __init__(
        self,
        machine_id: str = "PUMP-UNIT-01",
        webhook_url: Optional[str] = None,
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
        audit_log_path: Optional[str] = None,
    ):
        self.machine_id = machine_id
        self.webhook_url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
        self.mqtt_broker = mqtt_broker or os.environ.get("MQTT_BROKER_HOST")
        self.mqtt_port = mqtt_port

        if audit_log_path is None:
            from src import config
            audit_log_path = os.path.join(config.OUTPUT_DIR, "industrial_audit.log")
        self.audit_log_path = audit_log_path

    def dispatch(self, report_dict: Dict[str, Any], filename: str = "live_stream") -> Dict[str, Any]:
        """
        Package telemetry and dispatch alerts if condition requires action.
        """
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "asset_id": self.machine_id,
            "source_sample": filename,
            "metric": "MSE_Reconstruction_Loss",
            "mse_loss": report_dict.get("mse_loss"),
            "threshold": report_dict.get("threshold"),
            "severity": report_dict.get("severity"),
            "status": report_dict.get("status_summary"),
            "dominant_band": report_dict.get("dominant_fault_band"),
            "action_required": report_dict.get("recommended_action"),
        }

        # 1. Audit Log (always logged for ISO compliance)
        self._write_audit_log(payload)

        # 2. Dispatch webhook if WARNING or CRITICAL
        webhook_status = "skipped"
        if report_dict.get("severity") in ["WARNING", "CRITICAL"]:
            if self.webhook_url:
                webhook_status = self._send_webhook(payload)
            else:
                webhook_status = "no_url_configured"

        # 3. MQTT Broadcast
        mqtt_status = "skipped"
        if self.mqtt_broker:
            mqtt_status = self._publish_mqtt(payload)

        return {
            "telemetry_dispatched": True,
            "payload": payload,
            "webhook_status": webhook_status,
            "mqtt_status": mqtt_status,
        }

    def _write_audit_log(self, payload: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.warning("Audit log write failed: %s", e)

    def _send_webhook(self, payload: Dict[str, Any]) -> str:
        try:
            # Format payload for standard webhook endpoints (Slack/Discord/Teams compatible)
            body = {
                "text": f"🚨 *INDUSTRIAL ASSET ALERT* — `{payload['asset_id']}`\n"
                        f"*Severity:* {payload['severity']}\n"
                        f"*Loss:* {payload['mse_loss']} (Threshold: {payload['threshold']})\n"
                        f"*Fault Zone:* {payload['dominant_band']}\n"
                        f"*Action:* {payload['action_required']}",
                "telemetry": payload
            }
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "AcousticDiagnostic/2.0"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return f"delivered_status_{response.status}"
        except Exception as e:
            logger.warning("Webhook dispatch failed: %s", e)
            return f"failed_{str(e)}"

    def _publish_mqtt(self, payload: Dict[str, Any]) -> str:
        # Graceful MQTT publish if paho-mqtt is available, otherwise skipped
        try:
            import paho.mqtt.publish as publish
            topic = f"factory/telemetry/{self.machine_id}/fault"
            publish.single(
                topic,
                payload=json.dumps(payload),
                hostname=self.mqtt_broker,
                port=self.mqtt_port,
                qos=1
            )
            return "published"
        except ImportError:
            return "paho_mqtt_not_installed"
        except Exception as e:
            logger.warning("MQTT broadcast failed: %s", e)
            return f"mqtt_error_{str(e)}"
