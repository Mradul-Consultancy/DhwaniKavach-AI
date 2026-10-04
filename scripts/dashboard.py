#!/usr/bin/env python
"""
Entry point: Start the Acoustic Machine Fault Detection and Monitoring System web dashboard.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --port 8080
    python scripts/dashboard.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Launch DhwaniKavach AI (ध्वनिकवच) Industrial Dashboard")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to listen on (default: 5000)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from dashboard.app import create_app, app

    create_app()

    logger.info("=" * 60)
    logger.info("  DhwaniKavach AI (ध्वनिकवच) — Industrial Diagnostic Dashboard")
    logger.info("  Open: http://%s:%d", args.host, args.port)
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
