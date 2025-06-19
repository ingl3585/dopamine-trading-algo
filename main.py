#!/usr/bin/env python3

import argparse
import logging
import shutil
import signal
import sys
from pathlib import Path

from trading_system import TradingSystem


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading.log'),
            logging.StreamHandler()
        ]
    )


def reset_workspace():
    dirs = ["models", "data", "logs"]
    for d in dirs:
        path = Path(d)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Black Box Trading System")
    parser.add_argument("--reset", action="store_true", help="Reset workspace")
    args = parser.parse_args()

    if args.reset:
        reset_workspace()
        print("Workspace reset complete")
        return

    # Ensure directories exist
    for d in ["models", "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    setup_logging()
    
    system = TradingSystem()
    
    def shutdown_handler(signum, frame):
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        system.start()
    except Exception as e:
        logging.error(f"System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()