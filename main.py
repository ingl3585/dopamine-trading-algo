# main.py

import argparse
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from trading_system import create_pure_blackbox_system

# Simplified logging setup - remove duplicate logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pure_blackbox_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

RESET_DIRS = ["models", "patterns", "meta_learning", "data"]

def reset_workspace() -> None:
    """Delete and recreate the state directories specified in RESET_DIRS."""
    for d in RESET_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            log.info(f"Removed {d}/")
        os.makedirs(d, exist_ok=True)
        log.info(f"Re-created empty {d}/")

def handle_shutdown(signum, frame):
    """Clean shutdown handler"""
    log.info("Shutdown signal received")
    if 'trading_system' in globals():
        try:
            trading_system.shutdown_and_save()
            log.info("Emergency save completed")
        except Exception as e:
            log.error(f"Emergency save failed: {e}")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Pure Black-Box Trading System")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete models/patterns/dbs/etc before starting",
    )
    args = parser.parse_args()

    if args.reset:
        log.warning("--reset supplied: wiping workspace")
        reset_workspace()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        global trading_system
        log.info("Creating trading system")
        trading_system = create_pure_blackbox_system()
        log.info("System created")
        trading_system.start()
    except KeyboardInterrupt:
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()