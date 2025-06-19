# main.py

import argparse
import logging
import os
import shutil
import signal
import sys
from datetime import datetime

def reset_workspace():
    """Reset workspace directories"""
    dirs = ["models", "patterns", "meta_learning", "data", "logs"]
    for d in dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"Removed {d}/")
            except:
                print(f"Couldn't remove {d}/")
        os.makedirs(d, exist_ok=True)

def setup_logging():
    """Simple logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/blackbox_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Pure Black-Box Trading System")
    parser.add_argument("--reset", action="store_true", help="Reset workspace")
    args = parser.parse_args()

    if args.reset:
        print("Resetting workspace...")
        reset_workspace()
        print("✅ Reset complete")
    
    setup_logging()
    log = logging.getLogger(__name__)

    # Import here to avoid circular imports
    from trading_system import BlackBoxTradingSystem
    
    def shutdown_handler(signum, frame):
        log.info("Shutdown signal received")
        if 'system' in globals():
            system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        log.info("Starting Black Box Trading System")
        system = BlackBoxTradingSystem()
        system.start()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()