# main.py - FIXED

import argparse
import logging
import os
import shutil
import signal
import sys
from datetime import datetime
from trading_system import create_pure_blackbox_system

RESET_DIRS = ["models", "patterns", "meta_learning", "data", "logs"]  # Added logs

def reset_workspace() -> None:
    """Reset workspace - don't worry about permission errors"""
    for d in RESET_DIRS:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"Removed {d}/")
            except:
                print(f"{d}/ couldn't be removed")
        
        os.makedirs(d, exist_ok=True)
        print(f"✅ {d}/ ready")

def setup_logging_after_reset():
    """Setup logging AFTER reset to avoid file locking issues"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pure_blackbox_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ],
        force=True  # Override any existing logging config
    )

def handle_shutdown(signum, frame):
    """Clean shutdown handler"""
    if 'log' in globals():
        log.info("Shutdown signal received")
    if 'trading_system' in globals():
        try:
            trading_system.shutdown_and_save()
            if 'log' in globals():
                log.info("Emergency save completed")
        except Exception as e:
            if 'log' in globals():
                log.error(f"Emergency save failed: {e}")
            else:
                print(f"Emergency save failed: {e}")
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
        print("--reset supplied: wiping workspace")
        reset_workspace()
        print("✅ Reset complete")
    
    # Setup logging AFTER reset
    setup_logging_after_reset()
    global log
    log = logging.getLogger(__name__)

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