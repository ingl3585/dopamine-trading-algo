# main.py

import logging
import signal
import sys
import time
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

def handle_shutdown(signum, frame):
    """Clean shutdown handler"""
    log.info("Shutdown signal received")
    if 'trading_system' in globals():
        try:
            trading_system.meta_learner.force_save()
            trading_system.trade_manager.force_save_all_adaptive_learning()
            log.info("Emergency save completed")
        except Exception as e:
            log.error(f"Emergency save failed: {e}")
    sys.exit(0)

def main():
    """Main entry point - SIMPLIFIED"""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        global trading_system
        log.info("Creating trading system...")
        trading_system = create_pure_blackbox_system()
        log.info("System created, starting...")
        trading_system.start()
    except KeyboardInterrupt:
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 