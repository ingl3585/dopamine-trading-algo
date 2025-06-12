# main.py

import sys
import os
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

# Global reference for signal handler
trading_system = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global trading_system

    if trading_system and hasattr(trading_system, 'shutdown_event'):
        trading_system.shutdown_event.set()

def main():
    """Main entry point"""
    global trading_system

    # Signal handler for shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("Initializing trading system...")
        print("Press Ctrl+C to stop the system")
        trading_system = TradingSystem()
        trading_system.start()

    except Exception as e:
        print(f"System error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 