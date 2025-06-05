# main.py

import sys
import os
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

system = None  # Global reference for signal handler

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutdown signal received...")
    if system:
        system.shutdown()
    sys.exit(0)

def main():
    """Main entry point"""
    global system
    
    print("="*60)
    print("RESEARCH-ALIGNED TRADING SYSTEM")
    print("="*60)
    print("Features: RSI + Bollinger Bands + EMA + SMA + Volume")
    print("Timeframes: 15min (trend) + 5min (entry)")
    print("ML Model: Logistic Regression")
    print("Architecture: Modular Python + NinjaScript")
    print("="*60)
    
    # Setup signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    try:
        print("Initializing trading system...")
        system = TradingSystem()
        print("Starting TCP server and waiting for NinjaTrader connection...")
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        return 1
    finally:
        # Cleanup in finally block
        if system:
            system.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())