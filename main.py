# main.py

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

def main():
    """Main entry point"""    
    try:
        print("Initializing trading system...")
        trading_system = TradingSystem()
        trading_system.start()
    except Exception as e:
        print(f"System error: {e}")
        return 1
    finally:
        if trading_system:
            trading_system.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())