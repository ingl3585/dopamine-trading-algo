# main.py

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

def main():
    """Main entry point"""
    print("="*60)
    print("RESEARCH-ALIGNED TRADING SYSTEM")
    print("="*60)
    print("Features: RSI + Bollinger Bands + EMA + SMA + Volume")
    print("Timeframes: 15min (trend) + 5min (entry)")
    print("ML Model: Logistic Regression")
    print("Architecture: Modular Python + NinjaScript")
    print("="*60)
    
    try:
        system = TradingSystem()
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())