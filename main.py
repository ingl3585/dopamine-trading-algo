# main.py - SIMPLE VERSION
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

def main():
    """Simple main entry point"""
    try:
        print("=" * 60)
        print("ADVANCED MARKET INTELLIGENCE ENGINE")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        print()
        
        trading_system = TradingSystem()
        trading_system.start()  # This handles Ctrl+C internally
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())