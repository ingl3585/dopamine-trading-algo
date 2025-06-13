# main.py - Enhanced version
import sys
import os
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_system import TradingSystem

# Global reference for signal handler
enhanced_system = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global enhanced_system
    if enhanced_system:
        enhanced_system.stop()

def main():
    """Enhanced main entry point with Advanced Market Intelligence"""
    global enhanced_system

    # Signal handler for shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("=" * 60)
        print("ADVANCED MARKET INTELLIGENCE ENGINE")
        print("Multi-Layer AI System for MNQ Futures Trading")
        print("=" * 60)
        print("Features:")
        print("  ✓ DNA Sequencing System (Market Genetics)")
        print("  ✓ Micro-Pattern Memory Network") 
        print("  ✓ Temporal Pattern Archaeologist")
        print("  ✓ Market Immune System")
        print("  ✓ Meta-Learning Director")
        print("  ✓ Permanent Memory (Never Forgets)")
        print("  ✓ Continuous Learning & Adaptation")
        print("=" * 60)
        print("Press Ctrl+C to stop the system")
        print()
        
        trading_system = TradingSystem()
        trading_system.start()

    except Exception as e:
        print(f"System error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())