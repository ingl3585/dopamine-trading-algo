# main.py - Clean startup for pure black box AI learning

import sys
import os
import time
import signal
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_system import TradingSystem
from config import BlackBoxConfig

def print_startup_banner():
    """Print clean startup banner for black box AI"""
    print("=" * 80)
    print("🤖 BLACK BOX AI TRADING SYSTEM - PURE LEARNING")
    print("=" * 80)
    print("🧬 DNA Sequencing System     - Market pattern genetics")
    print("🔬 Micro Pattern Network     - Short-term behavior detection") 
    print("⏰ Temporal Archaeologist    - Time-based pattern discovery")
    print("🛡️  Market Immune System     - Loss prevention & pattern immunity")
    print()
    print("🎯 AI DISCOVERS EVERYTHING:")
    print("   • Which tools to use when (DNA vs Micro vs Temporal vs Immune)")
    print("   • Whether to use stops/targets at all")
    print("   • Optimal stop/target distances if beneficial")
    print("   • Tool combinations that work together")
    print("   • Exit timing for each strategy")
    print("   • Risk management per market condition")
    print()
    print("🚀 NO RULES - AI LEARNS OPTIMAL TRADING FROM PURE EXPERIENCE")
    print("=" * 80)

def setup_graceful_shutdown(trading_system):
    """Setup graceful shutdown with AI model saving"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Shutdown signal received")
        print("💾 Saving AI learning progress...")
        
        try:
            # Save final performance report
            if hasattr(trading_system.trade_manager, 'get_performance_report'):
                final_report = trading_system.trade_manager.get_performance_report()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"reports/final_ai_report_{timestamp}.txt"
                
                with open(report_file, 'w') as f:
                    f.write("BLACK BOX AI - FINAL LEARNING REPORT\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Session ended: {datetime.now()}\n\n")
                    f.write(final_report)
                
                print(f"📊 Final report saved: {report_file}")
            
            # Save AI model
            if hasattr(trading_system.trade_manager, 'black_box_ai'):
                model_path = f"models/final_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                trading_system.trade_manager.black_box_ai.save_model(model_path)
                print(f"🧠 AI model saved: {model_path}")
            
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
        
        print("✅ Graceful shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Clean main entry point for pure black box AI"""
    try:
        print_startup_banner()
        
        print("🔄 Initializing Black Box AI...")
        print("   • Loading permanent memory patterns...")
        print("   • Starting neural networks...")
        print("   • Connecting to NinjaTrader...")
        print()
        
        trading_system = TradingSystem()
        setup_graceful_shutdown(trading_system)
        
        print("✅ BLACK BOX AI READY - PURE LEARNING MODE")
        print()
        print("🎯 AI WILL DISCOVER:")
        print("   • Optimal tool usage patterns")
        print("   • Whether stops/targets help or hurt")
        print("   • Best risk management for each situation")
        print("   • Market regime adaptation strategies")
        print()
        print("📊 Progress reports every 5 minutes")
        print("🛑 Press Ctrl+C for graceful shutdown")
        print("=" * 80)
        print()
        
        trading_system.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())