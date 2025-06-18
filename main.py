# main.py - Simple Pure Black Box

import logging
import signal
import sys
import time
from datetime import datetime
from trading_system import PureBlackBoxTradingSystem

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def handle_shutdown(signum, frame):
    """Simple shutdown handler"""
    print(f"\nShutdown signal received. Saving AI learning...")
    
    if 'trading_system' in globals():
        try:
            # Save AI model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trading_system.trade_manager.agent.save_model(f"ai_model_{timestamp}.pt")
            print(f"AI model saved.")
            
            # Emergency close
            if trading_system.trade_manager.current_position['in_position']:
                trading_system.trade_manager.emergency_close_all()
                
        except Exception as e:
            print(f"Shutdown error: {e}")
    
    print("Pure Black Box AI shutdown complete.")
    sys.exit(0)

def main():
    """Simple main function"""
    
    # Register shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        print("🤖 Pure Black Box AI Trading System")
        print("AI learns everything from scratch - zero hardcoded knowledge")
        print("Press Ctrl+C to stop\n")
        
        # Initialize and start
        global trading_system
        trading_system = PureBlackBoxTradingSystem()
        trading_system.start()
        
        print("✅ System started - AI learning active")
        
        # Simple monitoring loop
        while True:
            time.sleep(60)  # Check every minute
            
            # Simple status every 10 minutes
            if hasattr(trading_system, 'signal_count') and trading_system.signal_count % 10 == 0:
                trades = trading_system.trade_manager.trade_stats['total_trades']
                pnl = trading_system.trade_manager.trade_stats['total_pnl']
                phase = trading_system.trade_manager.safety_manager.current_phase
                print(f"Status: {trades} trades, ${pnl:.0f} P&L, {phase} phase")
                
    except Exception as e:
        log.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()