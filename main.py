# main.py - Enhanced with Advanced Position Management

import logging
import os
import signal
import sys
import time
from datetime import datetime
from trading_system import TradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(__name__)

def print_startup_banner():
    """Enhanced startup banner showing all AI capabilities"""
    print("=" * 80)
    print("🤖 BLACK BOX AI TRADING SYSTEM - ADVANCED POSITION MANAGEMENT")
    print("=" * 80)
    print("🧬 DNA Sequencing System     - Market pattern genetics")
    print("🔬 Micro Pattern Network     - Short-term behavior detection") 
    print("⏰ Temporal Archaeologist    - Time-based pattern discovery")
    print("🛡️  Market Immune System     - Loss prevention & pattern immunity")
    print("📈 Position Management AI    - Scaling, exits, and trail stops")
    print("🎯 Strategic Tool Learning   - Optimal subsystem coordination")
    print()
    print("🎯 AI DISCOVERS EVERYTHING:")
    print("   • Which tools to use when")
    print("   • Entry timing and confidence")
    print("   • Risk management strategies")
    print("   • When to scale into winning positions")
    print("   • When to take partial profits")
    print("   • Optimal exit timing strategies")
    print("   • Dynamic trailing stop management")
    print("   • Multi-level position management")
    print()
    print("📊 LEARNING PROGRESSION:")
    print("   Trades 1-5:    Basic entries and exits")
    print("   Trades 5-15:   + Partial exit strategies")
    print("   Trades 15-30:  + Position scaling")
    print("   Trades 30-50:  + Trailing stop optimization")
    print("   Trades 50+:    + Advanced multi-level management")
    print()
    print("💡 The AI learns optimal trading strategies through experience")
    print("🚀 No hardcoded rules - pure pattern discovery and adaptation")
    print("=" * 80)

def print_system_status(trading_system):
    """Print enhanced system status with position management info"""
    print("\n" + "=" * 60)
    print("📊 ENHANCED SYSTEM STATUS")
    print("=" * 60)
    
    # Core system status
    print(f"🔌 NinjaTrader Connection: {'Connected' if trading_system.tcp_bridge.connected else 'Disconnected'}")
    print(f"🧠 Intelligence Engine: {'Active' if trading_system.intelligence_engine else 'Inactive'}")
    print(f"🎯 AI Trade Manager: {'Active' if trading_system.trade_manager else 'Inactive'}")
    
    # Position status
    if trading_system.trade_manager:
        position = trading_system.trade_manager.current_position
        if position['in_position']:
            print(f"📍 Current Position: {position['tool_used'].upper()} - {'LONG' if position['action'] == 1 else 'SHORT'}")
            print(f"   Entry: ${position['entry_price']:.2f}")
            print(f"   Size: {position['size']:.1f}")
            print(f"   Scales: {position['scales_added']}")
            print(f"   Exits: {position['partial_exits']}")
            
            # Time in position
            if position['entry_time']:
                duration = (datetime.now() - position['entry_time']).total_seconds() / 3600
                print(f"   Duration: {duration:.1f}h")
        else:
            print("📍 Current Position: None")
        
        # Trade statistics
        stats = trading_system.trade_manager.trade_stats
        print(f"📈 Trade Statistics:")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Total P&L: ${stats['total_pnl']:.2f}")
        if stats['total_trades'] > 0:
            avg_pnl = stats['total_pnl'] / stats['total_trades']
            print(f"   Average P&L: ${avg_pnl:.2f}")
        print(f"   Best Trade: ${stats['best_trade']:.2f}")
        print(f"   Worst Trade: ${stats['worst_trade']:.2f}")
        
        # AI Learning Status (no artificial thresholds - AI learns everything from scratch)
        print(f"🎓 AI Learning Status:")
        print(f"   All Features Active: Entry/Exit, Scaling, Partial Exits, Risk Management")
        print(f"   Learning Mode: Pure discovery - no training wheels")
        
        # Position AI status
        if trading_system.trade_manager.position_ai.current_position:
            pos_ai = trading_system.trade_manager.position_ai.current_position
            print(f"🤖 Position AI:")
            print(f"   P&L: {pos_ai.current_pnl:.2%}")
            print(f"   Max Profit: {pos_ai.max_favorable_excursion:.2%}")
            print(f"   Max Loss: {pos_ai.max_adverse_excursion:.2%}")
    
    print("=" * 60)

def print_performance_summary(trading_system):
    """Print comprehensive performance summary"""
    if not trading_system.trade_manager:
        return
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 80)
    
    # Get full performance report from trade manager
    report = trading_system.trade_manager.get_performance_report()
    print(report)
    
    # Additional AI insights
    if hasattr(trading_system.trade_manager.agent, 'get_current_tool_preferences'):
        prefs = trading_system.trade_manager.agent.get_current_tool_preferences()
        print("\n🎯 CURRENT AI TOOL PREFERENCES:")
        for tool, preference in prefs.items():
            confidence_level = "High" if preference > 0.7 else "Medium" if preference > 0.5 else "Learning"
            print(f"   {tool.upper()}: {preference:.2f} ({confidence_level})")
    
    print("=" * 80)

def handle_shutdown(signum, frame):
    """Enhanced shutdown handler"""
    print(f"\n🛑 Shutdown signal received ({signum})")
    print("📊 Generating final performance report...")
    
    # Print final performance if system exists
    if 'trading_system' in globals():
        try:
            print_performance_summary(trading_system)
            
            # Save AI models
            if hasattr(trading_system.trade_manager, 'agent'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"ai_model_{timestamp}.pt"
                trading_system.trade_manager.agent.save_model(model_path)
                print(f"💾 AI model saved to {model_path}")
            
            # Emergency close any open positions
            if (hasattr(trading_system.trade_manager, 'current_position') and 
                trading_system.trade_manager.current_position['in_position']):
                print("⚠️  Closing open positions...")
                trading_system.trade_manager.emergency_close_all()
                time.sleep(2)  # Give time for close signal
                
        except Exception as e:
            print(f"Error during shutdown: {e}")
    
    print("👋 Black Box AI Trading System shutdown complete")
    sys.exit(0)

def main():
    """Enhanced main function with position management"""
    
    # Register enhanced shutdown handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        print_startup_banner()
        
        print("🚀 Initializing Enhanced Black Box AI Trading System...")
        
        # Ensure all required directories exist
        required_dirs = ['patterns', 'data', 'models', 'logs', 'reports', 'experience']
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified directories: {', '.join(required_dirs)}")
        
        # Initialize trading system
        global trading_system
        trading_system = TradingSystem()
        
        print("✅ Trading system initialized")
        print("🔌 Connecting to NinjaTrader...")
        
        # Start the system
        trading_system.start()
        
        print("✅ System started successfully")
        print("🤖 AI learning and position management active")
        print("\nPress Ctrl+C for graceful shutdown with performance report")
        print("=" * 80)
        
        # Enhanced monitoring loop
        last_status_time = time.time()
        last_performance_time = time.time()
        
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                
                # Print status every 5 minutes
                if current_time - last_status_time > 300:
                    print_system_status(trading_system)
                    last_status_time = current_time
                
                # Print performance summary every 30 minutes
                if current_time - last_performance_time > 1800:
                    print_performance_summary(trading_system)
                    last_performance_time = current_time
                
                # Check for system health
                if not trading_system.tcp_bridge.connected:
                    log.warning("⚠️  Connection to NinjaTrader lost - attempting reconnect...")
                    try:
                        trading_system.tcp_bridge.connect()
                    except Exception as e:
                        log.error(f"Reconnection failed: {e}")
                
            except KeyboardInterrupt:
                handle_shutdown(signal.SIGINT, None)
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before continuing
                
    except Exception as e:
        log.error(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_performance_analysis():
    """Standalone performance analysis tool"""
    print("📊 BLACK BOX AI PERFORMANCE ANALYZER")
    print("=" * 50)
    
    try:
        # Load saved AI model for analysis
        import glob
        model_files = glob.glob("ai_model_*.pt")
        
        if not model_files:
            print("❌ No saved AI models found")
            return
        
        latest_model = max(model_files)
        print(f"📂 Loading latest model: {latest_model}")
        
        # Initialize minimal system for analysis
        from rl_agent import StrategicToolLearningAgent
        agent = StrategicToolLearningAgent()
        agent.load_model(latest_model)
        
        # Print tool performance
        print(agent.get_tool_performance_report())
        
        # Print tool preferences
        prefs = agent.get_current_tool_preferences()
        print("\n🎯 LEARNED TOOL PREFERENCES:")
        for tool, pref in prefs.items():
            print(f"   {tool.upper()}: {pref:.3f}")
        
    except Exception as e:
        print(f"❌ Error in performance analysis: {e}")

if __name__ == "__main__":
    # Check for analysis mode
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        run_performance_analysis()
    else:
        main()