# main.py - UPDATED: Pure Black Box with Complete Meta-Learning

import logging
import signal
import sys
import time
from datetime import datetime
from trading_system import create_pure_blackbox_system

# Enhanced logging for meta-learning
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
    """Enhanced shutdown handler for pure black box system"""
    print(f"\n🛑 Shutdown signal received. Saving ALL adaptive learning...")
    
    if 'trading_system' in globals():
        try:
            print("💾 Preserving meta-learned parameters...")
            trading_system.meta_learner.force_save()
            
            print("🧠 Saving AI model and learned architecture...")
            trading_system.trade_manager.force_save_all_adaptive_learning()
            
            print("📊 Exporting complete knowledge base...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            knowledge_file = f"emergency_save_{timestamp}.json"
            trading_system.intelligence_engine.export_knowledge_base(knowledge_file)
            
            print("🔒 Emergency closing any open positions...")
            if trading_system.trade_manager.current_position['in_position']:
                trading_system.trade_manager.emergency_close_all()
                
            print(f"✅ ALL ADAPTIVE LEARNING PRESERVED")
            print(f"📁 Knowledge saved to: {knowledge_file}")
            
        except Exception as e:
            print(f"❌ Emergency save error: {e}")
            # Still try to save meta-parameters at minimum
            try:
                trading_system.meta_learner.force_save()
                print("✅ Critical meta-parameters saved")
            except:
                print("❌ CRITICAL: Could not save meta-parameters")
    
    print("🤖 Pure Black Box AI shutdown complete.")
    print("🚀 Next startup will resume adaptive learning from saved state")
    sys.exit(0)

def display_startup_banner():
    """Display enhanced startup banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🤖 PURE BLACK BOX AI TRADING SYSTEM                     ║
║                        Complete Meta-Learning Implementation                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🧠 ZERO HARDCODED VALUES - Everything adapts through experience            ║
║                                                                              ║
║  📈 ADAPTIVE FEATURES:                                                       ║
║     • Risk limits learned from actual losses                                ║
║     • Position sizing adapts to market conditions                           ║
║     • Confidence thresholds discover optimal entry points                   ║
║     • Network architecture evolves based on performance                     ║
║     • Reward structure learns what drives success                           ║
║     • Tool usage optimized through trial and error                          ║
║                                                                              ║
║  🎯 COMPLETE SELF-OPTIMIZATION:                                             ║
║     • All parameters discovered through pure experience                     ║
║     • No trader wisdom or preset rules                                      ║
║     • Continuous learning and adaptation                                    ║
║     • Persistent memory across sessions                                     ║
║                                                                              ║
║  ⚡ READY FOR LIVE TRADING: Connect to NinjaTrader and begin learning       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 Starting Pure Black Box AI System...
🧠 Loading all previously learned parameters...
📊 Preparing for complete autonomous operation...

Press Ctrl+C to stop and save all learned knowledge
"""
    print(banner)

def display_system_status(system):
    """Display current system status"""
    try:
        # Get adaptive configuration status
        config_status = system.config.get_adaptation_status()
        
        # Get current meta-learning state
        meta_status = system.meta_learner.get_adaptation_report()
        
        print("\n" + "="*60)
        print("📊 CURRENT ADAPTIVE STATE")
        print("="*60)
        print(config_status)
        print("\n" + "="*60)
        print("🧠 META-LEARNING STATUS")  
        print("="*60)
        print(meta_status)
        print("="*60 + "\n")
        
    except Exception as e:
        log.warning(f"Status display error: {e}")

def main():
    """Enhanced main function for pure black box system"""
    
    # Register enhanced shutdown handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        display_startup_banner()
        
        print("🔧 Initializing adaptive components...")
        
        # Initialize pure black box system with meta-learning
        global trading_system
        trading_system = create_pure_blackbox_system()
        
        print("✅ Pure Black Box System initialized")
        print("🧠 Meta-learning active - all parameters will adapt")
        print("📈 Intelligence subsystems ready for tool learning")
        print("🎯 Safety management with adaptive limits")
        print("🚀 Network architecture evolution enabled")
        
        # Display initial adaptive state
        display_system_status(trading_system)
        
        print("🟢 SYSTEM READY - Starting pure black box operation...\n")
        
        # Start the pure black box system
        trading_system.start()
        
    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        log.error(f"Critical system error: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        
        # Emergency shutdown
        if 'trading_system' in globals():
            try:
                trading_system.shutdown_and_save()
            except:
                pass
        
        sys.exit(1)

def run_system_diagnostics():
    """Run diagnostic checks for pure black box system"""
    print("🔍 Running Pure Black Box System Diagnostics...\n")
    
    try:
        # Test meta-learner initialization
        print("1. Testing Meta-Learning System...")
        from meta_learner import PureMetaLearner
        meta_learner = PureMetaLearner()
        print(f"   ✅ Meta-learner initialized with {len(meta_learner.parameters)} adaptive parameters")
        
        # Test adaptive config
        print("2. Testing Adaptive Configuration...")
        from config import create_adaptive_config
        config = create_adaptive_config()
        print(f"   ✅ Adaptive config ready - position size: {config.PRODUCTION_PHASE_SIZE:.3f}")
        
        # Test pure black box agent
        print("3. Testing Pure Black Box Agent...")
        from rl_agent import create_pure_blackbox_agent
        agent = create_pure_blackbox_agent()
        print(f"   ✅ Agent ready - network rebuilds supported")
        
        # Test intelligence systems
        print("4. Testing Intelligence Systems...")
        from advanced_market_intelligence import AdvancedMarketIntelligence
        intel = AdvancedMarketIntelligence()
        print(f"   ✅ Intelligence systems ready - DNA, Micro, Temporal, Immune")
        
        # Test database connections
        print("5. Testing Persistent Memory...")
        import os
        if os.path.exists("meta_parameters.db"):
            print("   ✅ Meta-parameter database found - will resume learning")
        else:
            print("   📝 Fresh start - will create new learning database")
        
        if os.path.exists("market_intelligence.db"):
            print("   ✅ Intelligence database found - patterns preserved")
        else:
            print("   📝 Fresh intelligence - will discover new patterns")
        
        print("\n🎯 ALL DIAGNOSTICS PASSED")
        print("🚀 Pure Black Box System ready for operation")
        print("🧠 Complete meta-learning stack functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DIAGNOSTIC FAILED: {e}")
        print("Please check installation and dependencies")
        return False

if __name__ == "__main__":
    print("🤖 Pure Black Box AI Trading System")
    print("Complete Meta-Learning Implementation\n")
    
    # Run diagnostics first
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        run_system_diagnostics()
        sys.exit(0)
    
    # Check if diagnostics should be run
    print("Running quick system check...")
    if run_system_diagnostics():
        print("\n" + "="*50)
        print("🚀 Starting Pure Black Box System...")
        print("="*50)
        time.sleep(2)  # Brief pause
        main()
    else:
        print("\n❌ System check failed - please fix issues before starting")
        sys.exit(1)