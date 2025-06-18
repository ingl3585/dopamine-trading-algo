# main.py

import logging
import signal
import sys
import time
from datetime import datetime
from trading_system import create_pure_blackbox_system

# Logging setup
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
    """Clean shutdown handler"""
    log.info("Shutdown signal received - preserving learning state")
    if 'trading_system' in globals():
        try:
            trading_system.meta_learner.force_save()
            trading_system.trade_manager.force_save_all_adaptive_learning()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trading_system.intelligence_engine.export_knowledge_base(f"emergency_save_{timestamp}.json")

            if trading_system.trade_manager.current_position['in_position']:
                trading_system.trade_manager.emergency_close_all()

            log.info("All adaptive learning saved and emergency close (if any) triggered")
        except Exception as e:
            log.error(f"Emergency save failed: {e}")
    sys.exit(0)

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        global trading_system
        trading_system = create_pure_blackbox_system()
        log.info("System initialized with meta-learning")
        trading_system.start()
    except KeyboardInterrupt:
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        if 'trading_system' in globals():
            try:
                trading_system.shutdown_and_save()
            except:
                pass
        sys.exit(1)

def run_system_diagnostics():
    """Minimal diagnostics"""
    try:
        from meta_learner import PureMetaLearner
        meta_learner = PureMetaLearner()

        from config import create_adaptive_config
        config = create_adaptive_config()

        from rl_agent import create_pure_blackbox_agent
        agent = create_pure_blackbox_agent()

        from advanced_market_intelligence import AdvancedMarketIntelligence
        intel = AdvancedMarketIntelligence()

        log.info("Diagnostics passed - all subsystems operational")
        return True
    except Exception as e:
        print(f"Diagnostics failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        run_system_diagnostics()
        sys.exit(0)

    if run_system_diagnostics():
        log.info("Starting Pure Black Box AI System...")
        main()
    else:
        print("Startup aborted due to failed diagnostics.")
        sys.exit(1)
