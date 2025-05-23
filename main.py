# main.py

import argparse
import logging

from utils.logging_config import setup_logging
from core.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(
        description='Reinforcement Learning Trading System with Ichimoku/EMA Features'
    )
    
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Force full model retraining (required when switching to Ichimoku/EMA features)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level"
    )
    
    parser.add_argument(
        "--validate-features",
        action="store_true",
        help="Enable strict feature validation for debugging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulation mode without sending actual signals"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging with specified level
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    
    log.info("="*60)
    log.info("Starting RL Trading System with Ichimoku/EMA Features")
    log.info("="*60)
    
    if args.reset:
        log.info("Model reset requested - will retrain from scratch")
    
    if args.dry_run:
        log.info("DRY RUN MODE - No actual trades will be executed")
    
    if args.validate_features:
        log.info("Strict feature validation enabled")
    
    try:
        # Initialize and run the system
        runner = Runner(args)
        log.info("System initialized successfully")
        
        log.info("Waiting for NinjaTrader connection and market data...")
        runner.run()
        
    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
    except Exception as e:
        log.error(f"System error: {e}")
        raise
    finally:
        log.info("System shutdown complete")

if __name__ == "__main__":
    main()