# main.py

import argparse
import logging

from utils.logging_config import setup_logging
from core.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pure ML Signal Generation System for Ichimoku/EMA Trading'
    )
    
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Force full model retraining (recommended when switching architectures)"
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
        help="Run ML training without sending signals to NinjaScript"
    )
    
    parser.add_argument(
        "--signal-analysis",
        action="store_true", 
        help="Enable detailed signal quality analysis logging"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging with specified level
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)
    
    log.info("="*70)
    log.info("PURE ML SIGNAL GENERATION SYSTEM")
    log.info("Ichimoku/EMA Features → ML Predictions → NinjaScript Execution")
    log.info("="*70)
    
    log.info("Architecture: Python (ML) ↔ NinjaScript (Position Management)")
    log.info("Python Role: Feature processing, ML predictions, signal generation")
    log.info("NinjaScript Role: Entry/exit execution, stops, targets, position sizing")
    
    if args.reset:
        log.info("Model reset requested - will retrain from scratch")
    
    if args.dry_run:
        log.warning("DRY RUN MODE - ML training only, no signals sent to NinjaScript")
    
    if args.validate_features:
        log.info("Strict feature validation enabled")
        
    if args.signal_analysis:
        log.info("Detailed signal quality analysis enabled")
    
    try:
        # Initialize and run the pure ML system
        runner = Runner(args)
        log.info("Pure ML system initialized successfully")
        
        log.info("System Status:")
        log.info("  • Waiting for NinjaScript connection...")
        log.info("  • Ready to process Ichimoku/EMA features")
        log.info("  • ML model will generate confidence-based signals")
        log.info("  • NinjaScript will handle all position management")
        
        runner.run()
        
    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
    except Exception as e:
        log.error(f"System error: {e}")
        raise
    finally:
        log.info("Pure ML signal generation system shutdown complete")

if __name__ == "__main__":
    main()