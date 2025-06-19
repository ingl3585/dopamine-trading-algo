# trading_system.py

import time
import logging
import signal
import sys
from datetime import datetime
from typing import Dict

from config import create_config
from meta_learner import create_meta_learner
from advanced_market_intelligence import create_intelligence_engine
from rl_agent import create_rl_agent
from tcp_bridge import create_tcp_bridge
from trade_manager_ai import create_trade_manager
from market_env import create_market_environment

log = logging.getLogger(__name__)

class BlackBoxTradingSystem:
    """
    Simplified black box trading system that orchestrates all components
    """
    
    def __init__(self):
        log.info("Initializing Black Box Trading System...")
        
        # Core components
        self.config = create_config()
        self.meta_learner = create_meta_learner()
        self.intelligence_engine = create_intelligence_engine()
        self.rl_agent = create_rl_agent(self.meta_learner, self.config)
        self.tcp_bridge = create_tcp_bridge()
        self.market_env = create_market_environment()
        
        # Trade manager coordinates everything
        self.trade_manager = create_trade_manager(
            self.rl_agent, self.intelligence_engine, self.tcp_bridge, self.config
        )
        
        # Set up callbacks
        self.tcp_bridge.on_market_data = self.on_market_data
        self.tcp_bridge.on_trade_completion = self.on_trade_completion
        
        # State tracking
        self.running = False
        self.market_updates = 0
        self.trades_completed = 0
        
        log.info("Black Box Trading System initialized")
    
    def start(self):
        """Start the trading system"""
        log.info("Starting Black Box Trading System...")
        
        # Setup shutdown handler
        def shutdown_handler(signum, frame):
            log.info("Shutdown requested")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        try:
            # Start TCP bridge
            self.tcp_bridge.start()
            self.running = True
            
            log.info("System started successfully")
            log.info("Waiting for market data from NinjaTrader...")
            
            # Main loop
            while self.running:
                time.sleep(1)
                
                # Periodic maintenance
                if self.market_updates % 100 == 0 and self.market_updates > 0:
                    self._periodic_maintenance()
                
        except KeyboardInterrupt:
            log.info("Shutdown requested by user")
            self.shutdown()
        except Exception as e:
            log.error(f"System error: {e}")
            self.shutdown()
            raise
    
    def on_market_data(self, data: Dict):
        """Handle incoming market data"""
        try:
            self.market_updates += 1
            
            # Extract price data and update market environment
            prices_1m = data.get('price_1m', [])
            if prices_1m:
                current_price = prices_1m[-1]
                volume = data.get('volume_1m', [1000])[-1] if data.get('volume_1m') else 1000
                self.market_env.update(current_price, volume)
            
            # Let trade manager handle the rest
            signal_sent = self.trade_manager.process_market_data(data)
            
            # Log progress
            if self.market_updates % 25 == 0:
                self._log_progress()
                
        except Exception as e:
            log.error(f"Error processing market data: {e}")
    
    def on_trade_completion(self, completion_data: Dict):
        """Handle trade completion from NinjaTrader"""
        try:
            self.trades_completed += 1
            
            pnl = completion_data.get('final_pnl', 0.0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            tool_used = completion_data.get('tool_used', 'unknown')
            
            # Let trade manager handle learning
            self.trade_manager.process_trade_completion(completion_data)
            
            log.info(f"Trade completed: P&L=${pnl:.2f}, Exit={exit_reason}, Tool={tool_used}")
            
        except Exception as e:
            log.error(f"Error processing trade completion: {e}")
    
    def _log_progress(self):
        """Log current progress"""
        
        rl_status = self.rl_agent.get_status()
        tcp_status = self.tcp_bridge.get_status()
        trade_status = self.trade_manager.get_status()
        
        log.info(f"Progress: {self.market_updates} updates, "
                f"{rl_status['total_decisions']} decisions, "
                f"{self.trades_completed} trades completed")
        
        log.info(f"Success rate: {trade_status['success_rate']:.1%}, "
                f"Daily P&L: ${trade_status['daily_pnl']:.2f}, "
                f"Exploration: {rl_status['epsilon']:.3f}")
        
        log.info(f"Account: ${trade_status['account_balance']:.0f} balance, "
                f"${trade_status['buying_power']:.0f} buying power")
    
    def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        try:
            # Save learning progress
            self.config.save_config()
            self.meta_learner.save_all()
            self.intelligence_engine.save_memory()
            self.rl_agent.save_model("models/rl_agent.pt")
            
            # Generate status report
            if self.market_updates % 500 == 0:
                self._generate_status_report()
                
        except Exception as e:
            log.error(f"Maintenance error: {e}")
    
    def _generate_status_report(self):
        """Generate comprehensive status report"""
        
        log.info("=" * 60)
        log.info("BLACK BOX TRADING SYSTEM STATUS")
        log.info("=" * 60)
        
        # System stats
        uptime = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)
        log.info(f"Uptime: {uptime}")
        log.info(f"Market updates: {self.market_updates}")
        log.info(f"Trades completed: {self.trades_completed}")
        
        # Component status
        rl_status = self.rl_agent.get_status()
        intel_status = self.intelligence_engine.get_status()
        trade_status = self.trade_manager.get_status()
        market_status = self.market_env.get_status()
        
        log.info(f"Daily signals: {trade_status['daily_signals']}")
        log.info(f"Daily P&L: ${trade_status['daily_pnl']:.2f}")
        
        log.info(f"\nRL Agent: {rl_status['success_rate']:.1%} success, "
                f"${rl_status['total_pnl']:.2f} total P&L")
        
        log.info(f"Intelligence: {intel_status['dna_patterns']} DNA patterns, "
                f"{intel_status['micro_patterns']} micro patterns")
        
        log.info(f"Market Environment: {market_status['bars_processed']} bars processed")
        
        log.info(f"Meta-learner: {self.meta_learner.successful_adaptations} adaptations")
        
        log.info("=" * 60)
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        log.info("Shutting down Black Box Trading System...")
        
        self.running = False
        
        try:
            # Save all learning progress
            log.info("Saving learning progress...")
            self.config.save_config()
            self.meta_learner.save_all()
            self.intelligence_engine.save_memory()
            self.rl_agent.save_model("models/rl_agent_final.pt")
            
            # Cleanup trade manager
            self.trade_manager.cleanup()
            
            # Stop TCP bridge
            self.tcp_bridge.stop()
            
            # Final status
            self._generate_status_report()
            
            # Trade manager performance report
            log.info(self.trade_manager.get_performance_report())
            
            log.info("Shutdown complete - all learning saved")
            
        except Exception as e:
            log.error(f"Shutdown error: {e}")

# Factory function
def create_trading_system():
    return BlackBoxTradingSystem()