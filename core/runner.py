# core/runner.py

import time
import logging

from config import Config
from model.agent import RLAgent
from services.portfolio import Portfolio
from services.tcp_bridge import TCPBridge
from services.tick_processor import TickProcessor
from utils.feature_logger import FeatureLogger
from handlers.live_feature_handler import LiveFeatureHandler

log = logging.getLogger(__name__)

class Runner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config()
        
        log.info("Initializing RL Trading System with Ichimoku/EMA features")
        log.info(f"Input dimensions: {self.cfg.INPUT_DIM} features")
        log.info(f"Model architecture: {self.cfg.HIDDEN_DIM} hidden units")
        
        # Initialize core components
        self.agent = RLAgent(self.cfg)
        self.portfolio = Portfolio(self.cfg)
        self.logger = FeatureLogger(self.cfg.FEATURE_FILE, self.cfg.BATCH_SIZE)
        
        # Network components
        self.tcp = TCPBridge("localhost", 5556, 5557)
        self.tick_processor = TickProcessor()
        
        # Main handler
        self.handler = LiveFeatureHandler(
            self.cfg, self.agent, self.portfolio, 
            self.logger, self.tcp, self.args
        )
        
        # System status
        self.start_time = time.time()
        self.is_running = False
        
        log.info("System initialization complete")

    def run(self):
        """Main execution loop"""
        log.info("Starting trading system...")
        
        try:
            # Wait for tick processor to be ready
            if not self.tick_processor.wait_until_ready(timeout=30):
                log.error("Tick processor failed to initialize")
                return
            
            # Set up TCP handler
            self.tcp.on_features = self.handler.handle_live_feature
            
            self.is_running = True
            log.info("System is now active and waiting for market data")
            
            # Log initial status
            self._log_startup_status()
            
            # Main loop with periodic status updates
            last_status_time = time.time()
            status_interval = 3600  # 1 hour
            
            while self.is_running:
                current_time = time.time()
                
                # Periodic status logging
                if current_time - last_status_time >= status_interval:
                    self._log_periodic_status()
                    last_status_time = current_time
                
                # Sleep for a reasonable interval
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            log.info("Shutdown signal received")
        except Exception as e:
            log.error(f"Runtime error: {e}")
            raise
        finally:
            self.shutdown()

    def _log_startup_status(self):
        """Log system status at startup"""
        try:
            log.info("=== System Status ===")
            log.info(f"Model ready: {self.agent.model is not None}")
            log.info(f"Feature dimensions: {self.cfg.INPUT_DIM}")
            log.info(f"Confidence threshold: {self.cfg.CONFIDENCE_THRESHOLD}")
            log.info(f"Reset mode: {self.args.reset}")
            
            if hasattr(self.args, 'dry_run') and self.args.dry_run:
                log.warning("RUNNING IN DRY-RUN MODE - No actual trades will be executed")
            
            # Portfolio status
            portfolio_status = self.portfolio.get_portfolio_status()
            log.info(f"Portfolio: {portfolio_status}")
            
        except Exception as e:
            log.warning(f"Startup status logging failed: {e}")

    def _log_periodic_status(self):
        """Log periodic system status"""
        try:
            uptime = time.time() - self.start_time
            uptime_hours = uptime / 3600
            
            log.info(f"=== Periodic Status (Uptime: {uptime_hours:.1f}h) ===")
            
            # Handler status
            if hasattr(self.handler, 'get_status_report'):
                status = self.handler.get_status_report()
                log.info(f"Steps processed: {status.get('step_counter', 0)}")
                log.info(f"Ready for trading: {status.get('is_ready_for_trading', False)}")
                log.info(f"Experience buffer: {status.get('experience_buffer_size', 0)}")
                
                # Feature importance
                importance = status.get('feature_importance', {})
                if importance:
                    top_feature = max(importance.items(), key=lambda x: x[1])
                    log.info(f"Top feature: {top_feature[0]} ({top_feature[1]:.3f})")
            
            # Portfolio status
            portfolio_status = self.portfolio.get_portfolio_status()
            log.info(f"Position utilization: {portfolio_status.get('utilization_pct', 0):.1f}%")
            
            # Signal handler performance
            if hasattr(self.handler, 'dispatcher'):
                performance = self.handler.dispatcher.get_performance_summary()
                log.info("Signal Performance:")
                for line in performance.split('\n'):
                    if line.strip():
                        log.info(f"  {line.strip()}")
            
        except Exception as e:
            log.warning(f"Periodic status logging failed: {e}")

    def shutdown(self):
        """Graceful shutdown"""
        log.info("Initiating system shutdown...")
        self.is_running = False
        
        try:
            # Close network connections
            if hasattr(self, 'tcp'):
                self.tcp.close()
                log.info("TCP connections closed")
            
            # Flush and save data
            if hasattr(self, 'logger'):
                self.logger.flush()
                log.info("Feature data flushed")
            
            # Save model
            if hasattr(self, 'agent'):
                self.agent.save_model()
                log.info("Model saved")
            
            # Final performance report
            self._generate_shutdown_report()
            
        except Exception as e:
            log.error(f"Shutdown error: {e}")
        
        log.info("System shutdown complete")

    def _generate_shutdown_report(self):
        """Generate final performance report"""
        try:
            uptime = time.time() - self.start_time
            uptime_hours = uptime / 3600
            
            log.info("=== Shutdown Report ===")
            log.info(f"Total uptime: {uptime_hours:.2f} hours")
            
            # Get final status from handler
            if hasattr(self.handler, 'get_status_report'):
                final_status = self.handler.get_status_report()
                log.info(f"Total steps processed: {final_status.get('step_counter', 0)}")
                
                # Feature statistics
                if hasattr(self.logger, 'get_feature_statistics'):
                    stats = self.logger.get_feature_statistics()
                    if stats:
                        log.info(f"Final statistics: {stats}")
            
            # Signal handler final performance
            if hasattr(self.handler, 'dispatcher'):
                final_performance = self.handler.dispatcher.get_performance_summary()
                log.info("Final Signal Performance:")
                for line in final_performance.split('\n'):
                    if line.strip():
                        log.info(f"  {line.strip()}")
            
        except Exception as e:
            log.warning(f"Shutdown report generation failed: {e}")

    def get_system_health(self):
        """Get current system health status"""
        try:
            health = {
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'is_running': self.is_running,
                'tcp_connected': self.tcp.fsock is not None and self.tcp.ssock is not None,
                'model_loaded': self.agent.model is not None,
                'ready_for_trading': self.handler.trainer.is_ready_for_trading() if hasattr(self.handler, 'trainer') else False
            }
            
            return health
            
        except Exception as e:
            log.warning(f"Health check failed: {e}")
            return {'error': str(e)}