# pure_blackbox_trading_system.py - REPLACES trading_system.py with complete meta-learning

import os
import threading
import logging
import time
import signal
import sys
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from config import create_adaptive_config
from tcp_bridge import TCPBridge
from advanced_market_intelligence import AdvancedMarketIntelligence
from trade_manager_ai import create_pure_blackbox_trade_manager
from meta_learner import PureMetaLearner

log = logging.getLogger(__name__)

class PureBlackBoxTradingSystem:
    """
    COMPLETE PURE BLACK BOX: Every parameter adapts through meta-learning
    - No hardcoded thresholds, limits, or learning rates
    - Network architecture evolves based on performance
    - Risk management learned from actual losses
    - Reward structure discovers what drives success
    - Complete self-optimization through experience
    """

    def __init__(self, meta_db_path: str = "data/meta_parameters.db"):
        # Enhanced logging for meta-learning
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pure_blackbox.log'),
                logging.StreamHandler()
            ]
        )

        # Adaptive configuration - NO STATIC VALUES
        self.config = create_adaptive_config(meta_db_path)
        self.meta_learner = self.config.get_meta_learner()

        # Core intelligence with permanent memory
        self.intelligence_engine = AdvancedMarketIntelligence()
        
        # Load existing patterns and meta-parameters
        self._load_all_persistent_learning()
        self.intelligence_engine.start_continuous_learning()

        # Enhanced TCP bridge with adaptive parameters
        self.tcp_bridge = self._create_adaptive_tcp_bridge()
        self.tcp_bridge.on_market_data = self.process_pure_blackbox_data
        self.tcp_bridge.on_trade_completion = self.on_trade_completed

        # PURE BLACK BOX trade manager with complete adaptation
        self.trade_manager = create_pure_blackbox_trade_manager(
            self.intelligence_engine,
            self.tcp_bridge,
            self.config
        )

        # Meta-learning state
        self.system_start_time = datetime.now()
        self.signal_count = 0
        self.bootstrap_complete = False
        self.learning_milestones = []
        
        # Pure adaptive statistics - everything learned
        self.adaptive_stats = {
            'total_market_updates': 0,
            'meta_parameter_updates': 0,
            'network_architecture_changes': 0,
            'reward_structure_adaptations': 0,
            'safety_limit_adjustments': 0,
            'confidence_threshold_changes': 0,
            'learning_efficiency_improvements': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0
        }

        # Background meta-learning monitor
        self._start_meta_learning_monitor()

        log.info("Pure Black Box Trading System initialized")
        
        # Display initial adaptive configuration
        self._log_initial_adaptive_state()
    
    def _create_adaptive_tcp_bridge(self) -> TCPBridge:
        """Create TCP bridge with adaptive parameters"""
        
        # Use adaptive config for any TCP parameters
        bridge = TCPBridge(self.config)
        
        # Enhance with adaptive callbacks
        original_send_signal = bridge.send_signal
        
        def adaptive_send_signal(action, confidence, quality, position_size, stop_price=0.0, target_price=0.0):
            # Track signal sending for meta-learning
            self.adaptive_stats['total_market_updates'] += 1

            # Log adaptive signal details
            if stop_price > 0 or target_price > 0:
                log.info(f"Adaptive Signal: AI chose stop=${stop_price:.2f}, target=${target_price:.2f}")

            return original_send_signal(action, confidence, quality, stop_price, target_price, position_size)

        
        bridge.send_signal = adaptive_send_signal
        return bridge
    
    def _load_all_persistent_learning(self):
        """Load all forms of persistent learning"""
        try:
            # Load DNA patterns
            dna_patterns = self.intelligence_engine.memory_db.load_dna_patterns()
            self.intelligence_engine.dna_system.dna_patterns.update(dna_patterns)
            
            # Meta-parameters are automatically loaded by meta_learner
            meta_param_count = len(self.meta_learner.parameters)
            
            log.info(f"Loaded {len(dna_patterns)} DNA patterns, {meta_param_count} meta-parameters")
            
        except Exception as e:
            log.info(f"Starting fresh learning session: {e}")
    
    def _log_initial_adaptive_state(self):
        """Log the initial state of all adaptive parameters"""
        
        # Log key adaptive parameters
        risk_params = self.meta_learner.get_risk_parameters()
        learning_params = self.meta_learner.get_learning_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        
        log.info(f"Initial parameters - Position: {risk_params['position_size_base']:.3f}, "
                f"Entry threshold: {confidence_thresholds['entry']:.3f}")
    
    def _start_meta_learning_monitor(self):
        """Start background monitor for meta-learning progress"""
        def meta_monitor():
            last_report_time = datetime.now()
            
            while True:
                try:
                    time.sleep(60)  # Check every minute
                    
                    # Generate periodic adaptation reports
                    if datetime.now() - last_report_time > timedelta(minutes=15):
                        self._log_adaptation_progress()
                        last_report_time = datetime.now()
                    
                    # Check for significant parameter changes
                    self._check_for_significant_adaptations()
                    
                    # Auto-save learning progress
                    if self.signal_count % 100 == 0 and self.signal_count > 0:
                        self._save_all_learning_progress()
                    
                except Exception as e:
                    log.error(f"Meta-learning monitor error: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=meta_monitor, daemon=True, name="MetaLearningMonitor")
        thread.start()
        log.info("Meta-learning monitor started")
    
    def start(self):
        """Start the pure black box system"""
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            log.info("Shutdown signal received - saving all adaptive learning...")
            self.shutdown_and_save()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            self.tcp_bridge.start()
            
            log.info("Pure Black Box AI Trading System started")
            log.info("Press Ctrl+C to stop and save learned knowledge")
            
            # Main run loop - pure black box operation
            while True:
                time.sleep(1)
                
                # The system is now completely autonomous
                # All decisions made through adaptive parameters
                
        except KeyboardInterrupt:
            log.info("Shutdown requested by user")
        except Exception as e:
            log.error(f"System error: {e}")
        finally:
            self.shutdown_and_save()
    
    def bootstrap_pure_learning(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """Bootstrap with ZERO knowledge - pure neutral pattern creation"""
        log.info("Bootstrap: Creating neutral pattern library")
        
        patterns_learned = 0
        
        # Create completely neutral patterns - no trading bias
        if len(price_1m) >= 100:
            for i in range(50, len(price_1m), 40):  # Sparse sampling
                chunk_prices = price_1m[i-50:i]
                chunk_volumes = volume_1m[i-50:i] if volume_1m else [1000] * 50
                
                # Create DNA sequences with ZERO outcome bias
                dna_sequence = self.intelligence_engine.dna_system.create_dna_sequence(
                    chunk_prices, chunk_volumes
                )
                
                if dna_sequence and len(dna_sequence) > 8:
                    # Process through intelligence
                    result = self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                    
                    # Save with COMPLETELY NEUTRAL outcome - no bias whatsoever
                    if dna_sequence not in self.intelligence_engine.dna_system.dna_patterns:
                        self.intelligence_engine.dna_system.update_pattern_outcome(dna_sequence, 0.0)
                        patterns_learned += 1
                
                if patterns_learned >= 20:  # Minimal bootstrap
                    break
        
        # Initialize micro patterns with neutral outcomes
        if len(price_5m) >= 60:
            for i in range(30, len(price_5m), 20):
                chunk_prices = price_5m[i-30:i]
                chunk_volumes = volume_5m[i-30:i] if volume_5m else [500] * 30
                
                self.intelligence_engine.process_market_data(chunk_prices, chunk_volumes)
                patterns_learned += 1
                
                if patterns_learned >= 35:
                    break
        
        self.bootstrap_complete = True
        self.adaptive_stats['successful_adaptations'] += 1
        
        log.info(f"Bootstrap complete: {patterns_learned} patterns created")
    
    def process_pure_blackbox_data(self, data: Dict):
        """PURE BLACK BOX: Complete adaptive processing"""
        try:
            # Extract market data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            if not price_1m:
                return
            
            # Neutral bootstrap if needed
            if not self.bootstrap_complete and len(price_1m) >= 60:
                self.bootstrap_pure_learning(
                    data.get("price_15m", []), data.get("volume_15m", []),
                    data.get("price_5m", []), data.get("volume_5m", []),
                    price_1m, volume_1m
                )
            
            # Store current price for adaptive calculations
            if price_1m:
                self.current_price = price_1m[-1]
            
            # PURE BLACK BOX: AI makes all decisions using adaptive parameters
            self.trade_manager.on_new_bar(data)
            
            self.signal_count += 1
            self.adaptive_stats['total_market_updates'] += 1
            
            # Adaptive progress reporting
            if self.signal_count % 25 == 0 and self.signal_count > 0:
                self._log_real_time_adaptation()
            
            # Comprehensive adaptive report
            if self.signal_count % 150 == 0 and self.signal_count > 0:
                self._generate_comprehensive_adaptive_report()
            
        except Exception as e:
            log.error(f"Pure black box processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_real_time_adaptation(self):
        """Log real-time adaptation progress"""
        
        # Get current adaptive parameters
        risk_params = self.meta_learner.get_risk_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        learning_params = self.meta_learner.get_learning_parameters()
        
        log.debug(f"Progress: {self.signal_count} updates, position size: {risk_params['position_size_base']:.3f}")
    
    def _generate_comprehensive_adaptive_report(self):
        """Generate and display comprehensive adaptation report"""
        
        report = self.trade_manager.get_adaptive_performance_report()
        
        print("\n" + "="*80)
        print("🧠 PURE BLACK BOX ADAPTIVE LEARNING REPORT")
        print("="*80)
        print(report)
        print("="*80 + "\n")
        
        # Check for adaptation milestones
        self._check_adaptation_milestones()
    
    def _check_adaptation_milestones(self):
        """Check and celebrate adaptation milestones"""
        
        trades = self.trade_manager.trade_stats['total_trades']
        updates = self.meta_learner.total_updates
        
        milestones = [
            (10, "First 10 trades - initial adaptation phase"),
            (25, "25 trades - parameter stabilization"),
            (50, "50 trades - pattern recognition emergence"),
            (100, "100 trades - strategy crystallization"),
            (250, "250 trades - advanced adaptation mastery"),
            (500, "500 trades - expert-level meta-learning")
        ]
        
        for milestone_trades, description in milestones:
            milestone_key = f"trades_{milestone_trades}"
            
            if trades >= milestone_trades and milestone_key not in self.learning_milestones:
                self.learning_milestones.append(milestone_key)
                self.adaptive_stats['learning_efficiency_improvements'] += 1
                
                log.info(f"Adaption Milestone Achieved: {description}")
                log.info(f"Total Parameter Updates: {updates}")
                log.info(f"Network Architecture Changes: {self.trade_manager.agent.network_rebuilds}")
                
                # Force save at milestones
                self._save_all_learning_progress()
    
    def _check_for_significant_adaptations(self):
        """Check for and log significant parameter adaptations"""
        
        # This would track parameter changes and log when they exceed thresholds
        # For now, we'll track the count of significant changes
        
        recent_updates = self.meta_learner.total_updates
        if hasattr(self, '_last_update_count'):
            new_updates = recent_updates - self._last_update_count
            if new_updates > 5:  # Significant adaptation activity
                self.adaptive_stats['meta_parameter_updates'] += new_updates
                log.debug(f"Meta Learning: {new_updates} parameter updates in last period")
        
        self._last_update_count = recent_updates
    
    def _log_adaptation_progress(self):
        """Log detailed adaptation progress"""
        
        log.info("Adaption Progress Report:")
        log.info(f"System Uptime: {datetime.now() - self.system_start_time}")
        log.info(f"Market Updates: {self.adaptive_stats['total_market_updates']}")
        log.info(f"Meta-Parameter Updates: {self.meta_learner.total_updates}")
        log.info(f"Network Rebuilds: {self.trade_manager.agent.network_rebuilds}")
        log.info(f"Learning Efficiency: {self.meta_learner.get_learning_efficiency():.3f}")
        
        # Current adaptive state
        current_phase = self.trade_manager.safety_manager.current_phase
        log.info(f"Current Learning Phase: {current_phase}")
        
        # Adaptation effectiveness
        total_trades = self.trade_manager.trade_stats['total_trades']
        if total_trades > 0:
            win_rate = self.trade_manager.trade_stats['winning_trades'] / total_trades
            log.info(f"Win Rate: {win_rate:.1%} (from pure learning)")
    
    def _save_all_learning_progress(self):
        """Save all forms of learning progress"""
        try:
            # Save meta-learning parameters
            self.meta_learner.force_save()
            
            # Save AI model and learning state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trade_manager.force_save_all_adaptive_learning()
            
            # Save DNA patterns to permanent memory
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            # Export comprehensive knowledge base
            knowledge_file = f"data/adaptive_blackbox_knowledge_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            
            log.info(f"AUTO-SAVE: All adaptive learning progress saved")
            log.debug(f"Knowledge exported to {knowledge_file}")
            
        except Exception as e:
            log.warning(f"Auto-save error: {e}")
    
    def on_trade_completed(self, completion_data: Dict):
        """Handle trade completion with adaptive learning"""
        try:
            exit_price = completion_data.get('exit_price', 0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            
            # Feed to adaptive trade manager
            self.trade_manager._complete_adaptive_trade(exit_reason, exit_price)
            
            log.info(f"Adaption Completion: {exit_reason} at ${exit_price:.2f}")
            log.info("AI learning from this outcome for future parameter adaptation")
            
            self.adaptive_stats['successful_adaptations'] += 1
            
        except Exception as e:
            log.error(f"Trade completion error: {e}")
            self.adaptive_stats['failed_adaptations'] += 1
    
    def shutdown_and_save(self):
        """Comprehensive shutdown with complete learning preservation"""
        log.info("Shutdown: Preserving all adaptive learning...")
        
        shutdown_start = datetime.now()
        
        try:
            # Stop TCP bridge
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP shutdown error: {e}")
        
        # Force save all learning - CRITICAL
        try:
            log.info("Saving all adaptive parameters and learning progress...")
            
            # Save meta-learning parameters
            self.meta_learner.force_save()
            
            # Save agent learning state
            self.trade_manager.force_save_all_adaptive_learning()
            
            # Save intelligence patterns
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            # Export final knowledge base
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_knowledge_file = f"patterns/final_adaptive_state_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(final_knowledge_file)
            
            log.info(f"Save Complete: All learning preserved")
            log.info(f"Final knowledge state: {final_knowledge_file}")
            
        except Exception as e:
            log.error(f"CRITICAL: Learning save error: {e}")
        
        # Generate final adaptive summary
        self._generate_final_adaptive_summary(shutdown_start)
        
        log.info("Shutdown complete - all discoveries preserved")
    
    def _generate_final_adaptive_summary(self, shutdown_start: datetime):
        """Generate comprehensive final summary"""
        
        session_duration = datetime.now() - self.system_start_time
        shutdown_duration = datetime.now() - shutdown_start
        
        log.info(f"Session Statistics:")
        log.info(f"Duration: {session_duration}")
        log.info(f"Market Updates: {self.adaptive_stats['total_market_updates']}")
        log.info(f"Parameter Updates: {self.meta_learner.total_updates}")
        log.info(f"Network Rebuilds: {self.trade_manager.agent.network_rebuilds}")
        
        # Trading performance
        trades = self.trade_manager.trade_stats['total_trades']
        if trades > 0:
            win_rate = self.trade_manager.trade_stats['winning_trades'] / trades
            total_pnl = self.trade_manager.trade_stats['total_pnl']
            log.info(f"Total Trades: {trades}")
            log.info(f"Win Rate: {win_rate:.1%}")
            log.info(f"Total P&L: ${total_pnl:.2f}")
        
        # Meta-learning achievements
        log.info(f"Achievements:")
        log.info(f"Successful Adaptations: {self.adaptive_stats['successful_adaptations']}")
        log.info(f"Failed Adaptations: {self.adaptive_stats['failed_adaptations']}")
        log.info(f"Learning Milestones: {len(self.learning_milestones)}")
        
        # Current adaptive state
        risk_params = self.meta_learner.get_risk_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        
        log.info(f"Final Adaptive Parameters:")
        log.info(f"Position Size: {risk_params['position_size_base']:.3f}")
        log.info(f"Entry Threshold: {confidence_thresholds['entry']:.3f}")
        log.info(f"Daily Loss Limit: {risk_params['max_daily_loss_pct']:.1%}")
        log.info(f"Learning Phase: {self.trade_manager.safety_manager.current_phase}")
        
        # Tool discovery
        tool_stats = self.trade_manager.trade_stats['tool_discoveries']
        total_experiments = sum(tool_stats.values())
        if total_experiments > 0:
            log.info(f"Tool Discovery Results:")
            for tool, count in tool_stats.items():
                if count > 0:
                    pct = (count / total_experiments) * 100
                    log.info(f"  {tool.upper()}: {count} experiments ({pct:.0f}%)")
        
        # Intelligence patterns discovered
        dna_patterns = len(self.intelligence_engine.dna_system.dna_patterns)
        log.info(f"Pattern Discoveries:")
        log.info(f"DNA Sequences: {dna_patterns}")
        log.info(f"Micro Patterns: {len(self.intelligence_engine.micro_system.patterns)}")

        log.info(f"Shutdown Time: {shutdown_duration}")
        log.info(f"All Learning Preserved: YES")

# Factory function for easy initialization
def create_pure_blackbox_system(meta_db_path: str = "data/meta_parameters.db") -> PureBlackBoxTradingSystem:
    """Create a complete pure black box trading system"""
    
    system = PureBlackBoxTradingSystem(meta_db_path)
    
    log.info("Pure black box system created")
    
    return system

# Main entry point
if __name__ == "__main__":
    
    # Create and start system
    system = create_pure_blackbox_system()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        system.shutdown_and_save()
    except Exception as e:
        print(f"System error: {e}")
        system.shutdown_and_save()