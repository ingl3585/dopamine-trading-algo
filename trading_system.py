# FIXED trading_system.py - Complete integration with account data and proper learning correlation

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
    FIXED: Complete pure black box system with proper account data integration
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

        # Core intelligence with permanent memory and proper meta-learner integration
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
            'failed_adaptations': 0,
            'account_data_integrations': 0,  # FIXED: Track account data usage
            'position_sizing_calculations': 0  # FIXED: Track position sizing
        }

        # FIXED: Account data tracking
        self.current_account_data = {}
        self.account_data_history = []
        
        # Background meta-learning monitor
        self._start_meta_learning_monitor()

        log.info("Trading system initialized with complete account data integration")
        
        # Display initial adaptive configuration
        self._log_initial_adaptive_state()
    
    def _create_adaptive_tcp_bridge(self) -> TCPBridge:
        """FIXED: Create TCP bridge with enhanced account data processing"""
        
        bridge = TCPBridge(self.config)
        
        # Enhance with adaptive callbacks
        original_send_signal = bridge.send_signal
        
        def adaptive_send_signal(action, confidence, quality, stop_price=0.0, target_price=0.0, position_size=1.0, meta_learner=None):
            # Track signal sending for meta-learning
            self.adaptive_stats['total_market_updates'] += 1
            self.adaptive_stats['position_sizing_calculations'] += 1

            # Log adaptive signal details with account context
            if self.current_account_data:
                log.info(f"Adaptive signal with account context: Balance=${self.current_account_data.get('account_balance', 0):.0f}, "
                        f"BP=${self.current_account_data.get('buying_power', 0):.0f}, Size={position_size:.1f}")

            if stop_price > 0 or target_price > 0:
                log.info(f"AI risk management: Stop=${stop_price:.2f}, Target=${target_price:.2f}")

            return original_send_signal(action, confidence, quality, stop_price, target_price, position_size, meta_learner)
        
        bridge.send_signal = adaptive_send_signal
        return bridge
    
    def _load_all_persistent_learning(self):
        """FIXED: Load all forms of persistent learning with proper integration"""
        try:
            # Load DNA patterns
            dna_patterns = self.intelligence_engine.memory_db.load_dna_patterns()
            self.intelligence_engine.dna_system.dna_patterns.update(dna_patterns)
            
            # Meta-parameters are automatically loaded by meta_learner
            meta_param_count = len(self.meta_learner.parameters)
            
            log.info(f"Loaded {len(dna_patterns)} DNA patterns, {meta_param_count} meta-parameters")
            log.info("Account data integration enabled for position sizing")
            
        except Exception as e:
            log.info(f"Starting fresh learning session: {e}")
    
    def _log_initial_adaptive_state(self):
        """FIXED: Log the initial state with account data integration info"""
        
        # Log key adaptive parameters
        risk_params = self.meta_learner.get_risk_parameters()
        learning_params = self.meta_learner.get_learning_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        
        log.info(f"Initial adaptive parameters:")
        log.info(f"Position size base: {risk_params['position_size_base']:.3f}")
        log.info(f"Entry threshold: {confidence_thresholds['entry']:.3f}")
        log.info(f"Risk per trade: {risk_params['risk_per_trade_pct']:.3f}")
        log.info("Account data integration: ACTIVE for position sizing")
    
    def _start_meta_learning_monitor(self):
        """FIXED: Enhanced background monitor with account data tracking"""
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
                    
                    # FIXED: Monitor account data freshness
                    self._monitor_account_data_freshness()
                    
                except Exception as e:
                    log.error(f"Meta-learning monitor error: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=meta_monitor, daemon=True, name="MetaLearningMonitor")
        thread.start()
        log.info("Enhanced meta-learning monitor started with account data tracking")

    def _monitor_account_data_freshness(self):
        """FIXED: Monitor account data freshness for position sizing"""
        if self.current_account_data and 'timestamp' in self.current_account_data:
            age_seconds = time.time() - self.current_account_data['timestamp']
            if age_seconds > 300:  # 5 minutes old
                log.warning(f"Account data is {age_seconds:.0f}s old - position sizing may be stale")
    
    def start(self):
        """FIXED: Start the enhanced pure black box system"""

        # Setup graceful shutdown via OS signals only
        def signal_handler(signum, frame):
            log.info("Shutdown signal received - saving all adaptive learning including account data")
            self.shutdown_and_save()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Kick off the TCP bridge and then run forever; only the signal handler can exit
        self.tcp_bridge.start()

        log.info("Enhanced trading system started with account data integration")
        log.info("AI will use real account/margin data for position sizing")
        log.info("Press Ctrl+C to stop and save learned knowledge")

        while True:
            time.sleep(1)
    
    def bootstrap_pure_learning(self, price_15m, volume_15m, price_5m, volume_5m, price_1m, volume_1m):
        """FIXED: Bootstrap with ZERO knowledge - pure neutral pattern creation"""
        log.info("Bootstrap: Creating neutral pattern library with account data awareness")
        
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
        
        log.info(f"Bootstrap complete: {patterns_learned} patterns created with account data integration ready")
    
    def process_pure_blackbox_data(self, data: Dict):
        """FIXED: Complete adaptive processing with account data integration"""
        try:
            # Extract market data
            price_1m = data.get("price_1m", [])
            volume_1m = data.get("volume_1m", [])
            
            if not price_1m:
                return
            
            # FIXED: Extract and validate account data
            account_data = {
                'buying_power': data.get('buying_power', 25000),
                'account_balance': data.get('account_balance', 25000),
                'daily_pnl': data.get('daily_pnl', 0.0),
                'cash_value': data.get('cash_value', 25000),
                'excess_liquidity': data.get('excess_liquidity', 25000),
                'net_liquidation': data.get('net_liquidation', 25000),
                'timestamp': time.time()  # Add timestamp for freshness monitoring
            }
            
            # Update current account data
            self.current_account_data = account_data
            self.account_data_history.append(account_data.copy())
            
            # Keep only recent account history
            if len(self.account_data_history) > 100:
                self.account_data_history = self.account_data_history[-100:]
            
            self.adaptive_stats['account_data_integrations'] += 1
            
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
            
            # FIXED: Pass account data to trade manager
            self.trade_manager.on_new_bar(data, account_data)
            
            self.signal_count += 1
            self.adaptive_stats['total_market_updates'] += 1
            
            # Enhanced progress reporting with account data
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
        """FIXED: Enhanced real-time adaptation progress with account data"""
        
        # Get current adaptive parameters
        risk_params = self.meta_learner.get_risk_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        
        # Log with account context
        account_info = ""
        if self.current_account_data:
            balance = self.current_account_data.get('account_balance', 0)
            buying_power = self.current_account_data.get('buying_power', 0)
            daily_pnl = self.current_account_data.get('daily_pnl', 0)
            account_info = f", Account: ${balance:.0f}, BP: ${buying_power:.0f}, Daily: ${daily_pnl:.2f}"
        
        log.info(f"Progress: {self.signal_count} updates, position size: {risk_params['position_size_base']:.3f}{account_info}")
    
    def _generate_comprehensive_adaptive_report(self):
        """FIXED: Enhanced comprehensive adaptation report with account data insights"""
        
        report = self.trade_manager.get_adaptive_performance_report()
        
        # Add account data analysis
        if self.account_data_history:
            recent_accounts = self.account_data_history[-10:]
            avg_balance = np.mean([a['account_balance'] for a in recent_accounts])
            avg_daily_pnl = np.mean([a['daily_pnl'] for a in recent_accounts])
            
            account_analysis = f"""
ACCOUNT DATA INTEGRATION ANALYSIS:
  Average Account Balance: ${avg_balance:.0f}
  Average Daily P&L: ${avg_daily_pnl:.2f}
  Account Data Updates: {self.adaptive_stats['account_data_integrations']}
  Position Sizing Calculations: {self.adaptive_stats['position_sizing_calculations']}
"""
            report += account_analysis
        
        print("\n" + "="*80)
        print("🧠 PURE BLACK BOX WITH ACCOUNT DATA INTEGRATION")
        print("="*80)
        print(report)
        print("="*80 + "\n")
        
        # Check for adaptation milestones
        self._check_adaptation_milestones()
    
    def _check_adaptation_milestones(self):
        """FIXED: Enhanced adaptation milestones with account data integration"""
        
        signals = self.trade_manager.signal_stats['total_signals']
        updates = self.meta_learner.total_updates
        account_integrations = self.adaptive_stats['account_data_integrations']
        
        milestones = [
            (10, "First 10 signals - initial account data integration"),
            (25, "25 signals - account-based position sizing stabilization"),
            (50, "50 signals - pattern recognition with account awareness"),
            (100, "100 signals - advanced account-based strategy crystallization"),
            (250, "250 signals - expert-level account data integration"),
            (500, "500 signals - master-level adaptive account management")
        ]
        
        for milestone_signals, description in milestones:
            milestone_key = f"signals_{milestone_signals}"
            
            if signals >= milestone_signals and milestone_key not in self.learning_milestones:
                self.learning_milestones.append(milestone_key)
                self.adaptive_stats['learning_efficiency_improvements'] += 1
                
                log.info(f"Account-integrated adaptation milestone: {description}")
                log.info(f"Total parameter updates: {updates}")
                log.info(f"Account data integrations: {account_integrations}")
                log.info(f"Network architecture changes: {self.trade_manager.agent.network_rebuilds}")
                
                # Force save at milestones
                self._save_all_learning_progress()
    
    def _check_for_significant_adaptations(self):
        """FIXED: Enhanced adaptation tracking with account data awareness"""
        
        recent_updates = self.meta_learner.total_updates
        if hasattr(self, '_last_update_count'):
            new_updates = recent_updates - self._last_update_count
            if new_updates > 5:  # Significant adaptation activity
                self.adaptive_stats['meta_parameter_updates'] += new_updates
                log.debug(f"Meta Learning with account data: {new_updates} parameter updates in last period")
        
        self._last_update_count = recent_updates
    
    def _log_adaptation_progress(self):
        """FIXED: Enhanced adaptation progress with account data insights"""
        
        log.info("Enhanced adaptation progress report:")
        log.info(f"System uptime: {datetime.now() - self.system_start_time}")
        log.info(f"Market updates with account data: {self.adaptive_stats['total_market_updates']}")
        log.info(f"Account data integrations: {self.adaptive_stats['account_data_integrations']}")
        log.info(f"Position sizing calculations: {self.adaptive_stats['position_sizing_calculations']}")
        log.info(f"Meta parameter updates: {self.meta_learner.total_updates}")
        log.info(f"Network rebuilds: {self.trade_manager.agent.network_rebuilds}")
        log.info(f"Learning efficiency: {self.meta_learner.get_learning_efficiency():.3f}")
        
        # Current adaptive state with account context
        current_phase = self.trade_manager.safety_manager.current_phase
        log.info(f"Current learning phase: {current_phase}")
        
        if self.current_account_data:
            log.info(f"Current account balance: ${self.current_account_data.get('account_balance', 0):.0f}")
            log.info(f"Current buying power: ${self.current_account_data.get('buying_power', 0):.0f}")
        
        # Adaptation effectiveness
        total_signals = self.trade_manager.signal_stats['total_signals']
        if total_signals > 0:
            success_rate = self.trade_manager.signal_stats['successful_signals'] / total_signals
            log.info(f"Signal success rate: {success_rate:.1%}")
    
    def _save_all_learning_progress(self):
        """FIXED: Enhanced save with account data integration tracking"""
        try:
            # Save meta-learning parameters
            self.meta_learner.force_save()
            
            # Save AI model and learning state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.trade_manager.force_save_all_adaptive_learning()
            
            # Save DNA patterns to permanent memory
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            # Export comprehensive knowledge base with account data insights
            knowledge_file = f"data/adaptive_blackbox_with_account_data_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(knowledge_file)
            
            # Save account data integration statistics
            account_data_stats = {
                'total_account_integrations': self.adaptive_stats['account_data_integrations'],
                'position_sizing_calculations': self.adaptive_stats['position_sizing_calculations'],
                'current_account_data': self.current_account_data,
                'account_data_history_length': len(self.account_data_history)
            }
            
            import json
            with open(f"data/account_integration_stats_{timestamp}.json", 'w') as f:
                json.dump(account_data_stats, f, indent=2)
            
            log.info(f"All adaptive learning with account data integration saved")
            log.debug(f"Knowledge exported to {knowledge_file}")
            
        except Exception as e:
            log.warning(f"Auto-save error: {e}")
    
    def on_trade_completed(self, completion_data: Dict):
        """Handle trade completion with adaptive learning"""
        try:
            # Extract completion data
            final_pnl = completion_data.get('final_pnl', 0.0)
            exit_price = completion_data.get('exit_price', 0.0)
            exit_reason = completion_data.get('exit_reason', 'unknown')
            duration_minutes = completion_data.get('duration_minutes', 0)
            tool_used = completion_data.get('tool_used', 'unknown')
            
            # CRITICAL: Convert to format expected by trade manager
            formatted_outcome = {
                'signal_timestamp': completion_data.get('signal_timestamp', 0),
                'final_pnl': final_pnl,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'duration_minutes': duration_minutes,
                'entry_price': completion_data.get('entry_price', 0.0),
                'tool_used': tool_used,
                'used_ai_stop': completion_data.get('used_ai_stop', False),
                'used_ai_target': completion_data.get('used_ai_target', False)
            }
            
            # Feed to trade manager for agent learning
            self.trade_manager.learn_from_execution_outcome(formatted_outcome)
            
            # Also feed to intelligence engine for pattern learning
            if hasattr(self, 'intelligence_engine') and tool_used != 'unknown':
                # Find the corresponding DNA sequence if available
                current_dna = completion_data.get('dna_sequence', '')
                if current_dna:
                    self.intelligence_engine.dna_system.update_pattern_outcome(current_dna, final_pnl)
            
            log.info(f"Adaptive trade completion learned:")
            log.info(f"Tool: {tool_used}, P&L: ${final_pnl:.2f}, Exit: {exit_reason}")
            
            self.adaptive_stats['successful_adaptations'] += 1
            
        except Exception as e:
            log.error(f"Trade completion learning error: {e}")
            self.adaptive_stats['failed_adaptations'] += 1
    
    def shutdown_and_save(self):
        """FIXED: Enhanced shutdown with complete learning preservation including account data"""
        log.info("Shutdown: Preserving all adaptive learning including account data integration")
        
        shutdown_start = datetime.now()
        
        try:
            # Stop TCP bridge
            self.tcp_bridge.stop()
        except Exception as e:
            log.warning(f"TCP shutdown error: {e}")
        
        # Force save all learning - CRITICAL
        try:
            log.info("Saving all adaptive parameters and account data integration learning")
            
            # Save meta-learning parameters
            self.meta_learner.force_save()
            
            # Save agent learning state
            self.trade_manager.force_save_all_adaptive_learning()
            
            # Save intelligence patterns
            for seq, pattern in self.intelligence_engine.dna_system.dna_patterns.items():
                self.intelligence_engine.memory_db.save_dna_pattern(pattern)
            
            # Export final knowledge base with account data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_knowledge_file = f"patterns/final_adaptive_state_with_account_data_{timestamp}.json"
            self.intelligence_engine.export_knowledge_base(final_knowledge_file)
            
            # Save final account data integration summary
            final_account_summary = {
                'total_account_integrations': self.adaptive_stats['account_data_integrations'],
                'position_sizing_calculations': self.adaptive_stats['position_sizing_calculations'],
                'final_account_data': self.current_account_data,
                'account_data_history_count': len(self.account_data_history),
                'session_start_time': self.system_start_time.isoformat(),
                'session_end_time': datetime.now().isoformat()
            }
            
            import json
            with open(f"patterns/final_account_integration_summary_{timestamp}.json", 'w') as f:
                json.dump(final_account_summary, f, indent=2)
            
            log.info(f"Save complete: All learning including account data preserved")
            log.info(f"Final knowledge state: {final_knowledge_file}")
            
        except Exception as e:
            log.error(f"CRITICAL: Learning save error: {e}")
        
        # Generate final adaptive summary
        self._generate_final_adaptive_summary(shutdown_start)
        
        log.info("Enhanced shutdown complete - all discoveries including account data preserved")
    
    def _generate_final_adaptive_summary(self, shutdown_start: datetime):
        """FIXED: Enhanced final summary with account data integration insights"""
        
        session_duration = datetime.now() - self.system_start_time
        shutdown_duration = datetime.now() - shutdown_start
        
        log.info(f"Enhanced session statistics:")
        log.info(f"Duration: {session_duration}")
        log.info(f"Market updates with account data: {self.adaptive_stats['total_market_updates']}")
        log.info(f"Account data integrations: {self.adaptive_stats['account_data_integrations']}")
        log.info(f"Position sizing calculations: {self.adaptive_stats['position_sizing_calculations']}")
        log.info(f"Parameter updates: {self.meta_learner.total_updates}")
        log.info(f"Network rebuilds: {self.trade_manager.agent.network_rebuilds}")
        
        # Trading performance with account context
        signals = self.trade_manager.signal_stats['total_signals']
        if signals > 0:
            success_rate = self.trade_manager.signal_stats['successful_signals'] / signals
            total_pnl = self.trade_manager.signal_stats['total_pnl_from_signals']
            log.info(f"Total signals: {signals}")
            log.info(f"Success rate: {success_rate:.1%}")
            log.info(f"Total P&L: ${total_pnl:.2f}")
        
        # Account data integration insights
        if self.account_data_history:
            initial_balance = self.account_data_history[0]['account_balance']
            final_balance = self.current_account_data.get('account_balance', initial_balance)
            balance_change = final_balance - initial_balance
            
            log.info(f"Account integration insights:")
            log.info(f"Initial balance: ${initial_balance:.0f}")
            log.info(f"Final balance: ${final_balance:.0f}")
            log.info(f"Balance change: ${balance_change:.2f}")
        
        # Meta-learning achievements
        log.info(f"Adaptive achievements:")
        log.info(f"Successful adaptations: {self.adaptive_stats['successful_adaptations']}")
        log.info(f"Failed adaptations: {self.adaptive_stats['failed_adaptations']}")
        log.info(f"Learning milestones: {len(self.learning_milestones)}")
        
        # Current adaptive state
        risk_params = self.meta_learner.get_risk_parameters()
        confidence_thresholds = self.meta_learner.get_confidence_thresholds()
        
        log.info(f"Final adaptive parameters:")
        log.info(f"Position size base: {risk_params['position_size_base']:.3f}")
        log.info(f"Risk per trade: {risk_params['risk_per_trade_pct']:.3f}")
        log.info(f"Entry threshold: {confidence_thresholds['entry']:.3f}")
        log.info(f"Daily loss limit: {risk_params['max_daily_loss_pct']:.1%}")
        log.info(f"Learning phase: {self.trade_manager.safety_manager.current_phase}")
        
        # Tool discovery with account awareness
        tool_stats = self.trade_manager.signal_stats['tool_discoveries']
        total_experiments = sum(tool_stats.values())
        if total_experiments > 0:
            log.info(f"Tool discovery with account data:")
            for tool, count in tool_stats.items():
                if count > 0:
                    pct = (count / total_experiments) * 100
                    log.info(f"  {tool.upper()}: {count} experiments ({pct:.0f}%)")
        
        # Intelligence patterns discovered
        dna_patterns = len(self.intelligence_engine.dna_system.dna_patterns)
        log.info(f"Pattern discoveries:")
        log.info(f"DNA sequences: {dna_patterns}")
        log.info(f"Micro patterns: {len(self.intelligence_engine.micro_system.patterns)}")

        log.info(f"Enhanced shutdown time: {shutdown_duration}")
        log.info("Account data integration: COMPLETE")

# Factory function for easy initialization
def create_pure_blackbox_system(meta_db_path: str = "data/meta_parameters.db") -> PureBlackBoxTradingSystem:
    """Create a complete pure black box trading system with account data integration"""
    
    system = PureBlackBoxTradingSystem(meta_db_path)
    
    return system

# Main entry point
if __name__ == "__main__":
    
    # Create and start enhanced system
    system = create_pure_blackbox_system()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        system.shutdown_and_save()
    except Exception as e:
        print(f"System error: {e}")
        system.shutdown_and_save()