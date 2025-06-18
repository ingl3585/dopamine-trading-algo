# trade_manager_ai.py - REFACTORED: Pure signal generation, position management to NinjaTrader

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
from rl_agent import PureBlackBoxStrategicAgent
from market_env import MarketEnv
from collections import deque
from meta_learner import PureMetaLearner, AdaptiveRewardLearner

log = logging.getLogger(__name__)

class AdaptiveSafetyManager:
    """PURE adaptive safety - no hardcoded limits, everything learned from losses"""
    
    def __init__(self, meta_learner, config):
        self.meta_learner = meta_learner
        self.config = config
        
        # Add trade frequency as a meta-learned parameter
        if 'trade_frequency_multiplier' not in meta_learner.parameters:
            meta_learner.parameters['trade_frequency_multiplier'] = 1.0
            meta_learner.parameter_gradients['trade_frequency_multiplier'] = 0.0
            meta_learner.parameter_outcomes['trade_frequency_multiplier'] = deque(maxlen=200)
        
        self.current_phase = 'exploration'
        
        # Dynamic tracking - NO POSITION TRACKING HERE
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.signals_today = 0  # Changed from trades_today
        self.last_date = datetime.now().date()
        
        # Adaptive learning from losses
        self.loss_history = []
        self.phase_performance_history = {
            'exploration': [],
            'development': [], 
            'production': []
        }
    
    def can_generate_signal(self) -> bool:
        """Adaptive signal generation permission - NO position checks"""
        
        # Reset daily counters
        current_date = datetime.now().date()
        if current_date != self.last_date:
            if self.signals_today > 0:
                daily_performance = self.daily_pnl / max(1, self.signals_today)
                self.meta_learner.update_parameter('max_daily_loss_pct', daily_performance / 100.0)
                
                if self.daily_pnl > 0:
                    self.meta_learner.update_parameter('trade_frequency_multiplier', 0.05)
                else:
                    self.meta_learner.update_parameter('trade_frequency_multiplier', -0.02)
            
            self.daily_pnl = 0.0
            self.signals_today = 0
            self.last_date = current_date
        
        # Get adaptive limits from meta-learner
        risk_params = self.meta_learner.get_risk_parameters()
        
        max_daily_loss = risk_params['max_daily_loss_pct'] * 10000
        max_consecutive = risk_params['max_consecutive_losses']
        
        if self.daily_pnl <= -max_daily_loss:
            log.warning(f"ADAPTIVE SAFETY: Daily loss limit hit (${self.daily_pnl:.2f})")
            self.meta_learner.update_parameter('max_daily_loss_pct', -0.5)
            return False
        
        if self.consecutive_losses >= max_consecutive:
            log.warning(f"ADAPTIVE SAFETY: Consecutive loss limit hit ({self.consecutive_losses})")
            self.meta_learner.update_parameter('max_consecutive_losses', -0.3)
            return False
        
        # Use config signal limits
        if hasattr(self.config, 'MAX_DAILY_TRADES_EXPLORATION'):
            phase_limits = {
                'exploration': self.config.MAX_DAILY_TRADES_EXPLORATION,
                'development': self.config.MAX_DAILY_TRADES_DEVELOPMENT, 
                'production': self.config.MAX_DAILY_TRADES_PRODUCTION
            }
            base_max_signals = phase_limits[self.current_phase]
            
            performance_factor = max(0.5, 1.0 + (self.daily_pnl / 100.0))
            frequency_multiplier = self.meta_learner.get_parameter('trade_frequency_multiplier')
            
            max_signals_today = int(base_max_signals * performance_factor * frequency_multiplier)
            
            if self.signals_today >= max_signals_today:
                log.info(f"Adaptive signal limit: {self.signals_today} >= {max_signals_today}")
                return False
        
        return True
    
    def record_signal_outcome(self, pnl: float):
        """Record signal outcome and adapt"""
        self.daily_pnl += pnl
        self.signals_today += 1
        self.loss_history.append(pnl)
        
        if len(self.loss_history) > 50:
            self.loss_history = self.loss_history[-30:]
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if len(self.loss_history) >= 5:
            recent_performance = np.mean(self.loss_history[-5:])
            
            if recent_performance > 10:
                frequency_signal = 0.05
            elif recent_performance < -20:
                frequency_signal = -0.1
            else:
                frequency_signal = 0.01 if pnl > 0 else -0.02
            
            self.meta_learner.update_parameter('trade_frequency_multiplier', frequency_signal)
        
        self._adapt_learning_phase()
        
        if abs(pnl) > 20:
            normalized_outcome = np.tanh(pnl / 50.0)
            self.meta_learner.update_parameter('position_size_base', normalized_outcome)
    
    def _adapt_learning_phase(self):
        """Adapt learning phase based on actual performance"""
        if len(self.loss_history) < 10:
            return
        
        recent_performance = self.loss_history[-10:]
        win_rate = sum(1 for pnl in recent_performance if pnl > 0) / len(recent_performance)
        avg_pnl = np.mean(recent_performance)
        consistency = 1.0 - (np.std(recent_performance) / 50.0)
        
        readiness_score = (win_rate * 0.4) + (max(0, avg_pnl / 50.0) * 0.4) + (consistency * 0.2)
        self.phase_performance_history[self.current_phase].append(readiness_score)
        
        if self.current_phase == 'exploration':
            if readiness_score > 0.4 and avg_pnl > 5:
                self.current_phase = 'development'
                log.info(f"Adaptive progression: Advanced to development phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', 0.2)
        
        elif self.current_phase == 'development':
            if readiness_score > 0.6 and avg_pnl > 10 and win_rate > 0.5:
                self.current_phase = 'production'
                log.info(f"Adaptive progression: Advanced to production phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', 0.3)
            
            elif readiness_score < 0.2 and avg_pnl < -10:
                self.current_phase = 'exploration'
                log.warning(f"Adaptive regression: Dropped to exploration phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', -0.3)
        
        elif self.current_phase == 'production':
            if readiness_score < 0.3 or avg_pnl < -15:
                self.current_phase = 'development'
                log.warning(f"ADAPTIVE REGRESSION: Dropped to development phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', -0.2)

    def get_position_size(self, account_data: Dict = None, current_price: float = 4000.0) -> float:
        """FIXED: Return adaptive position size based on account data when available"""
        
        # If account data is available, use account-based sizing (PROMPT REQUIREMENT)
        if account_data and 'buying_power' in account_data:
            return self.calculate_account_based_position_size(
                account_data, current_price, self.current_phase
            )
        
        # Fallback to phase-based sizing with adaptive multipliers
        base_sizes = {
            'exploration': self.config.EXPLORATION_PHASE_SIZE,
            'development': self.config.DEVELOPMENT_PHASE_SIZE,
            'production': self.config.PRODUCTION_PHASE_SIZE
        }
        
        base_size = base_sizes.get(self.current_phase, 1.0)
        
        # Apply adaptive multiplier
        adaptive_multiplier = self.meta_learner.get_parameter('position_size_base')
        
        return max(1.0, base_size * adaptive_multiplier)

class PureBlackBoxSignalGenerator:
    """
    REFACTORED: Pure signal generation with NO position tracking
    
    What this does:
    - Generate trading signals using all existing subsystems
    - Learn from trade outcomes streamed back from NinjaTrader
    - Adapt all meta-parameters based on real execution results
    
    What this does NOT do:
    - Track position sizes or entry prices
    - Calculate P&L
    - Manage account balance
    - Execute trades or manage orders
    """

    def __init__(self, intelligence_engine, tcp_bridge, config):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        self.config = config
        
        # Pure black box agent with meta-learning
        self.agent = PureBlackBoxStrategicAgent(
            market_obs_size=15,
            subsystem_features_size=16
        )
        
        # Access to meta-learner
        self.meta_learner = self.agent.meta_learner
        self.reward_learner = self.agent.reward_learner
        
        # Adaptive safety manager
        self.safety_manager = AdaptiveSafetyManager(self.meta_learner, self.config)
        
        # Environment for state tracking
        self.env = MarketEnv()
        
        # State tracking for signal generation only
        self.last_market_obs = self.env.get_obs()
        self.last_subsystem_features = None
        self.last_decision = {}
        
        # Signal tracking for learning - NO POSITION TRACKING
        self.pending_signals = {}  # Signals waiting for outcomes
        self.signal_counter = 0
        
        # Pure statistics
        self.signal_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'total_pnl_from_signals': 0.0,
            'tool_discoveries': {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0},
            'adaptive_features_used': {
                'dynamic_stops': 0,
                'dynamic_targets': 0,
                'confidence_based_signals': 0,
                'meta_learned_entries': 0
            }
        }
        
        log.info("Signal generator initialized")

    def on_new_bar(self, msg: Dict[str, Any]):
        """Pure black box signal generation with complete adaptation"""
        try:
            # Adaptive safety check
            if not self.safety_manager.can_generate_signal():
                return
            
            # Extract price data
            price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
            prices = msg.get("price_1m", [price])
            volumes = msg.get("volume_1m", [1000])
            
            # Update environment state
            market_obs = self.env.get_obs()
            self.env.step(price, 0)
            
            # Process with intelligence systems
            intelligence_result = self.intel.process_market_data(prices, volumes, datetime.now())
            subsystem_features = self.extract_adaptive_subsystem_features(intelligence_result)
            
            # Pure black box decision making
            decision = self.agent.select_action_and_strategy(
                market_obs=market_obs,
                subsystem_features=subsystem_features,
                current_price=price,
                in_position=False  # We don't track positions
            )
            
            # Store decision data for learning
            self.last_decision = decision
            self.last_market_obs = market_obs
            self.last_subsystem_features = subsystem_features
            
            # Generate signal if AI wants to trade
            if decision['action'] != 0:
                self._generate_clean_signal(decision, price, intelligence_result)
            
        except Exception as e:
            log.error(f"Pure signal generation error: {e}")
    
    def calculate_account_based_position_size(self, account_data: Dict, current_price: float, 
                                            current_phase: str) -> float:
        """Calculate position size based on account/margin data as required by prompt"""
        
        # Extract account information
        available_capital = account_data.get('buying_power', 25000)
        account_balance = account_data.get('account_balance', 25000)
        daily_pnl = account_data.get('daily_pnl', 0.0)
        
        # Get adaptive risk parameters
        risk_per_trade_pct = self.meta_learner.get_parameter('risk_per_trade_pct')
        position_size_base = self.meta_learner.get_parameter('position_size_base')
        
        # Calculate base position size from available capital
        risk_amount = available_capital * risk_per_trade_pct
        
        # MNQ futures specifics
        estimated_margin_per_contract = 50.0  # Approximate margin requirement
        point_value = 2.0  # $2 per point for MNQ
        
        # Maximum contracts by margin
        max_contracts_by_margin = available_capital / estimated_margin_per_contract
        
        # Phase-based adjustments (adaptive)
        phase_multipliers = {
            'exploration': 0.3,
            'development': 0.6, 
            'production': 1.0
        }
        phase_multiplier = phase_multipliers.get(current_phase, 0.5)
        
        # Account performance adjustment
        if account_balance > 0:
            performance_factor = 1.0 + min(0.5, max(-0.5, daily_pnl / account_balance))
        else:
            performance_factor = 0.5
        
        # Calculate final position size
        base_contracts = (risk_amount / (current_price * point_value)) * position_size_base
        
        final_size = base_contracts * phase_multiplier * performance_factor
        
        # Safety bounds
        final_size = max(1, min(final_size, max_contracts_by_margin * 0.8))  # Never use more than 80% of margin
        final_size = min(final_size, 10)  # Absolute safety cap
        
        return int(final_size)

    def _generate_clean_signal(self, decision: Dict, current_price: float, intelligence_result: Dict):
        """Generate clean signal for NinjaTrader execution"""
        
        action = decision['action']
        confidence = decision['confidence']
        tool_name = decision['primary_tool']
        
        # AI calculates position size
        ai_position_size = self.safety_manager.get_position_size()
        
        # Build signal data
        signal_data = {
            "action": action,
            "confidence": confidence,
            "tool_used": tool_name,
            "position_size": ai_position_size,
            "ai_suggested_stop": decision.get('stop_price', 0.0),
            "ai_suggested_target": decision.get('target_price', 0.0),
            "use_stop": decision.get('use_stop', False),
            "use_target": decision.get('use_target', False),
            "reasoning": self._generate_adaptive_reasoning(decision, intelligence_result),
            "timestamp": datetime.now().timestamp()
        }
        
        # Send to NinjaTrader with AI position size
        self.tcp_bridge.send_signal(
            action=action,
            confidence=confidence,
            quality=f"{tool_name}_adaptive",
            stop_price=decision.get('stop_price', 0.0),
            target_price=decision.get('target_price', 0.0),
            position_size=ai_position_size
        )
        
        # Store signal for learning
        self.signal_counter += 1
        signal_id = f"signal_{self.signal_counter}"
        
        self.pending_signals[signal_id] = {
            'signal_data': signal_data,
            'decision': decision,
            'intelligence_result': intelligence_result,
            'timestamp': datetime.now(),
            'outcome_received': False
        }
        
        # Update stats
        self.signal_stats['total_signals'] += 1
        self.signal_stats['tool_discoveries'][tool_name] += 1
        self.signal_stats['adaptive_features_used']['meta_learned_entries'] += 1
        
        if decision.get('use_stop'):
            self.signal_stats['adaptive_features_used']['dynamic_stops'] += 1
        if decision.get('use_target'):
            self.signal_stats['adaptive_features_used']['dynamic_targets'] += 1
        
        log.info(f"Signal #{self.signal_counter}: {tool_name.upper()} {['EXIT', 'BUY', 'SELL'][action]}")
        log.info(f"Confidence: {confidence:.3f}")
        log.info(f"AI position size: {ai_position_size:.2f}")
        log.info(f"AI stop: ${decision.get('stop_price', 0):.2f}" if decision.get('use_stop') else "   No AI Stop")
        log.info(f"AI target: ${decision.get('target_price', 0):.2f}" if decision.get('use_target') else "   No AI Target")
        
        # Clean up old pending signals
        self._cleanup_old_pending_signals()

    def _generate_adaptive_reasoning(self, decision: Dict, intelligence_result: Dict) -> str:
        """Generate reasoning using existing code patterns"""
        
        primary_tool = decision.get('primary_tool', 'unknown')
        confidence = decision['confidence']
        exploration = decision.get('exploration_taken', False)
        
        reasoning_parts = []
        
        if exploration:
            reasoning_parts.append(f"EXPLORATION: Random {primary_tool.upper()} tool selection")
        else:
            reasoning_parts.append(f"LEARNED: {primary_tool.upper()} tool")
        
        threshold_used = decision.get('entry_threshold_used', 0.5)
        if confidence >= threshold_used:
            margin = confidence - threshold_used
            reasoning_parts.append(f"High confidence ({confidence:.2f} > {threshold_used:.2f})")
        else:
            reasoning_parts.append(f"Marginal confidence ({confidence:.2f})")
        
        # Add meta-learning insight
        learning_efficiency = self.meta_learner.get_learning_efficiency()
        if learning_efficiency > 0.5:
            reasoning_parts.append("FAST LEARNING MODE")
        elif learning_efficiency < -0.2:
            reasoning_parts.append("ADAPTATION MODE")
        
        return " | ".join(reasoning_parts)

    def learn_from_execution_outcome(self, outcome_data: Dict):
        """Learn from NinjaTrader execution outcome"""
        try:
            # Extract outcome data
            signal_timestamp = outcome_data.get('signal_timestamp')
            final_pnl = outcome_data.get('final_pnl', 0.0)
            exit_reason = outcome_data.get('exit_reason', 'unknown')
            duration_minutes = outcome_data.get('duration_minutes', 0)
            
            # Find matching signal
            matching_signal = None
            for signal_id, signal_data in self.pending_signals.items():
                if abs(signal_data['timestamp'].timestamp() - signal_timestamp) < 30:  # 30 sec tolerance
                    matching_signal = signal_data
                    break
            
            if not matching_signal:
                log.warning(f"No matching signal found for outcome")
                return
            
            # Safety manager learning
            self.safety_manager.record_signal_outcome(final_pnl)
            
            # Create trade outcome for agent learning
            trade_outcome = {
                'pnl': final_pnl,
                'hold_time_hours': duration_minutes / 60.0,
                'used_stop': matching_signal['decision'].get('use_stop', False),
                'used_target': matching_signal['decision'].get('use_target', False),
                'tool_confidence': matching_signal['decision'].get('confidence', 0.5),
                'primary_tool': matching_signal['decision'].get('primary_tool', 'unknown'),
                'exit_reason': exit_reason,
                'exploration_taken': matching_signal['decision'].get('exploration_taken', False),
                'meta_params_used': matching_signal['decision'].get('meta_parameters_used', {})
            }
            
            # Let agent learn from outcome
            self.agent.store_experience_and_learn(matching_signal['decision'], trade_outcome)
            
            # Update signal stats
            if final_pnl > 0:
                self.signal_stats['successful_signals'] += 1
            else:
                self.signal_stats['failed_signals'] += 1
            
            self.signal_stats['total_pnl_from_signals'] += final_pnl
            
            log.info(f"Signal outcome learned:")
            log.info(f"P&L: ${final_pnl:.2f}")
            log.info(f"Tool: {trade_outcome['primary_tool'].upper()}")
            log.info(f"Exit: {exit_reason}")
            log.info(f"Duration: {duration_minutes}min")
            
            # Mark as processed
            matching_signal['outcome_received'] = True
            
        except Exception as e:
            log.error(f"Error learning from outcome: {e}")

    def extract_adaptive_subsystem_features(self, intelligence_result: Dict) -> np.ndarray:
        """Use existing feature extraction method"""
        features = []
        
        subsystem_signals = intelligence_result.get('subsystem_signals', {})
        subsystem_scores = intelligence_result.get('subsystem_scores', {})
        
        # DNA features
        dna_signal = subsystem_signals.get('dna', 0.0)
        dna_confidence = subsystem_scores.get('dna', 0.0)
        dna_patterns = min(intelligence_result.get('similar_patterns_count', 0) / 10.0, 1.0)
        dna_length = min(intelligence_result.get('dna_sequence_length', 0) / 20.0, 1.0)
        features.extend([dna_signal, dna_confidence, dna_patterns, dna_length])
        
        # Micro features
        micro_signal = subsystem_signals.get('micro', 0.0)
        micro_confidence = subsystem_scores.get('micro', 0.0)
        micro_strength = abs(micro_signal) * micro_confidence
        micro_active = 1.0 if intelligence_result.get('micro_pattern_id') else 0.0
        features.extend([micro_signal, micro_confidence, micro_strength, micro_active])
        
        # Temporal features
        temporal_signal = subsystem_signals.get('temporal', 0.0)
        temporal_confidence = subsystem_scores.get('temporal', 0.0)
        temporal_strength = abs(temporal_signal) * temporal_confidence
        temporal_active = 1.0 if abs(temporal_signal) > 0.1 else 0.0
        features.extend([temporal_signal, temporal_confidence, temporal_strength, temporal_active])
        
        # Immune features
        immune_signal = subsystem_signals.get('immune', 0.0)
        immune_confidence = subsystem_scores.get('immune', 0.0)
        danger_flag = 1.0 if intelligence_result.get('is_dangerous_pattern') else 0.0
        benefit_flag = 1.0 if intelligence_result.get('is_beneficial_pattern') else 0.0
        features.extend([immune_signal, immune_confidence, danger_flag, benefit_flag])
        
        return np.array(features, dtype=np.float32)

    def _cleanup_old_pending_signals(self):
        """Clean up old pending signals"""
        if len(self.pending_signals) > 50:
            # Remove oldest 25 signals
            oldest_signals = sorted(self.pending_signals.keys())[:25]
            for old_signal in oldest_signals:
                del self.pending_signals[old_signal]

    def get_adaptive_performance_report(self) -> str:
        """Generate performance report"""
        
        # Agent status
        agent_status = self.agent.get_pure_blackbox_status()
        
        # Safety status
        safety_status = self.safety_manager.get_adaptive_status() if hasattr(self.safety_manager, 'get_adaptive_status') else "Safety status not available"
        
        # Signal statistics
        signals = self.signal_stats['total_signals']
        success_rate = self.signal_stats['successful_signals'] / max(1, signals)
        avg_pnl = self.signal_stats['total_pnl_from_signals'] / max(1, signals)
        
        signal_report = f"""
=== PURE SIGNAL GENERATION PERFORMANCE ===

Signal Statistics:
  Total Signals: {signals}
  Success Rate: {success_rate:.1%} ({self.signal_stats['successful_signals']} wins, {self.signal_stats['failed_signals']} losses)
  Average P&L per Signal: ${avg_pnl:.2f}
  Total P&L from Signals: ${self.signal_stats['total_pnl_from_signals']:.2f}

Tool Discovery Progress:
  DNA: {self.signal_stats['tool_discoveries']['dna']} signals
  MICRO: {self.signal_stats['tool_discoveries']['micro']} signals
  TEMPORAL: {self.signal_stats['tool_discoveries']['temporal']} signals
  IMMUNE: {self.signal_stats['tool_discoveries']['immune']} signals

Adaptive Features Used:
  Meta-Learned Entries: {self.signal_stats['adaptive_features_used']['meta_learned_entries']}
  Dynamic Stops: {self.signal_stats['adaptive_features_used']['dynamic_stops']}
  Dynamic Targets: {self.signal_stats['adaptive_features_used']['dynamic_targets']}

Pending Signals: {len(self.pending_signals)}
"""
        
        combined_report = f"""
{agent_status}

{signal_report}

NO POSITION TRACKING - Pure signal generation with adaptive learning!
All execution and risk management handled by NinjaTrader!
"""
        
        return combined_report

    def force_save_all_adaptive_learning(self):
        """Force save all adaptive learning"""
        self.agent.force_save_all_learning()
        log.info("All progress saved")

# Factory function
def create_pure_blackbox_trade_manager(intelligence_engine, tcp_bridge, config):
    """Create pure black box signal generator"""
    
    manager = PureBlackBoxSignalGenerator(intelligence_engine, tcp_bridge, config)
    
    return manager

# Usage example
if __name__ == "__main__":
    print("Testing Pure Black Box Trade Manager...")
    
    # This would normally be created with real intelligence engine and TCP bridge
    # For testing, we'll create a minimal version
    
    class MockIntelligence:
        def process_market_data(self, prices, volumes, timestamp):
            return {
                'subsystem_signals': {'dna': 0.2, 'micro': -0.1, 'temporal': 0.0, 'immune': 0.3},
                'subsystem_scores': {'dna': 0.7, 'micro': 0.5, 'temporal': 0.3, 'immune': 0.8},
                'similar_patterns_count': 5,
                'dna_sequence_length': 25,
                'micro_pattern_id': 'test_pattern',
                'is_dangerous_pattern': False,
                'is_beneficial_pattern': True
            }
    
    class MockTCPBridge:
        def send_signal(self, action, confidence, quality, stop_price=0, target_price=0):
            print(f"Mock signal: {action}, {confidence:.3f}, {quality}")
    
    mock_intel = MockIntelligence()
    mock_tcp = MockTCPBridge()
    
    manager = create_pure_blackbox_trade_manager(mock_intel, mock_tcp)
    
    print("Initial status:")
    print(manager.get_adaptive_performance_report())
    
    # Simulate some market data processing
    for i in range(5):
        mock_msg = {
            'price_1m': [4000 + i, 4001 + i, 4002 + i],
            'volume_1m': [1000, 1100, 1200]
        }
        
        manager.on_new_bar(mock_msg)
        
        # Simulate trade completion every other iteration
        if i % 2 == 1 and manager.current_position['in_position']:
            manager._complete_adaptive_trade("test_exit", 4005 + i)
    
    print("\nFinal status after adaptive learning:")
    print(manager.get_adaptive_performance_report())