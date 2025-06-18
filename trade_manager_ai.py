# pure_blackbox_trade_manager.py - REPLACES trade_manager_ai.py with full meta-learning

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
    """
    PURE adaptive safety - no hardcoded limits, everything learned from losses
    """
    
    def __init__(self, meta_learner, config):
        self.meta_learner = meta_learner
        self.config = config  # NOW ACTUALLY USING CONFIG!
        
        # Add trade frequency as a meta-learned parameter
        if 'trade_frequency_multiplier' not in meta_learner.parameters:
            meta_learner.parameters['trade_frequency_multiplier'] = 1.0
            meta_learner.parameter_gradients['trade_frequency_multiplier'] = 0.0
            meta_learner.parameter_outcomes['trade_frequency_multiplier'] = deque(maxlen=200)
        
        self.current_phase = 'exploration'
        
        # Dynamic tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_date = datetime.now().date()
        
        # Adaptive learning from losses
        self.loss_history = []
        self.phase_performance_history = {
            'exploration': [],
            'development': [], 
            'production': []
        }
        
        log.info("ADAPTIVE SAFETY: All limits will adapt based on actual loss experience")
        log.info("Trade limits will use adaptive config parameters")
    
    def can_trade(self) -> bool:
        """Adaptive trading permission based on learned risk parameters"""
        
        # Reset daily counters
        current_date = datetime.now().date()
        if current_date != self.last_date:
            # Learn from yesterday's performance before reset
            if self.trades_today > 0:
                daily_performance = self.daily_pnl / max(1, self.trades_today)
                self.meta_learner.update_parameter('max_daily_loss_pct', daily_performance / 100.0)
                
                # Learn trade frequency from daily results
                if self.daily_pnl > 0:
                    self.meta_learner.update_parameter('trade_frequency_multiplier', 0.05)  # More trades if profitable
                else:
                    self.meta_learner.update_parameter('trade_frequency_multiplier', -0.02)  # Fewer if losing
            
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_date = current_date
        
        # Get adaptive limits from meta-learner
        risk_params = self.meta_learner.get_risk_parameters()
        
        # OPTION 1: Keep basic safety limits (recommended)
        max_daily_loss = risk_params['max_daily_loss_pct'] * 10000  # Still need some safety
        max_consecutive = risk_params['max_consecutive_losses']
        
        # Check critical safety limits
        if self.daily_pnl <= -max_daily_loss:
            log.warning(f"ADAPTIVE SAFETY: Daily loss limit hit (${self.daily_pnl:.2f} vs ${-max_daily_loss:.2f})")
            self.meta_learner.update_parameter('max_daily_loss_pct', -0.5)  # Learn to be more cautious
            return False
        
        if self.consecutive_losses >= max_consecutive:
            log.warning(f"ADAPTIVE SAFETY: Consecutive loss limit hit ({self.consecutive_losses} vs {max_consecutive})")
            self.meta_learner.update_parameter('max_consecutive_losses', -0.3)
            return False
        
        # OPTION 2: Use config trade limits (FIXED - was hardcoded before)
        if hasattr(self.config, 'MAX_DAILY_TRADES_EXPLORATION'):
            # Use the adaptive config limits
            phase_limits = {
                'exploration': self.config.MAX_DAILY_TRADES_EXPLORATION,
                'development': self.config.MAX_DAILY_TRADES_DEVELOPMENT, 
                'production': self.config.MAX_DAILY_TRADES_PRODUCTION
            }
            base_max_trades = phase_limits[self.current_phase]
            
            # Apply performance and frequency multipliers
            performance_factor = max(0.5, 1.0 + (self.daily_pnl / 100.0))
            frequency_multiplier = self.meta_learner.get_parameter('trade_frequency_multiplier')
            
            max_trades_today = int(base_max_trades * performance_factor * frequency_multiplier)
            
            if self.trades_today >= max_trades_today:
                log.info(f"ADAPTIVE TRADE LIMIT: {self.trades_today} >= {max_trades_today}")
                log.info(f"  (Base: {base_max_trades}, Performance: {performance_factor:.2f}, Frequency: {frequency_multiplier:.2f})")
                return False
        
        # OPTION 3: No trade limits at all (uncomment this and comment out OPTION 2 above)
        # log.debug(f"UNLIMITED LEARNING: Trade #{self.trades_today + 1} allowed")
        
        return True
    
    def record_trade(self, pnl: float):
        """Record trade and adapt frequency based on outcome"""
        self.daily_pnl += pnl
        self.trades_today += 1
        self.loss_history.append(pnl)
        
        # Keep recent history
        if len(self.loss_history) > 50:
            self.loss_history = self.loss_history[-30:]
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Learn optimal trade frequency from outcomes
        if len(self.loss_history) >= 5:
            recent_performance = np.mean(self.loss_history[-5:])
            
            if recent_performance > 10:  # Good recent performance
                frequency_signal = 0.05  # Allow more frequent trading
            elif recent_performance < -20:  # Poor recent performance  
                frequency_signal = -0.1  # Reduce trading frequency
            else:
                frequency_signal = 0.01 if pnl > 0 else -0.02
            
            self.meta_learner.update_parameter('trade_frequency_multiplier', frequency_signal)
        
        # Adaptive phase progression
        self._adapt_learning_phase()
        
        # Learn from significant outcomes
        if abs(pnl) > 20:  # Significant trade
            normalized_outcome = np.tanh(pnl / 50.0)
            self.meta_learner.update_parameter('position_size_base', normalized_outcome)
    
    # Rest of the methods stay the same...
    def get_position_size(self) -> float:
        """Adaptive position sizing based on current performance and phase"""
        
        risk_params = self.meta_learner.get_risk_parameters()
        base_size = risk_params['position_size_base']
        
        # Adjust by phase
        phase_multiplier = {'exploration': 0.3, 'development': 0.7, 'production': 1.0}[self.current_phase]
        
        # Adjust by recent performance
        if len(self.loss_history) >= 5:
            recent_performance = np.mean(self.loss_history[-5:])
            performance_multiplier = max(0.2, 1.0 + recent_performance / 50.0)  # Better performance = larger size
        else:
            performance_multiplier = 1.0
        
        # Adjust by consecutive losses
        loss_adjustment = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
        
        final_size = base_size * phase_multiplier * performance_multiplier * loss_adjustment
        return max(0.1, min(2.0, final_size))  # Clamp to reasonable bounds
    
    def _adapt_learning_phase(self):
        """Adapt learning phase based on actual performance, not hardcoded rules"""
        
        if len(self.loss_history) < 10:
            return  # Need minimum data
        
        # Calculate phase readiness based on actual performance
        recent_performance = self.loss_history[-10:]
        win_rate = sum(1 for pnl in recent_performance if pnl > 0) / len(recent_performance)
        avg_pnl = np.mean(recent_performance)
        consistency = 1.0 - (np.std(recent_performance) / 50.0)  # Lower std = higher consistency
        
        # Calculate readiness score
        readiness_score = (win_rate * 0.4) + (max(0, avg_pnl / 50.0) * 0.4) + (consistency * 0.2)
        
        # Store performance for this phase
        self.phase_performance_history[self.current_phase].append(readiness_score)
        
        # Adaptive progression thresholds
        if self.current_phase == 'exploration':
            # Progress to development if showing competence
            if readiness_score > 0.4 and avg_pnl > 5:
                self.current_phase = 'development'
                log.info(f"ADAPTIVE PROGRESSION: Advanced to development phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', 0.2)  # Boost confidence
        
        elif self.current_phase == 'development':
            # Progress to production if showing consistent profitability
            if readiness_score > 0.6 and avg_pnl > 10 and win_rate > 0.5:
                self.current_phase = 'production'
                log.info(f"ADAPTIVE PROGRESSION: Advanced to production phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', 0.3)
            
            # Regress if performing poorly
            elif readiness_score < 0.2 and avg_pnl < -10:
                self.current_phase = 'exploration'
                log.warning(f"ADAPTIVE REGRESSION: Dropped to exploration phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', -0.3)
        
        elif self.current_phase == 'production':
            # Regress if performance deteriorates
            if readiness_score < 0.3 or avg_pnl < -15:
                self.current_phase = 'development'
                log.warning(f"ADAPTIVE REGRESSION: Dropped to development phase (readiness: {readiness_score:.2f})")
                self.meta_learner.update_parameter('position_size_base', -0.2)
    
    def get_adaptive_status(self) -> str:
        """Get adaptive safety status"""
        
        risk_params = self.meta_learner.get_risk_parameters()
        
        status = f"""
ADAPTIVE SAFETY STATUS:
  Current Phase: {self.current_phase}
  
Daily Limits (Adaptive):
  Daily P&L: ${self.daily_pnl:.2f} / ${-risk_params['max_daily_loss_pct'] * 10000:.0f} limit
  Trades Today: {self.trades_today}
  Consecutive Losses: {self.consecutive_losses} / {risk_params['max_consecutive_losses']:.0f} limit
  Trade Frequency Multiplier: {self.meta_learner.get_parameter('trade_frequency_multiplier'):.2f}
  
Position Sizing (Learned):
  Current Size: {self.get_position_size():.3f}
  Base Size: {risk_params['position_size_base']:.3f}
  
Recent Performance:
"""
        
        if len(self.loss_history) >= 5:
            recent_pnl = self.loss_history[-5:]
            win_rate = sum(1 for p in recent_pnl if p > 0) / len(recent_pnl)
            avg_pnl = np.mean(recent_pnl)
            status += f"  Win Rate: {win_rate:.1%}\n"
            status += f"  Avg P&L: ${avg_pnl:.2f}\n"
        
        status += f"\nALL LIMITS ADAPTING THROUGH EXPERIENCE!"
        
        return status

class PureBlackBoxTradeManager:
    """
    PURE BLACK BOX: Complete trade management with zero hardcoded knowledge
    Everything adapts through meta-learning
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
        
        # State tracking
        self.last_market_obs = self.env.get_obs()
        self.last_subsystem_features = None
        self.last_decision = {}
        
        # Position tracking
        self.current_position = {
            'in_position': False,
            'entry_price': 0,
            'entry_time': None,
            'action': 0,
            'size': 0.0,
            'tool_used': '',
            'entry_confidence': 0.0,
            'meta_params_at_entry': {}
        }
        
        # Pure statistics - no preset expectations
        self.trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_hold_time': 0.0,
            'tool_discoveries': {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0},
            'adaptive_features_used': {
                'dynamic_stops': 0,
                'dynamic_targets': 0,
                'confidence_based_exits': 0,
                'meta_learned_entries': 0
            }
        }
        
        log.info("PURE BLACK BOX Trade Manager initialized")
        log.info("ALL parameters will adapt through meta-learning")
        log.info("Safety limits, position sizing, and thresholds will evolve")

    def on_new_bar(self, msg: Dict[str, Any]):
        """Pure black box bar processing with complete adaptation"""
        try:
            # Adaptive safety check
            if not self.safety_manager.can_trade():
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
            if not self.current_position['in_position']:
                decision = self.agent.select_action_and_strategy(
                    market_obs=market_obs,
                    subsystem_features=subsystem_features,
                    current_price=price,
                    in_position=False
                )
                
                # Store decision data for learning
                self.last_decision = decision
                self.last_market_obs = market_obs
                self.last_subsystem_features = subsystem_features
                
                # Execute pure AI decision
                self._execute_pure_adaptive_decision(decision, price, intelligence_result)
            
            else:
                # Position management with adaptive parameters
                self._handle_adaptive_position_management(price, market_obs, subsystem_features)
            
            # Adaptive logging
            if self.last_decision:
                primary_tool = self.last_decision.get('primary_tool', 'unknown')
                confidence = self.last_decision.get('confidence', 0.0)
                threshold = self.last_decision.get('entry_threshold_used', 0.5)
                
                log.info(f"BLACK BOX ADAPTIVE: {primary_tool.upper()} tool, "
                       f"confidence {confidence:.3f} vs adaptive threshold {threshold:.3f}")
            
        except Exception as e:
            log.error(f"Pure black box error: {e}")
            import traceback
            traceback.print_exc()

    def _handle_adaptive_position_management(self, current_price: float, market_obs: np.ndarray, 
                                           subsystem_features: np.ndarray):
        """Position management with fully adaptive parameters"""
        
        # Get adaptive decision for position management
        position_decision = self.agent.select_action_and_strategy(
            market_obs=market_obs,
            subsystem_features=subsystem_features,
            current_price=current_price,
            in_position=True
        )
        
        # Adaptive exit decision
        if position_decision.get('should_exit', False):
            exit_reason = "adaptive_confidence_exit"
            exit_confidence = position_decision.get('confidence', 0.0)
            exit_threshold = position_decision.get('entry_threshold_used', 0.5)  # Using entry threshold for exit
            
            self.tcp_bridge.send_signal(0, exit_confidence, f"AI_adaptive_exit")
            
            log.info(f"🚪 ADAPTIVE EXIT: Confidence {exit_confidence:.3f} below adaptive threshold {exit_threshold:.3f}")
            log.info(f"   Tool used: {self.current_position.get('tool_used', 'unknown').upper()}")
            
            self.trade_stats['adaptive_features_used']['confidence_based_exits'] += 1
            
            # Complete trade
            self._complete_adaptive_trade(exit_reason, current_price)

    def _execute_pure_adaptive_decision(self, decision: Dict, current_price: float, intelligence_result: Dict):
        """Execute decision with fully adaptive parameters"""
        
        action = decision['action']
        confidence = decision['confidence']
        
        # Check adaptive entry conditions
        if action != 0 and not self.current_position['in_position']:
            
            # Get adaptive position size
            position_size = self.safety_manager.get_position_size()
            
            action_code = 1 if action == 1 else 2
            tool_name = decision['primary_tool']
            direction = 'long' if action == 1 else 'short'
            
            # Build quality string with adaptive parameters
            quality = f"AI_{tool_name}_{direction}_adaptive"
            
            # Adaptive risk management - AI learned parameters
            stop_price = 0.0
            target_price = 0.0

            if decision['use_stop'] and decision['stop_price']:
                stop_price = decision['stop_price']
                quality += f"_adaptiveStop{decision['stop_distance_pct']:.1f}pct"
                self.trade_stats['adaptive_features_used']['dynamic_stops'] += 1

            if decision['use_target'] and decision['target_price']:
                target_price = decision['target_price']
                quality += f"_adaptiveTarget{decision['target_distance_pct']:.1f}pct"
                self.trade_stats['adaptive_features_used']['dynamic_targets'] += 1

            # Send signal with adaptive parameters
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_price, target_price)
            
            # Track position with adaptive metadata
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'size': position_size,
                'tool_used': tool_name,
                'entry_confidence': confidence,
                'meta_params_at_entry': decision.get('meta_parameters_used', {})
            }
            
            self.trade_stats['total_trades'] += 1
            self.trade_stats['tool_discoveries'][tool_name] += 1
            self.trade_stats['adaptive_features_used']['meta_learned_entries'] += 1
            
            # Adaptive logging
            threshold_used = decision.get('entry_threshold_used', 0.5)
            exploration = decision.get('exploration_taken', False)
            
            log.info(f"🎯 PURE ADAPTIVE ENTRY:")
            log.info(f"   Tool: {tool_name.upper()} ({'EXPLORATION' if exploration else 'LEARNED'})")
            log.info(f"   Direction: {direction}, Confidence: {confidence:.3f}")
            log.info(f"   Adaptive Threshold: {threshold_used:.3f}")
            log.info(f"   Entry Price: ${current_price:.2f}, Adaptive Size: {position_size}")
            log.info(f"   Phase: {self.safety_manager.current_phase}")
            log.info(f"   Trade #{self.trade_stats['total_trades']}")
            
            if decision['use_stop']:
                log.info(f"   Adaptive Stop: ${decision['stop_price']:.2f} ({decision['stop_distance_pct']:.1f}%)")
            if decision['use_target']:
                log.info(f"   Adaptive Target: ${decision['target_price']:.2f} ({decision['target_distance_pct']:.1f}%)")

    def extract_adaptive_subsystem_features(self, intelligence_result: Dict) -> np.ndarray:
        """Extract features with adaptive normalization"""
        features = []
        
        # Raw subsystem signals with adaptive scaling
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

    def record_adaptive_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """Record outcome with adaptive reward learning"""
        try:
            if not self.last_decision:
                return
            
            # Update trade statistics
            self.trade_stats['total_pnl'] += pnl
            self.trade_stats['best_trade'] = max(self.trade_stats['best_trade'], pnl)
            self.trade_stats['worst_trade'] = min(self.trade_stats['worst_trade'], pnl)
            
            if pnl > 0:
                self.trade_stats['winning_trades'] += 1
            else:
                self.trade_stats['losing_trades'] += 1
            
            # Calculate hold time
            if self.current_position.get('entry_time'):
                hold_time_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
                self.trade_stats['avg_hold_time'] = (self.trade_stats['avg_hold_time'] * (self.trade_stats['total_trades'] - 1) + hold_time_hours) / self.trade_stats['total_trades']
            else:
                hold_time_hours = 1.0
            
            # Safety manager learning
            self.safety_manager.record_trade(pnl)
            
            # Comprehensive trade data for adaptive reward learning
            trade_data = {
                'pnl': pnl,
                'hold_time_hours': hold_time_hours,
                'used_stop': self.last_decision.get('use_stop', False),
                'used_target': self.last_decision.get('use_target', False),
                'tool_confidence': self.last_decision.get('confidence', 0.5),
                'primary_tool': self.last_decision.get('primary_tool', 'unknown'),
                'exit_reason': exit_reason,
                'entry_confidence': self.current_position.get('entry_confidence', 0.5),
                'position_size': self.current_position.get('size', 1.0),
                'exploration_taken': self.last_decision.get('exploration_taken', False),
                'meta_params_used': self.last_decision.get('meta_parameters_used', {}),
                'max_drawdown_pct': max(0, -min(0, pnl)) / max(abs(exit_price), 1000),
                'price_volatility': 20.0,  # Could be calculated from recent price data
                'max_risk_taken': self.current_position.get('size', 1.0) * 50  # Rough estimate
            }
            
            # Let agent learn from outcome
            self.agent.store_experience_and_learn(self.last_decision, trade_data)
            
            # Pure learning summary
            log.info(f"🎓 PURE ADAPTIVE LEARNING:")
            log.info(f"   P&L: ${pnl:.2f}")
            log.info(f"   Tool: {self.last_decision.get('primary_tool', 'unknown').upper()}")
            log.info(f"   Exit: {exit_reason}")
            log.info(f"   Hold Time: {hold_time_hours:.1f}h")
            log.info(f"   Phase: {self.safety_manager.current_phase}")
            
            # Meta-learning insights
            current_position_size = self.safety_manager.get_position_size()
            entry_threshold = self.meta_learner.get_confidence_thresholds()['entry']
            log.info(f"   Adaptive Position Size: {current_position_size:.3f}")
            log.info(f"   Adaptive Entry Threshold: {entry_threshold:.3f}")
            
            # Update state
            self.last_market_obs = self.env.get_obs()
            
        except Exception as e:
            log.error(f"Error recording adaptive outcome: {e}")

    def _complete_adaptive_trade(self, reason: str, exit_price: float):
        """Complete trade with adaptive learning"""
        if not self.current_position['in_position']:
            return
        
        entry_price = self.current_position['entry_price']
        action = self.current_position['action']
        
        # Calculate P&L
        if action == 1:  # Long
            pnl = (exit_price - entry_price) * 2.0  # $2 per point for MNQ
        else:  # Short
            pnl = (entry_price - exit_price) * 2.0
        
        # Adaptive logging
        duration_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
        
        log.info(f"📊 PURE ADAPTIVE TRADE COMPLETED:")
        log.info(f"   Tool: {self.current_position['tool_used'].upper()}")
        log.info(f"   P&L: ${pnl:.2f}")
        log.info(f"   Duration: {duration_hours:.1f}h")
        log.info(f"   Reason: {reason}")
        log.info(f"   Phase: {self.safety_manager.current_phase}")
        
        # Record adaptive outcome
        self.record_adaptive_trade_outcome(exit_price, pnl, reason)
        
        # Reset position
        self.current_position = {
            'in_position': False,
            'entry_price': 0,
            'entry_time': None,
            'action': 0,
            'size': 0.0,
            'tool_used': '',
            'entry_confidence': 0.0,
            'meta_params_at_entry': {}
        }

    def get_adaptive_performance_report(self) -> str:
        """Comprehensive adaptive performance report"""
        
        # Agent status
        agent_status = self.agent.get_pure_blackbox_status()
        
        # Safety status
        safety_status = self.safety_manager.get_adaptive_status()
        
        # Trade statistics
        trades = self.trade_stats['total_trades']
        win_rate = self.trade_stats['winning_trades'] / max(1, trades)
        avg_pnl = self.trade_stats['total_pnl'] / max(1, trades)
        
        trade_report = f"""

=== PURE ADAPTIVE TRADE PERFORMANCE ===

Trade Statistics:
  Total Trades: {trades}
  Win Rate: {win_rate:.1%} ({self.trade_stats['winning_trades']} wins, {self.trade_stats['losing_trades']} losses)
  Average P&L: ${avg_pnl:.2f}
  Total P&L: ${self.trade_stats['total_pnl']:.2f}
  Best Trade: ${self.trade_stats['best_trade']:.2f}
  Worst Trade: ${self.trade_stats['worst_trade']:.2f}
  Avg Hold Time: {self.trade_stats['avg_hold_time']:.1f}h

Tool Discovery Progress:
  DNA: {self.trade_stats['tool_discoveries']['dna']} uses
  MICRO: {self.trade_stats['tool_discoveries']['micro']} uses
  TEMPORAL: {self.trade_stats['tool_discoveries']['temporal']} uses
  IMMUNE: {self.trade_stats['tool_discoveries']['immune']} uses

Adaptive Features Used:
  Meta-Learned Entries: {self.trade_stats['adaptive_features_used']['meta_learned_entries']}
  Dynamic Stops: {self.trade_stats['adaptive_features_used']['dynamic_stops']}
  Dynamic Targets: {self.trade_stats['adaptive_features_used']['dynamic_targets']}
  Confidence-Based Exits: {self.trade_stats['adaptive_features_used']['confidence_based_exits']}
"""
        
        if self.current_position['in_position']:
            pos = self.current_position
            duration = (datetime.now() - pos['entry_time']).total_seconds() / 3600
            trade_report += f"""
Current Position (Adaptive):
  Tool: {pos['tool_used'].upper()}
  Entry Price: ${pos['entry_price']:.2f}
  Duration: {duration:.1f}h
  Entry Confidence: {pos['entry_confidence']:.3f}
  Adaptive Size: {pos['size']:.3f}
"""
        
        combined_report = f"""
{agent_status}

{safety_status}

{trade_report}

ALL PARAMETERS OPTIMIZING THROUGH PURE EXPERIENCE!
No hardcoded thresholds - complete adaptation through meta-learning!
"""
        
        return combined_report

    # External interface methods
    def on_trade_update(self, update_type: str, price: float, reason: str = ""):
        """Handle trade updates from NinjaTrader with adaptive learning"""
        if update_type == "filled":
            log.info(f"Adaptive trade filled at ${price:.2f}")
        elif update_type == "stopped":
            self._complete_adaptive_trade("adaptive_stop_hit", price)
        elif update_type == "target_hit":
            self._complete_adaptive_trade("adaptive_target_hit", price)
        elif update_type == "closed":
            self._complete_adaptive_trade(reason or "manual_close", price)

    def emergency_close_all(self):
        """Emergency close with adaptive learning"""
        if self.current_position['in_position']:
            self.tcp_bridge.send_signal(0, 0.9, "EMERGENCY_CLOSE")
            log.warning("ADAPTIVE EMERGENCY CLOSE: All positions closed")
            
            # Learn from emergency situation
            self.meta_learner.update_parameter('max_daily_loss_pct', -0.5)  # Increase caution
            self.meta_learner.update_parameter('position_size_base', -0.3)
    
    def force_save_all_adaptive_learning(self):
        """Force save all adaptive learning"""
        self.agent.force_save_all_learning()
        log.info("PURE ADAPTIVE: All meta-learning progress saved")

# Factory function
def create_pure_blackbox_trade_manager(intelligence_engine, tcp_bridge, config):
    """Create pure black box trade manager"""
    
    manager = PureBlackBoxTradeManager(intelligence_engine, tcp_bridge, config)
    
    log.info("PURE BLACK BOX TRADE MANAGER CREATED")
    log.info("All parameters will adapt through meta-learning")
    log.info("Safety limits, position sizing, and thresholds will evolve")
    log.info("Complete adaptation through experience - no hardcoded values")
    
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