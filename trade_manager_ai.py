# trade_manager_ai.py - PURE BLACK BOX: No wisdom, no biases, pure learning

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
from rl_agent import StrategicToolLearningAgent
from market_env import MarketEnv
from advanced_position_management import PositionManagementLearner, PositionState

log = logging.getLogger(__name__)

class PureBlackBoxConfidenceLearner:
    """
    Pure black box confidence learning - no trader wisdom, just experience
    """
    
    def __init__(self):
        # Start with completely neutral thresholds - AI discovers everything
        self.confidence_thresholds = {
            'exit': {'dna': 0.5, 'micro': 0.5, 'temporal': 0.5, 'immune': 0.5},
            'scale': {'dna': 0.5, 'micro': 0.5, 'temporal': 0.5, 'immune': 0.5}
        }
        
        # Track outcomes for threshold adjustment
        self.threshold_outcomes = {
            'exit': {'dna': [], 'micro': [], 'temporal': [], 'immune': []},
            'scale': {'dna': [], 'micro': [], 'temporal': [], 'immune': []}
        }
        
        # Pure learning parameters
        self.learning_rate = 0.02
        self.min_samples = 5  # Need more samples since no wisdom bias
        
        log.info("PURE BLACK BOX: AI starts with zero trading knowledge")
        log.info("All strategies will be discovered through pure experience")
    
    def get_threshold(self, action_type: str, tool: str) -> float:
        """Get current learned threshold for tool/action"""
        return self.confidence_thresholds[action_type].get(tool, 0.5)
    
    def should_take_action(self, action_type: str, tool: str, confidence: float) -> bool:
        """Pure experience-based decision - no market regime adjustments"""
        threshold = self.get_threshold(action_type, tool)
        return confidence >= threshold
    
    def record_outcome(self, action_type: str, tool: str, confidence: float, outcome: float):
        """Record outcome and adjust thresholds based purely on results"""
        
        # Store outcome
        self.threshold_outcomes[action_type][tool].append({
            'confidence': confidence,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Keep recent history only
        if len(self.threshold_outcomes[action_type][tool]) > 30:
            self.threshold_outcomes[action_type][tool] = \
                self.threshold_outcomes[action_type][tool][-20:]
        
        # Learn from outcomes
        self._update_threshold(action_type, tool)
    
    def _update_threshold(self, action_type: str, tool: str):
        """Pure learning - adjust thresholds based only on what works"""
        
        outcomes = self.threshold_outcomes[action_type][tool]
        if len(outcomes) < self.min_samples:
            return
        
        current_threshold = self.confidence_thresholds[action_type][tool]
        
        # Analyze recent performance
        recent_outcomes = outcomes[-15:]
        
        # Separate by confidence level relative to current threshold
        above_threshold = [o for o in recent_outcomes if o['confidence'] >= current_threshold]
        below_threshold = [o for o in recent_outcomes if o['confidence'] < current_threshold]
        
        # Calculate success rates
        above_success = sum(1 for o in above_threshold if o['outcome'] > 0) / max(len(above_threshold), 1)
        below_success = sum(1 for o in below_threshold if o['outcome'] > 0) / max(len(below_threshold), 1) if below_threshold else 0
        
        # Adjust threshold based on what's working
        if len(below_threshold) >= 3 and below_success > above_success + 0.2:
            # Lower confidence decisions working better - lower threshold
            new_threshold = current_threshold * (1 - self.learning_rate)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"BLACK BOX LEARNING: Lowering {action_type} threshold for {tool.upper()} to {new_threshold:.3f}")
            
        elif len(above_threshold) >= 3 and above_success < 0.4:
            # High confidence decisions failing - raise threshold
            new_threshold = current_threshold * (1 + self.learning_rate)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"BLACK BOX LEARNING: Raising {action_type} threshold for {tool.upper()} to {new_threshold:.3f}")
        
        elif len(above_threshold) >= 3 and above_success > 0.7:
            # High confidence working well - slightly lower threshold
            new_threshold = current_threshold * (1 - self.learning_rate * 0.5)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"BLACK BOX LEARNING: Optimizing {action_type} threshold for {tool.upper()} to {new_threshold:.3f}")
        
        # Keep thresholds in bounds
        self.confidence_thresholds[action_type][tool] = max(0.05, min(0.95, self.confidence_thresholds[action_type][tool]))
    
    def get_learning_status(self) -> str:
        """Pure learning status - no wisdom comparisons"""
        
        status = "PURE BLACK BOX CONFIDENCE LEARNING:\n"
        
        for action_type in ['exit', 'scale']:
            status += f"\n{action_type.upper()} Thresholds (Learned from Experience):\n"
            for tool, threshold in self.confidence_thresholds[action_type].items():
                samples = len(self.threshold_outcomes[action_type][tool])
                
                if samples > 0:
                    recent_success = sum(1 for o in self.threshold_outcomes[action_type][tool][-5:] if o['outcome'] > 0) / min(5, samples)
                    status += f"  {tool.upper()}: {threshold:.3f} ({samples} samples, {recent_success:.1%} recent success)\n"
                else:
                    status += f"  {tool.upper()}: {threshold:.3f} (no experience yet)\n"
        
        return status

class SafetyManager:
    """
    Pure safety constraints - no trading wisdom, just risk management
    """
    
    def __init__(self):
        # Progressive position sizing based on learning phase
        self.learning_phases = {
            'exploration': {'max_position_size': 0.1, 'max_daily_trades': 3},
            'development': {'max_position_size': 0.5, 'max_daily_trades': 8},
            'production': {'max_position_size': 1.0, 'max_daily_trades': 15}
        }
        
        # Hard safety limits
        self.max_daily_loss = 200  # Hard stop
        self.max_consecutive_losses = 6
        self.current_phase = 'exploration'
        
        # Tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_date = datetime.now().date()
        
        log.info("SAFETY MANAGER: Progressive risk limits enabled")
        log.info(f"Starting phase: {self.current_phase}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on safety limits"""
        
        # Reset daily counters
        current_date = datetime.now().date()
        if current_date != self.last_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_date = current_date
        
        # Check limits
        phase_limits = self.learning_phases[self.current_phase]
        
        if self.daily_pnl <= -self.max_daily_loss:
            log.warning(f"SAFETY: Daily loss limit hit (${self.daily_pnl:.2f})")
            return False
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            log.warning(f"SAFETY: Consecutive loss limit hit ({self.consecutive_losses})")
            return False
        
        if self.trades_today >= phase_limits['max_daily_trades']:
            log.info(f"SAFETY: Daily trade limit hit ({self.trades_today})")
            return False
        
        return True
    
    def get_position_size(self) -> float:
        """Get allowed position size for current learning phase"""
        phase_limits = self.learning_phases[self.current_phase]
        return phase_limits['max_position_size']
    
    def record_trade(self, pnl: float):
        """Record trade outcome for safety tracking"""
        self.daily_pnl += pnl
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Phase progression based on performance
        self._update_learning_phase()
    
    def _update_learning_phase(self):
        """Progress through learning phases based on performance"""
        
        if self.trades_today < 20:  # Need minimum experience
            return
        
        # Simple progression: positive daily P&L advances phase
        if self.current_phase == 'exploration' and self.daily_pnl > 50:
            self.current_phase = 'development'
            log.info("SAFETY: Advanced to development phase - increasing position sizes")
        
        elif self.current_phase == 'development' and self.daily_pnl > 100:
            self.current_phase = 'production'
            log.info("SAFETY: Advanced to production phase - full position sizes")
        
        # Regression: big losses drop back a phase
        elif self.daily_pnl < -100:
            if self.current_phase == 'production':
                self.current_phase = 'development'
                log.warning("SAFETY: Dropped to development phase due to losses")
            elif self.current_phase == 'development':
                self.current_phase = 'exploration'
                log.warning("SAFETY: Dropped to exploration phase due to losses")

class PureBlackBoxTradeManager:
    """
    PURE BLACK BOX: No wisdom, no biases, just learning from experience
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        
        # Pure black box AI - no wisdom initialization
        self.agent = StrategicToolLearningAgent(
            market_obs_size=15,
            subsystem_features_size=16
        )
        
        # Advanced position management AI
        self.position_ai = PositionManagementLearner(self.agent)
        
        # PURE learning systems - no wisdom
        self.confidence_learner = PureBlackBoxConfidenceLearner()
        self.safety_manager = SafetyManager()
        
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
            'scales_added': 0,
            'partial_exits': 0
        }
        
        # Pure statistics - no preset expectations
        self.trade_stats = {
            'total_trades': 0,
            'scaling_trades': 0,
            'partial_exit_trades': 0,
            'full_exit_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'tool_discovery': {'dna': 0, 'micro': 0, 'temporal': 0, 'immune': 0}
        }
        
        log.info("PURE BLACK BOX Trade Manager initialized")
        log.info("AI will discover ALL trading strategies from scratch")
        log.info("No hardcoded knowledge - pure pattern discovery")

    def on_new_bar(self, msg: Dict[str, Any]):
        """Pure black box processing - AI discovers everything"""
        try:
            # Safety check
            if not self.safety_manager.can_trade():
                return
            
            # Extract price data
            price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
            prices = msg.get("price_1m", [price])
            volumes = msg.get("volume_1m", [1000])
            
            # Update environment state
            market_obs = self.env.get_obs()
            self.env.step(price, 0)
            
            # Position management with pure learning
            if self.current_position['in_position']:
                self._handle_pure_position_management(price, market_obs)
            
            # New entry decisions with pure black box learning
            if not self.current_position['in_position']:
                decision = self.agent.select_action_and_strategy(
                    market_obs=market_obs,
                    subsystem_features=self.extract_subsystem_features(
                        self.intel.process_market_data(prices, volumes, datetime.now())
                    ),
                    current_price=price,
                    in_position=self.current_position['in_position']
                )
                
                # Store decision data for pure learning
                self.last_decision = decision
                self.last_market_obs = market_obs
                self.last_subsystem_features = decision['raw_subsystem_features']
                
                # Execute decision
                self._execute_pure_ai_decision(decision, price)
            
            # Pure discovery logging
            if self.last_decision:
                primary_tool = self.last_decision.get('primary_tool', 'unknown')
                confidence = self.last_decision.get('confidence', 0.0)
                log.info(f"BLACK BOX DISCOVERY: {primary_tool.upper()} tool, confidence {confidence:.3f}")
            
        except Exception as e:
            log.error(f"Pure black box error: {e}")
            import traceback
            traceback.print_exc()

    def _handle_pure_position_management(self, current_price: float, market_obs: np.ndarray):
        """Position management with pure learning - no wisdom biases"""
        
        if self.position_ai.current_position:
            self.position_ai.update_position(current_price)
            tool_used = self.position_ai.current_position.tool_used
            
            # SCALING with pure learning
            scale_decision = self.position_ai.should_scale_position(market_obs, current_price)
            
            # Pure experience-based threshold check
            if (scale_decision['action'] != 'no_scale' and 
                self.confidence_learner.should_take_action(
                    'scale', tool_used, scale_decision['confidence'])):
                
                scale_amount = scale_decision['scale_amount']
                action_code = 1 if self.current_position['action'] == 1 else 2
                quality = f"AI_scale_{scale_decision['action']}_{tool_used}"
                
                self.tcp_bridge.send_signal(action_code, scale_decision['confidence'], quality)
                
                # Update tracking
                old_size = self.current_position['size']
                self.current_position['size'] += scale_amount
                self.current_position['scales_added'] += 1
                self.trade_stats['scaling_trades'] += 1
                
                threshold_used = self.confidence_learner.get_threshold('scale', tool_used)
                
                log.info(f"🔄 PURE AI SCALING: {scale_decision['reasoning']}")
                log.info(f"   Confidence: {scale_decision['confidence']:.3f} vs learned threshold {threshold_used:.3f}")
                log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                
                # Record for pure learning
                self._pending_scale_decision = {
                    'tool': tool_used,
                    'confidence': scale_decision['confidence'],
                    'action_type': 'scale'
                }
            
            # EXITS with pure learning
            exit_decision = self.position_ai.should_exit_position(market_obs, current_price)
            
            # Pure experience-based threshold check
            if (exit_decision['action'] != 'hold' and 
                self.confidence_learner.should_take_action(
                    'exit', tool_used, exit_decision['confidence'])):
                
                threshold_used = self.confidence_learner.get_threshold('exit', tool_used)
                
                if exit_decision['action'] == 'exit_100%':
                    # Full exit
                    self.tcp_bridge.send_signal(0, exit_decision['confidence'], 
                                              f"AI_exit_full_{tool_used}")
                    
                    self.trade_stats['full_exit_trades'] += 1
                    log.info(f"🚪 PURE AI FULL EXIT: {exit_decision['reasoning']}")
                    log.info(f"   Confidence: {exit_decision['confidence']:.3f} vs learned threshold {threshold_used:.3f}")
                    
                    # Record for pure learning
                    self._pending_exit_decision = {
                        'tool': tool_used,
                        'confidence': exit_decision['confidence'],
                        'action_type': 'exit'
                    }
                    
                    # Complete trade
                    self._complete_trade("AI_full_exit", current_price)
                    
                elif 'exit_' in exit_decision['action']:
                    # Partial exit
                    exit_amount = exit_decision['exit_amount']
                    exit_quality = f"AI_exit_{exit_decision['action']}_{tool_used}"
                    
                    self.tcp_bridge.send_signal(0, exit_decision['confidence'], exit_quality)
                    
                    # Update position
                    old_size = self.current_position['size']
                    self.current_position['size'] *= (1.0 - exit_amount)
                    self.current_position['partial_exits'] += 1
                    self.trade_stats['partial_exit_trades'] += 1
                    
                    log.info(f"💰 PURE AI PARTIAL EXIT: {exit_decision['reasoning']}")
                    log.info(f"   Confidence: {exit_decision['confidence']:.3f} vs learned threshold {threshold_used:.3f}")
                    log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                    
                    # Record partial exit for learning
                    self.confidence_learner.record_outcome(
                        'exit', tool_used, exit_decision['confidence'], 0.1
                    )

    def _execute_pure_ai_decision(self, decision: Dict, current_price: float):
        """Execute AI decision with pure black box learning"""
        
        action = decision['action']
        confidence = decision['confidence']
        
        # Check if AI wants to enter
        if action != 0 and not self.current_position['in_position']:
            
            # Get safety-managed position size
            position_size = self.safety_manager.get_position_size()
            
            action_code = 1 if action == 1 else 2
            tool_name = decision['primary_tool']
            direction = 'long' if action == 1 else 'short'
            
            # Build quality string
            quality = f"AI_{tool_name}_{direction}"
            
            # AI-learned risk management (no hardcoded rules)
            stop_price = 0.0
            target_price = 0.0

            if decision['use_stop'] and decision['stop_price']:
                stop_price = decision['stop_price']
                quality += f"_stop{decision['stop_distance_pct']:.1f}pct"

            if decision['use_target'] and decision['target_price']:
                target_price = decision['target_price']
                quality += f"_target{decision['target_distance_pct']:.1f}pct"

            # Send signal
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_price, target_price)
            
            # Track position
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'size': position_size,
                'tool_used': tool_name,
                'scales_added': 0,
                'partial_exits': 0
            }
            
            # Start position AI tracking
            self.position_ai.start_position(
                entry_price=current_price,
                initial_size=position_size,
                tool_used=tool_name,
                entry_confidence=confidence,
                market_regime='unknown'  # No regime classification
            )
            
            self.trade_stats['total_trades'] += 1
            self.trade_stats['tool_discovery'][tool_name] += 1
            
            log.info(f"🎯 PURE BLACK BOX ENTRY:")
            log.info(f"   Tool Discovered: {tool_name.upper()}")
            log.info(f"   Direction: {direction}, Confidence: {confidence:.3f}")
            log.info(f"   Entry Price: ${current_price:.2f}, Size: {position_size}")
            log.info(f"   Trade #{self.trade_stats['total_trades']}")
            log.info(f"   Phase: {self.safety_manager.current_phase}")
            
            if decision['use_stop']:
                log.info(f"   AI Stop: ${decision['stop_price']:.2f}")
            if decision['use_target']:
                log.info(f"   AI Target: ${decision['target_price']:.2f}")

    def record_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """Pure learning from trade outcomes - no preset rewards"""
        try:
            if not self.last_decision:
                return
            
            # Update statistics
            self.trade_stats['total_pnl'] += pnl
            self.trade_stats['best_trade'] = max(self.trade_stats['best_trade'], pnl)
            self.trade_stats['worst_trade'] = min(self.trade_stats['worst_trade'], pnl)
            
            # Safety tracking
            self.safety_manager.record_trade(pnl)
            
            # Pure P&L-based reward (no bonuses, no wisdom validation)
            base_reward = pnl / 50.0  # Normalize for MNQ
            
            # ONLY position management rewards (universal principles)
            bonus_reward = 0.0
            
            # Risk management (universal principle: limit losses)
            if exit_reason == "stop_hit" and self.last_decision.get('use_stop', False):
                if pnl > -30:  # Stop limited loss
                    bonus_reward += 0.2
                    log.info("PURE LEARNING: +0.2 for protective stop")
            
            # Profit taking (universal principle: secure gains)
            if exit_reason == "target_hit" and self.last_decision.get('use_target', False):
                bonus_reward += 0.2
                log.info("PURE LEARNING: +0.2 for hitting target")
            
            # Position scaling outcomes
            if self.current_position['scales_added'] > 0:
                if pnl > 10:
                    bonus_reward += 0.3 * self.current_position['scales_added']
                    log.info(f"PURE LEARNING: +{0.3 * self.current_position['scales_added']:.1f} for successful scaling")
                else:
                    bonus_reward -= 0.1 * self.current_position['scales_added']
                    log.info(f"PURE LEARNING: -{0.1 * self.current_position['scales_added']:.1f} for failed scaling")
            
            # Partial exits
            if self.current_position['partial_exits'] > 0:
                bonus_reward += 0.1  # Small reward for risk management
                log.info("PURE LEARNING: +0.1 for partial exits")
            
            total_reward = base_reward + bonus_reward
            
            # Update pure confidence thresholds
            primary_tool = self.last_decision.get('primary_tool', '')
            normalized_outcome = pnl / 50.0
            
            # Record outcomes for pure learning
            if hasattr(self, '_pending_exit_decision'):
                decision = self._pending_exit_decision
                self.confidence_learner.record_outcome(
                    decision['action_type'],
                    decision['tool'], 
                    decision['confidence'],
                    normalized_outcome
                )
                log.info(f"PURE THRESHOLD LEARNING: Updated {decision['action_type']} for {decision['tool'].upper()}")
                delattr(self, '_pending_exit_decision')
            
            if hasattr(self, '_pending_scale_decision'):
                decision = self._pending_scale_decision
                self.confidence_learner.record_outcome(
                    decision['action_type'],
                    decision['tool'], 
                    decision['confidence'],
                    normalized_outcome
                )
                log.info(f"PURE THRESHOLD LEARNING: Updated {decision['action_type']} for {decision['tool'].upper()}")
                delattr(self, '_pending_scale_decision')
            
            # Store experience for AI learning
            next_market_obs = self.env.get_obs()
            next_subsystem_features = self.last_subsystem_features
            
            self.agent.store_experience(
                state_data=self.last_decision,
                reward=total_reward,
                next_market_obs=next_market_obs,
                next_subsystem_features=next_subsystem_features,
                done=True
            )
            
            # Pure learning summary
            log.info(f"🎓 PURE BLACK BOX LEARNING:")
            log.info(f"   P&L: ${pnl:.2f} -> Reward: {total_reward:.3f}")
            log.info(f"   Tool: {primary_tool.upper()}")
            log.info(f"   Exit: {exit_reason}")
            log.info(f"   Base: {base_reward:.3f}, Bonus: {bonus_reward:.3f}")
            
            # Tool discovery progress
            discoveries = self.trade_stats['tool_discovery']
            total_discoveries = sum(discoveries.values())
            if total_discoveries > 0:
                log.info(f"   Tool Discovery: DNA({discoveries['dna']}) MICRO({discoveries['micro']}) TEMPORAL({discoveries['temporal']}) IMMUNE({discoveries['immune']})")
            
            # Update state
            self.last_market_obs = next_market_obs
            
        except Exception as e:
            log.error(f"Error recording pure outcome: {e}")

    def _complete_trade(self, reason: str, exit_price: float):
        """Complete trade with pure learning"""
        if not self.current_position['in_position']:
            return
        
        entry_price = self.current_position['entry_price']
        action = self.current_position['action']
        
        # Calculate P&L
        if action == 1:  # Long
            pnl = (exit_price - entry_price) * 2.0  # $2 per point for MNQ
        else:  # Short
            pnl = (entry_price - exit_price) * 2.0
        
        # Pure logging
        duration_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
        
        log.info(f"📊 PURE BLACK BOX TRADE COMPLETED:")
        log.info(f"   Tool: {self.current_position['tool_used'].upper()}")
        log.info(f"   P&L: ${pnl:.2f}")
        log.info(f"   Duration: {duration_hours:.1f}h")
        log.info(f"   Scales: {self.current_position['scales_added']}")
        log.info(f"   Exits: {self.current_position['partial_exits']}")
        log.info(f"   Reason: {reason}")
        log.info(f"   Phase: {self.safety_manager.current_phase}")
        
        # Record pure outcome
        self.record_trade_outcome(exit_price, pnl, reason)
        
        # Position AI learning
        if self.position_ai.current_position:
            self.position_ai.close_position(exit_price)
        
        # Reset position
        self.current_position = {
            'in_position': False,
            'entry_price': 0,
            'entry_time': None,
            'action': 0,
            'size': 0.0,
            'tool_used': '',
            'scales_added': 0,
            'partial_exits': 0
        }

    def extract_subsystem_features(self, intelligence_result):
        """Extract features from subsystems - no interpretation"""
        features = []
        
        # Raw subsystem signals
        dna_signal = intelligence_result['subsystem_signals'].get('dna', 0.0)
        dna_confidence = intelligence_result['subsystem_scores'].get('dna', 0.0)
        dna_patterns = intelligence_result.get('similar_patterns_count', 0) / 10.0
        dna_length = min(intelligence_result.get('dna_sequence_length', 0) / 20.0, 1.0)
        features.extend([dna_signal, dna_confidence, dna_patterns, dna_length])
        
        micro_signal = intelligence_result['subsystem_signals'].get('micro', 0.0)
        micro_confidence = intelligence_result['subsystem_scores'].get('micro', 0.0)
        micro_strength = abs(micro_signal) * micro_confidence
        micro_active = 1.0 if intelligence_result.get('micro_pattern_id') else 0.0
        features.extend([micro_signal, micro_confidence, micro_strength, micro_active])
        
        temporal_signal = intelligence_result['subsystem_signals'].get('temporal', 0.0)
        temporal_confidence = intelligence_result['subsystem_scores'].get('temporal', 0.0)
        temporal_strength = abs(temporal_signal) * temporal_confidence
        temporal_active = 1.0 if abs(temporal_signal) > 0.1 else 0.0
        features.extend([temporal_signal, temporal_confidence, temporal_strength, temporal_active])
        
        immune_signal = intelligence_result['subsystem_signals'].get('immune', 0.0)
        immune_confidence = intelligence_result['subsystem_scores'].get('immune', 0.0)
        danger_flag = 1.0 if intelligence_result.get('is_dangerous_pattern') else 0.0
        benefit_flag = 1.0 if intelligence_result.get('is_beneficial_pattern') else 0.0
        features.extend([immune_signal, immune_confidence, danger_flag, benefit_flag])
        
        return np.array(features, dtype=np.float32)

    def get_performance_report(self) -> str:
        """Pure black box performance report"""
        
        base_report = self.agent.get_tool_performance_report()
        
        trades = self.trade_stats['total_trades']
        avg_pnl = self.trade_stats['total_pnl'] / max(1, trades)
        
        pure_stats = f"""

=== PURE BLACK BOX PERFORMANCE ===

Total Trades: {trades}
Average P&L: ${avg_pnl:.2f}
Best Trade: ${self.trade_stats['best_trade']:.2f}
Worst Trade: ${self.trade_stats['worst_trade']:.2f}
Total P&L: ${self.trade_stats['total_pnl']:.2f}

Tool Discovery Progress:
- DNA: {self.trade_stats['tool_discovery']['dna']} uses
- MICRO: {self.trade_stats['tool_discovery']['micro']} uses
- TEMPORAL: {self.trade_stats['tool_discovery']['temporal']} uses
- IMMUNE: {self.trade_stats['tool_discovery']['immune']} uses

Advanced Features:
- Scaling Trades: {self.trade_stats['scaling_trades']}
- Partial Exits: {self.trade_stats['partial_exit_trades']}
- Full Exits: {self.trade_stats['full_exit_trades']}

Safety Status:
- Current Phase: {self.safety_manager.current_phase}
- Daily P&L: ${self.safety_manager.daily_pnl:.2f}
- Consecutive Losses: {self.safety_manager.consecutive_losses}

AI discovering trading strategies through pure experience!
"""
        
        threshold_report = f"""

{self.confidence_learner.get_learning_status()}
"""
        
        if self.position_ai.current_position:
            pos = self.position_ai.current_position
            pure_stats += f"""
Current Position:
- Tool: {pos.tool_used.upper()}
- P&L: {pos.current_pnl:.2%}
- Duration: {(datetime.now() - pos.entry_time).total_seconds()/3600:.1f}h
- Scales: {pos.scales_added}
- Exits: {pos.partial_exits}
"""
        
        return base_report + pure_stats + threshold_report

    # External interface methods
    def on_trade_update(self, update_type: str, price: float, reason: str = ""):
        """Handle trade updates from NinjaTrader"""
        if update_type == "filled":
            log.info(f"Trade filled at ${price:.2f}")
        elif update_type == "stopped":
            self._complete_trade("stop_hit", price)
        elif update_type == "target_hit":
            self._complete_trade("target_hit", price)
        elif update_type == "closed":
            self._complete_trade(reason or "manual_close", price)

    def emergency_close_all(self):
        """Emergency close all positions"""
        if self.current_position['in_position']:
            self.tcp_bridge.send_signal(0, 0.9, "EMERGENCY_CLOSE")
            log.warning("EMERGENCY CLOSE: All positions closed")
        
        if self.position_ai.current_position:
            self.position_ai.close_position(0)