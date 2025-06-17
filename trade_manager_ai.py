# trade_manager_ai.py - Enhanced with Trader Wisdom Bootstrap (COMPLETED)

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
from rl_agent import StrategicToolLearningAgent
from market_env import MarketEnv
from advanced_position_management import PositionManagementLearner, PositionState

log = logging.getLogger(__name__)

class TraderWisdomBootstrap:
    """
    Bootstrap AI with trader wisdom as starting biases that can be learned away from
    """
    
    def __init__(self):
        # WISDOM: Tool-specific confidence biases (can be overridden by experience)
        self.tool_wisdom = {
            'dna': {
                'description': "DNA patterns work best in trending markets with clear momentum",
                'confidence_bias': {
                    'exit': 0.35,    # DNA needs higher confidence for exits (momentum can continue)
                    'scale': 0.45,   # DNA good for scaling in trends
                },
                'market_conditions': ['trending', 'strong_momentum'],
                'tips': [
                    "DNA patterns excel when market has clear direction",
                    "Scale into DNA trades when momentum is strong", 
                    "Don't exit DNA trades too early in trends"
                ]
            },
            
            'micro': {
                'description': "Micro patterns catch quick reversals and short-term moves",
                'confidence_bias': {
                    'exit': 0.25,    # Exit micro trades quickly 
                    'scale': 0.55,   # Be careful scaling micro patterns
                },
                'market_conditions': ['volatile', 'ranging'],
                'tips': [
                    "Micro patterns are for quick in-and-out trades",
                    "Take profits fast on micro patterns",
                    "Don't hold micro trades too long"
                ]
            },
            
            'temporal': {
                'description': "Temporal patterns work at specific times/sessions",
                'confidence_bias': {
                    'exit': 0.30,    # Hold temporal trades during optimal times
                    'scale': 0.35,   # Good for scaling during session times
                },
                'market_conditions': ['session_open', 'session_close', 'high_volume_times'],
                'tips': [
                    "Temporal patterns strongest during market open/close",
                    "Hold temporal trades during their optimal time windows",
                    "Scale temporal trades when volume is high"
                ]
            },
            
            'immune': {
                'description': "Immune system avoids danger and finds safe opportunities", 
                'confidence_bias': {
                    'exit': 0.20,    # Exit quickly when immune detects danger
                    'scale': 0.60,   # Very careful about scaling when immune is active
                },
                'market_conditions': ['uncertain', 'high_volatility', 'risk_off'],
                'tips': [
                    "When immune system activates, be defensive",
                    "Exit fast when immune detects danger patterns",
                    "Don't scale positions when immune system is warning"
                ]
            }
        }
        
        # WISDOM: Market regime biases
        self.regime_wisdom = {
            'trending': {
                'preferred_tools': ['dna', 'temporal'],
                'confidence_adjustments': {
                    'exit': +0.1,    # Hold longer in trends
                    'scale': -0.1    # Easier to scale in trends
                },
                'tips': "In trending markets, let winners run and scale into momentum"
            },
            
            'volatile': {
                'preferred_tools': ['micro', 'immune'],
                'confidence_adjustments': {
                    'exit': -0.15,   # Exit faster in volatility
                    'scale': +0.2    # Much harder to scale in volatility
                },
                'tips': "In volatile markets, take profits quickly and avoid scaling"
            },
            
            'sideways': {
                'preferred_tools': ['micro', 'immune'],
                'confidence_adjustments': {
                    'exit': -0.05,   # Exit moderately fast in range
                    'scale': +0.1    # Careful scaling in ranges
                },
                'tips': "In ranging markets, trade the bounces and don't overstay"
            }
        }
        
        # WISDOM: Time-based biases
        self.time_wisdom = {
            'market_open': {  # 9:30-10:30 AM
                'confidence_adjustments': {'exit': +0.05, 'scale': -0.05},
                'tip': "Market open has strong momentum - hold winners longer"
            },
            'mid_day': {      # 11:00-2:00 PM  
                'confidence_adjustments': {'exit': -0.10, 'scale': +0.15},
                'tip': "Mid-day is choppy - take profits faster, scale carefully"
            },
            'power_hour': {   # 3:00-4:00 PM
                'confidence_adjustments': {'exit': +0.1, 'scale': -0.1}, 
                'tip': "Power hour can trend - let good trades run"
            }
        }

class WisdomEnhancedConfidenceLearner:
    """Enhanced confidence learner that starts with trader wisdom"""
    
    def __init__(self):
        # Initialize trader wisdom
        self.wisdom = TraderWisdomBootstrap()
        
        # Start with wisdom-based thresholds instead of random
        self.confidence_thresholds = {
            'exit': {},
            'scale': {}
        }
        
        # Initialize thresholds based on trader wisdom
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            wisdom_data = self.wisdom.tool_wisdom[tool]
            self.confidence_thresholds['exit'][tool] = wisdom_data['confidence_bias']['exit']
            self.confidence_thresholds['scale'][tool] = wisdom_data['confidence_bias']['scale']
        
        # Track outcomes for threshold adjustment
        self.threshold_outcomes = {
            'exit': {'dna': [], 'micro': [], 'temporal': [], 'immune': []},
            'scale': {'dna': [], 'micro': [], 'temporal': [], 'immune': []}
        }
        
        # Learning parameters
        self.learning_rate = 0.02
        self.min_samples = 3
        
        # Track if wisdom is still being used vs learned
        self.wisdom_override_count = {'exit': {}, 'scale': {}}
        for action_type in ['exit', 'scale']:
            for tool in ['dna', 'micro', 'temporal', 'immune']:
                self.wisdom_override_count[action_type][tool] = 0
        
        log.info("TRADER WISDOM: AI initialized with expert trading knowledge")
        log.info("AI can learn to override wisdom when data suggests better approaches")
    
    def get_threshold(self, action_type: str, tool: str) -> float:
        """Get current learned threshold for tool/action"""
        return self.confidence_thresholds[action_type].get(tool, 0.2)
    
    def get_market_adjusted_threshold(self, action_type: str, tool: str, 
                                    market_regime: str = None, current_time: datetime = None) -> float:
        """Get threshold adjusted for market conditions and time"""
        
        base_threshold = self.get_threshold(action_type, tool)
        adjustments = []
        
        # Apply market regime wisdom
        if market_regime and market_regime in self.wisdom.regime_wisdom:
            regime_data = self.wisdom.regime_wisdom[market_regime]
            if tool in regime_data['preferred_tools']:
                # Tool is preferred for this regime - lower threshold
                adjustment = regime_data['confidence_adjustments'][action_type] * 0.5
                adjustments.append(f"regime_preferred:{adjustment:+.2f}")
            else:
                # Tool not preferred - use full adjustment
                adjustment = regime_data['confidence_adjustments'][action_type]
                adjustments.append(f"regime_caution:{adjustment:+.2f}")
            base_threshold += adjustment
        
        # Apply time-based wisdom
        if current_time:
            hour = current_time.hour
            if 9 <= hour <= 10:  # Market open
                time_adj = self.wisdom.time_wisdom['market_open']['confidence_adjustments'][action_type]
                adjustments.append(f"market_open:{time_adj*0.3:+.2f}")
            elif 11 <= hour <= 14:  # Mid day
                time_adj = self.wisdom.time_wisdom['mid_day']['confidence_adjustments'][action_type]
                adjustments.append(f"mid_day:{time_adj*0.3:+.2f}")
            elif 15 <= hour <= 16:  # Power hour
                time_adj = self.wisdom.time_wisdom['power_hour']['confidence_adjustments'][action_type]
                adjustments.append(f"power_hour:{time_adj*0.3:+.2f}")
            else:
                time_adj = 0
                adjustments.append("off_hours:0.00")
            
            base_threshold += time_adj * 0.3  # Reduced impact
        
        # Keep in bounds
        final_threshold = max(0.05, min(0.95, base_threshold))
        
        return final_threshold
    
    def should_take_action(self, action_type: str, tool: str, confidence: float,
                          market_regime: str = None, current_time: datetime = None) -> bool:
        """Enhanced decision with market wisdom"""
        
        adjusted_threshold = self.get_market_adjusted_threshold(
            action_type, tool, market_regime, current_time
        )
        
        return confidence >= adjusted_threshold
    
    def record_outcome(self, action_type: str, tool: str, confidence: float, outcome: float):
        """Record outcome and adjust thresholds"""
        
        # Store outcome with confidence level
        self.threshold_outcomes[action_type][tool].append({
            'confidence': confidence,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Keep recent history only
        if len(self.threshold_outcomes[action_type][tool]) > 20:
            self.threshold_outcomes[action_type][tool] = \
                self.threshold_outcomes[action_type][tool][-15:]
        
        # Learn from outcomes
        self._update_threshold(action_type, tool)
    
    def _update_threshold(self, action_type: str, tool: str):
        """Enhanced threshold update that can override wisdom"""
        
        outcomes = self.threshold_outcomes[action_type][tool]
        if len(outcomes) < self.min_samples:
            return
        
        current_threshold = self.confidence_thresholds[action_type][tool]
        wisdom_threshold = self.wisdom.tool_wisdom[tool]['confidence_bias'][action_type]
        
        # Analyze recent performance
        recent_outcomes = outcomes[-10:]
        
        # Separate by confidence level relative to current threshold
        above_threshold = [o for o in recent_outcomes if o['confidence'] >= current_threshold]
        below_threshold = [o for o in recent_outcomes if o['confidence'] < current_threshold]
        
        # Calculate success rates
        above_success = sum(1 for o in above_threshold if o['outcome'] > 0) / max(len(above_threshold), 1)
        below_success = sum(1 for o in below_threshold if o['outcome'] > 0) / max(len(below_threshold), 1) if below_threshold else 0
        
        # Check if we should override wisdom
        wisdom_performance = [o for o in recent_outcomes if abs(o['confidence'] - wisdom_threshold) < 0.1]
        if len(wisdom_performance) >= 5:
            wisdom_success = sum(1 for o in wisdom_performance if o['outcome'] > 0) / len(wisdom_performance)
            
            if wisdom_success < 0.3:  # Wisdom failing
                self.wisdom_override_count[action_type][tool] += 1
                log.info(f"WISDOM OVERRIDE: {tool.upper()} {action_type} wisdom failing ({wisdom_success:.1%}), learning new threshold")
        
        # Adjust threshold based on what's working
        if len(below_threshold) >= 2 and below_success > above_success + 0.2:
            # Lower confidence decisions are working better - lower threshold
            new_threshold = current_threshold * (1 - self.learning_rate)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"THRESHOLD LEARNING: Lowering {action_type} threshold for {tool.upper()} to {new_threshold:.3f} (low confidence working)")
            
        elif len(above_threshold) >= 2 and above_success < 0.4:
            # High confidence decisions are failing - raise threshold to be more selective
            new_threshold = current_threshold * (1 + self.learning_rate)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"THRESHOLD LEARNING: Raising {action_type} threshold for {tool.upper()} to {new_threshold:.3f} (high confidence failing)")
        
        elif len(above_threshold) >= 3 and above_success > 0.7:
            # High confidence decisions working well - slightly lower threshold to capture more
            new_threshold = current_threshold * (1 - self.learning_rate * 0.5)
            self.confidence_thresholds[action_type][tool] = new_threshold
            log.info(f"THRESHOLD LEARNING: Slightly lowering {action_type} threshold for {tool.upper()} to {new_threshold:.3f} (high confidence successful)")
        
        # Keep thresholds in reasonable bounds
        self.confidence_thresholds[action_type][tool] = max(0.05, min(0.95, self.confidence_thresholds[action_type][tool]))
        
        # Log when we deviate significantly from wisdom
        new_threshold = self.confidence_thresholds[action_type][tool]
        deviation = abs(new_threshold - wisdom_threshold)
        
        if deviation > 0.15:  # Significant deviation from wisdom
            log.info(f"WISDOM EVOLUTION: {tool.upper()} {action_type} learned threshold {new_threshold:.3f} " +
                    f"vs wisdom {wisdom_threshold:.3f} (deviation: {deviation:.3f})")
    
    def get_learning_status(self) -> str:
        """Enhanced status showing wisdom vs learned"""
        
        status = "WISDOM-ENHANCED CONFIDENCE LEARNING:\n"
        
        for action_type in ['exit', 'scale']:
            status += f"\n{action_type.upper()} Thresholds (Wisdom → Learned):\n"
            for tool, threshold in self.confidence_thresholds[action_type].items():
                wisdom_threshold = self.wisdom.tool_wisdom[tool]['confidence_bias'][action_type]
                samples = len(self.threshold_outcomes[action_type][tool])
                overrides = self.wisdom_override_count[action_type][tool]
                
                if samples > 0:
                    recent_success = sum(1 for o in self.threshold_outcomes[action_type][tool][-5:] if o['outcome'] > 0) / min(5, samples)
                    deviation = threshold - wisdom_threshold
                    status += f"  {tool.upper()}: {wisdom_threshold:.3f} → {threshold:.3f} " + \
                             f"(Δ{deviation:+.3f}, {samples} samples, {recent_success:.1%} success, {overrides} overrides)\n"
                else:
                    status += f"  {tool.upper()}: {wisdom_threshold:.3f} → {threshold:.3f} (using wisdom)\n"
        
        return status
    
    def get_wisdom_tip(self, tool: str, action_type: str, market_regime: str = None) -> str:
        """Get relevant wisdom tip for current situation"""
        
        tool_tips = self.wisdom.tool_wisdom[tool]['tips']
        base_tip = f"{tool.upper()}: {tool_tips[0]}"
        
        if market_regime and market_regime in self.wisdom.regime_wisdom:
            regime_tip = self.wisdom.regime_wisdom[market_regime]['tips']
            return f"{base_tip} | Market: {regime_tip}"
        
        return base_tip

class StrategicRLTradeManager:
    """
    Enhanced trade manager with black box AI and wisdom-enhanced confidence learning
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        
        # Black box AI that learns to use your subsystems
        self.agent = StrategicToolLearningAgent(
            market_obs_size=15,
            subsystem_features_size=16
        )
        
        # Advanced position management AI
        self.position_ai = PositionManagementLearner(self.agent)
        
        # NEW: Wisdom-enhanced confidence learning
        self.confidence_learner = WisdomEnhancedConfidenceLearner()
        
        # Environment for state tracking
        self.env = MarketEnv()
        
        # Enhanced state tracking
        self.last_market_obs = self.env.get_obs()
        self.last_subsystem_features = None
        self.last_decision = {}
        
        # Position tracking with more detail
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
        
        # Trade statistics for learning
        self.trade_stats = {
            'total_trades': 0,
            'scaling_trades': 0,
            'partial_exit_trades': 0,
            'full_exit_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_trade_duration_hours': 0.0
        }
        
        log.info("Enhanced Black Box Trade Manager with Trader Wisdom initialized")
        log.info("AI starts with trading wisdom and learns to improve upon it")

    def on_new_bar(self, msg: Dict[str, Any]):
        """
        ENHANCED: Process new bar with wisdom-enhanced position management
        """
        try:
            # Extract price data
            price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
            prices = msg.get("price_1m", [price])
            volumes = msg.get("volume_1m", [1000])
            
            # Update environment state
            market_obs = self.env.get_obs()
            self.env.step(price, 0)
            
            # Advanced position management logic with wisdom
            if self.current_position['in_position']:
                self._handle_position_management(price, market_obs)
            
            # Standard AI decision making for new entries
            if not self.current_position['in_position']:
                decision = self.agent.select_action_and_strategy(
                    market_obs=market_obs,
                    subsystem_features=self.extract_subsystem_features(
                        self.intel.process_market_data(prices, volumes, datetime.now())
                    ),
                    current_price=price,
                    in_position=self.current_position['in_position']
                )
                
                # Store decision data for learning
                self.last_decision = decision
                self.last_market_obs = market_obs
                self.last_subsystem_features = decision['raw_subsystem_features']
                
                # Execute entry decision
                self._execute_ai_decision(decision, price)
            
            # Log tool usage
            if self.last_decision:
                trust = self.last_decision.get('tool_trust', {})
                log.info(f"AI TOOL USAGE: DNA({trust.get('dna', 0):.2f}) "
                        f"MICRO({trust.get('micro', 0):.2f}) "
                        f"TEMPORAL({trust.get('temporal', 0):.2f}) "
                        f"IMMUNE({trust.get('immune', 0):.2f})")
                log.info(f"PRIMARY TOOL: {self.last_decision.get('primary_tool', 'UNKNOWN').upper()}")
            
        except Exception as e:
            log.error(f"Error processing new bar: {e}")
            import traceback
            traceback.print_exc()

    def _handle_position_management(self, current_price: float, market_obs: np.ndarray):
        """Handle position management with wisdom-enhanced thresholds"""
        
        if self.position_ai.current_position:
            self.position_ai.update_position(current_price)
            tool_used = self.position_ai.current_position.tool_used
            market_regime = getattr(self.position_ai.current_position, 'current_market_regime', 'unknown')
            
            # SCALING with wisdom-enhanced thresholds
            scale_decision = self.position_ai.should_scale_position(market_obs, current_price)
            
            # Use wisdom-enhanced threshold check
            if (scale_decision['action'] != 'no_scale' and 
                self.confidence_learner.should_take_action(
                    'scale', tool_used, scale_decision['confidence'], 
                    market_regime, datetime.now())):
                
                scale_amount = scale_decision['scale_amount']
                
                # Send scaling signal to NinjaTrader
                action_code = 1 if self.current_position['action'] == 1 else 2
                quality = f"AI_scale_{scale_decision['action']}_{tool_used}"
                
                self.tcp_bridge.send_signal(action_code, scale_decision['confidence'], quality)
                
                # Update internal tracking
                old_size = self.current_position['size']
                self.current_position['size'] += scale_amount
                self.current_position['scales_added'] += 1
                self.trade_stats['scaling_trades'] += 1
                
                threshold_used = self.confidence_learner.get_market_adjusted_threshold(
                    'scale', tool_used, market_regime, datetime.now()
                )
                wisdom_tip = self.confidence_learner.get_wisdom_tip(tool_used, 'scale', market_regime)
                
                log.info(f"🔄 AI SCALING (Wisdom-Enhanced): {scale_decision['reasoning']}")
                log.info(f"   Confidence: {scale_decision['confidence']:.3f} vs wisdom threshold {threshold_used:.3f}")
                log.info(f"   Wisdom: {wisdom_tip}")
                log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                
                # Record decision for learning
                self._pending_scale_decision = {
                    'tool': tool_used,
                    'confidence': scale_decision['confidence'],
                    'action_type': 'scale'
                }
            
            # EXITS with wisdom-enhanced thresholds
            exit_decision = self.position_ai.should_exit_position(market_obs, current_price)
            
            # Use wisdom-enhanced threshold check
            if (exit_decision['action'] != 'hold' and 
                self.confidence_learner.should_take_action(
                    'exit', tool_used, exit_decision['confidence'], 
                    market_regime, datetime.now())):
                
                threshold_used = self.confidence_learner.get_market_adjusted_threshold(
                    'exit', tool_used, market_regime, datetime.now()
                )
                wisdom_tip = self.confidence_learner.get_wisdom_tip(tool_used, 'exit', market_regime)
                
                if exit_decision['action'] == 'exit_100%':
                    # Full exit
                    self.tcp_bridge.send_signal(0, exit_decision['confidence'], 
                                              f"AI_exit_full_{tool_used}")
                    
                    self.trade_stats['full_exit_trades'] += 1
                    log.info(f"🚪 AI FULL EXIT (Wisdom-Enhanced): {exit_decision['reasoning']}")
                    log.info(f"   Confidence: {exit_decision['confidence']:.3f} vs wisdom threshold {threshold_used:.3f}")
                    log.info(f"   Wisdom: {wisdom_tip}")
                    
                    # Record decision for learning
                    self._pending_exit_decision = {
                        'tool': tool_used,
                        'confidence': exit_decision['confidence'],
                        'action_type': 'exit'
                    }
                    
                    # Complete the trade
                    self._complete_trade("AI_full_exit", current_price)
                    
                elif 'exit_' in exit_decision['action']:
                    # Partial exit
                    exit_amount = exit_decision['exit_amount']
                    exit_quality = f"AI_exit_{exit_decision['action']}_{tool_used}"
                    
                    # Send partial exit signal to NinjaTrader
                    self.tcp_bridge.send_signal(0, exit_decision['confidence'], exit_quality)
                    
                    # Update position size
                    old_size = self.current_position['size']
                    self.current_position['size'] *= (1.0 - exit_amount)
                    self.current_position['partial_exits'] += 1
                    self.trade_stats['partial_exit_trades'] += 1
                    
                    log.info(f"💰 AI PARTIAL EXIT (Wisdom-Enhanced): {exit_decision['reasoning']}")
                    log.info(f"   Confidence: {exit_decision['confidence']:.3f} vs wisdom threshold {threshold_used:.3f}")
                    log.info(f"   Wisdom: {wisdom_tip}")
                    log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                    log.info(f"   Exit amount: {exit_amount:.1%}")
                    
                    # Record partial exit for learning
                    normalized_outcome = 0.1  # Small positive for risk management
                    self.confidence_learner.record_outcome(
                        'exit', tool_used, exit_decision['confidence'], normalized_outcome
                    )
            
            # TRAILING STOPS ALWAYS AVAILABLE
            trail_distance = self.position_ai.get_trail_stop_distance(market_obs, current_price)
            
            if trail_distance > 0:
                quality = f"AI_trail_{trail_distance:.3f}_{tool_used}"
                log.info(f"🎯 AI TRAIL STOP: Distance {trail_distance:.2%} from current profit")
            
            # Log wisdom status every 10 minutes for monitoring
            if hasattr(self, '_last_wisdom_log'):
                if (datetime.now() - self._last_wisdom_log).total_seconds() > 600:
                    log.info(self.confidence_learner.get_learning_status())
                    self._last_wisdom_log = datetime.now()
            else:
                self._last_wisdom_log = datetime.now()

    def _execute_ai_decision(self, decision: Dict, current_price: float):
        """Execute AI decision with position tracking"""
        
        action = decision['action']
        confidence = decision['confidence']
        
        # Enter new position
        if action != 0 and not self.current_position['in_position']:
            
            action_code = 1 if action == 1 else 2
            tool_name = decision['primary_tool']
            direction = 'long' if action == 1 else 'short'
            
            # Build quality string
            quality = f"AI_{tool_name}_{direction}"
            
            # Risk management
            stop_price = 0.0
            target_price = 0.0

            if decision['use_stop'] and decision['stop_price']:
                stop_price = decision['stop_price']
                quality += f"_stop{decision['stop_distance_pct']:.1f}pct"

            if decision['use_target'] and decision['target_price']:
                target_price = decision['target_price']
                quality += f"_target{decision['target_distance_pct']:.1f}pct"

            # Send signal to NinjaTrader with actual prices
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_price, target_price)
            
            # Track position with full details
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'size': 1.0,
                'tool_used': tool_name,
                'scales_added': 0,
                'partial_exits': 0
            }
            
            # Start position tracking in AI
            self.position_ai.start_position(
                entry_price=current_price,
                initial_size=1.0,
                tool_used=tool_name,
                entry_confidence=confidence,
                market_regime=decision.get('market_regime', 'unknown')
            )
            
            self.trade_stats['total_trades'] += 1
            
            log.info(f"🎯 BLACK BOX ENTRY using {tool_name.upper()} tool:")
            log.info(f"   Direction: {direction}, Confidence: {confidence:.3f}")
            log.info(f"   Entry Price: ${current_price:.2f}")
            log.info(f"   Trade #{self.trade_stats['total_trades']}")
            
            # Show wisdom for this tool
            wisdom_tip = self.confidence_learner.get_wisdom_tip(
                tool_name, 'exit', decision.get('market_regime', 'unknown')
            )
            log.info(f"   Trader Wisdom: {wisdom_tip}")
            
            # Show learning progression
            self._log_learning_progression()
            
            if decision['use_stop']:
                log.info(f"   Stop: ${decision['stop_price']:.2f}")
            if decision['use_target']:
                log.info(f"   Target: ${decision['target_price']:.2f}")

    def _log_learning_progression(self):
        """Log what AI features are active"""
        trades = self.trade_stats['total_trades']
        
        active_features = [
            "Entry/Exit", 
            "Partial Exits", 
            "Position Scaling", 
            "Trailing Stops", 
            "Trader Wisdom Learning"
        ]
        
        log.info(f"   All AI Features Active: {' + '.join(active_features)}")
        log.info(f"   Trade #{trades} - Learning to improve upon trader wisdom")

    def record_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """
        Record trade outcome with wisdom-enhanced learning
        """
        try:
            if not self.last_decision:
                return
            
            # Update trade statistics
            self.trade_stats['total_pnl'] += pnl
            self.trade_stats['best_trade'] = max(self.trade_stats['best_trade'], pnl)
            self.trade_stats['worst_trade'] = min(self.trade_stats['worst_trade'], pnl)
            
            # Calculate trade duration
            if self.current_position['entry_time']:
                duration_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
                current_avg = self.trade_stats['avg_trade_duration_hours']
                total_trades = self.trade_stats['total_trades']
                self.trade_stats['avg_trade_duration_hours'] = (
                    (current_avg * (total_trades - 1) + duration_hours) / total_trades
                )
            
            # Base reward from P&L
            base_reward = pnl / 50.0  # Normalize for MNQ
            
            # Enhanced bonus rewards with WISDOM EVALUATION
            bonus_reward = 0.0
            
            # Evaluate if AI followed or contradicted trader wisdom correctly
            primary_tool = self.last_decision.get('primary_tool', '')
            market_regime = self.last_decision.get('market_regime', 'unknown')
            tool_trust = self.last_decision.get('tool_trust', {}).get(primary_tool, 0.0)
            
            # WISDOM VALIDATION: Check if AI's tool choice matched wisdom
            if market_regime in self.confidence_learner.wisdom.regime_wisdom:
                preferred_tools = self.confidence_learner.wisdom.regime_wisdom[market_regime]['preferred_tools']
                
                if primary_tool in preferred_tools and pnl > 10:
                    bonus_reward += 0.4
                    log.info(f"WISDOM VALIDATION: +0.4 for correctly using {primary_tool.upper()} in {market_regime} market")
                elif primary_tool not in preferred_tools and pnl > 10:
                    bonus_reward += 0.6
                    log.info(f"WISDOM OVERRIDE SUCCESS: +0.6 for successful override - {primary_tool.upper()} worked better than wisdom in {market_regime}")
                elif primary_tool in preferred_tools and pnl < -20:
                    bonus_reward -= 0.2
                    log.info(f"WISDOM FAILURE: -0.2 for wisdom tool {primary_tool.upper()} failing in expected {market_regime} market")
                elif primary_tool not in preferred_tools and pnl < -20:
                    bonus_reward -= 0.5
                    log.info(f"WISDOM IGNORED: -0.5 for ignoring wisdom and using {primary_tool.upper()} poorly in {market_regime}")
            
            # TIME-BASED WISDOM VALIDATION
            current_hour = datetime.now().hour
            time_period = None
            if 9 <= current_hour <= 10:
                time_period = 'market_open'
            elif 11 <= current_hour <= 14:
                time_period = 'mid_day'
            elif 15 <= current_hour <= 16:
                time_period = 'power_hour'
            
            if time_period and time_period in self.confidence_learner.wisdom.time_wisdom:
                time_wisdom = self.confidence_learner.wisdom.time_wisdom[time_period]
                
                # Check if AI's exit timing aligned with time wisdom
                if exit_reason == "AI_full_exit":
                    if time_period == 'market_open' and pnl > 20:
                        bonus_reward += 0.3
                        log.info(f"TIME WISDOM: +0.3 for holding during {time_period} momentum")
                    elif time_period == 'mid_day' and pnl > 0:
                        bonus_reward += 0.3
                        log.info(f"TIME WISDOM: +0.3 for taking profits during choppy {time_period}")
                    elif time_period == 'power_hour' and pnl > 30:
                        bonus_reward += 0.4
                        log.info(f"TIME WISDOM: +0.4 for riding {time_period} trend")
            
            # TOOL-SPECIFIC WISDOM EVALUATION
            tool_tips = self.confidence_learner.wisdom.tool_wisdom[primary_tool]['tips']
            
            # DNA tool wisdom evaluation
            if primary_tool == 'dna':
                if market_regime == 'trending' and pnl > 20:
                    bonus_reward += 0.5
                    log.info("DNA WISDOM: +0.5 for using DNA in trending market as wisdom suggests")
                elif self.current_position['scales_added'] > 0 and pnl > 30:
                    bonus_reward += 0.4
                    log.info("DNA WISDOM: +0.4 for scaling DNA trade as wisdom suggests")
                elif exit_reason in ["AI_full_exit", "stop_hit"] and duration_hours < 1 and market_regime == 'trending':
                    bonus_reward -= 0.3
                    log.info("DNA WISDOM: -0.3 for exiting DNA trade too early in trend (against wisdom)")
            
            # Micro tool wisdom evaluation
            elif primary_tool == 'micro':
                if duration_hours < 2 and pnl > 0:
                    bonus_reward += 0.4
                    log.info("MICRO WISDOM: +0.4 for quick micro trade exit as wisdom suggests")
                elif self.current_position['scales_added'] > 0 and pnl < 0:
                    bonus_reward -= 0.4
                    log.info("MICRO WISDOM: -0.4 for scaling micro pattern (against wisdom)")
                elif duration_hours > 4 and pnl < 0:
                    bonus_reward -= 0.3
                    log.info("MICRO WISDOM: -0.3 for holding micro trade too long (against wisdom)")
            
            # Temporal tool wisdom evaluation
            elif primary_tool == 'temporal':
                if time_period in ['market_open', 'power_hour'] and pnl > 15:
                    bonus_reward += 0.4
                    log.info("TEMPORAL WISDOM: +0.4 for using temporal during optimal session time")
                elif time_period == 'mid_day' and pnl < -10:
                    bonus_reward -= 0.3
                    log.info("TEMPORAL WISDOM: -0.3 for temporal trade during weak session time")
            
            # Immune tool wisdom evaluation
            elif primary_tool == 'immune':
                if market_regime in ['volatile', 'uncertain'] and pnl > -5:
                    bonus_reward += 0.4
                    log.info("IMMUNE WISDOM: +0.4 for defensive immune play limiting losses")
                elif self.current_position['scales_added'] == 0 and market_regime == 'volatile':
                    bonus_reward += 0.3
                    log.info("IMMUNE WISDOM: +0.3 for not scaling during immune warning")
                elif pnl < -30:
                    bonus_reward -= 0.4
                    log.info("IMMUNE WISDOM: -0.4 for immune system failing to protect")
            
            # CONFIDENCE THRESHOLD WISDOM EVALUATION
            primary_confidence = self.last_decision.get('confidence', 0.0)
            wisdom_threshold = self.confidence_learner.wisdom.tool_wisdom[primary_tool]['confidence_bias']['exit']
            
            if primary_confidence > wisdom_threshold and pnl > 10:
                bonus_reward += 0.2
                log.info(f"CONFIDENCE WISDOM: +0.2 for high confidence ({primary_confidence:.2f}) above wisdom threshold ({wisdom_threshold:.2f})")
            elif primary_confidence < wisdom_threshold and pnl < -15:
                bonus_reward -= 0.2
                log.info(f"CONFIDENCE WISDOM: -0.2 for low confidence ({primary_confidence:.2f}) below wisdom threshold ({wisdom_threshold:.2f})")
            
            # Standard position management rewards
            if self.current_position['scales_added'] > 0:
                if pnl > 20:  # Successful scaling
                    bonus_reward += 0.3 * self.current_position['scales_added']
                    log.info(f"SCALING: +{0.3 * self.current_position['scales_added']:.1f} for successful scaling")
                else:  # Failed scaling
                    bonus_reward -= 0.2 * self.current_position['scales_added']
                    log.info(f"SCALING: -{0.2 * self.current_position['scales_added']:.1f} for failed scaling")
            
            if self.current_position['partial_exits'] > 0:
                if pnl > 0:
                    bonus_reward += 0.3
                    log.info("PARTIAL EXITS: +0.3 for smart partial exits")
                else:
                    bonus_reward += 0.1  # Still reward for risk management
                    log.info("PARTIAL EXITS: +0.1 for taking some risk off")
            
            # Risk management rewards
            if exit_reason == "stop_hit" and self.last_decision.get('use_stop', False):
                if pnl > -30:
                    bonus_reward += 0.3
                    log.info("RISK MGMT: +0.3 for protective stop limiting loss")
                else:
                    bonus_reward += 0.1
            
            if exit_reason == "target_hit" and self.last_decision.get('use_target', False):
                bonus_reward += 0.3
                log.info("RISK MGMT: +0.3 for hitting profit target")
            
            # Penalize big losses without stops
            if pnl < -50 and not self.last_decision.get('use_stop', False):
                bonus_reward -= 0.6
                log.info("RISK MGMT: -0.6 for big loss without stop")
            
            total_reward = base_reward + bonus_reward
            
            # NEW: Update confidence thresholds based on outcome with WISDOM CONTEXT
            normalized_outcome = pnl / 50.0  # Normalize P&L for learning
            
            # Record outcomes for confidence learning with wisdom enhancement
            if hasattr(self, '_pending_exit_decision'):
                decision = self._pending_exit_decision
                self.confidence_learner.record_outcome(
                    decision['action_type'],
                    decision['tool'], 
                    decision['confidence'],
                    normalized_outcome
                )
                log.info(f"CONFIDENCE LEARNING: Updated {decision['action_type']} threshold for {decision['tool'].upper()} based on P&L ${pnl:.2f}")
                delattr(self, '_pending_exit_decision')
            
            if hasattr(self, '_pending_scale_decision'):
                decision = self._pending_scale_decision
                self.confidence_learner.record_outcome(
                    decision['action_type'],
                    decision['tool'], 
                    decision['confidence'],
                    normalized_outcome
                )
                log.info(f"CONFIDENCE LEARNING: Updated {decision['action_type']} threshold for {decision['tool'].upper()} based on P&L ${pnl:.2f}")
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
            
            # Enhanced logging with wisdom evaluation
            log.info(f"🎓 TRADER WISDOM LEARNING SUMMARY:")
            log.info(f"   P&L: ${pnl:.2f} -> Total Reward: {total_reward:.3f}")
            log.info(f"   Tool: {primary_tool.upper()} (trust: {tool_trust:.2f})")
            log.info(f"   Market: {market_regime}, Duration: {duration_hours:.1f}h")
            log.info(f"   Base reward: {base_reward:.3f}, Wisdom bonus: {bonus_reward:.3f}")
            
            # Show wisdom evolution
            current_threshold = self.confidence_learner.get_threshold('exit', primary_tool)
            wisdom_threshold = self.confidence_learner.wisdom.tool_wisdom[primary_tool]['confidence_bias']['exit']
            deviation = current_threshold - wisdom_threshold
            
            if abs(deviation) > 0.1:
                log.info(f"   WISDOM EVOLUTION: Exit threshold {current_threshold:.3f} vs original wisdom {wisdom_threshold:.3f} (Δ{deviation:+.3f})")
            else:
                log.info(f"   WISDOM STATUS: Still following wisdom threshold {wisdom_threshold:.3f}")
            
            # Update last state
            self.last_market_obs = next_market_obs
            
        except Exception as e:
            log.error(f"Error recording trade outcome: {e}")

    def _complete_trade(self, reason: str, exit_price: float):
        """Complete trade with position AI learning"""
        if not self.current_position['in_position']:
            return
        
        entry_price = self.current_position['entry_price']
        action = self.current_position['action']
        
        # Calculate P&L
        if action == 1:  # Long
            pnl = (exit_price - entry_price) * 2.0  # $2 per point for MNQ
        else:  # Short
            pnl = (entry_price - exit_price) * 2.0
        
        # Enhanced logging with wisdom context
        duration_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
        
        log.info(f"📊 WISDOM-ENHANCED TRADE COMPLETED:")
        log.info(f"   Tool: {self.current_position['tool_used'].upper()}")
        log.info(f"   P&L: ${pnl:.2f} ({pnl/50:.1%})")  # % assuming $5k account
        log.info(f"   Duration: {duration_hours:.1f}h")
        log.info(f"   Scales Added: {self.current_position['scales_added']}")
        log.info(f"   Partial Exits: {self.current_position['partial_exits']}")
        log.info(f"   Exit Reason: {reason}")
        
        # Show applicable wisdom
        tool_used = self.current_position['tool_used']
        current_hour = datetime.now().hour
        
        if 9 <= current_hour <= 10:
            wisdom_tip = self.confidence_learner.wisdom.time_wisdom['market_open']['tip']
            log.info(f"   Market Open Wisdom: {wisdom_tip}")
        elif 11 <= current_hour <= 14:
            wisdom_tip = self.confidence_learner.wisdom.time_wisdom['mid_day']['tip']
            log.info(f"   Mid-Day Wisdom: {wisdom_tip}")
        elif 15 <= current_hour <= 16:
            wisdom_tip = self.confidence_learner.wisdom.time_wisdom['power_hour']['tip']
            log.info(f"   Power Hour Wisdom: {wisdom_tip}")
        
        # Record outcome for both AIs
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
        """Extract features from existing subsystems"""
        features = []
        
        # DNA system features
        dna_signal = intelligence_result['subsystem_signals'].get('dna', 0.0)
        dna_confidence = intelligence_result['subsystem_scores'].get('dna', 0.0)
        dna_patterns_found = intelligence_result.get('similar_patterns_count', 0) / 10.0
        dna_sequence_quality = min(intelligence_result.get('dna_sequence_length', 0) / 20.0, 1.0)
        features.extend([dna_signal, dna_confidence, dna_patterns_found, dna_sequence_quality])
        
        # Micro pattern features
        micro_signal = intelligence_result['subsystem_signals'].get('micro', 0.0)
        micro_confidence = intelligence_result['subsystem_scores'].get('micro', 0.0)
        micro_pattern_strength = abs(micro_signal) * micro_confidence
        micro_pattern_reliability = 1.0 if intelligence_result.get('micro_pattern_id') else 0.0
        features.extend([micro_signal, micro_confidence, micro_pattern_strength, micro_pattern_reliability])
        
        # Temporal features
        temporal_signal = intelligence_result['subsystem_signals'].get('temporal', 0.0)
        temporal_confidence = intelligence_result['subsystem_scores'].get('temporal', 0.0)
        temporal_timing_quality = temporal_confidence
        temporal_session_relevance = 1.0 if abs(temporal_signal) > 0.1 else 0.0
        features.extend([temporal_signal, temporal_confidence, temporal_timing_quality, temporal_session_relevance])
        
        # Immune system features
        immune_signal = intelligence_result['subsystem_signals'].get('immune', 0.0)
        immune_confidence = intelligence_result['subsystem_scores'].get('immune', 0.0)
        danger_detected = 1.0 if intelligence_result.get('is_dangerous_pattern') else 0.0
        beneficial_detected = 1.0 if intelligence_result.get('is_beneficial_pattern') else 0.0
        features.extend([immune_signal, immune_confidence, danger_detected, beneficial_detected])
        
        return np.array(features, dtype=np.float32)

    def get_performance_report(self) -> str:
        """Enhanced performance report with wisdom learning status"""
        
        base_report = self.agent.get_tool_performance_report()
        
        # Add position management stats
        trades = self.trade_stats['total_trades']
        if trades > 0:
            avg_pnl = self.trade_stats['total_pnl'] / trades
        else:
            avg_pnl = 0
        
        position_stats = f"""

=== WISDOM-ENHANCED POSITION MANAGEMENT ===

Total Trades: {trades}
Average P&L per Trade: ${avg_pnl:.2f}
Best Trade: ${self.trade_stats['best_trade']:.2f}
Worst Trade: ${self.trade_stats['worst_trade']:.2f}
Total P&L: ${self.trade_stats['total_pnl']:.2f}
Average Duration: {self.trade_stats['avg_trade_duration_hours']:.1f}h

Advanced Features Usage:
- Scaling Trades: {self.trade_stats['scaling_trades']} ({self.trade_stats['scaling_trades']/max(1,trades):.1%})
- Partial Exit Trades: {self.trade_stats['partial_exit_trades']} ({self.trade_stats['partial_exit_trades']/max(1,trades):.1%})
- Full Exit Trades: {self.trade_stats['full_exit_trades']} ({self.trade_stats['full_exit_trades']/max(1,trades):.1%})

AI starts with trader wisdom and learns to improve upon it!
"""
        
        # Add wisdom learning status
        wisdom_report = f"""

{self.confidence_learner.get_learning_status()}

CURRENT TRADER WISDOM TIPS:
"""
        
        for tool in ['dna', 'micro', 'temporal', 'immune']:
            tips = self.confidence_learner.wisdom.tool_wisdom[tool]['tips']
            wisdom_report += f"  {tool.upper()}: {tips[0]}\n"
        
        wisdom_report += f"""
MARKET REGIME WISDOM:
  Trending Markets: Use DNA and Temporal tools - let winners run
  Volatile Markets: Use Micro and Immune tools - take profits quickly  
  Sideways Markets: Use Micro and Immune tools - trade the bounces

TIME-BASED WISDOM:
  Market Open (9:30-10:30): Strong momentum - hold winners longer
  Mid-Day (11:00-2:00): Choppy action - take profits faster
  Power Hour (3:00-4:00): Can trend - let good trades run
"""
        
        if self.position_ai.current_position:
            pos = self.position_ai.current_position
            position_stats += f"""
Current Position:
- Tool: {pos.tool_used.upper()}
- P&L: {pos.current_pnl:.2%}
- Time: {(datetime.now() - pos.entry_time).total_seconds()/3600:.1f}h
- Scales: {pos.scales_added}
- Exits: {pos.partial_exits}
- Max Profit: {pos.max_favorable_excursion:.2%}
- Max Loss: {pos.max_adverse_excursion:.2%}
"""
        else:
            position_stats += "\nCurrent Position: None"
        
        return base_report + position_stats + wisdom_report

    def get_wisdom_evolution_report(self) -> str:
        """Get report on how AI has evolved beyond initial wisdom"""
        
        report = "🎓 TRADER WISDOM EVOLUTION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for action_type in ['exit', 'scale']:
            report += f"{action_type.upper()} Threshold Evolution:\n"
            
            for tool in ['dna', 'micro', 'temporal', 'immune']:
                current = self.confidence_learner.get_threshold(action_type, tool)
                wisdom = self.confidence_learner.wisdom.tool_wisdom[tool]['confidence_bias'][action_type]
                deviation = current - wisdom
                samples = len(self.confidence_learner.threshold_outcomes[action_type][tool])
                overrides = self.confidence_learner.wisdom_override_count[action_type][tool]
                
                status = "EVOLVED" if abs(deviation) > 0.1 else "FOLLOWING WISDOM"
                
                report += f"  {tool.upper()}: {wisdom:.3f} → {current:.3f} "
                report += f"({status}, {samples} samples, {overrides} overrides)\n"
            
            report += "\n"
        
        # Show wisdom tips vs learned behavior
        report += "WISDOM vs LEARNED BEHAVIOR:\n\n"
        
        tool_prefs = self.agent.get_current_tool_preferences()
        for tool, preference in tool_prefs.items():
            wisdom_desc = self.confidence_learner.wisdom.tool_wisdom[tool]['description']
            usage = self.agent.tool_usage_count[tool]
            success = self.agent.successful_tool_usage[tool]
            success_rate = success / usage if usage > 0 else 0.0
            
            report += f"{tool.upper()}:\n"
            report += f"  Wisdom: {wisdom_desc}\n"
            report += f"  Learned: {usage} uses, {success_rate:.1%} success, {preference:.2f} preference\n"
            
            if success_rate > 0.7:
                report += f"  Status: AI VALIDATES wisdom - performing well\n"
            elif success_rate < 0.4 and usage > 10:
                report += f"  Status: AI CHALLENGES wisdom - exploring better approaches\n"
            else:
                report += f"  Status: AI LEARNING - gathering more data\n"
            report += "\n"
        
        return report

    # External interface methods for NinjaTrader callbacks
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
            log.warning("EMERGENCY CLOSE: All positions closed by user request")
        
        if self.position_ai.current_position:
            self.position_ai.close_position(0)  # Emergency close in AI tracking