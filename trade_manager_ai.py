# trade_manager_ai.py - Enhanced with Advanced Position Management

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
from rl_agent import StrategicToolLearningAgent
from market_env import MarketEnv
from advanced_position_management import PositionManagementLearner, PositionState

log = logging.getLogger(__name__)

class BlackBoxTradeManagerWithSubsystems:
    """
    Enhanced trade manager with black box AI and advanced position management
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        
        # Black box AI that learns to use your subsystems
        self.agent = StrategicToolLearningAgent(
            market_obs_size=15,
            subsystem_features_size=16
        )
        
        # NEW: Advanced position management AI
        self.position_ai = PositionManagementLearner(self.agent)
        
        # Environment for state tracking
        self.env = MarketEnv()
        
        # Enhanced state tracking
        self.last_market_obs = self.env.get_obs()
        self.last_subsystem_features = None
        self.last_decision = {}
        
        # ENHANCED: Position tracking with more detail
        self.current_position = {
            'in_position': False,
            'entry_price': 0,
            'entry_time': None,
            'action': 0,
            'size': 0.0,  # NEW: Track size
            'tool_used': '',  # NEW: Track which tool
            'scales_added': 0,  # NEW: Track scaling
            'partial_exits': 0  # NEW: Track exits
        }
        
        # NEW: Trade statistics for learning
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
        
        # Learning progression thresholds
        self.learning_thresholds = {
            'partial_exits': 5,      # After 5 trades
            'position_scaling': 15,  # After 15 trades  
            'trailing_stops': 30,    # After 30 trades
            'advanced_management': 50 # After 50 trades
        }
        
        log.info("Enhanced Black Box Trade Manager with Position AI initialized")
        log.info("Learning progression: Basic -> Exits -> Scaling -> Trailing -> Advanced")

    def on_new_bar(self, msg: Dict[str, Any]):
        """
        ENHANCED: Process new bar with advanced position management
        """
        try:
            # Extract price data
            price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
            prices = msg.get("price_1m", [price])
            volumes = msg.get("volume_1m", [1000])
            
            # Update environment state
            market_obs = self.env.get_obs()
            self.env.step(price, 0)
            
            # NEW: Advanced position management logic
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
        """NEW: Handle advanced position management"""
        
        # Update position AI state
        if self.position_ai.current_position:
            self.position_ai.update_position(current_price)
            
            # Check for scaling opportunities (only after enough experience)
            if self.trade_stats['total_trades'] >= self.learning_thresholds['position_scaling']:
                scale_decision = self.position_ai.should_scale_position(market_obs, current_price)
                
                if (scale_decision['action'] != 'no_scale' and 
                    scale_decision['confidence'] > 0.65):
                    
                    scale_amount = scale_decision['scale_amount']
                    
                    # Send scaling signal to NinjaTrader
                    action_code = 1 if self.current_position['action'] == 1 else 2
                    quality = f"AI_scale_{scale_decision['action']}_{self.position_ai.current_position.tool_used}"
                    
                    self.tcp_bridge.send_signal(action_code, scale_decision['confidence'], quality)
                    
                    # Update internal tracking
                    old_size = self.current_position['size']
                    self.current_position['size'] += scale_amount
                    self.current_position['scales_added'] += 1
                    self.trade_stats['scaling_trades'] += 1
                    
                    log.info(f"🔄 AI SCALING: {scale_decision['reasoning']}")
                    log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                    log.info(f"   Confidence: {scale_decision['confidence']:.3f}")
            
            # Check for exit opportunities (available earlier than scaling)
            if self.trade_stats['total_trades'] >= self.learning_thresholds['partial_exits']:
                exit_decision = self.position_ai.should_exit_position(market_obs, current_price)
                
                if (exit_decision['action'] != 'hold' and 
                    exit_decision['confidence'] > 0.7):
                    
                    if exit_decision['action'] == 'exit_100%':
                        # Full exit
                        self.tcp_bridge.send_signal(0, exit_decision['confidence'], 
                                                  f"AI_exit_full_{self.position_ai.current_position.tool_used}")
                        
                        self.trade_stats['full_exit_trades'] += 1
                        log.info(f"🚪 AI FULL EXIT: {exit_decision['reasoning']}")
                        
                        # Complete the trade
                        self._complete_trade("AI_full_exit", current_price)
                        
                    elif 'exit_' in exit_decision['action']:
                        # Partial exit
                        exit_amount = exit_decision['exit_amount']
                        exit_quality = f"AI_exit_{exit_decision['action']}_{self.position_ai.current_position.tool_used}"
                        
                        # Send partial exit signal to NinjaTrader
                        self.tcp_bridge.send_signal(0, exit_decision['confidence'], exit_quality)
                        
                        # Update position size
                        old_size = self.current_position['size']
                        self.current_position['size'] *= (1.0 - exit_amount)
                        self.current_position['partial_exits'] += 1
                        self.trade_stats['partial_exit_trades'] += 1
                        
                        log.info(f"💰 AI PARTIAL EXIT: {exit_decision['reasoning']}")
                        log.info(f"   Size: {old_size:.1f} -> {self.current_position['size']:.1f}")
                        log.info(f"   Exit amount: {exit_amount:.1%}")
            
            # Check for trailing stop adjustments (advanced feature)
            if self.trade_stats['total_trades'] >= self.learning_thresholds['trailing_stops']:
                trail_distance = self.position_ai.get_trail_stop_distance(market_obs, current_price)
                
                if trail_distance > 0:
                    # Send trailing stop update to NinjaTrader
                    quality = f"AI_trail_{trail_distance:.3f}_{self.position_ai.current_position.tool_used}"
                    # You could send this as a special signal type for trailing stops
                    log.info(f"🎯 AI TRAIL STOP: Distance {trail_distance:.2%} from current profit")

    def _execute_ai_decision(self, decision: Dict, current_price: float):
        """ENHANCED: Execute AI decision with position tracking"""
        
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
                stop_price = decision['stop_price']  # Use the actual price
                quality += f"_stop{decision['stop_distance_pct']:.1f}pct"

            if decision['use_target'] and decision['target_price']:
                target_price = decision['target_price']  # Use the actual price
                quality += f"_target{decision['target_distance_pct']:.1f}pct"

            # Send signal to NinjaTrader with actual prices
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_price, target_price)
            
            # ENHANCED: Track position with full details
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'size': 1.0,  # Base size
                'tool_used': tool_name,
                'scales_added': 0,
                'partial_exits': 0
            }
            
            # NEW: Start position tracking in AI
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
            
            # Show learning progression
            self._log_learning_progression()
            
            if decision['use_stop']:
                log.info(f"   Stop: ${decision['stop_price']:.2f}")
            if decision['use_target']:
                log.info(f"   Target: ${decision['target_price']:.2f}")

    def _log_learning_progression(self):
        """Log what AI features are currently active"""
        trades = self.trade_stats['total_trades']
        
        active_features = ["Entry/Exit"]
        
        if trades >= self.learning_thresholds['partial_exits']:
            active_features.append("Partial Exits")
        if trades >= self.learning_thresholds['position_scaling']:
            active_features.append("Position Scaling")
        if trades >= self.learning_thresholds['trailing_stops']:
            active_features.append("Trailing Stops")
        if trades >= self.learning_thresholds['advanced_management']:
            active_features.append("Advanced Management")
        
        log.info(f"   Active AI Features: {' + '.join(active_features)}")
        
        # Show next unlock
        for feature, threshold in self.learning_thresholds.items():
            if trades < threshold:
                remaining = threshold - trades
                log.info(f"   Next unlock: {feature.replace('_', ' ').title()} in {remaining} trades")
                break

    def record_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """
        ENHANCED: Record trade outcome with position management learning
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
            
            # Enhanced bonus rewards for position management
            bonus_reward = 0.0
            
            # Reward successful tool selection
            primary_tool = self.last_decision.get('primary_tool', '')
            tool_trust = self.last_decision.get('tool_trust', {}).get(primary_tool, 0.0)
            
            if pnl > 10 and tool_trust > 0.6:
                bonus_reward += 0.3
                log.info(f"BLACK BOX LEARNING: +0.3 for trusting {primary_tool} tool")
            
            if pnl < -20 and tool_trust > 0.8:
                bonus_reward -= 0.4
                log.info(f"BLACK BOX LEARNING: -0.4 for over-trusting {primary_tool} tool")
            
            # NEW: Reward good position management
            if self.current_position['scales_added'] > 0:
                if pnl > 20:  # Successful scaling
                    bonus_reward += 0.4 * self.current_position['scales_added']
                    log.info(f"BLACK BOX LEARNING: +{0.4 * self.current_position['scales_added']:.1f} for successful scaling")
                else:  # Failed scaling
                    bonus_reward -= 0.2 * self.current_position['scales_added']
                    log.info(f"BLACK BOX LEARNING: -{0.2 * self.current_position['scales_added']:.1f} for failed scaling")
            
            if self.current_position['partial_exits'] > 0:
                if pnl > 0:  # Good partial exit strategy
                    bonus_reward += 0.3
                    log.info("BLACK BOX LEARNING: +0.3 for smart partial exits")
                else:
                    bonus_reward += 0.1  # Still reward for risk management
                    log.info("BLACK BOX LEARNING: +0.1 for taking some risk off")
            
            # Enhanced risk management rewards
            if exit_reason == "stop_hit" and self.last_decision.get('use_stop', False):
                if pnl > -30:
                    bonus_reward += 0.3
                    log.info("BLACK BOX LEARNING: +0.3 for protective stop limiting loss")
                else:
                    bonus_reward += 0.1
            
            if exit_reason == "target_hit" and self.last_decision.get('use_target', False):
                bonus_reward += 0.3
                log.info("BLACK BOX LEARNING: +0.3 for hitting profit target")
            
            if exit_reason == "AI_full_exit":
                if pnl > 30:
                    bonus_reward += 0.4
                    log.info("BLACK BOX LEARNING: +0.4 for smart AI exit with profits")
                elif pnl > 0:
                    bonus_reward += 0.2
                    log.info("BLACK BOX LEARNING: +0.2 for AI exit with small profit")
                else:
                    bonus_reward += 0.1
                    log.info("BLACK BOX LEARNING: +0.1 for AI cutting losses")
            
            # Penalize big losses without stops
            if pnl < -50 and not self.last_decision.get('use_stop', False):
                bonus_reward -= 0.6
                log.info("BLACK BOX LEARNING: -0.6 for big loss without stop")
            
            total_reward = base_reward + bonus_reward
            
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
            
            log.info(f"BLACK BOX LEARNING: P&L=${pnl:.2f} -> Reward={total_reward:.3f}")
            log.info(f"Tool used: {primary_tool.upper()} (trust: {tool_trust:.2f})")
            log.info(f"Base reward: {base_reward:.3f}, Bonus: {bonus_reward:.3f}")
            
            # Update last state
            self.last_market_obs = next_market_obs
            
        except Exception as e:
            log.error(f"Error recording trade outcome: {e}")

    def _complete_trade(self, reason: str, exit_price: float):
        """ENHANCED: Complete trade with position AI learning"""
        if not self.current_position['in_position']:
            return
        
        entry_price = self.current_position['entry_price']
        action = self.current_position['action']
        
        # Calculate P&L
        if action == 1:  # Long
            pnl = (exit_price - entry_price) * 2.0  # $2 per point for MNQ
        else:  # Short
            pnl = (entry_price - exit_price) * 2.0
        
        # Enhanced logging
        duration_hours = (datetime.now() - self.current_position['entry_time']).total_seconds() / 3600
        
        log.info(f"📊 TRADE COMPLETED:")
        log.info(f"   Tool: {self.current_position['tool_used'].upper()}")
        log.info(f"   P&L: ${pnl:.2f} ({pnl/50:.1%})")  # % assuming $5k account
        log.info(f"   Duration: {duration_hours:.1f}h")
        log.info(f"   Scales Added: {self.current_position['scales_added']}")
        log.info(f"   Partial Exits: {self.current_position['partial_exits']}")
        log.info(f"   Exit Reason: {reason}")
        
        # Record outcome for both AIs
        self.record_trade_outcome(exit_price, pnl, reason)
        
        # NEW: Position AI learning
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
        """ENHANCED: Performance report with position management stats"""
        
        base_report = self.agent.get_tool_performance_report()
        
        # Add position management stats
        trades = self.trade_stats['total_trades']
        if trades > 0:
            avg_pnl = self.trade_stats['total_pnl'] / trades
            win_rate = "N/A"  # Would need to track wins/losses separately
        else:
            avg_pnl = 0
            win_rate = "N/A"
        
        position_stats = f"""

=== ENHANCED POSITION MANAGEMENT STATS ===

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

Learning Progression:
"""
        
        # Show feature availability
        for feature, threshold in self.learning_thresholds.items():
            status = "✅ ACTIVE" if trades >= threshold else f"🔒 Unlocks at {threshold} trades"
            position_stats += f"- {feature.replace('_', ' ').title()}: {status}\n"
        
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
        
        return base_report + position_stats

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