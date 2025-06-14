# trade_manager_ai.py - Complete black box trade management

# trade_manager_with_blackbox.py
# Updated trade manager that uses the black box AI with your existing subsystems

from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
from rl_agent import StrategicToolLearningAgent
from market_env import MarketEnv

log = logging.getLogger(__name__)

class BlackBoxTradeManagerWithSubsystems:
    """
    Trade manager that uses black box AI to strategically employ your existing subsystems
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        
        # Black box AI that learns to use your subsystems
        self.agent = StrategicToolLearningAgent(
            market_obs_size=15,
            subsystem_features_size=16
        )
        
        # Environment for state tracking
        self.env = MarketEnv()
        
        # State tracking
        self.last_market_obs = self.env.get_obs()
        self.last_subsystem_features = None
        self.last_decision = {}
        self.current_position = {
            'in_position': False, 
            'entry_price': 0, 
            'entry_time': None,
            'action': 0
        }
        
        log.info("Black Box Trade Manager with Subsystem Tools initialized")
        log.info("AI will learn to strategically use DNA, Micro, Temporal, and Immune systems")

    def on_new_bar(self, msg: Dict[str, Any]):
        """
        Process new bar with black box AI using your subsystems as tools
        """
        try:
            # Extract price data
            price = msg.get("price_1m", [4000.0])[-1] if msg.get("price_1m") else 4000.0
            prices = msg.get("price_1m", [price])
            volumes = msg.get("volume_1m", [1000])
            
            # Update environment state
            market_obs = self.env.get_obs()
            self.env.step(price, 0)
            
            # AI makes complete decision using your subsystems as tools
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
            
            # Execute AI's decision
            self._execute_ai_decision(decision, price)
            
            # Log tool usage
            trust = decision['tool_trust']
            log.info(f"AI TOOL USAGE: DNA({trust['dna']:.2f}) MICRO({trust['micro']:.2f}) "
                    f"TEMPORAL({trust['temporal']:.2f}) IMMUNE({trust['immune']:.2f})")
            log.info(f"PRIMARY TOOL: {decision['primary_tool'].upper()}")
            
        except Exception as e:
            log.error(f"Error processing new bar: {e}")
            import traceback
            traceback.print_exc()

    def _execute_ai_decision(self, decision: Dict, current_price: float):
        """Execute the AI's complete trading decision"""
        
        action = decision['action']
        confidence = decision['confidence']
        
        # Enter new position if AI signals entry
        if action != 0 and not self.current_position['in_position']:
            
            action_code = 1 if action == 1 else 2
            
            # AI's risk management decisions
            stop_points = 0.0
            tp_points = 0.0
            
            # Build quality string with tool and risk management info
            tool_name = decision['primary_tool']
            direction = 'long' if action == 1 else 'short'
            quality = f"AI_{tool_name}_{direction}"
            
            # Add stop/target info to quality string
            if decision['use_stop'] and decision['stop_price']:
                stop_points = abs(current_price - decision['stop_price'])
                quality += f"_stop{decision['stop_distance_pct']:.1f}pct"
            
            if decision['use_target'] and decision['target_price']:
                tp_points = abs(decision['target_price'] - current_price)
                quality += f"_target{decision['target_distance_pct']:.1f}pct"
            
            # Send signal to NinjaTrader
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_points, tp_points)
            
            # Track position
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'primary_tool': tool_name,
                'stop_price': decision.get('stop_price'),
                'target_price': decision.get('target_price')
            }
            
            log.info(f"BLACK BOX TRADE using {tool_name.upper()} tool:")
            log.info(f"  Direction: {direction}")
            log.info(f"  Confidence: {confidence:.3f}")
            log.info(f"  Position Size: {decision['position_size']:.2f}")
            if decision['use_stop']:
                log.info(f"  Stop: ${decision['stop_price']:.2f} ({decision['stop_distance_pct']:.1f}%)")
            if decision['use_target']:
                log.info(f"  Target: ${decision['target_price']:.2f} ({decision['target_distance_pct']:.1f}%)")

    def record_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """
        Record trade outcome and teach AI about tool effectiveness
        """
        try:
            if not self.last_decision:
                return
            
            # Base reward from P&L
            base_reward = pnl / 50.0  # Normalize for MNQ
            
            # Bonus rewards for good tool usage and risk management
            bonus_reward = 0.0
            
            # Reward successful tool selection
            primary_tool = self.last_decision.get('primary_tool', '')
            tool_trust = self.last_decision.get('tool_trust', {}).get(primary_tool, 0.0)
            
            if pnl > 10 and tool_trust > 0.6:  # Profitable trade with high tool trust
                bonus_reward += 0.3
                log.info(f"BLACK BOX LEARNING: +0.3 for trusting {primary_tool} tool")
            
            if pnl < -20 and tool_trust > 0.8:  # Loss despite high trust
                bonus_reward -= 0.4
                log.info(f"BLACK BOX LEARNING: -0.4 for over-trusting {primary_tool} tool")
            
            # Reward good risk management
            if exit_reason == "stop_hit" and self.last_decision.get('use_stop', False):
                if pnl > -30:  # Small loss with stop
                    bonus_reward += 0.2
                    log.info("BLACK BOX LEARNING: +0.2 for using protective stop")
                else:
                    bonus_reward += 0.1  # Still reward stop usage even if bigger loss
            
            if exit_reason == "target_hit" and self.last_decision.get('use_target', False):
                bonus_reward += 0.2
                log.info("BLACK BOX LEARNING: +0.2 for using profit target")
            
            # Penalize not using stops on big losses
            if pnl < -50 and not self.last_decision.get('use_stop', False):
                bonus_reward -= 0.5
                log.info("BLACK BOX LEARNING: -0.5 for not using stop on big loss")
            
            # Penalize not taking profits on big wins that reverse
            if exit_reason == "AI_exit" and pnl > 80 and not self.last_decision.get('use_target', False):
                bonus_reward -= 0.3
                log.info("BLACK BOX LEARNING: -0.3 for not taking profits early enough")
            
            total_reward = base_reward + bonus_reward
            
            # Store experience for AI learning
            next_market_obs = self.env.get_obs()
            next_subsystem_features = self.last_subsystem_features  # Would get fresh in real implementation
            
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
        """Complete trade and calculate outcome"""
        if not self.current_position['in_position']:
            return
        
        entry_price = self.current_position['entry_price']
        action = self.current_position['action']
        
        # Calculate P&L
        if action == 1:  # Long
            pnl = (exit_price - entry_price) * 2.0  # $2 per point for MNQ
        else:  # Short
            pnl = (entry_price - exit_price) * 2.0
        
        # Record outcome for AI learning
        self.record_trade_outcome(exit_price, pnl, reason)
        
        # Reset position
        self.current_position = {
            'in_position': False, 
            'entry_price': 0, 
            'entry_time': None,
            'action': 0
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
        """Get comprehensive performance report"""
        return self.agent.get_tool_performance_report()