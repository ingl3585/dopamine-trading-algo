# trade_manager_ai.py - Complete black box trade management

from datetime import datetime
from typing import Dict, Any
import time
import logging
from rl_agent import BlackBoxRLAgent
from market_env import MarketEnv

log = logging.getLogger(__name__)

class BlackBoxTradeManager:
    """
    PURE black box trade manager - AI makes ALL decisions
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge
        
        # Pure black box components
        self.env = MarketEnv()
        self.black_box_agent = BlackBoxRLAgent()
        
        # State tracking
        self.last_obs = self.env.get_obs()
        self.last_decision = {}
        self.current_position = {'in_position': False, 'entry_price': 0, 'entry_time': None}
        
        log.info("BLACK BOX Trade Manager initialized - AI controls EVERYTHING")

    def on_new_bar(self, msg: Dict[str, Any]):
        """
        PURE BLACK BOX - AI makes all decisions
        """
        price = msg["price_1m"][-1] if msg.get("price_1m") else 4000.0
        
        # Update environment
        obs = self.env.get_obs()
        self.env.step(price, 0)  # Update with current price
        
        # AI makes COMPLETE decision
        decision = self.black_box_agent.make_decision(
            obs, 
            price, 
            self.current_position['in_position']
        )
        
        self.last_decision = decision
        self.last_obs = obs
        
        # Execute AI's decision
        self._execute_black_box_decision(decision, price)
    
    def _execute_black_box_decision(self, decision: Dict, current_price: float):
        """Execute the AI's complete decision"""
        
        action = decision['action']
        confidence = decision['overall_confidence']
        
        # Exit current position if AI says so
        if decision['should_exit'] and self.current_position['in_position']:
            self.tcp_bridge.send_signal(0, confidence, "AI_exit")
            log.info(f"BLACK BOX: AI decided to exit (confidence: {confidence:.3f})")
            self._complete_trade("AI_exit", current_price)
            return
        
        # Enter new position if AI says so
        if action != 0 and not self.current_position['in_position']:
            
            # Prepare signal with AI's complete decision
            action_code = 1 if action == 1 else 2
            
            # AI-determined stop/target levels (in points for MNQ)
            stop_points = 0.0
            tp_points = 0.0
            
            quality = f"AI_{'long' if action==1 else 'short'}"
            
            if decision['use_stop'] and decision['stop_price']:
                stop_points = abs(current_price - decision['stop_price'])
                quality += f"_stop{decision['stop_distance_pct']:.1f}pct"
            
            if decision['use_target'] and decision['target_price']:
                tp_points = abs(decision['target_price'] - current_price)
                quality += f"_target{decision['target_distance_pct']:.1f}pct"
            
            if not decision['use_stop'] and not decision['use_target']:
                quality += "_naked"
            
            # Send AI's complete decision to NinjaTrader
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_points, tp_points)
            
            # Track position
            self.current_position = {
                'in_position': True,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'action': action,
                'stop_price': decision['stop_price'],
                'target_price': decision['target_price']
            }
            
            log.info(f"BLACK BOX TRADE: {quality}")
            log.info(f"  AI Decision: Size={decision['position_size']:.2f}, Conf={confidence:.3f}")
            log.info(f"  Stop: ${decision['stop_price']:.2f} ({decision['stop_distance_pct']:.1f}%)")
            log.info(f"  Target: ${decision['target_price']:.2f} ({decision['target_distance_pct']:.1f}%)")
    
    def record_trade_outcome(self, exit_price: float, pnl: float, exit_reason: str = "unknown"):
        """Feed outcome back to black box for learning"""
        
        if not self.last_decision:
            return
        
        # Calculate comprehensive reward for the AI
        base_reward = pnl / 50.0  # Normalize for MNQ
        
        # Reward AI for good risk management decisions
        bonus_reward = 0.0
        
        # Reward using stops when they save money
        if exit_reason == "stop_hit" and pnl > -30:  # Small loss
            if self.last_decision.get('use_stop', False):
                bonus_reward += 0.5  # Reward AI for using stops
                log.info("BLACK BOX LEARNING: +0.5 for using protective stop")
        
        # Reward using targets when they lock profits
        if exit_reason == "target_hit" and pnl > 0:
            if self.last_decision.get('use_target', False):
                bonus_reward += 0.3  # Reward AI for using targets
                log.info("BLACK BOX LEARNING: +0.3 for using profit target")
        
        # Penalize not using stops on big losses
        if pnl < -50 and not self.last_decision.get('use_stop', False):
            bonus_reward -= 0.8  # Teach AI to use stops
            log.info("BLACK BOX LEARNING: -0.8 for not using stop on big loss")
        
        # Penalize not using targets on big wins that reverse
        if exit_reason == "AI_exit" and pnl > 100 and not self.last_decision.get('use_target', False):
            bonus_reward -= 0.4  # Teach AI to lock profits
            log.info("BLACK BOX LEARNING: -0.4 for not taking profits")
        
        total_reward = base_reward + bonus_reward
        
        # Store experience for learning
        next_obs = self.env.get_obs()
        self.black_box_agent.store_experience(
            self.last_obs,
            self.last_decision,
            total_reward,
            next_obs,
            True  # Trade completed
        )
        
        log.info(f"BLACK BOX LEARNING: P&L=${pnl:.2f} -> Reward={total_reward:.3f}")
        log.info(f"  Base: {base_reward:.3f}, Bonus: {bonus_reward:.3f}")
        
        self.last_obs = next_obs
    
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
        
        # Feed back to AI for learning
        self.record_trade_outcome(exit_price, pnl, reason)
        
        # Reset position
        self.current_position = {'in_position': False, 'entry_price': 0, 'entry_time': None}