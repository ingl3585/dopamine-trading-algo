# trade_manager_ai.py - Complete black box trade management

from datetime import datetime
from typing import Dict, Any
import time

# Black box RL components
from rl_agent import RLTradingAgent
from market_env import MarketEnv

class TradeManagerAI:
    """
    Pure black box trade manager that delegates ALL decisions to RL agent
    with intelligent immune system safeguards
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        # Existing dependency injection
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge

        # Pure black box RL components
        self.env = MarketEnv()           # Clean environment, no indicators
        self.agent = RLTradingAgent()    # Self-learning agent

        # Runtime tracking
        self._last_obs = self.env.get_obs()
        self._last_action = 0
        self._last_action_details = {}

    # ------------------------------------------------------------------
    # MAIN DECISION LOOP ------------------------------------------------
    # ------------------------------------------------------------------
    def on_new_bar(self, msg: Dict[str, Any]):
        """
        Pure black box decision making - no indicators, pure RL
        """
        price = msg["price_1m"][-1]
        obs = self.env.get_obs()
        
        # Get pure RL decision (no indicators involved)
        action, stop_points, tp_points = self.agent.act(obs)
        
        # Store details for learning feedback
        self._last_action_details = {
            'action': action,
            'stop_points': stop_points,
            'tp_points': tp_points,
            'price': price,
            'timestamp': time.time()
        }
        
        # Update environment state
        self.env.step(price, action)
        self._last_obs = obs
        self._last_action = action
        
        # Send signal if RL agent wants to trade
        if action != MarketEnv.FLAT:
            action_code = 1 if action == MarketEnv.LONG else 2
            
            # Confidence based on risk management usage
            confidence = 0.6
            if stop_points > 0.5:
                confidence += 0.2  # Bonus for using stops
            if tp_points > 0.5:
                confidence += 0.2  # Bonus for using targets
            
            # Quality description for logging
            quality = f"rl_{'long' if action_code==1 else 'short'}"
            if stop_points > 0.5 and tp_points > 0.5:
                quality += "_bracket"
            elif stop_points > 0.5:
                quality += "_stop"
            elif tp_points > 0.5:
                quality += "_target"
            else:
                quality += "_naked"
            
            # Send to NinjaTrader
            self.tcp_bridge.send_signal(action_code, confidence, quality, stop_points, tp_points)
            
            print(f"BLACK BOX RL SIGNAL: {quality}")
            print(f"  Stop: ${stop_points:.2f}, Target: ${tp_points:.2f}, Price: ${price:.2f}")

    # ------------------------------------------------------------------
    # LEARNING FEEDBACK ------------------------------------------------
    # ------------------------------------------------------------------
    def record_trade_outcome(self, exit_price: float, pnl: float, 
                           exit_reason: str = "unknown", done: bool = True):
        """
        Pure black box learning from trade outcomes
        """
        next_obs = self.env.get_obs()
        
        # Base reward from raw P&L (no indicator normalization)
        base_reward = pnl / 100.0  # Normalize by $100 for MNQ scaling
        
        # Extract last action details for learning
        stop_points = self._last_action_details.get('stop_points', 0.0)
        tp_points = self._last_action_details.get('tp_points', 0.0)
        used_stop = stop_points > 0.5
        used_target = tp_points > 0.5
        
        # REWARD ENGINEERING - teach optimal stop/target usage
        bonus_reward = 0.0
        
        if exit_reason == "stop_hit":
            if pnl < 0:
                # Good stop loss - prevented bigger disaster
                bonus_reward = +0.5 if used_stop else -0.3
                print(f"RL LEARN: Stop saved ${abs(pnl):.2f} loss → reward +0.5")
            else:
                # Stop too tight - cut a winner
                bonus_reward = -0.4 if used_stop else 0.0
                print(f"RL LEARN: Stop cut ${pnl:.2f} winner → penalty -0.4")
                
        elif exit_reason == "target_hit":
            if pnl > 0:
                # Good target - locked in profit
                bonus_reward = +0.4 if used_target else -0.2
                print(f"RL LEARN: Target locked ${pnl:.2f} profit → bonus +0.4")
            else:
                # Shouldn't happen but penalize if it does
                bonus_reward = -0.2
                
        elif exit_reason == "intelligence_exit":
            # Manual exit by intelligence system
            if not used_stop and pnl < -20.0:
                # Lost >$20 without stop - bad risk management
                bonus_reward = -0.6
                print(f"RL LEARN: Lost ${abs(pnl):.2f} without stop → big penalty -0.6")
            elif not used_target and pnl > 50.0:
                # Made >$50 but no target - could have locked profits earlier
                bonus_reward = -0.3
                print(f"RL LEARN: Made ${pnl:.2f} but no target → penalty -0.3")
            elif used_stop and used_target:
                # Good risk management + intelligence exit
                bonus_reward = +0.2
                print(f"RL LEARN: Good risk mgmt + smart exit → bonus +0.2")
            else:
                # Neutral manual exit
                bonus_reward = 0.0
                
        elif exit_reason == "session_close":
            # Time-based exit
            if used_stop or used_target:
                bonus_reward = +0.1  # Reward risk management
            else:
                bonus_reward = -0.1 if abs(pnl) > 10.0 else 0.0
        
        # Calculate total reward
        total_reward = base_reward + bonus_reward
        
        # Feed back to RL agent for learning
        self.agent.store(self._last_obs, self._last_action, total_reward, next_obs, done)
        
        # Logging for monitoring
        print(f"RL REWARD BREAKDOWN:")
        print(f"  P&L: ${pnl:.2f} → Base Reward: {base_reward:.3f}")
        print(f"  Bonus: {bonus_reward:.3f} → Total: {total_reward:.3f}")
        print(f"  Exit: {exit_reason}, Stop: ${stop_points:.2f}, Target: ${tp_points:.2f}")
        
        # Update for next iteration
        self._last_obs = next_obs

    # ------------------------------------------------------------------
    # IMMUNE SYSTEM SAFEGUARDS (LEGACY SAFETY NET) ---------------------
    # ------------------------------------------------------------------
    def should_exit_now(self, live_prices, live_volumes, entry_time):
        """
        Intelligence-based safety net - overrides RL in dangerous situations
        Keeps the black box from nuking itself while learning
        """
        now = datetime.now()
        duration = (now - entry_time).total_seconds() / 60

        # Get intelligence assessment
        current_result = self.intel.process_market_data(live_prices, live_volumes, now)
        signal = current_result['signal_strength']
        confidence = current_result['confidence']
        is_dangerous = current_result.get('is_dangerous_pattern', False)

        # Emergency exits
        if is_dangerous:
            return True, "immune_system_danger"
        if confidence < 0.3 and duration > 3:
            return True, f"low_confidence_{duration:.1f}m"
        if abs(signal) < 0.1 and duration > 5:
            return True, f"neutral_signal_{duration:.1f}m"
        if duration > 20:
            return True, "max_duration_20m"

        return False, "continue_holding"