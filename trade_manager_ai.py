# trade_manager_ai.py

from datetime import datetime
from typing import Dict, Any

# 🔗 NEW: RL learning + environment
from rl_agent import RLTradingAgent
from market_env import MarketEnv


class TradeManagerAI:
    """Live execution wrapper that now delegates *all* entry/exit logic
    to an online‑learning RL agent while keeping your original immune‑system
    safeguards as a last‑resort kill‑switch.
    """

    def __init__(self, intelligence_engine, tcp_bridge):
        # ← existing dependency injection stays the same
        self.intel = intelligence_engine
        self.tcp_bridge = tcp_bridge

        # 🆕 self‑learning components
        self.env = MarketEnv()           # creates ATR‑aware obs vector
        self.agent = RLTradingAgent()    # starts learner thread internally

        # runtime book‑keeping
        self._last_obs = self.env._get_obs()  # seed for first act()
        self._last_action = 0   

    # ------------------------------------------------------------------
    # PUBLIC API – called once per *new* 1‑minute bar (or tick) ---------
    # ------------------------------------------------------------------
    def on_new_bar(self, msg: Dict[str, Any]):
        """
        Main decision loop.
        `msg["price_1m"]` must contain the latest close as list/array.
        """
        price = msg["price_1m"][-1]

        # 1️⃣ observation BEFORE env.step
        obs = self.env._get_obs()

        # 2️⃣ ask the RL agent
        action, stop_atr, tp_atr = self.agent.act(obs)
        self._last_action = action            # <-- track for learning

        # 3️⃣ keep env history up-to-date
        self.env.step(price, action)
        self._last_obs = obs

        # 4️⃣ send order if action is BUY or SELL
        if action != MarketEnv.FLAT:
            action_code = 1 if action == MarketEnv.LONG else 2
            self.tcp_bridge.send_signal(          # positional args safer
                action_code,                      # action
                1.0,                              # confidence placeholder
                f"rl_{'long' if action_code==1 else 'short'}",
                stop_atr,                         # stop offset (ATR)
                tp_atr                            # take-profit offset (ATR)
            )

    # ------------------------------------------------------------------
    # (optional) call this after a trade closes to feed back reward ------
    # ------------------------------------------------------------------
    def record_trade_outcome(
        self,
        exit_price: float,
        pnl: float,
        done: bool = True,
    ):
        """
        Call this from NinjaScript when a position closes.
        `pnl` = raw $ or ticks per contract.
        """
        next_obs = self.env._get_obs()
        atr = self.env.atr_values[-1] if self.env.atr_values else 1.0
        reward = pnl / atr if atr else pnl

        # push (state, action, reward, next_state, done)
        self.agent.store(self._last_obs, self._last_action, reward, next_obs, done)

        # keep bookkeeping tidy for the next transition
        self._last_obs = next_obs

    # ------------------------------------------------------------------
    # ORIGINAL IMMUNE‑SYSTEM EXIT FILTER --------------------------------
    # ------------------------------------------------------------------
    def should_exit_now(self, live_prices, live_volumes, entry_time):
        """Legacy safety cage – still runs *after* RL decision to force an
        exit on catastrophic patterns, low confidence, or max duration.
        Disable if you want pure black‑box behaviour, but recommended to
        keep capital from nuking itself on day‑one.
        """
        now = datetime.now()
        duration = (now - entry_time).total_seconds() / 60

        current_result = self.intel.process_market_data(live_prices, live_volumes, now)
        signal = current_result['signal_strength']
        confidence = current_result['confidence']
        is_dangerous = current_result.get('is_dangerous_pattern', False)

        if is_dangerous:
            return True, "immune system warning"
        if confidence < 0.3 and duration > 3:
            return True, f"low confidence after {duration:.1f}m"
        if abs(signal) < 0.1 and duration > 5:
            return True, f"neutral signal for {duration:.1f}m"
        if duration > 20:
            return True, "max duration exceeded"

        return False, "hold"
