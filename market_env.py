# market_env.py

import numpy as np
from typing import List, Tuple

class MarketEnv:
    """
    Mini environment that:
    • keeps the last 15 closes
    • maintains a simple ATR (avg absolute change) for reward scaling
    • tracks whether we’re flat / long / short so we can measure PnL
    It is *not* an order-manager – just lightweight context for RL.
    """
    FLAT, LONG, SHORT = 0, 1, 2

    def __init__(self, atr_window: int = 14):
        self.atr_window = atr_window
        self.reset()

    # ---------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------
    def reset(self):
        self.prices: List[float] = []
        self.atr_values: List[float] = []
        self.position = self.FLAT
        self.entry_price = None
        self.hold_bars = 0
        return self._obs()

    def step(self, price: float, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance one bar.  
        • action is usually FLAT when called from TradeManagerAI;  
          the RL agent’s chosen *trade* action is handled separately.
        • returns (obs, reward, done, info) just like OpenAI-Gym.
        """
        self.prices.append(price)
        # maintain ATR
        if len(self.prices) > self.atr_window:
            atr = np.mean(np.abs(np.diff(self.prices[-self.atr_window:])))
            self.atr_values.append(atr)
        else:
            atr = 0.0

        reward = 0.0
        done = False

        # (Optional) basic position bookkeeping so rewards work
        if self.position != self.FLAT:
            self.hold_bars += 1
            side = 1 if self.position == self.LONG else -1
            reward = side * (price - self.entry_price) / max(atr, 1e-6)

        return self._obs(), reward, done, {"atr": atr}

    # ---------------------------------------------------------------
    # internal
    # ---------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        """Return a fixed-length (15,) price vector, zero-padded on the left."""
        pad = [0.0] * (15 - len(self.prices)) if len(self.prices) < 15 else []
        return np.array(pad + self.prices[-15:], dtype=np.float32)
