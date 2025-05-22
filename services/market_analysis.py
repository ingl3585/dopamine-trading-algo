# services/market_analysis.py

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

class MarketAnalysis:
    @staticmethod
    def detect_regime(prices, window=20):
        """
        Simple regime detection using moving averages
        Returns 0 for uptrend, 1 for downtrend
        """
        try:
            if len(prices) < window:
                return 0

            # Simple and robust: short MA vs long MA
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-window:])
            
            return 0 if ma_short > ma_long else 1

        except Exception as e:
            log.warning(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        """
        Simple exponential moving average of returns volatility
        """
        try:
            if len(prices) < window + 1:
                return 0.01

            returns = pd.Series(prices).pct_change().dropna()
            
            if len(returns) < 5:
                return 0.01
                
            # Simple exponential weighted volatility
            volatility = returns.ewm(span=window).std().iloc[-1]
            
            # Ensure reasonable bounds
            return max(0.001, min(volatility, 0.1))

        except Exception as e:
            log.warning(f"Volatility forecasting error: {e}")
            return 0.01

    @staticmethod
    def calculate_momentum(prices, window=10):
        """
        Simple momentum indicator
        """
        try:
            if len(prices) < window + 1:
                return 0.0
                
            current_price = prices[-1]
            past_price = prices[-window-1]
            
            momentum = (current_price - past_price) / past_price
            
            # Normalize to reasonable range
            return np.clip(momentum, -0.1, 0.1)
            
        except Exception as e:
            log.warning(f"Momentum calculation error: {e}")
            return 0.0