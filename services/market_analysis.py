# services/market_analysis.py

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

class MarketAnalysis:
    @staticmethod
    def detect_regime(prices, window=20):
        try:
            if len(prices) < window:
                return 0

            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-window:])
            
            return 0 if ma_short > ma_long else 1

        except Exception as e:
            log.warning(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        try:
            if len(prices) < window + 1:
                return 0.01

            returns = pd.Series(prices).pct_change().dropna()
            
            if len(returns) < 5:
                return 0.01
                
            volatility = returns.ewm(span=window).std().iloc[-1]
            
            return max(0.001, min(volatility, 0.1))

        except Exception as e:
            log.warning(f"Volatility forecasting error: {e}")
            return 0.01

    @staticmethod
    def calculate_momentum(prices, window=10):
        try:
            if len(prices) < window + 1:
                return 0.0
                
            current_price = prices[-1]
            past_price = prices[-window-1]
            
            momentum = (current_price - past_price) / past_price
            
            return np.clip(momentum, -0.1, 0.1)
            
        except Exception as e:
            log.warning(f"Momentum calculation error: {e}")
            return 0.0