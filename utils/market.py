# utils/market.py

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

try:
    from arch import arch_model
    has_arch = True
except ImportError:
    log.warning("arch package not found. Using simple volatility estimation.")
    has_arch = False

try:
    from hmmlearn.hmm import GaussianHMM
    has_hmm = True
except ImportError:
    log.warning("hmmlearn package not found. Using simple regime detection.")
    has_hmm = False

class MarketUtils:
    @staticmethod
    def detect_regime(prices, window=50):
        try:
            if len(prices) < 20:
                return 0

            if has_hmm and len(prices) > window:
                try:
                    returns = np.diff(np.log(prices))
                    features = np.column_stack([
                        returns,
                        pd.Series(returns).rolling(5).std().fillna(0).values,
                        pd.Series(prices).pct_change().rolling(10).mean().fillna(0).values
                    ])[1:]

                    if len(features) > 10:
                        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
                        model.fit(features)
                        return model.predict(features)[-1]
                except Exception as e:
                    log.warning(f"HMM regime detection failed: {e}")

            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            return 0 if ma_short > ma_long else 1

        except Exception as e:
            log.error(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        try:
            if len(prices) < window + 1:
                return 0.01

            returns = 100 * pd.Series(prices).pct_change().dropna()

            if has_arch and len(returns) > 30:
                try:
                    model = arch_model(returns, vol='Garch', p=1, q=1)
                    res = model.fit(disp='off')
                    return np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])
                except Exception as e:
                    log.warning(f"GARCH failed: {e}")

            return returns.ewm(span=window).std().iloc[-1] if not returns.empty else 0.01

        except Exception as e:
            log.error(f"Volatility forecasting error: {e}")
            return 0.01
