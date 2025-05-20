# utils/market.py

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

try:
    from arch import arch_model
    has_arch = True
except ImportError:
    log.warning("Arch package not found. Using simple volatility estimation.")
    has_arch = False

try:
    from hmmlearn.hmm import GaussianHMM
    has_hmm = True
except ImportError:
    log.warning("HmmLearn package not found. Using simple regime detection.")
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
                    roll_std = pd.Series(returns).rolling(5).std().fillna(0).values
                    roll_mean = pd.Series(prices).pct_change().rolling(10).mean().fillna(0).values

                    min_len = min(len(returns), len(roll_std), len(roll_mean))
                    features = np.column_stack([
                        returns[-min_len:],
                        roll_std[-min_len:],
                        roll_mean[-min_len:]
                    ])

                    if len(features) > 10 and not np.isnan(features).any():
                        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
                        model.fit(features)
                        return model.predict(features)[-1]
                except Exception as e:
                    log.warning(f"HMM regime detection failed: {e}")

            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            return 0 if ma_short > ma_long else 1

        except Exception as e:
            log.warning(f"Regime detection error: {e}")
            return 0

    @staticmethod
    def forecast_volatility(prices, window=14):
        try:
            if len(prices) < window + 1:
                return 0.01

            returns = 1000 * pd.Series(prices).pct_change().dropna()

            if has_arch and len(returns) > 30:
                try:
                    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off')
                    forecast = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])
                    if not np.isfinite(forecast) or forecast > 5: 
                        raise ValueError("Unrealistic GARCH output")
                    return forecast
                except Exception as e:
                    log.warning(f"GARCH failed: {e}")
                    return returns.ewm(span=window).std().iloc[-1]

            return returns.ewm(span=window).std().iloc[-1] if not returns.empty else 0.01

        except Exception as e:
            log.warning(f"Volatility forecasting error: {e}")
            return 0.01
