# services/feature_processor.py

import time
import logging

from model.reward import RewardCalculator
from model.agent import RLAgent

log = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self, agent: RLAgent, rewarder: RewardCalculator):
        self.agent = agent
        self.rewarder = rewarder
        self.last_price = None
        self.last_features = None
        self.last_action = None
        self.last_signals = None

    def process_and_predict(self, feat):
        """
        Process new feature vector with Ichimoku/EMA signals
        
        Expected feature vector (9 elements):
        [0] close, [1] normalized_volume, [2] tenkan_kijun_signal, 
        [3] price_cloud_signal, [4] future_cloud_signal, [5] ema_cross_signal,
        [6] tenkan_momentum, [7] kijun_momentum, [8] lwpe
        """
        if len(feat) != 9:
            log.warning(f"Expected 9 features, got {len(feat)}. Padding or truncating.")
            feat = self._normalize_feature_vector(feat)
        
        close = feat[0]
        lwpe = feat[8] if len(feat) > 8 else 0.5
        
        # Extract signals for reward calculation
        current_signals = self._extract_signals(feat)

        reward = 0.0
        if self.last_price is not None and self.last_action is not None:
            price_change = close - self.last_price

            # Use LWPE as volatility proxy (replacing ATR)
            volatility_proxy = max(abs(lwpe - 0.5) * 2, 0.01)  # Convert LWPE to volatility measure

            if self.last_action == 1:  # Long
                reward = self.rewarder.compute_reward(
                    price_change, volatility_proxy, self.last_action, self.last_signals
                )
            elif self.last_action == 2:  # Short
                reward = self.rewarder.compute_reward(
                    -price_change, volatility_proxy, self.last_action, self.last_signals
                )
            else:  # Hold
                reward = self.rewarder.compute_reward(
                    0, volatility_proxy, self.last_action, self.last_signals
                )
            
            # Add experience to agent
            if self.last_features is not None:
                self.agent.add_experience(self.last_features, self.last_action, reward, feat)

        # Get new action and confidence
        action, conf = self.agent.predict_single(feat)
        
        # Update state
        self.last_price = close
        self.last_features = feat.copy()
        self.last_action = action
        self.last_signals = current_signals

        # Create row for logging (timestamp + features + reward)
        row = [time.time(), *feat, reward]

        return row, action, conf, close

    def _normalize_feature_vector(self, feat):
        """Ensure feature vector has exactly 9 elements"""
        if len(feat) < 9:
            # Pad with default values
            padded = list(feat) + [0.0] * (9 - len(feat))
            # Set LWPE default if missing
            if len(feat) <= 8:
                padded[8] = 0.5  # Default LWPE
            return padded
        elif len(feat) > 9:
            # Truncate to 9 elements
            return feat[:9]
        return feat

    def _extract_signals(self, feat):
        """Extract and validate signal values for reward calculation"""
        try:
            signals = {
                'tenkan_kijun': int(round(feat[2])),
                'price_cloud': int(round(feat[3])),
                'future_cloud': int(round(feat[4])),
                'ema_cross': int(round(feat[5])),
                'tenkan_momentum': int(round(feat[6])),
                'kijun_momentum': int(round(feat[7])),
                'normalized_volume': feat[1],
                'lwpe': feat[8]
            }
            
            # Validate all signals are in valid range
            for signal_name, signal_value in signals.items():
                if signal_name not in ['normalized_volume', 'lwpe']:
                    if signal_value not in [-1, 0, 1]:
                        log.warning(f"Invalid {signal_name} signal: {signal_value}, setting to 0")
                        signals[signal_name] = 0
            
            return signals
            
        except (IndexError, ValueError) as e:
            log.warning(f"Feature vector incomplete for signal extraction: {e}")
            return {
                'tenkan_kijun': 0, 'price_cloud': 0, 'future_cloud': 0,
                'ema_cross': 0, 'tenkan_momentum': 0, 'kijun_momentum': 0,
                'normalized_volume': 0, 'lwpe': 0.5
            }

    def get_signal_summary(self):
        """Enhanced signal state summary with neutral signal handling"""
        if self.last_signals is None:
            return "No signals available"
        
        summary = []
        
        # Ichimoku signals with neutral handling
        tk_signal = self.last_signals.get('tenkan_kijun', 0)
        if tk_signal > 0:
            summary.append("Tenkan>Kijun (Bullish)")
        elif tk_signal < 0:
            summary.append("Tenkan<Kijun (Bearish)")
        else:
            summary.append("Tenkan≈Kijun (Neutral)")
        
        # Price vs Cloud
        pc_signal = self.last_signals.get('price_cloud', 0)
        if pc_signal > 0:
            summary.append("Price above Cloud")
        elif pc_signal < 0:
            summary.append("Price below Cloud")
        else:
            summary.append("Price in Cloud (Neutral)")
        
        # EMA signal with neutral handling
        ema_signal = self.last_signals.get('ema_cross', 0)
        if ema_signal > 0:
            summary.append("EMA Bullish")
        elif ema_signal < 0:
            summary.append("EMA Bearish")
        else:
            summary.append("EMA Neutral")
        
        # Momentum signals
        momentum_signals = []
        tm_signal = self.last_signals.get('tenkan_momentum', 0)
        if tm_signal > 0:
            momentum_signals.append("Tenkan+")
        elif tm_signal < 0:
            momentum_signals.append("Tenkan-")
        else:
            momentum_signals.append("Tenkan=")
            
        km_signal = self.last_signals.get('kijun_momentum', 0)
        if km_signal > 0:
            momentum_signals.append("Kijun+")
        elif km_signal < 0:
            momentum_signals.append("Kijun-")
        else:
            momentum_signals.append("Kijun=")
        
        if momentum_signals:
            summary.append(f"Momentum: {', '.join(momentum_signals)}")
        
        return " | ".join(summary) if summary else "All signals neutral"