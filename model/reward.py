# model/reward.py

import numpy as np

class RewardCalculator:
    def __init__(self):
        self.transaction_cost = 0.001
        
    def compute_reward(self, price_change, volatility_proxy, action_taken=None, signal_features=None):
        """
        Compute reward with Ichimoku/EMA signal alignment bonus
        
        Args:
            price_change: Price movement
            volatility_proxy: Use LWPE or volume as volatility measure (replacing ATR)
            action_taken: Action that was taken
            signal_features: Dict with Ichimoku/EMA signals for alignment bonus
        """
        # Use LWPE-based volatility normalization instead of ATR
        volatility = max(volatility_proxy, 1e-6) if volatility_proxy > 0 else 0.01
        
        base_reward = price_change / volatility
        
        # Apply transaction costs
        if action_taken is not None and action_taken != 0:
            base_reward -= self.transaction_cost
        
        # Add signal alignment bonus
        if signal_features and action_taken is not None:
            alignment_bonus = self._calculate_signal_alignment_bonus(
                action_taken, signal_features, price_change
            )
            base_reward += alignment_bonus
            
        return np.clip(base_reward, -5.0, 5.0)
    
    def _calculate_signal_alignment_bonus(self, action, signals, price_change):
        """
        Reward when traditional signals align with profitable outcomes
        """
        try:
            if action == 0:  # No position, no bonus
                return 0.0
            
            # Determine if the action was profitable
            is_profitable = (action == 1 and price_change > 0) or (action == 2 and price_change < 0)
            
            if not is_profitable:
                return 0.0  # No bonus for unprofitable trades
            
            alignment_score = 0.0
            signal_count = 0
            
            # Check Ichimoku alignment
            if 'tenkan_kijun' in signals:
                expected_signal = 1 if action == 1 else -1
                if signals['tenkan_kijun'] == expected_signal:
                    alignment_score += 0.2
                signal_count += 1
            
            if 'price_cloud' in signals:
                expected_signal = 1 if action == 1 else -1
                if signals['price_cloud'] == expected_signal:
                    alignment_score += 0.15
                signal_count += 1
            
            if 'future_cloud' in signals:
                expected_signal = 1 if action == 1 else -1
                if signals['future_cloud'] == expected_signal:
                    alignment_score += 0.1
                signal_count += 1
            
            # Check EMA alignment
            if 'ema_cross' in signals:
                expected_signal = 1 if action == 1 else -1
                if signals['ema_cross'] == expected_signal:
                    alignment_score += 0.15
                signal_count += 1
            
            # Check momentum alignment
            if 'tenkan_momentum' in signals and 'kijun_momentum' in signals:
                expected_signal = 1 if action == 1 else -1
                momentum_alignment = 0
                if signals['tenkan_momentum'] == expected_signal:
                    momentum_alignment += 0.5
                if signals['kijun_momentum'] == expected_signal:
                    momentum_alignment += 0.5
                alignment_score += 0.1 * momentum_alignment
                signal_count += 1
            
            # Normalize by number of signals
            if signal_count > 0:
                return alignment_score / signal_count
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def compute_position_reward(self, position, price_change, volatility_proxy):
        """
        Compute reward for existing positions (replacing ATR with volatility proxy)
        """
        if position == 0:
            return 0.0
            
        pnl = position * price_change
        volatility = max(volatility_proxy, 1e-6) if volatility_proxy > 0 else 0.01
        normalized_pnl = pnl / volatility
        
        return np.clip(normalized_pnl, -5.0, 5.0)
    
    def compute_ichimoku_specific_reward(self, price_change, ichimoku_signals, action_taken):
        """
        Specialized reward for Ichimoku strategy validation
        """
        try:
            if action_taken == 0:
                return 0.0
            
            base_reward = price_change if action_taken == 1 else -price_change
            
            # Bonus for following strong Ichimoku signals
            signal_strength = 0.0
            
            # Strong bullish Ichimoku setup
            if (ichimoku_signals.get('tenkan_kijun', 0) > 0 and 
                ichimoku_signals.get('price_cloud', 0) > 0 and
                ichimoku_signals.get('future_cloud', 0) > 0):
                if action_taken == 1:  # Long position
                    signal_strength = 0.3
            
            # Strong bearish Ichimoku setup  
            elif (ichimoku_signals.get('tenkan_kijun', 0) < 0 and 
                  ichimoku_signals.get('price_cloud', 0) < 0 and
                  ichimoku_signals.get('future_cloud', 0) < 0):
                if action_taken == 2:  # Short position
                    signal_strength = 0.3
            
            # Mixed signals - penalize slightly
            else:
                signal_strength = -0.1
            
            return base_reward + signal_strength
            
        except Exception:
            return price_change if action_taken == 1 else -price_change