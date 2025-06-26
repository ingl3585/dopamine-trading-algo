"""
Market Regime Detection - Volatility and trend regime classification
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Market regime detection and classification
    """
    
    def classify_regime(self, volatility: float, momentum: float, volume_momentum: float,
                       regime_history: deque) -> str:
        """Classify current market regime"""
        try:
            # Volatility-based classification
            if volatility > 0.05:
                vol_regime = "high_volatility"
            elif volatility < 0.01:
                vol_regime = "low_volatility"
            else:
                vol_regime = "normal_volatility"
            
            # Momentum-based classification
            if abs(momentum) > 0.03:
                if momentum > 0:
                    momentum_regime = "trending_up"
                else:
                    momentum_regime = "trending_down"
            elif abs(momentum) < 0.005:
                momentum_regime = "ranging"
            else:
                momentum_regime = "weak_trend"
            
            # Volume momentum consideration
            if abs(volume_momentum) > 0.2:
                volume_regime = "high_volume"
            elif abs(volume_momentum) < 0.05:
                volume_regime = "low_volume"
            else:
                volume_regime = "normal_volume"
            
            # Combine regimes into primary classification
            primary_regime = self._determine_primary_regime(
                vol_regime, momentum_regime, volume_regime
            )
            
            # Check for regime changes
            if len(regime_history) >= 5:
                regime_change = self._detect_regime_change(regime_history, primary_regime)
                if regime_change:
                    primary_regime = "transition"
            
            # Special pattern detection
            breakout_detected = self._detect_breakout(volatility, momentum, volume_momentum)
            if breakout_detected:
                primary_regime = "breakout"
            
            return primary_regime
            
        except Exception as e:
            logger.error(f"Error in regime classification: {e}")
            return "unknown"

    def _determine_primary_regime(self, vol_regime: str, momentum_regime: str, 
                                 volume_regime: str) -> str:
        """Determine primary regime from component regimes"""
        
        # High volatility overrides other considerations
        if vol_regime == "high_volatility":
            if momentum_regime in ["trending_up", "trending_down"]:
                return momentum_regime + "_volatile"
            else:
                return "high_volatility"
        
        # Strong trends
        if momentum_regime in ["trending_up", "trending_down"]:
            if volume_regime == "high_volume":
                return momentum_regime + "_strong"
            else:
                return momentum_regime
        
        # Ranging markets
        if momentum_regime == "ranging":
            if vol_regime == "low_volatility":
                return "low_volatility_range"
            else:
                return "ranging"
        
        # Default cases
        if vol_regime == "low_volatility":
            return "low_volatility"
        
        return "normal"

    def _detect_regime_change(self, regime_history: deque, current_regime: str) -> bool:
        """Detect if we're in a regime transition"""
        try:
            if len(regime_history) < 5:
                return False
            
            # Get recent regimes
            recent_regimes = [entry['regime'] for entry in list(regime_history)[-5:]]
            
            # Check for regime instability (frequent changes)
            unique_regimes = set(recent_regimes)
            if len(unique_regimes) >= 3:  # 3+ different regimes in last 5 periods
                return True
            
            # Check for significant volatility changes
            recent_volatilities = [entry['volatility'] for entry in list(regime_history)[-3:]]
            vol_change = max(recent_volatilities) / (min(recent_volatilities) + 1e-8)
            
            if vol_change > 2.0:  # 2x volatility change
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting regime change: {e}")
            return False

    def _detect_breakout(self, volatility: float, momentum: float, volume_momentum: float) -> bool:
        """Detect breakout conditions"""
        try:
            # Breakout criteria: high volatility + strong momentum + volume confirmation
            high_vol = volatility > 0.03
            strong_momentum = abs(momentum) > 0.025
            volume_confirmation = abs(volume_momentum) > 0.15
            
            # All three conditions must be met
            return high_vol and strong_momentum and volume_confirmation
            
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return False

    def get_regime_probabilities(self, volatility: float, momentum: float, 
                               volume_momentum: float) -> Dict[str, float]:
        """Get probabilities for different regime states"""
        try:
            probabilities = {
                'trending_up': 0.0,
                'trending_down': 0.0,
                'ranging': 0.0,
                'high_volatility': 0.0,
                'low_volatility': 0.0,
                'breakout': 0.0
            }
            
            # Momentum-based probabilities
            if momentum > 0:
                probabilities['trending_up'] = min(1.0, abs(momentum) * 20)
            else:
                probabilities['trending_down'] = min(1.0, abs(momentum) * 20)
            
            # Ranging probability (inverse of momentum)
            probabilities['ranging'] = max(0.0, 1.0 - abs(momentum) * 50)
            
            # Volatility probabilities
            if volatility > 0.03:
                probabilities['high_volatility'] = min(1.0, volatility * 20)
            else:
                probabilities['low_volatility'] = max(0.0, 1.0 - volatility * 50)
            
            # Breakout probability
            breakout_score = (
                min(1.0, volatility * 20) * 0.4 +
                min(1.0, abs(momentum) * 20) * 0.4 +
                min(1.0, abs(volume_momentum) * 5) * 0.2
            )
            probabilities['breakout'] = breakout_score
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {e}")
            return {regime: 0.2 for regime in ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']}