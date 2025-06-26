"""
Market Microstructure Domain - Order flow analysis per prompt.txt
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List

from .flow_analyzer import OrderFlowAnalyzer
from .regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

class MarketMicrostructureEngine:
    """
    Market Microstructure Intelligence per prompt.txt:
    - Smart money vs retail flow pattern detection
    - Market maker identification for accumulation/distribution phases
    - Liquidity depth analysis for intelligent position sizing
    - Real-time tape reading and momentum detection
    """
    
    def __init__(self):
        self.order_flow_history = deque(maxlen=1000)
        self.regime_history = deque(maxlen=100)
        self.liquidity_metrics = deque(maxlen=500)
        self.smart_money_indicators = {}
        self.market_maker_patterns = {}
        
        # Domain services
        self.flow_analyzer = OrderFlowAnalyzer()
        self.regime_detector = RegimeDetector()

    def analyze_order_flow(self, market_data: Dict) -> float:
        """Analyze order flow for smart money vs retail patterns"""
        try:
            # Extract market data
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            timestamps = market_data.get('timestamps', [])
            
            if len(prices) < 10 or len(volumes) < 10:
                return 0.0
            
            # Analyze order flow patterns
            smart_money_score = self.flow_analyzer.detect_smart_money_flow(
                prices, volumes, timestamps
            )
            
            # Market maker detection
            mm_score = self.flow_analyzer.detect_market_maker_activity(
                prices, volumes, self.market_maker_patterns
            )
            
            # Combine signals
            flow_signal = (smart_money_score * 0.7 + mm_score * 0.3)
            
            # Store in history
            self.order_flow_history.append({
                'smart_money_score': smart_money_score,
                'market_maker_score': mm_score,
                'combined_signal': flow_signal,
                'timestamp': timestamps[-1] if timestamps else 0
            })
            
            return flow_signal
            
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return 0.0

    def detect_regime(self, market_features: Dict) -> str:
        """Detect current market regime"""
        try:
            volatility = market_features.get('volatility', 0.02)
            momentum = market_features.get('price_momentum', 0.0)
            volume_momentum = market_features.get('volume_momentum', 0.0)
            
            regime = self.regime_detector.classify_regime(
                volatility, momentum, volume_momentum, self.regime_history
            )
            
            # Store regime in history
            self.regime_history.append({
                'regime': regime,
                'volatility': volatility,
                'momentum': momentum,
                'volume_momentum': volume_momentum
            })
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown"

    def analyze_liquidity(self, market_data: Dict) -> float:
        """Analyze market liquidity depth"""
        try:
            volumes = market_data.get('volumes', [])
            prices = market_data.get('prices', [])
            
            if len(volumes) < 5 or len(prices) < 5:
                return 0.5  # Neutral liquidity
            
            # Calculate liquidity metrics
            avg_volume = np.mean(volumes[-10:])
            volume_volatility = np.std(volumes[-10:]) / avg_volume if avg_volume > 0 else 1.0
            
            # Price impact analysis
            price_changes = np.diff(prices[-10:])
            volume_ratios = np.array(volumes[-9:]) / avg_volume
            
            # Liquidity score (higher = more liquid)
            liquidity_score = 1.0 / (1.0 + volume_volatility)
            
            # Adjust for consistent volume
            if np.std(volume_ratios) < 0.3:  # Consistent volume
                liquidity_score *= 1.2
            
            liquidity_score = min(1.0, max(0.0, liquidity_score))
            
            # Store liquidity metrics
            self.liquidity_metrics.append({
                'liquidity_score': liquidity_score,
                'avg_volume': avg_volume,
                'volume_volatility': volume_volatility
            })
            
            return liquidity_score
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return 0.5

    def get_microstructure_signal(self, market_data: Dict, market_features: Dict) -> float:
        """Get combined microstructure signal"""
        try:
            flow_signal = self.analyze_order_flow(market_data)
            regime = self.detect_regime(market_features)
            liquidity = self.analyze_liquidity(market_data)
            
            # Regime-based signal adjustment
            regime_multiplier = self._get_regime_multiplier(regime)
            
            # Combine signals with liquidity weighting
            combined_signal = flow_signal * regime_multiplier * liquidity
            
            # Validate signal
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                return 0.0
            
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error getting microstructure signal: {e}")
            return 0.0

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get multiplier based on market regime"""
        multipliers = {
            'trending_up': 1.2,
            'trending_down': 1.1,
            'high_volatility': 0.8,
            'low_volatility': 1.0,
            'ranging': 0.9,
            'breakout': 1.3,
            'unknown': 1.0
        }
        return multipliers.get(regime, 1.0)

    def get_microstructure_stats(self) -> Dict:
        """Get comprehensive microstructure statistics"""
        if not self.order_flow_history:
            return {}
        
        recent_flows = list(self.order_flow_history)[-20:]
        smart_money_scores = [flow['smart_money_score'] for flow in recent_flows]
        
        return {
            'avg_smart_money_score': np.mean(smart_money_scores),
            'order_flow_history_size': len(self.order_flow_history),
            'regime_history_size': len(self.regime_history),
            'current_regime': self.regime_history[-1]['regime'] if self.regime_history else 'unknown',
            'avg_liquidity': np.mean([liq['liquidity_score'] for liq in self.liquidity_metrics]) if self.liquidity_metrics else 0.5
        }