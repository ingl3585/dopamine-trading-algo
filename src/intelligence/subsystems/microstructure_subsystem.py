"""
Microstructure Intelligence Subsystem - Order flow and regime analysis
"""

import logging
from typing import Dict

from src.market_analysis.microstructure_analyzer import MarketMicrostructureEngine

logger = logging.getLogger(__name__)

class MicrostructureSubsystem:
    """
    Microstructure intelligence subsystem for order flow analysis and regime detection
    """
    
    def __init__(self, config):
        self.config = config
        self.microstructure_engine = MarketMicrostructureEngine()
        
    def analyze_order_flow(self, market_data: Dict) -> float:
        """Analyze order flow patterns"""
        try:
            return self.microstructure_engine.analyze_order_flow(market_data)
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            return 0.0
    
    def detect_regime(self, market_features: Dict) -> str:
        """Detect current market regime"""
        try:
            return self.microstructure_engine.detect_regime(market_features)
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown"
    
    def analyze_liquidity(self, market_data: Dict) -> float:
        """Analyze market liquidity"""
        try:
            return self.microstructure_engine.analyze_liquidity(market_data)
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return 0.5
    
    def get_signal(self, market_data: Dict, market_features: Dict) -> Dict:
        """Get microstructure intelligence signal"""
        try:
            return {
                'order_flow_signal': self.analyze_order_flow(market_data),
                'market_regime': self.detect_regime(market_features),
                'liquidity_score': self.analyze_liquidity(market_data),
                'microstructure_signal': self.microstructure_engine.get_microstructure_signal(
                    market_data, market_features
                ),
                'stats': self.microstructure_engine.get_microstructure_stats()
            }
        except Exception as e:
            logger.error(f"Error in microstructure signal generation: {e}")
            return {
                'order_flow_signal': 0.0,
                'market_regime': 'unknown',
                'liquidity_score': 0.5,
                'microstructure_signal': 0.0,
                'stats': {}
            }