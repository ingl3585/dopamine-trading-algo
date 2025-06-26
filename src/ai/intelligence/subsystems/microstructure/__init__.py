"""
Market Microstructure Subsystem - Order flow and regime detection

Public Interface:
- analyze_order_flow: Detect smart money vs retail patterns
- detect_regime: Identify market regime changes
- analyze_liquidity: Assess market liquidity depth
"""

from .domain import MarketMicrostructureEngine

# Public API
def analyze_order_flow(engine: MarketMicrostructureEngine, market_data):
    return engine.analyze_order_flow(market_data)

def detect_regime(engine: MarketMicrostructureEngine, market_features):
    return engine.detect_regime(market_features)

def analyze_liquidity(engine: MarketMicrostructureEngine, market_data):
    return engine.analyze_liquidity(market_data)

__all__ = ['MarketMicrostructureEngine', 'analyze_order_flow', 'detect_regime', 'analyze_liquidity']