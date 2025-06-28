"""
Market Domain - Market data processing

Public Interface:
- process_market_data: Process incoming market data
- extract_features: Extract trading features from data
"""

from .processor import MarketDataProcessor

def create_market_processor(config):
    """Factory to create configured market processor"""
    return MarketDataProcessor(config)

__all__ = ['MarketDataProcessor', 'create_market_processor']