"""
Market Domain - Market data processing and analysis

Public Interface:
- process_market_data: Process incoming market data
- extract_features: Extract trading features from data
- analyze_microstructure: Microstructure analysis
"""

from .market_data.processor import MarketDataProcessor
from .microstructure.analyzer import MicrostructureAnalyzer

def create_market_processor(config):
    """Factory to create configured market processor"""
    return MarketDataProcessor(config)

__all__ = ['MarketDataProcessor', 'MicrostructureAnalyzer', 'create_market_processor']