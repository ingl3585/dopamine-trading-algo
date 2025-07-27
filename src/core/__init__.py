# Core module - unified configuration and data processing
from .config_manager import Config, ConfigurationManager
from .trading_system import TradingSystem

# Market data processor available via direct import to avoid numpy dependency issues
# from .market_data_processor import MarketDataProcessor, MarketData

# Additional components available via direct imports
# from .component_integrator import ComponentIntegrator
# from .event_bus import EventBus

__all__ = ['Config', 'ConfigurationManager', 'TradingSystem']