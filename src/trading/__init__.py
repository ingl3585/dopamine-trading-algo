"""
Trading Domain - Trade execution and account management

Public Interface:
- execute_trade: Execute trades via NinjaTrader
- get_account_info: Get current account status
- manage_positions: Position management and tracking
"""

from .domain.services import TradingService
from .infrastructure.ninjatrader import NinjaTraderBridge

def create_trading_service(config):
    """Factory to create configured trading service"""
    bridge = NinjaTraderBridge(config)
    return TradingService(bridge)

__all__ = ['TradingService', 'NinjaTraderBridge', 'create_trading_service']