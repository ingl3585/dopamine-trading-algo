"""
Trading Domain - Trade execution and account management

Public Interface:
- execute_trade: Execute trades via NinjaTrader
- get_account_info: Get current account status
- manage_positions: Position management and tracking
"""

from .domain.services import TradingService
from .infrastructure.ninjatrader import NinjaTraderRepository

def create_trading_service(config):
    """Factory to create configured trading service"""
    repository = NinjaTraderRepository(config)
    return TradingService(repository)

__all__ = ['TradingService', 'NinjaTraderRepository', 'create_trading_service']