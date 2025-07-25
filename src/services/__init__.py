"""
Service Layer - Business logic and coordination
"""

from .trading_service import TradingService

def create_trading_service(config):
    """Factory to create configured trading service with real NinjaTrader integration"""
    from .trading_service import TradingService
    from src.repositories.ninjatrader_repository import NinjaTraderRepository
    
    # Create NinjaTrader repository for real TCP bridge communication
    ninja_repository = NinjaTraderRepository(config)
    
    # Create trading service with real repository
    trading_service = TradingService(ninja_repository)
    
    # Add convenience methods for account data integration
    def update_account_data(market_data):
        """Update account data from live market feed"""
        if hasattr(ninja_repository, 'last_account_data'):
            ninja_repository.last_account_data = {
                'account_balance': market_data.account_balance,
                'buying_power': market_data.buying_power,
                'total_position_size': market_data.total_position_size,
                'unrealized_pnl': market_data.unrealized_pnl,
                'daily_pnl': market_data.daily_pnl,
                'net_liquidation': market_data.net_liquidation,
                'margin_used': market_data.margin_used,
                'available_margin': market_data.available_margin,
                'open_positions': market_data.open_positions
            }
    
    # Attach convenience method to trading service
    trading_service.update_account_data = update_account_data
    
    return trading_service

__all__ = ['TradingService', 'create_trading_service']