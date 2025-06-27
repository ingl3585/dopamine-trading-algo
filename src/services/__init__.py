"""
Service Layer - Business logic and coordination
"""

from .trading_service import TradingService

def create_trading_service(trading_repository):
    """Factory to create configured trading service"""
    return TradingService(trading_repository)

__all__ = ['TradingService', 'create_trading_service']