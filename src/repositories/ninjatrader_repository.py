"""
NinjaTrader Infrastructure - TCP bridge implementation
"""

import json
import logging
from typing import Dict

from .trading_repository import TradingRepository, ExecutionResult
from src.data_models.trading_domain_models import Trade, Position, Account
from src.communication.tcp_bridge import TCPServer
from src.risk.risk_manager import Order

logger = logging.getLogger(__name__)

class NinjaTraderRepository(TradingRepository):
    """
    NinjaTrader implementation of trading repository using TCP bridge
    """
    
    def __init__(self, config):
        self.config = config
        self.tcp_server = None  # Will be set by orchestrator
        self.last_account_data = None
        self.last_positions = {}
    
    def execute_trade(self, trade: Trade) -> ExecutionResult:
        """Execute trade via NinjaTrader TCP bridge"""
        try:
            logger.info(f"Attempting to execute trade: {trade.action.value} {trade.size} @ {trade.price}")
            
            if not self.tcp_server:
                logger.error("TCP server not available for trade execution")
                return ExecutionResult(success=False, error="TCP server not available")
            
            # Create Order for TCP server
            order = Order(
                action=trade.action.value,
                size=trade.size,
                price=trade.price or 0.0,
                confidence=0.8  # Default confidence
            )
            
            logger.info(f"Sending order to TCP bridge: {order.action} {order.size} @ {order.price}")
            
            # Send signal via TCP server
            success = self.tcp_server.send_signal(order)
            
            if success:
                logger.info(f"Trade successfully sent to NinjaTrader: {trade.action.value} {trade.size}")
                return ExecutionResult(
                    success=True,
                    execution_price=trade.price or 0.0,
                    pnl=0.0,  # PnL calculated later
                    duration=0.0
                )
            else:
                logger.error(f"Failed to send trade signal to NinjaTrader: {trade.action.value} {trade.size}")
                return ExecutionResult(
                    success=False,
                    error="Failed to send signal to NinjaTrader"
                )
                
        except Exception as e:
            logger.error(f"Error executing trade via NinjaTrader: {e}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    def get_account_data(self) -> Account:
        """Get account data from NinjaTrader"""
        try:
            # For now, return default account data
            # In production, this would get data from TCP server
            return Account(
                cash=25000.0,
                buying_power=25000.0,
                total_value=25000.0,
                margin_used=0.0,
                day_trading_buying_power=25000.0
            )
                    
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return Account(
                cash=25000.0,
                buying_power=25000.0,
                total_value=25000.0,
                margin_used=0.0,
                day_trading_buying_power=25000.0
            )
    
    def get_current_positions(self) -> Dict[str, Position]:
        """Get current positions from NinjaTrader"""
        try:
            # For now, return empty positions
            # In production, this would get data from TCP server
            return {}
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_market_data(self) -> Dict:
        """Get market data from NinjaTrader"""
        try:
            # For now, return empty market data
            # In production, this would get data from TCP server
            return {}
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
    
    def set_tcp_server(self, tcp_server):
        """Set the TCP server reference"""
        self.tcp_server = tcp_server